from typing import List, Dict, Any, Optional
import asyncio
import logging

from sahayak_ai_v3.backend.services.retrieval.qdrant import QdrantDenseRetriever
from sahayak_ai_v3.backend.services.retrieval.sparse import BM25SparseRetriever
from sahayak_ai_v3.backend.services.retrieval.graph import Neo4jGraphRetriever
from sahayak_ai_v3.backend.services.retrieval.aggregator import RetrievalAggregator
from sahayak_ai_v3.backend.services.retrieval.router import QueryRouter
from sahayak_ai_v3.backend.services.retrieval.fusion import reciprocal_rank_fusion, rerank_with_bge
from sahayak_ai_v3.backend.services.crag_orchestrator.grader import grade_retrieval

logger = logging.getLogger(__name__)

# Singletons for retrievers (BM25 takes time to init, so we keep it globally)
_dense_retriever = None
_sparse_retriever = None
_graph_retriever = None
_aggregator = None
_router = None

def _get_aggregator():
    global _dense_retriever, _sparse_retriever, _graph_retriever, _aggregator, _router
    if not _aggregator:
        _dense_retriever = QdrantDenseRetriever()
        _sparse_retriever = BM25SparseRetriever()
        _graph_retriever = Neo4jGraphRetriever()
        _aggregator = RetrievalAggregator(_dense_retriever, _sparse_retriever, _graph_retriever)
        _router = QueryRouter()
    return _aggregator, _router

async def v3_retrieve_fused(query: str, top_k: int = 5, target: str = "auto", filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Executes the full V3 Multimodal Hybrid CRAG + GraphRAG retrieval pipeline.
    Returns standard list of dicts for backward compatibility.
    """
    from sahayak_ai_v3.backend.services.retrieval.decomposition import QueryDecomposer
    aggregator, router = _get_aggregator()
    decomposer = QueryDecomposer()
    
    # 1. Route Query and Decompose
    strategy = router.route_query(query)
    sub_queries = decomposer.decompose(query)
    
    # We will gather all sub-query results and merge them
    tasks = {}
    
    for idx, sq in enumerate(sub_queries):
        # Base retrievals
        if strategy.get("dense", True):
            tasks[f"dense_{idx}"] = aggregator.dense_retriever.retrieve(sq, top_k=top_k, filters=filters)
        if strategy.get("bm25", True):
            tasks[f"sparse_{idx}"] = aggregator.sparse_retriever.retrieve(sq, top_k=top_k, filters=filters)
        if strategy.get("graph", False) and aggregator.graph_retriever:
            tasks[f"graph_{idx}"] = aggregator.graph_retriever.retrieve(sq, top_k=top_k, filters=filters)
            
        # Modality-specific targeted retrievals
        targeted_filters = dict(filters or {})
        if strategy.get("code", False):
            targeted_filters["modality"] = "code"
            tasks[f"dense_code_{idx}"] = aggregator.dense_retriever.retrieve(sq, top_k=top_k, filters=targeted_filters)
        if strategy.get("table", False):
            targeted_filters["modality"] = "csv"
            tasks[f"dense_table_{idx}"] = aggregator.dense_retriever.retrieve(sq, top_k=top_k, filters=targeted_filters)
        if strategy.get("video", False):
            targeted_filters["modality"] = "video"
            tasks[f"dense_video_{idx}"] = aggregator.dense_retriever.retrieve(sq, top_k=top_k, filters=targeted_filters)
        if strategy.get("image", False):
            targeted_filters["modality"] = "image"
            tasks[f"dense_image_{idx}"] = aggregator.dense_retriever.retrieve(sq, top_k=top_k, filters=targeted_filters)
            
    results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)
    
    final_results = {}
    dense = []
    sparse = []
    graph = []
    
    for key, result in zip(tasks.keys(), results_list):
        if isinstance(result, Exception):
            logger.error(f"Error during parallel retrieval for {key}: {result}")
        else:
            if key.startswith("dense"):
                dense.extend(result)
            elif key.startswith("sparse"):
                sparse.extend(result)
            elif key.startswith("graph"):
                graph.extend(result)
    
    # 2. Fusion
    fused = reciprocal_rank_fusion(dense, sparse, graph)
    
    # 3. Rerank
    reranked = await rerank_with_bge(query, fused, top_n=top_k)
    
    # 4. Grade (CRAG)
    # grade_retrieval(query, reranked) # Can be used later for routing
    
    # Convert RetrievedDocument back to dict for legacy compatibility
    out = []
    for doc in reranked:
        out.append({
            "id": doc.id,
            "score": doc.rerank_score or doc.rrf_score or doc.retrieval_score or 0.0,
            "metadata": doc.metadata or {},
            "content": doc.text
        })
    return out

_crag_workflow = None

def _get_crag_workflow():
    global _crag_workflow
    if not _crag_workflow:
        from sahayak_ai_v3.backend.services.crag_orchestrator.workflow import build_crag_workflow
        _crag_workflow = build_crag_workflow()
    return _crag_workflow

async def v3_rag_answer(query: str, filters: dict = None, learning_mode: str = "student", user_mode: str = None) -> dict:
    """
    Executes the full CRAG state machine for generating an answer.
    """
    workflow = _get_crag_workflow()
    
    initial_state = {
        "original_query": query,
        "active_query": query,
        "rewrite_count": 0,
        "web_search_invoked": False,
        "filters": filters or {},
        "learning_mode": learning_mode,
        "user_mode": user_mode
    }
    
    # Run the graph asynchronously
    final_state = await workflow.ainvoke(initial_state)
    
    context = "\n\n".join([doc.text for doc in final_state.get("refined_pool", [])])
    
    return {
        "answer": final_state.get("final_response", ""),
        "sources": final_state.get("sources", []),
        "context": context,
        "is_faithful": final_state.get("is_faithful", True),
        "debug_info": {
            "rewrite_count": final_state.get("rewrite_count", 0),
            "web_search_invoked": final_state.get("web_search_invoked", False),
            "confidence_category": final_state.get("confidence_category", "UNKNOWN")
        }
    }
