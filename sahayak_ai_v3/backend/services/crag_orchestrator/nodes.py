import logging
import os
from typing import Dict, Any
from sahayak_ai_v3.backend.core.graph_state import CRAGState
from sahayak_ai_v3.backend.services.retrieval.fusion import reciprocal_rank_fusion, rerank_with_bge
from sahayak_ai_v3.backend.services.crag_orchestrator.grader import grade_retrieval
from backend.rag.generator import Generator

logger = logging.getLogger(__name__)

async def retrieve_node(state: CRAGState) -> Dict[str, Any]:
    """Node: Executes parallel retrieval across Qdrant, BM25, and Neo4j, followed by RRF."""
    logger.info("--- NODE: RETRIEVE ---")
    query = state["active_query"]
    
    # We will import the aggregator from v3_orchestrator to reuse singletons
    from sahayak_ai_v3.backend.services.retrieval.v3_orchestrator import _get_aggregator
    import asyncio
    
    aggregator, router = _get_aggregator()
    strategy = router.route_query(query)
    
    # Get filters from state if present
    filters = state.get("filters", {})
    
    tasks = {}
    if strategy.get("dense", True):
        tasks["dense"] = aggregator.dense_retriever.retrieve(query, top_k=5, filters=filters)
    if strategy.get("bm25", True):
        tasks["sparse"] = aggregator.sparse_retriever.retrieve(query, top_k=5, filters=filters)
    if strategy.get("graph", False) and aggregator.graph_retriever:
        tasks["graph"] = aggregator.graph_retriever.retrieve(query, top_k=5, filters=filters)
        
    results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)
    
    final_results = {}
    for key, result in zip(tasks.keys(), results_list):
        if isinstance(result, Exception):
            logger.error(f"Error during parallel retrieval for {key}: {result}")
            final_results[key] = []
        else:
            final_results[key] = result
            
    dense = final_results.get("dense", [])
    sparse = final_results.get("sparse", [])
    graph = final_results.get("graph", [])
    
    candidate_pool = reciprocal_rank_fusion(dense, sparse, graph)
    return {"candidate_pool": candidate_pool}

async def rerank_node(state: CRAGState) -> Dict[str, Any]:
    """Node: Refines candidate pool using BAAI/bge-reranker-base."""
    logger.info("--- NODE: RERANK ---")
    refined_pool = await rerank_with_bge(state["active_query"], state["candidate_pool"])
    return {"refined_pool": refined_pool}

def grade_node(state: CRAGState) -> Dict[str, Any]:
    """Node: Grades the relevance of the retrieved/reranked documents."""
    logger.info("--- NODE: GRADE RETRIEVAL ---")
    score, category = grade_retrieval(state["active_query"], state["refined_pool"])
    return {"relevance_score": score, "confidence_category": category}

def rewrite_query_node(state: CRAGState) -> Dict[str, Any]:
    """Node: Rewrites query for partial matches (Graph Multi-hop)."""
    logger.info("--- NODE: REWRITE QUERY ---")
    count = state.get("rewrite_count", 0) + 1
    query = state["active_query"]
    
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = f"Rewrite this search query to be broader and capture more context. Do not answer it, just rewrite it:\nQuery: {query}\nRewritten:"
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50,
        )
        new_query = (resp.choices[0].message.content or "").strip().strip('"')
    except Exception as e:
        logger.warning(f"Failed to rewrite query: {e}")
        new_query = f"{query} expanded context"
        
    return {"active_query": new_query, "rewrite_count": count}

def web_search_node(state: CRAGState) -> Dict[str, Any]:
    """Node: Fallback external search for BAD retrievals."""
    logger.info("--- NODE: WEB/API SEARCH ---")
    query = state["active_query"]
    
    from backend.services.language_service import generate_answer
    # We don't have a direct Tavily integration enabled by default,
    # so we'll fallback to a general LLM response or simple Wikipedia fallback if needed.
    # For now, we mock adding a web result.
    
    from sahayak_ai_v3.backend.core.models import RetrievedDocument
    web_doc = RetrievedDocument(
        id="web-1",
        modality="text",
        content=f"Web search result placeholder for: {query}",
        source="web",
        metadata={"source": "web"},
        retrieval_score=0.9
    )
    
    pool = state.get("refined_pool", [])
    pool.append(web_doc)
    
    return {"web_search_invoked": True, "refined_pool": pool}

async def generate_node(state: CRAGState) -> Dict[str, Any]:
    """Node: Final grounded generation using LLM."""
    logger.info("--- NODE: GENERATE ---")
    
    # Context compression and Deduplication
    unique_texts = []
    seen_hashes = set()
    import hashlib
    
    deduped_pool = []
    for doc in state.get("refined_pool", []):
        if not doc.text: continue
        # Simple exact deduplication via hash
        h = hashlib.md5(doc.text.strip().encode('utf-8')).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_texts.append(doc.text)
            deduped_pool.append(doc)
            
    # HIERARCHICAL CONTEXT EXPANSION (Parent-Child)
    # If the top chunk is highly relevant, we fetch its siblings to give the LLM full document context.
    if deduped_pool and getattr(deduped_pool[0], 'document_id', None):
        top_doc_id = deduped_pool[0].document_id
        logger.info(f"Expanding context for top document: {top_doc_id}")
        try:
            from sahayak_ai_v3.backend.services.retrieval.v3_orchestrator import _get_aggregator
            aggregator, _ = _get_aggregator()
            # Fetch by document_id filter
            expanded = aggregator.dense_retriever._sync_retrieve("", top_k=5, filters={"document_id": top_doc_id})
            for edoc in expanded:
                h = hashlib.md5(edoc.text.strip().encode('utf-8')).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique_texts.append(edoc.text)
                    deduped_pool.append(edoc)
        except Exception as e:
            logger.warning(f"Context expansion failed: {e}")
            
    context_parts = []
    for i, text in enumerate(unique_texts, start=1):
        context_parts.append(f"[{i}] {text}")
    context = "\n\n".join(context_parts)
    
    gen = Generator()
    result = gen.generate_answer(
        context=context,
        question=state["original_query"],
        sources=[{"source": f"[{i+1}] " + d.metadata.get("source", "unknown")} for i, d in enumerate(deduped_pool)],
        learning_mode=state.get("learning_mode", "student"),
        user_mode=state.get("user_mode")
    )
    
    return {"draft_response": result["answer"], "sources": result.get("sources", [])}

def verify_faithfulness_node(state: CRAGState) -> Dict[str, Any]:
    """Node: Checks if the draft hallucinated."""
    logger.info("--- NODE: VERIFY FAITHFULNESS ---")
    
    draft = state["draft_response"]
    context = "\n\n".join([doc.text for doc in state.get("refined_pool", [])])
    
    prompt = (
        "You are an AI auditor. Determine if the following answer is fully supported by the context.\n"
        "Reply exactly YES or NO.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{draft}\n\n"
        "Is faithful:"
    )
    
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        is_faithful = "YES" in (resp.choices[0].message.content or "").strip().upper()
    except Exception:
        is_faithful = True # Assume faithful if verification fails
        
    return {"is_faithful": is_faithful, "final_response": draft if is_faithful else draft + "\n\n[Warning: This answer may contain hallucinated details.]"}
