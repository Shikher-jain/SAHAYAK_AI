from typing import List, Dict
from sahayak_ai_v3.backend.core.models import RetrievedDocument
import logging

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(
    dense_results: List[RetrievedDocument],
    sparse_results: List[RetrievedDocument],
    graph_results: List[RetrievedDocument],
    k: int = 60
) -> List[RetrievedDocument]:
    """
    Fuses results from Qdrant, Elasticsearch, and Neo4j using Reciprocal Rank Fusion.
    Score = 1 / (k + rank)
    """
    logger.info("Executing Reciprocal Rank Fusion (RRF) across Dense, Sparse, and Graph results.")
    
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, RetrievedDocument] = {}
    
    for rank, chunk in enumerate(dense_results):
        if chunk.id not in rrf_scores:
            rrf_scores[chunk.id] = 0.0
            chunk_map[chunk.id] = chunk
        rrf_scores[chunk.id] += 1.0 / (k + rank + 1)
        
    for rank, chunk in enumerate(sparse_results):
        if chunk.id not in rrf_scores:
            rrf_scores[chunk.id] = 0.0
            chunk_map[chunk.id] = chunk
        rrf_scores[chunk.id] += 1.0 / (k + rank + 1)
        
    for rank, chunk in enumerate(graph_results):
        if chunk.id not in rrf_scores:
            rrf_scores[chunk.id] = 0.0
            chunk_map[chunk.id] = chunk
        rrf_scores[chunk.id] += 1.0 / (k + rank + 1)
        
    # Sort chunks by RRF score descending
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    
    final_results = []
    for cid in sorted_chunk_ids:
        chunk = chunk_map[cid]
        chunk.rrf_score = rrf_scores[cid]
        final_results.append(chunk)
        
    return final_results

async def rerank_with_bge(query: str, candidate_pool: List[RetrievedDocument], top_n: int = 15) -> List[RetrievedDocument]:
    """
    Executes Cross-Encoder reranking using BAAI/bge-reranker-base.
    This step refines the Top 30-100 candidates down to a highly relevant Top 10-20.
    """
    logger.info(f"Reranking candidate pool of size {len(candidate_pool)} using BAAI/bge-reranker-base.")
    
    if not candidate_pool:
        return []
        
    from sahayak_ai_v3.backend.services.external.hf_client import hf_client
    
    # Reranker only supports text
    text_docs = [chunk.text for chunk in candidate_pool if chunk.text]
    
    if not text_docs:
        return candidate_pool[:top_n]
        
    results = await hf_client.rerank(query, text_docs, top_k=top_n)
    
    # Map back scores to RetrievedDocuments
    # Hf client returns: [{"document": str, "score": float}]
    doc_to_score = {r["document"]: r["score"] for r in results}
    
    reranked = []
    for chunk in candidate_pool:
        if chunk.text in doc_to_score:
            chunk.rerank_score = doc_to_score[chunk.text]
            reranked.append(chunk)
            
    # Sort by rerank score
    reranked.sort(key=lambda d: d.rerank_score or -999.0, reverse=True)
    
    return reranked[:top_n]
