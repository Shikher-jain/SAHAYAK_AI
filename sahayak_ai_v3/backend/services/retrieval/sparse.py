import logging
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import numpy as np

from sahayak_ai_v3.backend.services.retrieval.interfaces import SparseRetriever
from sahayak_ai_v3.backend.core.models import RetrievedDocument
from backend.vector_store.qdrant_store import qdrant_store

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

logger = logging.getLogger(__name__)

class BM25SparseRetriever(SparseRetriever):
    """
    Lightweight Sparse Retriever using BM25.
    Fetches chunks from Qdrant on initialization (up to a limit to respect 512MB RAM limit).
    """
    def __init__(self, max_docs: int = 5000):
        self.bm25 = None
        self.documents: List[Dict[str, Any]] = []
        
        if not BM25Okapi:
            logger.warning("rank_bm25 is not installed. Sparse retrieval will return empty.")
            return
            
        try:
            # We initialize by scrolling Qdrant to get all payloads
            if qdrant_store.is_available:
                points, _ = qdrant_store._client.scroll(
                    collection_name=qdrant_store.collection_name,
                    limit=max_docs,
                    with_payload=True,
                    with_vectors=False
                )
                self.documents = [
                    {"id": str(p.id), "payload": p.payload or {}} 
                    for p in points if p.payload and "content" in p.payload
                ]
                
                if self.documents:
                    tokenized_corpus = [doc["payload"]["content"].lower().split() for doc in self.documents]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    logger.info(f"BM25 initialized with {len(self.documents)} documents.")
                else:
                    logger.info("No documents found in Qdrant for BM25 initialization.")
        except Exception as e:
            logger.error(f"Failed to initialize BM25 index from Qdrant: {e}")

    async def retrieve(
        self,
        query: str,
        top_k: int = 50,
        filters: Optional[Dict] = None,
    ) -> List[RetrievedDocument]:
        
        if not self.bm25 or not self.documents:
            return []
            
        logger.info(f"BM25 Sparse search called for: {query}")
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Apply filters by zeroing out scores for non-matching documents
        if filters:
            for i, doc in enumerate(self.documents):
                payload = doc["payload"]
                match = True
                for k, v in filters.items():
                    if payload.get(k) != v:
                        match = False
                        break
                if not match:
                    doc_scores[i] = -1.0 # Ensure it gets pushed to the bottom
        
        # Get top_k indices
        top_n = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_n):
            score = doc_scores[idx]
            if score <= 0:
                continue
            doc = self.documents[idx]
            payload = doc["payload"]
            results.append(RetrievedDocument(
                id=doc.get("id", str(i)),
                modality=doc.get("modality", "text"),
                content=doc.get("content", ""),
                source=doc.get("source", "unknown"),
                sparse_score=float(score),
                metadata=doc
            ))
            
        return results
