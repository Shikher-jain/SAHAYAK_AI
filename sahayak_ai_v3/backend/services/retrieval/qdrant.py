from typing import List, Dict, Optional
import numpy as np

from sahayak_ai_v3.backend.services.retrieval.interfaces import DenseRetriever
from sahayak_ai_v3.backend.core.models import RetrievedDocument

# Import the existing legacy Qdrant store to reuse working code
from backend.vector_store.qdrant_store import qdrant_store

class QdrantDenseRetriever(DenseRetriever):
    """
    Wraps the existing QdrantStore to implement the DenseRetriever interface
    without breaking legacy endpoints.
    """
    async def retrieve(
        self,
        query: str,
        top_k: int = 50,
        filters: Optional[Dict] = None,
    ) -> List[RetrievedDocument]:
        
        # In a real setup, we need the query embedding here.
        # For this wrapper, we assume the query embedding is injected, 
        # or we fetch it via the embedder.
        from backend.common.embedder import embed_text
        
        try:
            embedding = embed_text(query)
            
            # The legacy search method uses qdrant_client query_points
            hits = qdrant_store.search(embedding, top_k=top_k, filters=filters)
            
            results = []
            for hit in hits:
                # hit format: {"id": str, "score": float, "metadata": dict, "content": str}
                metadata = hit.get("metadata", {})
                doc = RetrievedDocument(
                    id=hit["id"],
                    modality=metadata.get("modality", "text"),
                    content=hit.get("content", ""),
                    source=metadata.get("source", "unknown"),
                    dense_score=hit.get("score", 0.0),
                    metadata=metadata
                )
                results.append(doc)
                
            return results
        except Exception as e:
            # Fallback or empty on failure to ensure degradation
            import logging
            logging.getLogger(__name__).warning(f"QdrantDenseRetriever failure: {e}")
            return []
