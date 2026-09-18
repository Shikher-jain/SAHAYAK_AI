import asyncio
import logging
from typing import List, Dict, Optional
from sahayak_ai_v3.backend.core.models import RetrievedDocument
from sahayak_ai_v3.backend.services.retrieval.interfaces import DenseRetriever, SparseRetriever, GraphRetriever

logger = logging.getLogger(__name__)

class RetrievalAggregator:
    """
    Executes multiple retrieval strategies in parallel to minimize latency.
    """
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        graph_retriever: Optional[GraphRetriever] = None
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.graph_retriever = graph_retriever

    async def retrieve_all(
        self,
        query: str,
        top_k: int = 50,
        filters: Optional[Dict] = None
    ) -> Dict[str, List[RetrievedDocument]]:
        """
        Runs configured retrievers concurrently.
        """
        logger.info(f"Executing parallel retrieval for query: {query}")
        
        tasks = {
            "dense": self.dense_retriever.retrieve(query, top_k, filters),
            "sparse": self.sparse_retriever.retrieve(query, top_k, filters)
        }
        
        if self.graph_retriever:
            tasks["graph"] = self.graph_retriever.retrieve(query, top_k)
            
        results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        final_results = {}
        for key, result in zip(tasks.keys(), results_list):
            if isinstance(result, Exception):
                logger.error(f"Retriever {key} failed: {result}")
                final_results[key] = []
            else:
                final_results[key] = result
                
        return final_results
