from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from sahayak_ai_v3.backend.core.models import RetrievedDocument

class DenseRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 50,
        filters: Optional[Dict] = None,
    ) -> List[RetrievedDocument]:
        """
        Perform dense vector retrieval.
        Returns a list of RetrievedDocument objects with `retrieval_score` populated.
        """
        pass

class SparseRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 50,
        filters: Optional[Dict] = None,
    ) -> List[RetrievedDocument]:
        """
        Perform sparse keyword/BM25 retrieval.
        Returns a list of RetrievedDocument objects with `retrieval_score` populated.
        """
        pass

class GraphRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[RetrievedDocument]:
        """
        Perform graph-based retrieval traversing relationships.
        """
        pass
