from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import datetime

class EvidenceUnit(BaseModel):
    id: str
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    
    modality: str
    content: str
    
    source: str
    source_type: Optional[str] = None
    
    parent_id: Optional[str] = None
    chunk_id: Optional[str] = None
    
    page: Optional[int] = None
    section: Optional[str] = None
    
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    
    bbox: Optional[list[float]] = None
    
    language: Optional[str] = None
    embedding_id: Optional[str] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    created_at: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())

class RetrievedDocument(EvidenceUnit):
    """
    Extends EvidenceUnit with retrieval-specific scoring metrics.
    Keeping the name 'RetrievedDocument' for backward compatibility.
    """
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    graph_score: Optional[float] = None
    rrf_score: Optional[float] = None
    reranker_score: Optional[float] = None
    
    @property
    def text(self) -> str:
        # Alias for backward compatibility
        return self.content
