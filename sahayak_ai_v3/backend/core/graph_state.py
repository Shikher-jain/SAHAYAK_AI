import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from sahayak_ai_v3.backend.core.models import RetrievedDocument

class CRAGState(TypedDict):
    """
    Represents the state of the Corrective RAG (CRAG) workflow at any given node.
    """
    # 1. Query Understanding Phase
    original_query: str
    active_query: str                  # The query currently being executed (can be original or rewritten)
    intent: Optional[str]
    detected_modality: Optional[str]
    extracted_entities: List[str]
    
    # 2. Retrieval & Reranking Phase
    candidate_pool: List[RetrievedDocument]  # Top 30-100 chunks from RRF (Qdrant + BM25 + Neo4j)
    refined_pool: List[RetrievedDocument]    # Top 10-20 chunks after cross-encoder (BAAI/bge-reranker-base)
    
    # 3. CRAG Grading Phase
    relevance_score: Optional[float]
    confidence_category: Optional[str]     # "GOOD" (>=0.85), "PARTIAL" (0.60-0.84), "BAD" (<0.60)
    
    # 4. Corrective Action Tracking (To prevent infinite loops)
    rewrite_count: int                     # Track how many times we've rewritten the query
    web_search_invoked: bool               # Flag if external fallback was required
    
    # 5. Context Engine Phase
    compressed_context: Optional[str]      # Final string passed to the LLM after deduplication/budgeting
    
    # 6. Generation & Faithfulness Phase
    draft_response: Optional[str]
    is_faithful: Optional[bool]            # Did the LLM hallucinate or stick to the context?
    final_response: Optional[str]          # The verified output sent to the client
    
    # 7. Metadata
    filters: Optional[Dict[str, Any]]
    sources: Optional[List[Dict[str, str]]]
    learning_mode: Optional[str]
    user_mode: Optional[str]
