from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class CitationBBox(BaseModel):
    """Represents a spatial citation within a specific PDF document page."""
    citation_index: int = Field(..., description="The inline citation number used in the text (e.g. [1]).")
    document_name: str = Field(..., description="The filename of the source document.")
    document_url: Optional[str] = Field(None, description="The URL to access the source document, if available.")
    page_number: int = Field(..., description="The 1-indexed page number where the citation is found.")
    bbox: List[float] = Field(
        ..., 
        description="The bounding box coordinates [x_min, y_min, x_max, y_max] highlighting the cited text.",
        min_length=4,
        max_length=4
    )
    snippet: Optional[str] = Field(None, description="A text snippet of the cited content for context.")

class GraphNode(BaseModel):
    """Represents a node (entity) in the Neo4j Knowledge Graph."""
    id: str = Field(..., description="Unique identifier for the node.")
    name: str = Field(..., description="Display label for the node.")
    type: str = Field(..., description="The category or label of the entity (e.g., User, Genre, Document).")
    val: Optional[float] = Field(1.0, description="The weight or importance value of the node, used for sizing in UI.")

class GraphLink(BaseModel):
    """Represents a directed edge (relationship) between two nodes in the graph."""
    source: str = Field(..., description="The ID of the source node.")
    target: str = Field(..., description="The ID of the target node.")
    label: str = Field(..., description="The relationship type (e.g., LIKES, CONTAINS).")

class GraphPayload(BaseModel):
    """Container for the nodes and links to be visualized by react-force-graph."""
    nodes: List[GraphNode] = Field(default_factory=list, description="List of nodes in the sub-graph.")
    links: List[GraphLink] = Field(default_factory=list, description="List of edges in the sub-graph.")

class ChatResponse(BaseModel):
    """Unified response contract for all Sahayak AI Chat endpoints."""
    message_id: str = Field(..., description="Unique identifier for this specific interaction/message.")
    routed_agent: str = Field(
        ..., 
        description="The LangGraph agent that handled the query (e.g., 'rag_agent', 'recommender_agent', 'counseling_agent')."
    )
    text: str = Field(..., description="The primary markdown-formatted response text.")
    citations: List[CitationBBox] = Field(
        default_factory=list, 
        description="List of bounding box citations for RAG responses."
    )
    graph_data: Optional[GraphPayload] = Field(
        None, 
        description="Sub-graph extraction for visual representation of Recommender or Knowledge Graph responses."
    )
    is_emergency: bool = Field(
        False, 
        description="Flag indicating if the query triggered crisis protocols and requires immediate intervention."
    )
