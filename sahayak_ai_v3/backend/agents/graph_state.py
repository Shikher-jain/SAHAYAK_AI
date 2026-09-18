from typing import Annotated, Sequence, Optional, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """Unified state flowing across supervisor and specialized sub-agents."""
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    visual_citations: list[dict[str, Any]]
    graph_data: Optional[dict[str, Any]]
    is_emergency: bool
    error: Optional[str]
    user_id: str
    tier: str
    iteration_count: int
