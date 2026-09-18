"""Typed state + routing contract shared across the canonical sub-agent package.

`AgentState` is langgraph-native (`operator.add` reducer on messages), one shared
schema that the supervisor, rag_agent, counseling_agent and recommender_agent
all read/write through — the unified typed contract the plan doc mandates
(no more per-module re-declarations that drifted apart).
"""
import operator
from typing import Annotated, Dict, List, Literal, TypedDict

from pydantic import BaseModel, Field

AGENT_NAMES = ("counseling_agent", "recommender_agent", "rag_agent")


class AgentState(TypedDict):
    """Unified state flowing across the supervisor and specialized sub-agents."""
    messages: Annotated[List[Dict[str, str]], operator.add]
    next_agent: Literal["counseling_agent", "recommender_agent", "rag_agent", "FINISH"]
    user_id: str
    user_tier: str                              # "free" | "premium"
    distress_level: float                       # 0.0 - 1.0 (safety guard for counseling)
    emergency_flag: bool                        # HITL interrupt trigger
    final_output: str
    citations: List[dict]                       # visual citations from the rag agent


class RouteDecision(BaseModel):
    """Structured runtime route decision returned by the supervisor."""
    next_agent: Literal["counseling_agent", "recommender_agent", "rag_agent", "FINISH"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
