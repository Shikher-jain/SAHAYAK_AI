import os
from typing import Annotated, Literal, Sequence
import operator
import logging

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from sahayak_ai_v3.backend.agents.graph_state import GraphState

logger = logging.getLogger(__name__)

# =====================================================================
# 1. Supervisor Routing Schema & Node
# =====================================================================

class RouteDecision(BaseModel):
    """Structured decision returned by the supervisor."""
    next_agent: Literal["rag_agent", "counseling_agent", "recommender_agent", "FINISH"] = Field(
        description="The target worker agent best suited to answer the user's intent."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Routing confidence score.")
    rationale: str = Field(description="Brief reason for routing choice.")

def get_llm():
    """Initializes a lightweight remote LLM client via Groq (Zero local RAM)."""
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY", "dummy_key_for_testing"),
        temperature=0.0,
    )

async def supervisor_node(state: GraphState) -> dict:
    """Analyzes intent and routes to RAG, Counseling, or Recommender."""
    messages = state.get("messages", [])
    if not messages:
        return {"next_agent": "FINISH"}
        
    last_user_message = messages[-1].content.lower()

    # Pre-screen for severe distress to avoid LLM latency/failures
    distress_keywords = ["suicide", "kill myself", "want to die", "end it all"]
    if any(keyword in last_user_message for keyword in distress_keywords):
        return {"next_agent": "counseling_agent", "is_emergency": True, "iteration_count": state.get("iteration_count", 0) + 1}

    # Circuit Breaker: prevent infinite recursion
    iteration_count = state.get("iteration_count", 0) + 1
    if iteration_count >= 3:
        return {"next_agent": "FINISH", "iteration_count": iteration_count, "error": "MAX_RECURSION_REACHED"}

    try:
        llm = get_llm().with_structured_output(RouteDecision)
        system_prompt = (
            "You are the Central Supervisor for Sahayak AI.\n"
            "Your role is to classify the user's latest message and route it to ONE worker:\n"
            "- 'counseling_agent': Emotional support, mental wellness, life guidance, active listening, venting.\n"
            "- 'recommender_agent': Suggestions for books, movies, courses, tech stacks, or user preferences.\n"
            "- 'rag_agent': Factual, technical, documentation, PDF/Audio/Video analysis, or project queries.\n"
            "- 'FINISH': Only if the user says pure pleasantries like 'bye' or 'thank you'."
        )
        
        input_messages = [SystemMessage(content=system_prompt)] + list(messages)
        decision: RouteDecision = await llm.ainvoke(input_messages)
        
        return {"next_agent": decision.next_agent, "iteration_count": iteration_count}
    except Exception as e:
        logger.error(f"Supervisor LLM error: {e}")
        # Fallback to rag_agent if Groq fails
        return {"next_agent": "rag_agent", "iteration_count": iteration_count, "error": "SUPERVISOR_ROUTING_FAILED"}

# =====================================================================
# 2. Specialized Sub-Agent Nodes
# =====================================================================

from .counseling_agent import counseling_agent as counseling_agent_node

from .recommender_agent import recommender_agent as recommender_agent_node

from .rag_agent import rag_agent as rag_agent_node

# =====================================================================
# 3. Routing Logic & StateGraph Assembly
# =====================================================================

def route_next(state: GraphState) -> str:
    """Routes state based on supervisor decision or safety flags."""
    if state.get("is_emergency") or state.get("iteration_count", 0) >= 3:
        return END
    
    target = state.get("next_agent")
    if target in ["counseling_agent", "recommender_agent", "rag_agent"]:
        return target
    return END

def build_sahayak_supervisor_graph():
    """Compiles the LangGraph Multi-Agent network."""
    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("counseling_agent", counseling_agent_node)
    workflow.add_node("recommender_agent", recommender_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "counseling_agent": "counseling_agent",
            "recommender_agent": "recommender_agent",
            "rag_agent": "rag_agent",
            END: END,
        }
    )

    # All sub-agents terminate directly at END
    workflow.add_edge("counseling_agent", END)
    workflow.add_edge("recommender_agent", END)
    workflow.add_edge("rag_agent", END)

    return workflow.compile()

sahayak_agent_app = build_sahayak_supervisor_graph()
