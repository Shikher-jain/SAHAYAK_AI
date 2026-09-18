import logging
from langgraph.graph import StateGraph, END
from sahayak_ai_v3.backend.core.graph_state import CRAGState
from sahayak_ai_v3.backend.services.crag_orchestrator.nodes import (
    retrieve_node, rerank_node, grade_node, 
    rewrite_query_node, web_search_node, 
    generate_node, verify_faithfulness_node
)

logger = logging.getLogger(__name__)

def route_after_grading(state: CRAGState):
    """
    Conditional Edge: Routes flow based on the CRAG grader category.
    """
    category = state.get("confidence_category")
    logger.info(f"Routing based on grade: {category}")
    
    # Enforce MAX_CRAG_ITERATIONS (prevent infinite loop)
    if state.get("web_search_invoked", False):
        return "generate"
    
    if category == "GOOD":
        return "generate"
    elif category == "PARTIAL":
        if state.get("rewrite_count", 0) > 1:
            return "web_search"  # Prevent infinite rewrite loops
        return "rewrite"
    else: # BAD
        return "web_search"

def build_crag_workflow() -> StateGraph:
    """
    Compiles the complete CRAG LangGraph state machine exactly matching
    the architectural blueprint.
    """
    workflow = StateGraph(CRAGState)

    # 1. Define Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_faithfulness_node)

    # 2. Define Edges (The flow)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade")

    # Conditional Routing based on Grade
    workflow.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "web_search": "web_search"
        }
    )

    # Recovery Edges
    workflow.add_edge("rewrite", "retrieve")        # Re-retrieve after rewrite
    workflow.add_edge("web_search", "retrieve")     # Re-retrieve after external search

    # Final Generation flow
    workflow.add_edge("generate", "verify")
    workflow.add_edge("verify", END)

    logger.info("Compiled CRAG LangGraph Workflow.")
    return workflow.compile()
