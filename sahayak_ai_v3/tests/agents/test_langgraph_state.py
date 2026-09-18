import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END

# Import the graph and State
from sahayak_ai_v3.backend.agents.supervisor import sahayak_agent_app, RouteDecision
from sahayak_ai_v3.backend.agents.graph_state import GraphState

@pytest.fixture
def mock_supervisor_llm():
    """Patches get_llm globally in supervisor.py to avoid actual Groq calls."""
    with patch("sahayak_ai_v3.backend.agents.supervisor.get_llm") as mock_get:
        # Create a mock LLM client
        mock_llm_instance = AsyncMock()
        
        # When with_structured_output is called, return another mock
        mock_structured_llm = AsyncMock()
        # Make ainvoke return a deterministic routing decision
        mock_structured_llm.ainvoke.return_value = RouteDecision(
            next_agent="rag_agent", 
            confidence=0.99, 
            rationale="Test"
        )
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm
        
        # Make plain ainvoke return a dummy AIMessage for the sub-agents
        mock_llm_instance.ainvoke.return_value = AIMessage(content="Mocked LLM Response")
        
        mock_get.return_value = mock_llm_instance
        yield mock_get

@pytest.mark.asyncio
async def test_message_reducer_accumulation(mock_supervisor_llm):
    """
    Action: Invoke the graph with an initial state containing one HumanMessage.
    Assertion: Verify the final state's messages list has a length of exactly 2,
               proving add_messages appended without overwriting.
    """
    initial_state = {
        "messages": [HumanMessage(content="What is Sahayak?")],
        "iteration_count": 0
    }
    
    # We invoke the graph. Supervisor routes to rag_agent (via our mock).
    # rag_agent processes and returns its AIMessage.
    final_state = await sahayak_agent_app.ainvoke(initial_state)
    
    # Check that messages have accumulated correctly
    messages = final_state.get("messages", [])
    assert len(messages) == 2, "Messages were overwritten; expected length 2."
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert "Mocked LLM Response" in messages[1].content or "I encountered an issue" in messages[1].content

@pytest.mark.asyncio
async def test_supervisor_circuit_breaker(mock_supervisor_llm):
    """
    Action: Initialize the state with iteration_count = 3 (the configured limit).
    Assertion: Verify conditional edge routes to END and populates MAX_RECURSION_REACHED.
    """
    initial_state = {
        "messages": [HumanMessage(content="Trigger recursion loop.")],
        "iteration_count": 3
    }
    
    # Run the graph
    final_state = await sahayak_agent_app.ainvoke(initial_state)
    
    # Since iteration_count was 3, the supervisor should hit the circuit breaker immediately.
    assert final_state.get("error") == "MAX_RECURSION_REACHED"
    # The supervisor's route_next logic states that if iteration >= 3, it routes to END.
    # Therefore, next_agent is returned as "FINISH" and no sub-agents are executed.
    assert final_state.get("next_agent") == "FINISH"

@pytest.mark.asyncio
async def test_emergency_distress_routing(mock_supervisor_llm):
    """
    Action: Send a HumanMessage containing crisis keywords.
    Assertion: Assert supervisor intercepts without calling ChatGroq and routes to counseling_agent.
               Verify helpline info is in the final response.
    """
    initial_state = {
        "messages": [HumanMessage(content="I can't take this anymore, I want to end it all")],
        "iteration_count": 0
    }
    
    # Run the graph
    final_state = await sahayak_agent_app.ainvoke(initial_state)
    
    # The keyword should trigger the pre-screen, bypassing the LLM in supervisor.
    # We can check if `with_structured_output().ainvoke()` was NOT called.
    mock_instance = mock_supervisor_llm.return_value
    mock_structured = mock_instance.with_structured_output.return_value
    mock_structured.ainvoke.assert_not_called()
    
    # The supervisor sets is_emergency = True
    assert final_state.get("is_emergency") is True
    
    # Check that the counseling agent returned the hardcoded helpline numbers
    messages = final_state.get("messages", [])
    last_msg = messages[-1]
    assert isinstance(last_msg, AIMessage)
    assert "Tele-MANAS:" in last_msg.content
    assert "14416" in last_msg.content
