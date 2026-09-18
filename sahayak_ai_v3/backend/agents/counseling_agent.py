import os
import logging
from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

from sahayak_ai_v3.backend.agents.graph_state import GraphState

logger = logging.getLogger("sahayak.counseling_agent")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

CRISIS_HELPLINES = (
    "\n\n***\n"
    "**Immediate Support is Available:**\n"
    "If you are feeling overwhelmed, hopeless, or in danger, please reach out to these free, confidential resources in India right now:\n"
    "• **Tele-MANAS:** 14416 or 1800-891-4416 (24/7)\n"
    "• **KIRAN Mental Health Helpline:** 1800-599-0019\n"
    "• **AASRA:** 9820466726\n"
    "You don't have to go through this alone.\n"
    "***\n"
)

class TriageResult(BaseModel):
    distress_score: float = Field(
        description="Score from 0.0 to 1.0 indicating emotional distress, hopelessness, or self-harm risk. 1.0 is highest risk."
    )
    is_crisis: bool = Field(
        description="Set to True if the user expresses self-harm, severe trauma, or explicit crisis."
    )

async def _evaluate_distress(query: str) -> TriageResult:
    """Uses a lightweight LLM call to score the distress level of the user's message."""
    try:
        # Using llama-3.1-8b-instant for lower latency on triage
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, max_retries=1)
        structured_llm = llm.with_structured_output(TriageResult)
        
        prompt = (
            "Analyze the following user message for emotional distress, self-harm, or crisis intent. "
            f"Message: '{query}'"
        )
        result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
        return result
    except Exception as exc:
        logger.error("Distress evaluation failed: %s. Defaulting to crisis protocol.", exc)
        # Fail-safe: Assume crisis if the safety classifier goes offline
        return TriageResult(distress_score=1.0, is_crisis=True)

async def counseling_agent(state: GraphState) -> dict[str, Any]:
    """LangGraph node: Empathetic response generation and crisis intervention."""
    
    query_text = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            query_text = str(msg.content)
            break

    if not query_text.strip():
        return {"messages": [AIMessage(content="I am here to listen. How are you feeling today?")]}

    # 1. Check for preemptive emergency flag from supervisor, or run internal triage
    is_emergency = state.get("is_emergency", False)
    if not is_emergency:
        triage = await _evaluate_distress(query_text)
        is_emergency = triage.is_crisis or triage.distress_score >= 0.8

    # 2. Formulate Response Constraints
    if is_emergency:
        system_prompt = (
            "You are a compassionate, grounded AI assistant. The user is experiencing severe distress or crisis. "
            "Your goal is to de-escalate, validate their pain, and gently encourage them to seek professional help. "
            "Keep your response warm, concise, and non-judgmental. Do NOT provide medical diagnosis or toxic positivity."
        )
    else:
        system_prompt = (
            "You are an empathetic, supportive AI companion. Validate the user's feelings, offer gentle reframing "
            "or grounding techniques if they are stressed, and maintain a warm, conversational tone. "
            "Do not act as a licensed therapist or offer medical advice."
        )

    # 3. Generate Empathetic Response
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4, max_retries=2)
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_text}
        ])
        final_answer = str(response.content)
    except Exception as exc:
        logger.error("Groq generation failed in counseling agent: %s", exc)
        final_answer = "I'm having a little trouble connecting right now, but I want you to know I'm here and listening."

    # 4. Append crisis helplines if emergency is detected
    if is_emergency:
        final_answer += CRISIS_HELPLINES

    return {
        "messages": [AIMessage(content=final_answer)],
        "is_emergency": is_emergency
    }
