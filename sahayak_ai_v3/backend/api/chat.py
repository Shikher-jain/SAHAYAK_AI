from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from sahayak_ai_v3.backend.agents.supervisor import sahayak_agent_app

router = APIRouter(prefix="/api/v2/chat", tags=["Agent Chat"])

class ChatRequest(BaseModel):
    message: str
    user_id: str = "guest_user"
    user_tier: str = "free"

class ChatResponse(BaseModel):
    routed_agent: str
    response: str
    emergency_triggered: bool

@router.post("/orchestrate", response_model=ChatResponse)
async def chat_with_supervisor(req: ChatRequest):
    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "next_agent": "",
        "user_id": req.user_id,
        "user_tier": req.user_tier,
        "distress_level": 0.0,
        "emergency_flag": False,
        "final_output": "",
    }
    
    result = await sahayak_agent_app.ainvoke(initial_state)
    
    return ChatResponse(
        routed_agent=result.get("next_agent", "unknown"),
        response=result.get("final_output", ""),
        emergency_triggered=result.get("emergency_flag", False),
    )
