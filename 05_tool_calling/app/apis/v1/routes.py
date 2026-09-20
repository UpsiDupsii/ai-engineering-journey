from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from app.services.llm_service import llm_service

router = APIRouter(prefix="/api/v1")


class ChatRequest(BaseModel):
    prompt: str
    user_role: Literal["guest", "admin"] = Field(
        default="guest", 
        description="The role of the user making the request."
    )


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Routes a user prompt through the LLM tool-calling loop with RBAC."""
    try:
        response_text = await llm_service.chat_with_tools(
            user_prompt=request.prompt,
            user_role=request.user_role
        )
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))