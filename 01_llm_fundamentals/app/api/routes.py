from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.llm_schemas import (
    GenerateTextRequest, 
    GenerateTextResponse,
    GenerateStructuredRequest,
    GenerateStructuredResponse,
    ChatConversationRequest,
    ChatConversationResponse
)
from app.services.llm_service import (
    generate_text_service, 
    stream_text_service,
    generate_structured_service,
    chat_conversation_service
)

router = APIRouter()


@router.post("/generate", response_model=GenerateTextResponse)
async def generate_text_endpoint(request: GenerateTextRequest):
    """
    Executes a standard LLM text generation request.
    """
    try:
        return await generate_text_service(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/generate/stream")
async def stream_text_endpoint(request: GenerateTextRequest):
    """
    Streams the LLM generation response token-by-token using Server-Sent Events (SSE).
    """
    try:
        return StreamingResponse(
            stream_text_service(request), 
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/structured", response_model=GenerateStructuredResponse)
async def generate_structured_endpoint(request: GenerateStructuredRequest):
    """
    Forces the LLM to output a strictly formatted JSON response.
    """
    return await generate_structured_service(request)


@router.post("/chat", response_model=ChatConversationResponse)
async def chat_conversation_endpoint(request: ChatConversationRequest):
    """
    Processes multi-turn conversations by evaluating the full message history.
    """
    return await chat_conversation_service(request)
