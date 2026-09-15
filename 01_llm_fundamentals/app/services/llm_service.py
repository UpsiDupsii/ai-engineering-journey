import json
import time
from typing import AsyncGenerator
import httpx
from fastapi import HTTPException

from app.core.config import settings
from app.core.llm import llm_client
from app.schemas.llm_schemas import (
    ChatMessage,
    ChatConversationRequest,
    ChatConversationResponse,
    GenerateStructuredRequest,
    GenerateStructuredResponse,
    GenerateTextRequest,
    GenerateTextResponse,
)


async def count_tokens(text: str) -> int:
    """
    Calculates token count using the backend tokenize endpoint.
    Falls back to a standard heuristic (1 token ≈ 4 characters) if unavailable.
    """
    base_url = settings.LLM_API_BASE.replace("/v1", "")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/tokenize",
                json={"model": settings.LLM_MODEL_NAME, "content": text},
            )
            if response.status_code == 200:
                return len(response.json().get("tokens", []))
    except Exception as e:
        print(f"Tokenization service unavailable, using character heuristic fallback: {e}")

    return len(text) // 4


async def generate_text_service(request_data: GenerateTextRequest) -> GenerateTextResponse:
    """
    Executes standard text completion with pre-flight context checks and latency tracking.
    """
    full_prompt = f"{request_data.system_prompt}\n{request_data.user_prompt}"
    input_tokens = await count_tokens(full_prompt)

    expected_total_tokens = input_tokens + request_data.max_tokens
    if expected_total_tokens > settings.LLM_CONTEXT_WINDOW:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Context overflow: Prompt tokens ({input_tokens}) + "
                f"Max tokens ({request_data.max_tokens}) = {expected_total_tokens}. "
                f"Configured limit is {settings.LLM_CONTEXT_WINDOW}."
            ),
        )

    messages = [
        {"role": "system", "content": request_data.system_prompt},
        {"role": "user", "content": request_data.user_prompt},
    ]

    start_time = time.time()

    response = await llm_client.chat.completions.create(
        model=settings.LLM_MODEL_NAME,
        messages=messages,
        temperature=request_data.temperature,
        top_p=request_data.top_p,
        max_tokens=request_data.max_tokens,
    )

    end_time = time.time()

    generated_text = response.choices[0].message.content
    usage = response.usage

    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0

    duration_seconds = end_time - start_time
    latency_ms = round(duration_seconds * 1000, 2)
    tokens_per_second = (
        round(completion_tokens / duration_seconds, 2)
        if duration_seconds > 0
        else 0.0
    )

    return GenerateTextResponse(
        generated_text=generated_text,
        model_used=settings.LLM_MODEL_NAME,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        tokens_per_second=tokens_per_second,
    )


async def stream_text_service(request_data: GenerateTextRequest) -> AsyncGenerator[str, None]:
    """
    Yields completion tokens as a Server-Sent Events (SSE) stream.
    """
    messages = [
        {"role": "system", "content": request_data.system_prompt},
        {"role": "user", "content": request_data.user_prompt},
    ]

    stream = await llm_client.chat.completions.create(
        model=settings.LLM_MODEL_NAME,
        messages=messages,
        temperature=request_data.temperature,
        top_p=request_data.top_p,
        max_tokens=request_data.max_tokens,
        stream=True,
    )

    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield f"data: {token}\n\n"

    yield "data: [DONE]\n\n"


async def generate_structured_service(
    request_data: GenerateStructuredRequest,
) -> GenerateStructuredResponse:
    """
    Executes JSON-constrained completion and returns parsed dictionary data.
    """
    full_prompt = f"{request_data.system_prompt}\n{request_data.user_prompt}"
    input_tokens = await count_tokens(full_prompt)

    if input_tokens + request_data.max_tokens > settings.LLM_CONTEXT_WINDOW:
        raise HTTPException(
            status_code=413,
            detail=f"Context overflow: Prompt tokens exceed window limit of {settings.LLM_CONTEXT_WINDOW}.",
        )

    messages = [
        {"role": "system", "content": request_data.system_prompt},
        {"role": "user", "content": request_data.user_prompt},
    ]

    response = await llm_client.chat.completions.create(
        model=settings.LLM_MODEL_NAME,
        messages=messages,
        temperature=request_data.temperature,
        max_tokens=request_data.max_tokens,
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content

    try:
        parsed_json = json.loads(raw_content)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Model failed to produce valid JSON: {raw_content}",
        )

    usage = response.usage
    return GenerateStructuredResponse(
        data=parsed_json,
        model_used=settings.LLM_MODEL_NAME,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
    )


async def chat_conversation_service(
    request_data: ChatConversationRequest,
) -> ChatConversationResponse:
    """
    Processes a multi-turn conversation maintaining full transcript context.
    """
    messages_payload = [msg.model_dump() for msg in request_data.messages]

    full_conversation_text = "\n".join(
        [f"{msg.role}: {msg.content}" for msg in request_data.messages]
    )
    input_tokens = await count_tokens(full_conversation_text)

    if input_tokens + request_data.max_tokens > settings.LLM_CONTEXT_WINDOW:
        raise HTTPException(
            status_code=413,
            detail=f"Conversation history exceeds context limit ({settings.LLM_CONTEXT_WINDOW} tokens).",
        )

    response = await llm_client.chat.completions.create(
        model=settings.LLM_MODEL_NAME,
        messages=messages_payload,
        temperature=request_data.temperature,
        max_tokens=request_data.max_tokens,
    )

    assistant_reply = response.choices[0].message.content
    usage = response.usage

    return ChatConversationResponse(
        reply=ChatMessage(role="assistant", content=assistant_reply),
        model_used=settings.LLM_MODEL_NAME,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
    )
    