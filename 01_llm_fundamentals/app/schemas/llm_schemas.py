from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field


class GenerateTextRequest(BaseModel):
    """
    Standard request payload for text generation.
    """
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="The system instruction that sets the behavior of the AI."
    )
    user_prompt: str = Field(
        ...,
        description="The actual question or instruction from the user.",
        min_length=1
    )
    temperature: float = Field(
        default=0.7, 
        ge=0.0,
        le=2.0,
        description="Controls randomness. 0.0 is deterministic, 2.0 is highly random."
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold."
    )
    max_tokens: int = Field(
        default=512,
        gt=0,
        description="The maximum number of tokens to generate."
    )


class GenerateTextResponse(BaseModel):
    """
    Standard response payload for text generation including performance metrics.
    """
    generated_text: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    tokens_per_second: float


class GenerateStructuredRequest(BaseModel):
    """
    Request payload for forcing structured JSON outputs.
    """
    system_prompt: str = Field(
        default="You are a data extraction engine. You must output only valid JSON.",
        description="System instruction. Must explicitly instruct the model to output JSON."
    )
    user_prompt: str = Field(
        ...,
        description="The input text and instructions on what JSON structure to produce."
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Deterministic temperature is recommended for JSON extraction."
    )
    max_tokens: int = Field(
        default=512, 
        gt=0
    )


class GenerateStructuredResponse(BaseModel):
    """
    Response payload containing parsed JSON object data.
    """
    data: Dict[str, Any]
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatMessage(BaseModel):
    """
    Individual message structure for multi-turn conversations.
    """
    role: Literal["system", "user", "assistant"]
    content: str = Field(
        ..., 
        min_length=1
    )


class ChatConversationRequest(BaseModel):
    """
    Request payload containing entire conversational history.
    """
    messages: List[ChatMessage] = Field(
        ...,
        description="Full conversation history in chronological order.",
        min_length=1
    )
    temperature: float = Field(
        default=0.7, 
        ge=0.0, 
        le=2.0
    )
    max_tokens: int = Field(
        default=512, 
        gt=0
    )


class ChatConversationResponse(BaseModel):
    """
    Response payload returning the assistant's next message.
    """
    reply: ChatMessage
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    