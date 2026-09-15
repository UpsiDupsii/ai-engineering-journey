from openai import AsyncOpenAI
from app.core.config import settings

llm_client = AsyncOpenAI(
    base_url=settings.LLM_API_BASE,
    api_key="ollama" 
)

