import asyncio
import logging
from typing import List, Dict, Any, Optional
import ollama
from app.core.config import settings

logger = logging.getLogger(__name__)
client = ollama.AsyncClient(host=settings.OLLAMA_HOST)

async def generate_response(prompt: str, system: str = "You are a helpful AI.", retries: int = 3) -> str:
    """Handles basic generation with fault tolerance (retries)."""
    for attempt in range(retries):
        try:
            response = await client.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                system=system,
                options={"temperature": 0.7}
            )
            return response['response']
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise Exception("LLM generation failed after maximum retries.")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

async def get_embedding(text: str) -> List[float]:
    """Generates embeddings for Milvus state persistence."""
    response = await client.embeddings(model=settings.OLLAMA_MODEL, prompt=text)
    return response['embedding']