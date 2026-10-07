import httpx
from app.core.config import settings

async def invoke_ai_agent(prompt: str) -> str:
    """Connects to local Ollama instance running qwen2.5:3b"""
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Agent execution failed: {str(e)}"