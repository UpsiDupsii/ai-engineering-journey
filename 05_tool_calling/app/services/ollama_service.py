import httpx
from app.core.config import settings


class OllamaService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL

    async def check_health(self) -> dict:
        """Verifies Ollama connectivity and confirms model availability."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return {
                        "status": "unhealthy",
                        "error": f"Ollama returned status {response.status_code}",
                    }

                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                
                model_found = any(self.model in name for name in models)

                return {
                    "status": "healthy" if model_found else "model_missing",
                    "available_models": models,
                    "target_model": self.model,
                    "model_ready": model_found,
                }
        except httpx.ConnectError:
            return {
                "status": "unreachable",
                "error": f"Could not connect to Ollama at {self.base_url}. Ensure 'ollama serve' is running.",
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
            }


ollama_service = OllamaService()