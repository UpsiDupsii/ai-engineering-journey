from ollama import AsyncClient
from core.config import settings

class LLMEngine:
    """Handles asynchronous communication with the local Ollama LLM engine."""
    
    def __init__(self):
        self.client = AsyncClient(host=settings.ollama_host)
        self.model = settings.model_name

    async def check_connection(self) -> bool:
        """Verifies server reachability and ensures the configured model is pulled."""
        try:
            response = await self.client.list()
            model_names = [m.get('name') for m in response.get('models', [])]
            
            # Substring match accommodates variant tags (e.g., 'qwen2.5:3b-instruct')
            return any(self.model in name for name in model_names)
        except Exception as e:
            print(f"LLM connection error: {e}")
            return False

    async def generate(self, prompt: str) -> str:
        """Executes a non-streaming text generation request for the agent loop."""
        response = await self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=False 
        )
        return response.get('response', '')

llm_engine = LLMEngine()