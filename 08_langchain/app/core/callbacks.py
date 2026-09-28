from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List

# --- Callbacks & Observability Concept ---
# This custom handler tracks LLM execution, which is crucial for observability.
class LoggingCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print(f"[Callback] LLM Started. Prompts: {prompts}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print(f"[Callback] LLM Ended. Generated: {response.generations[0][0].text}")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        print(f"[Callback] LLM Error: {error}")