import json
import httpx
from app.core.config import settings
from app.tools.schemas import AVAILABLE_TOOL_SCHEMAS
from app.services.executor import execute_tool


class LLMService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        self.chat_endpoint = f"{self.base_url}/api/chat"

    async def chat_with_tools(self, user_prompt: str, user_role: str = "guest", max_retries: int = 3) -> str:
        """
        Runs an agentic loop supporting multiple tools, error feedback, and retries.
        """
        messages = [
            {"role": "user", "content": user_prompt}
        ]

        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(max_retries):
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "tools": AVAILABLE_TOOL_SCHEMAS,
                    "stream": False,
                }

                response = await client.post(self.chat_endpoint, json=payload)
                response.raise_for_status()
                data = response.json()

                response_message = data.get("message", {})
                tool_calls = response_message.get("tool_calls", [])

                if not tool_calls:
                    return response_message.get("content", "")

                # Append assistant's tool call request to history
                messages.append(response_message)

                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    arguments = tool_call["function"]["arguments"]

                    print(f"[Attempt {attempt + 1}] LLM invoked '{function_name}' with args: {arguments}")

                    # Execute tool with RBAC validation
                    tool_result_str = await execute_tool(
                        tool_name=function_name, 
                        arguments=arguments,
                        user_role=user_role
                    )
                    print(f"[Attempt {attempt + 1}] Tool result: {tool_result_str}")

                    # Append tool result to history for the next iteration
                    messages.append({
                        "role": "tool",
                        "content": tool_result_str,
                        "name": function_name,
                    })

            return "Unable to complete request: maximum tool retry limit reached."


llm_service = LLMService()
