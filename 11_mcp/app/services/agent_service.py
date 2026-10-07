import httpx
import json
from app.core.config import settings
from app.services.mcp_client import mcp_client

async def run_mcp_agent(user_query: str, scopes: list[str]) -> dict:
    """Agent + MCP workflow:
    1. Tool Discovery: Agent queries MCP Client to inspect available tools.
    2. Model Reasoning: Sends user query + schemas to local Ollama (qwen2.5:3b).
    3. Tool Execution: Executes requested tool via MCP Client respecting permissions.
    4. Synthesis: Synthesizes final response back to user.
    """
    tools = mcp_client.discover_tools()
    
    # Expose tools in a structured prompt format to qwen2.5:3b
    tool_descriptions = json.dumps(tools, indent=2)
    system_instruction = (
        "You are an AI Agent with access to tools via the Model Context Protocol (MCP).\n"
        f"Available Tools:\n{tool_descriptions}\n\n"
        "If you need to call a tool, respond with ONLY a JSON block:\n"
        '{"action": "call_tool", "tool_name": "<name>", "arguments": {<args>}}\n'
        "If no tool is required, reply directly with plain text."
    )

    prompt = f"{system_instruction}\n\nUser Query: {user_query}"

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60.0
        )
        llm_output = response.json().get("response", "").strip()

    # Tool invocation parsing
    tool_execution_report = None
    final_output = llm_output

    if '{"action": "call_tool"' in llm_output:
        try:
            call_data = json.loads(llm_output)
            tool_name = call_data.get("tool_name")
            arguments = call_data.get("arguments", {})

            # Execute via MCP Client with permission checking
            execution_res = mcp_client.call_tool(
                name=tool_name, 
                arguments=arguments, 
                scopes=scopes
            )
            tool_execution_report = execution_res

            # Re-synthesize with the tool result
            synthesis_prompt = (
                f"User asked: {user_query}\n"
                f"Tool executed: {tool_name}\n"
                f"Result: {execution_res['result']}\n\n"
                "Provide a direct final answer to the user based on this result."
            )
            
            async with httpx.AsyncClient() as http_client:
                synth_response = await http_client.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": settings.ollama_model,
                        "prompt": synthesis_prompt,
                        "stream": False
                    },
                    timeout=60.0
                )
                final_output = synth_response.json().get("response", "").strip()

        except Exception as e:
            tool_execution_report = {"error": str(e)}
            final_output = f"Tool execution failed: {str(e)}"

    return {
        "query": user_query,
        "tool_execution": tool_execution_report,
        "agent_response": final_output
    }