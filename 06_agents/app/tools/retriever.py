from services.memory import memory_engine

# Description is crucial for the LLM to know when to use this
MEMORY_TOOL_DESCRIPTION = """
Tool Name: search_memory
Description: Useful for when you need to recall specific facts, past context, or project information from your database.
Action Input must be a clear search query string (e.g., 'What is the server password?').
"""

async def execute_search_memory(query: str) -> str:
    """Executes a vector search against the memory database."""
    try:
        result = await memory_engine.search_memory(query)
        return result
    except Exception as e:
        return f"Error accessing memory: {str(e)}"
    