from services.llm import llm_engine

CODING_SUBAGENT_DESCRIPTION = """
Tool Name: ask_coding_expert
Description: Useful for delegating programming, software architecture, or code-generation tasks to a Senior Developer subagent. 
Action Input must be a detailed description of the coding task or question.
"""

async def execute_coding_subagent(task: str) -> str:
    """A subagent that strictly acts as a senior developer."""
    
    # We define a specialized system prompt just for this subagent
    subagent_prompt = f"""You are a Senior Python/Django Platform Engineer. 
    Your job is to write clean, production-ready code.
    Do not use conversational filler. Return only the requested code and a brief architectural explanation.
    
    Task from Orchestrator: {task}
    """
    
    print("\n[SUBAGENT ACTIVATED] The Coding Expert is thinking...\n")
    
    try:
        # We call the LLM engine directly, creating a separate "thought process" 
        # from our main Orchestrator agent loop.
        response = await llm_engine.generate(subagent_prompt)
        
        # We return the subagent's work back to the Orchestrator as an "Observation"
        return f"Coding Expert Subagent provided this solution:\n{response}"
    except Exception as e:
        return f"Subagent failed: {str(e)}"
    