import asyncio
from app.core.llm import generate_response

async def sequential_workflow(topic: str) -> dict:
    """Sequential execution: Step 1 feeds into Step 2."""
    step1_outline = await generate_response(f"Write a 3-point outline about {topic}.")
    step2_expand = await generate_response(f"Expand on this outline: {step1_outline}")
    return {"outline": step1_outline, "final": step2_expand}

async def parallel_workflow(topic: str) -> dict:
    """Parallel execution: Independent tasks run concurrently."""
    task1 = generate_response(f"What are the pros of {topic}?")
    task2 = generate_response(f"What are the cons of {topic}?")
    
    pros, cons = await asyncio.gather(task1, task2)
    return {"pros": pros, "cons": cons}