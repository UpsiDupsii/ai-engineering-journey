from app.core.llm import generate_response

async def evaluator_optimizer_loop(goal: str, max_iterations: int = 3) -> str:
    """Planning, Reflection, and Iterative Execution."""
    
    # 1. Planning
    plan = await generate_response(f"Create a step-by-step plan to achieve this goal: {goal}")
    
    current_draft = await generate_response(f"Execute this plan: {plan}")
    
    for i in range(max_iterations):
        # 2. Evaluator
        evaluation = await generate_response(
            system="You are a harsh critic. If the draft fully achieves the goal, output 'PASS'. Otherwise, explain what is missing.",
            prompt=f"Goal: {goal}\nDraft: {current_draft}"
        )
        
        if "PASS" in evaluation.upper():
            return f"Success on iteration {i+1}:\n{current_draft}"
            
        # 3. Optimizer / Reflection (Iterative execution)
        current_draft = await generate_response(
            system="You are an optimizer. Improve the draft based on the critique.",
            prompt=f"Original Draft: {current_draft}\nCritique: {evaluation}"
        )
        
    return f"Finished after max iterations:\n{current_draft}"