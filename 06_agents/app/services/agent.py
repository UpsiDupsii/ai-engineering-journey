import re
import uuid
from services.llm import llm_engine
from tools.registry import TOOL_REGISTRY, AVAILABLE_TOOLS_PROMPT
from services.guardrails import guard

REACT_SYSTEM_PROMPT = f"""You are a helpful AI assistant that solves problems using tools.
You have access to the following tools:

{AVAILABLE_TOOLS_PROMPT}

Solve the user's problem by strictly using the following step-by-step format:
Question: the input question you must answer
Thought: always think about what to do next
Action: the action to take. (Must be exactly one of: {", ".join(TOOL_REGISTRY.keys())} or None)
Action Input: the input to the action
Observation: the result of the action (provided by the system)
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""

class ReActAgent:
    """Core orchestrator implementing the Reason-Act-Observe loop and state management."""
    
    def __init__(self):
        self.system_prompt = REACT_SYSTEM_PROMPT
        # In-memory store for session state during human-in-the-loop pauses.
        self.pending_sessions = {}

    async def run(self, question: str, max_iterations: int = 5) -> dict:
        """Entry point for agent execution. Validates input and initializes the prompt context."""
        if not guard.is_safe_prompt(question):
            return {"status": "error", "answer": "Request blocked due to security policy.", "iterations": 0}

        prompt = self.system_prompt + f"\nQuestion: {question}\n"
        return await self._run_loop(prompt, max_iterations, 0)

    async def _run_loop(self, prompt: str, max_iterations: int, current_iteration: int) -> dict:
        """Executes the iterative reasoning and tool execution loop."""
        consecutive_errors = 0
        
        for iteration in range(current_iteration, max_iterations):
            response = await llm_engine.generate(prompt)
            prompt += response + "\n"
            
            # Terminal condition: Agent reached a final answer
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                return {"status": "success", "answer": final_answer, "iterations": iteration + 1}
            
            # Parse ReAct format
            action_match = re.search(r"Action:\s*(.*)", response)
            input_match = re.search(r"Action Input:\s*(.*)", response)
            
            if action_match:
                action = action_match.group(1).strip()
                action_input = input_match.group(1).strip() if input_match else ""
                consecutive_errors = 0 
                
                if action in TOOL_REGISTRY:
                    func, is_async, requires_human = TOOL_REGISTRY[action]
                    
                    if not guard.validate_tool_input(action, action_input):
                        observation = f"Error: The input '{action_input}' was rejected."
                    
                    elif requires_human:
                        # Suspend execution and yield state for human approval
                        session_id = str(uuid.uuid4())
                        self.pending_sessions[session_id] = {
                            "prompt": prompt,
                            "action": action,
                            "action_input": action_input,
                            "max_iterations": max_iterations,
                            "iteration": iteration
                        }
                        return {
                            "status": "awaiting_approval",
                            "answer": f"Agent wants to use '{action}'. Approval required.",
                            "session_id": session_id,
                            "action": action,
                            "action_input": action_input
                        }
                    
                    else:
                        # Standard tool execution
                        try:
                            if is_async:
                                observation = await func(action_input)
                            else:
                                observation = func(action_input)
                        except Exception as e:
                            observation = f"Error executing tool '{action}': {str(e)}"
                else:
                    observation = f"Error: Tool '{action}' does not exist."
                
                prompt += f"Observation: {observation}\n"
            else:
                # Formatting error retry strategy
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    return {"status": "error", "answer": "Agent failed to follow formatting rules.", "iterations": iteration + 1}
                prompt += "Observation: Format error. You MUST strictly use 'Action: [tool]' and 'Action Input: [input]' or 'Final Answer: [answer]'.\n"

        # Terminal condition: Exceeded max iterations
        return {"status": "error", "answer": "Agent stopped: Reached maximum iterations.", "iterations": max_iterations}

    async def resume(self, session_id: str, is_approved: bool) -> dict:
        """Resumes a suspended agent session post-human intervention."""
        if session_id not in self.pending_sessions:
            return {"status": "error", "answer": "Invalid or expired session ID.", "iterations": 0}

        session = self.pending_sessions.pop(session_id)
        prompt = session["prompt"]
        action = session["action"]
        action_input = session["action_input"]

        if is_approved:
            func, is_async, _ = TOOL_REGISTRY[action]
            try:
                if is_async:
                    observation = await func(action_input)
                else:
                    observation = func(action_input)
            except Exception as e:
                observation = f"Error: {e}"
        else:
            observation = "Action denied by user."

        prompt += f"Observation: {observation}\n"
        
        return await self._run_loop(prompt, session["max_iterations"], session["iteration"] + 1)

agent = ReActAgent()