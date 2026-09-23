from tools.calculator import CALCULATOR_DESCRIPTION, execute_calculator
from tools.retriever import MEMORY_TOOL_DESCRIPTION, execute_search_memory
from tools.email import EMAIL_TOOL_DESCRIPTION, execute_send_email
from tools.subagent import CODING_SUBAGENT_DESCRIPTION, execute_coding_subagent

# Registry format: "tool_name": (execution_function, is_async, requires_human_approval)
TOOL_REGISTRY = {
    "calculator": (execute_calculator, False, False),
    "search_memory": (execute_search_memory, True, False),
    "send_email": (execute_send_email, False, True),
    "ask_coding_expert": (execute_coding_subagent, True, False)
}

# Consolidates all tool descriptions into a single string for injection into the system prompt.
AVAILABLE_TOOLS_PROMPT = "\n".join([
    CALCULATOR_DESCRIPTION, 
    MEMORY_TOOL_DESCRIPTION,
    EMAIL_TOOL_DESCRIPTION,
    CODING_SUBAGENT_DESCRIPTION
])