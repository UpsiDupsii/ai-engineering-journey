import re

class SecurityGuard:
    """Provides validation and security checks against prompt and tool injection attacks."""
    
    def __init__(self):
        # Heuristics for detecting prompt injection attempts
        self.forbidden_phrases = [
            "ignore previous instructions",
            "you are now",
            "system prompt",
            "bypass",
            "forget your instructions"
        ]

    def is_safe_prompt(self, user_input: str) -> bool:
        """Validates that the user input does not contain known prompt injection patterns."""
        lower_input = user_input.lower()
        for phrase in self.forbidden_phrases:
            if phrase in lower_input:
                return False
        return True

    def validate_tool_input(self, tool_name: str, action_input: str) -> bool:
        """Validates tool inputs to prevent arbitrary code execution or bad formatting."""
        if tool_name == "calculator":
            # Restrict to safe mathematical characters
            if not re.match(r"^[\d\s\+\-\*\/\(\)\.]+$", action_input):
                return False
        
        return True

guard = SecurityGuard()
