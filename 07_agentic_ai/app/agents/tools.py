import json

def calculate(expression: str) -> str:
    """Tool: Evaluates a mathematical expression."""
    try:
        # Warning: eval is used here for demonstration. Use safe_eval in production.
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {e}"

def get_weather(location: str) -> str:
    """Tool: Gets dummy weather data."""
    return f"The weather in {location} is 72°F and sunny."

AVAILABLE_TOOLS = {
    "calculate": calculate,
    "get_weather": get_weather
}
