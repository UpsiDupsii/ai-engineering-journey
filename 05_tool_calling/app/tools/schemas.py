WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": (
            "Get the current weather for a given city or location. "
            "This tool should be used whenever the user asks about the current weather, "
            "temperature, or climate conditions in a specific place."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": 'The name of the city and country, e.g., "San Francisco, USA" or "Hyderabad, India".'
                },
                "unit": {
                    "type": "string",
                    "description": 'The temperature unit to use. Must be either "celsius" or "fahrenheit".',
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
}

CALCULATOR_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Perform a basic mathematical operation on two numbers. "
            "This tool should be used when the user asks to add, subtract, multiply, or divide numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "The mathematical operation to perform.",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "a": {
                    "type": "number",
                    "description": "The first number in the operation."
                },
                "b": {
                    "type": "number",
                    "description": "The second number in the operation."
                }
            },
            "required": ["operation", "a", "b"]
        }
    }
}

# Registry of all available tool schemas exposed to the LLM
AVAILABLE_TOOL_SCHEMAS = [WEATHER_TOOL_SCHEMA, CALCULATOR_TOOL_SCHEMA]
