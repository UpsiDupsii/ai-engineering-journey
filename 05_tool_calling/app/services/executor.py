import json
import asyncio
from typing import Any, Callable, Dict, List
from pydantic import ValidationError

from app.tools.weather import get_current_weather
from app.tools.calculator import calculate
from app.tools.validators import WeatherArgs, CalculatorArgs

TOOL_MAP: Dict[str, Callable[..., Any]] = {
    "get_current_weather": get_current_weather,
    "calculate": calculate,
}

VALIDATOR_MAP: Dict[str, Any] = {
    "get_current_weather": WeatherArgs,
    "calculate": CalculatorArgs,
}

# Role-based access control mapping
TOOL_PERMISSIONS: Dict[str, List[str]] = {
    "get_current_weather": ["guest", "admin"], 
    "calculate": ["admin"],                    
}


async def execute_tool(
    tool_name: str, 
    arguments: dict | str, 
    user_role: str = "guest", 
    timeout_seconds: float = 5.0
) -> str:
    """
    Validates permissions, validates arguments, and executes the tool asynchronously.
    """
    # --- PERMISSION PHASE ---
    allowed_roles = TOOL_PERMISSIONS.get(tool_name, [])
    if user_role not in allowed_roles:
        return json.dumps({
            "error": "Permission Denied",
            "message": f"User with role '{user_role}' is not authorized to use the tool '{tool_name}'."
        })

    # --- EXECUTION PHASE ---
    if isinstance(arguments, str):
        try:
            parsed_args = json.loads(arguments)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON format in tool arguments."})
    else:
        parsed_args = arguments

    tool_function = TOOL_MAP.get(tool_name)
    validator_model = VALIDATOR_MAP.get(tool_name)

    if not tool_function or not validator_model:
        return json.dumps({"error": f"Tool '{tool_name}' not found."})

    try:
        validated_args = validator_model(**parsed_args)
    except ValidationError as err:
        error_details = [{"field": e["loc"], "message": e["msg"]} for e in err.errors()]
        return json.dumps({
            "error": "Validation failed. Please correct your arguments.",
            "details": error_details
        })

    try:
        def _run_sync():
            return tool_function(**validated_args.model_dump())

        result = await asyncio.wait_for(
            asyncio.to_thread(_run_sync), 
            timeout=timeout_seconds
        )
        return str(result)
    
    except asyncio.TimeoutError:
        return json.dumps({"error": f"Tool execution timed out after {timeout_seconds} seconds."})
    except Exception as err:
        return json.dumps({"error": f"Execution failed: {str(err)}"})
    