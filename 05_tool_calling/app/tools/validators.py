import re
from pydantic import BaseModel, Field, field_validator
from typing import Literal

# Regex pattern to block common command injection and path traversal characters
DANGEROUS_CHARS_REGEX = re.compile(r"[;<>&|`$]|(\.\.\/)")

def sanitize_string(value: str) -> str:
    """
    Scans input strings for dangerous characters or sequences.
    Raises a ValueError if a security violation is detected.
    """
    if DANGEROUS_CHARS_REGEX.search(value):
        raise ValueError("Security Violation: Input contains forbidden shell characters or path traversals.")
    return value


class WeatherArgs(BaseModel):
    """Validation model for the get_current_weather tool."""
    location: str = Field(..., description="The name of the city and country")
    unit: Literal["celsius", "fahrenheit"] = Field(default="celsius")

    @field_validator("location")
    @classmethod
    def secure_location_input(cls, v: str):
        return sanitize_string(v)


class CalculatorArgs(BaseModel):
    """Validation model for the calculate tool."""
    operation: Literal["add", "subtract", "multiply", "divide"]
    a: float
    b: float

    @field_validator("b")
    @classmethod
    def prevent_divide_by_zero(cls, v: float, info):
        operation = info.data.get("operation")
        if operation == "divide" and v == 0.0:
            raise ValueError("Mathematical error: Cannot divide by zero.")
        return v
    