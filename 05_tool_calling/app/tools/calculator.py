def calculate(operation: str, a: float, b: float) -> str:
    """
    Perform a basic mathematical operation on two numbers.
    
    This tool should be used when the user asks to add, subtract, multiply, or divide numbers.
    
    Args:
        operation (str): The mathematical operation to perform. Must be one of: 'add', 'subtract', 'multiply', 'divide'.
        a (float): The first number.
        b (float): The second number.
        
    Returns:
        str: A JSON-formatted string containing the result of the calculation.
    """
    try:
        a = float(a)
        b = float(b)
        
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return '{"error": "Cannot divide by zero"}'
            result = a / b
        else:
            return f'{{"error": "Unknown operation: {operation}"}}'
            
        return f'{{"operation": "{operation}", "a": {a}, "b": {b}, "result": {result}}}'
    except ValueError:
        return '{"error": "Invalid numbers provided"}'
    