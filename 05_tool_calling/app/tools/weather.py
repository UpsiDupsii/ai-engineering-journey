def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a given city or location.
    
    This tool should be used whenever the user asks about the current weather, 
    temperature, or climate conditions in a specific place.
    
    Args:
        location (str): The name of the city and country, e.g., "San Francisco, USA" or "Hyderabad, India".
        unit (str, optional): The temperature unit to use. Must be either "celsius" or "fahrenheit". Defaults to "celsius".
        
    Returns:
        str: A JSON-formatted string containing the weather information.
    """
    # Mock response for local execution
    mock_temperature = 28 if unit == "celsius" else 82
    mock_condition = "Sunny"
    
    return f'{{"location": "{location}", "temperature": {mock_temperature}, "unit": "{unit}", "condition": "{mock_condition}"}}'
