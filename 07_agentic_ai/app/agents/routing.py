from app.core.llm import generate_response

async def route_query(query: str) -> str:
    """Dynamic routing based on user intent."""
    system_prompt = """You are a router. Classify the user query into exactly one of these categories: 
    [MATH, WEATHER, GENERAL]. Return ONLY the category name."""
    
    category = await generate_response(query, system=system_prompt)
    category = category.strip().upper()
    
    if "MATH" in category:
        return "Routed to Math Agent."
    elif "WEATHER" in category:
        return "Routed to Weather Agent."
    else:
        return "Routed to General Agent."