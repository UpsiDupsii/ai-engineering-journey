from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Ultimate LangGraph API",
    description="FastAPI + LangGraph + Ollama + Milvus with HITL and Time Travel",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "Running", "db": "Milvus", "llm": "Ollama qwen2.5:3b"}