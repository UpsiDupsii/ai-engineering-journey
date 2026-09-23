from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.config import settings
from services.llm import llm_engine
from services.memory import memory_engine
from api.v1.routes import router as agent_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages startup and shutdown events, ensuring critical services are reachable."""
    print(f"Starting {settings.app_name} v{settings.app_version}...")
    
    # Verify LLM Engine
    is_llm_ready = await llm_engine.check_connection()
    if not is_llm_ready:
        raise RuntimeError(
            f"Failed to connect to Ollama at {settings.ollama_host} "
            f"or model '{settings.model_name}' is missing."
        )
    print("LLM Engine connection verified.")
    
    # Verify Vector Database
    print(f"Initializing Milvus Vector DB at {settings.milvus_uri}...")
    memory_engine.setup_collection()
    
    yield
    
    print("Cleaning up resources and shutting down...")

app = FastAPI(
    title=settings.app_name, 
    version=settings.app_version,
    lifespan=lifespan 
)

app.include_router(agent_router)

@app.get("/health/", tags=["System"])
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "model_configured": settings.model_name,
        "milvus_uri": settings.milvus_uri,
        "milvus_db": settings.milvus_db_name,
        "milvus_collection": settings.milvus_collection_name
    }