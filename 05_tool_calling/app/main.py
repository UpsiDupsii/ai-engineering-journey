from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.services.ollama_service import ollama_service
from app.apis.v1.routes import router as v1_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting {settings.APP_NAME}...")
    
    # Check Ollama health on startup
    status = await ollama_service.check_health()
    if status.get("status") == "healthy":
        print(f"Ollama connected successfully. Model '{settings.OLLAMA_MODEL}' is ready.")
    else:
        print(f"Warning: Ollama health check failed on startup. Details: {status}")
    
    yield
    
    print(f"Shutting down {settings.APP_NAME}...")


app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# Root health check endpoint
@app.get("/health")
async def health_check():
    """System health check verifying API status and Ollama availability."""
    ollama_status = await ollama_service.check_health()
    return {
        "app": settings.APP_NAME,
        "status": "online",
        "ollama": ollama_status,
    }

# Mount v1 API routes
app.include_router(v1_router)
