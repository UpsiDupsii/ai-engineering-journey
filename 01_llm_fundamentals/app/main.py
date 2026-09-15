from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.config import settings
from app.core.llm import llm_client
from app.api.routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle.
    Operations before the yield run on startup.
    Operations after the yield run on shutdown.
    """
    print(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    print(f"Connected to LLM server at {settings.LLM_API_BASE} using model {settings.LLM_MODEL_NAME}")
    
    yield  
    
    print("Shutting down and cleaning up resources...")
    await llm_client.close()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Validates API operational status and current LLM configuration.
    """
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "llm_base": settings.LLM_API_BASE,
        "model": settings.LLM_MODEL_NAME
    }
    

# Register API routers
app.include_router(api_router, prefix="/api/v1", tags=["LLM Generation"])
