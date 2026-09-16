from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.config import settings
from app.api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown logic."""
    print(f"Starting up {settings.APP_NAME}...")
    # Optional: You could pre-warm/preload the embedding model here if desired
    
    yield  # Application runs while execution rests here
    
    print(f"Shutting down {settings.APP_NAME}...")

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# Mount router under /api/v1
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {
        "app": settings.APP_NAME,
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
def health_check():
    """Health check endpoint to monitor application status."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "default_model": settings.DEFAULT_EMBEDDING_MODEL
    }
    