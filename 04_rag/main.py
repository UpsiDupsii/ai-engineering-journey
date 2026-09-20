# main.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.config import settings

# Updated import path to match our versioned directory structure
from api.v1.routes import router as api_router 

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    logger.info(f"Starting up {settings.PROJECT_NAME}...")
    logger.info(f"Documents directory ready at: {settings.DOCUMENTS_DIR}")
    # Future step: Connect to Milvus and Ping Ollama here
    
    yield # App runs here
    
    # --- Shutdown Logic ---
    logger.info(f"Shutting down {settings.PROJECT_NAME}...")
    # Future step: Close Milvus connections here

# Initialize FastAPI app with lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

# Health check endpoint remains in main.py
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "project": settings.PROJECT_NAME
    }

# Include external routes following industrial standards
app.include_router(api_router, prefix="/api/v1")
