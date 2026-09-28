from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
from pymilvus import connections

from app.core.config import settings
from app.core.memory import init_milvus
from app.api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.PROJECT_NAME}...")
    try:
        init_milvus()
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}. State persistence will be disabled.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    try:
        connections.disconnect("default")
    except Exception:
        pass

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Register all agent endpoints
app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"status": "online", "project": settings.PROJECT_NAME}