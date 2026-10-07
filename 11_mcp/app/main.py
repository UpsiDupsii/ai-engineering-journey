from fastapi import FastAPI
from contextlib import asynccontextmanager
from pymilvus import MilvusClient
from app.core.db import db
from app.core.config import settings
from app.services.milvus_service import setup_mcp_collection
from app.api.mcp_routes import router as mcp_router
from app.api.agent_routes import router as agent_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to Milvus and initialize schemas
    db.client = MilvusClient(uri=settings.milvus_uri)
    setup_mcp_collection()
    yield
    # Shutdown: Close connections cleanly
    db.client.close()

app = FastAPI(
    title="Model Context Protocol (MCP) FastAPI Engine",
    version="1.0.0",
    lifespan=lifespan
)

# Register routers
app.include_router(mcp_router)
app.include_router(agent_router)