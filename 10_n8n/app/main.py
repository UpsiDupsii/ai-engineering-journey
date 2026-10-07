from fastapi import FastAPI
from contextlib import asynccontextmanager
from pymilvus import MilvusClient
from app.core.db import db
from app.core.config import settings
from app.services.milvus_service import setup_collection
from app.api.router import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Milvus client and prepare collections
    db.client = MilvusClient(uri=settings.milvus_uri)
    setup_collection()
    yield
    # Shutdown: Close connections gracefully
    db.client.close()

app = FastAPI(
    title="n8n Agent Simulator",
    lifespan=lifespan
)

# Connect the routers
app.include_router(router, prefix="/api/v1")