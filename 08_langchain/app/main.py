import contextlib
from fastapi import FastAPI, Request
from core.config import settings
from apis.router import api_router
import time

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up LangChain API...")
    print(f"Using LLM Model: {settings.llm_model}")
    print(f"Milvus URI: {settings.milvus_uri}")
    yield
    print("Shutting down LangChain API...")

app = FastAPI(
    title="LangChain & Agents Course API",
    description="Complete FastAPI project implementing LangChain concepts.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Middleware Concept ---
# We add a simple middleware to track request execution time
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

app.include_router(api_router, prefix="/api/v1")