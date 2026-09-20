# Advanced Local RAG API

A production-ready FastAPI backend designed to implement an advanced Retrieval-Augmented Generation (RAG) pipeline entirely locally. This project demonstrates 24 core RAG engineering concepts, moving from basic chunking and ingestion to production-grade query routing, semantic caching, and automated LLM evaluation.

## Architecture & Tech Stack

* **Framework:** FastAPI
* **Validation:** Pydantic (v2)
* **Vector Database:** `pymilvus` (compatible with Milvus Standalone or Lite)
* **LLM & Embedding Engine:** Ollama (Local execution)
* **Embedding Model:** `nomic-embed-text` (or any Ollama-supported embedding model)
* **Generation Model:** `llama3` (or any Ollama-supported LLM)
* **Structure:** Clean Architecture (`api/`, `core/`, `schemas/`, `services/`)

## Features & Core RAG Concepts

This project physically implements the fundamental concepts necessary for enterprise-grade AI applications, completely bypassing black-box wrappers like LangChain in favor of native, highly controllable python functions:

* **Advanced Ingestion Pipeline:** Implements standard sliding-window chunking alongside Parent-Child (Small-to-Big) retrieval for highly accurate search with maximum context retention.
* **Complex Retrieval Strategies:** Fuses Dense and Sparse vector search (Hybrid Search via RRF) and applies Stage-2 Cross-Encoder reranking for deep relevance sorting.
* **Agentic Query Transformation:** Automatically contextualizes multi-turn chat history, generates Hypothetical Document Embeddings (HyDE) for query expansion, and extracts metadata filters directly from natural language (Self-Querying).
* **Production Orchestration & Speed:** Features an LLM-based Query Router (bypassing the DB for standard chitchat), a Milvus-backed Semantic Cache to serve repeated questions instantly, and Server-Sent Events (SSE) for token-by-token UI streaming.
* **Safety & Accountability:** Applies strict security guardrails to block prompt injection and dynamically enforces inline bracketed citations (e.g., `[1]`) tying claims to specific source files.
* **Automated Evaluation:** Natively implements Ragas-style evaluation, using the LLM as an objective judge to score Faithfulness and Answer Relevancy on a 0-10 scale.

## Directory Structure

```text
.
├── api/
│   └── v1/
│       └── routes.py         # Master endpoints (Upload, Ingest, Query, Chat, Evaluate)
├── core/
│   └── config.py             # Pydantic BaseSettings environment configuration
├── schemas/
│   ├── document.py           # Pydantic models for ingestion and chunking
│   └── query.py              # Pydantic models for search, memory, and routing
├── services/
│   ├── cache.py              # Milvus-backed semantic Q&A caching
│   ├── chunking.py           # Standard and Parent-Child text splitting
│   ├── embeddings.py         # Vector generation wrappers for Ollama
│   ├── evaluation.py         # Ragas-style metric scoring (LLM-as-a-judge)
│   ├── llm.py                # Prompt builders, routing, guardrails, and streaming
│   ├── loaders.py            # TXT and PDF parsing
│   ├── reranker.py           # Stage-2 sorting logic
│   └── vector_store.py       # Milvus collection setup, indexes, and search logic
├── data/
│   └── documents/            # Local storage for uploaded files
└── main.py                   # Application entry point and lifespan

```

## Setup & Installation

**1. Clone and navigate to the project directory**

**2. Create and activate a virtual environment**

```cmd
python -m venv venv
venv\Scripts\activate

```

**3. Install dependencies**

```cmd
pip install fastapi uvicorn pymilvus pydantic-settings python-multipart requests PyPDF2

```

**4. Start Background Services**
Ensure you have Ollama and Milvus running locally:

```cmd
# Start Ollama (ensure you have pulled your preferred models, e.g., llama3)
ollama serve

# Start Milvus (via Docker Compose or local instance)
docker compose up -d

```

**5. Environment Variables**
Create a `.env` file in the root directory (or rely on the defaults in `core/config.py`):

```env
MILVUS_URI="http://localhost:19530"
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="llama3"
DOCUMENTS_DIR="data/documents"
EMBEDDING_DIM=768

```

**6. Start the Server**

```cmd
uvicorn main:app --reload --host 0.0.0.0 --port 8000

```

## API Documentation

FastAPI automatically generates interactive API documentation based on standard OpenAPI specifications. Once the server is running, you can explore the endpoints, view request/response schemas, and test the API directly from your browser.

* **Swagger UI (Interactive docs):** [http://localhost:8000/docs](http://localhost:8000/docs?utm_source=gemini)
* **ReDoc (Alternative docs):** [http://localhost:8000/redoc](http://localhost:8000/redoc?utm_source=gemini)
