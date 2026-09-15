# LLM Fundamentals API

A production-ready FastAPI backend designed to interface with local Large Language Models (LLMs) via Ollama. This project demonstrates core AI engineering concepts, utilizing the OpenAI Python SDK for compatibility and standardizing API structures using Clean Architecture principles.

## Architecture & Tech Stack

*   **Framework:** FastAPI
*   **Validation:** Pydantic (v2)
*   **LLM Client:** `openai` Python SDK (configured for local Ollama `/v1` endpoints)
*   **Async HTTP:** `httpx` (for native tokenization API calls)
*   **Structure:** Clean Architecture (`app/core`, `app/api`, `app/schemas`, `app/services`)

## Features & Core AI Concepts

This project physically implements fundamental LLM concepts necessary for production AI engineering:

*   **Inference & Prompt Engineering:** Full support for separating `system`, `user`, and `assistant` roles.
*   **Tokenization & Context Management:** Uses Ollama's native `/api/tokenize` endpoint to count tokens precisely. Includes a pre-flight safeguard to block requests that exceed the model's configured context window.
*   **Deterministic Generation:** Fine-grained control over `temperature` and `top_p` parameters.
*   **Real-time Streaming:** Implements Server-Sent Events (SSE) via FastAPI's `StreamingResponse` for token-by-token generation.
*   **Structured Outputs:** Enforces strict JSON return formats using `response_format={"type": "json_object"}`.
*   **Multi-Turn Conversations:** Stateless endpoint that accepts full chronological message history arrays to simulate memory.
*   **Performance Metrics:** Automatically calculates internal latency (ms) and throughput (Tokens Per Second / TPS).

## Directory Structure

```text
app/
├── api/
│   └── routes.py           # FastAPI router and endpoint definitions
├── core/
│   ├── config.py           # Pydantic BaseSettings environment configuration
│   └── llm.py              # OpenAI AsyncClient singleton initialization
├── schemas/
│   └── llm_schemas.py      # Pydantic models for request/response validation
├── services/
│   └── llm_service.py      # Business logic, token counting, and LLM calls
└── main.py                 # Application entry point and lifespan manager

```

## Setup & Installation

**1. Clone and navigate to the project directory**

**2. Create and activate a virtual environment**

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```

**3. Install dependencies**

```bash
pip install fastapi "uvicorn[standard]" pydantic-settings openai httpx

```

**4. Environment Variables**
Create a `.env` file in the root directory:

```env
PROJECT_NAME="LLM Fundamentals API"
VERSION="0.1.0"
LLM_API_BASE="http://localhost:11434/v1"
LLM_MODEL_NAME="qwen2.5:3b"
LLM_CONTEXT_WINDOW=8192

```

**5. Start the Server**

```bash
uvicorn app.main:app --reload

```

## API Documentation

FastAPI automatically generates interactive API documentation based on standard OpenAPI specifications. Once the server is running, you can explore the endpoints, view request/response schemas, and test the API directly from your browser.

* **Swagger UI (Interactive docs):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc (Alternative docs):** [http://localhost:8000/redoc](http://localhost:8000/redoc)
