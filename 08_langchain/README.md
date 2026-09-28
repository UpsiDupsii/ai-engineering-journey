# LangChain & Agents Course API

A production-ready FastAPI backend designed to implement core LangChain and AI Agent concepts step-by-step. This project bridges the gap between AI theory and practical backend engineering, demonstrating everything from basic model invocations to complete Retrieval-Augmented Generation (RAG) pipelines and autonomous tool-calling agents.

## Architecture & Tech Stack

* **Framework:** FastAPI
* **Orchestration:** LangChain (`langchain`, `langchain-core`)
* **Validation:** Pydantic (v2)
* **Vector Database:** Milvus Lite (Local SQLite-based via `langchain-milvus`)
* **LLM & Embedding Engine:** Ollama (Local execution via `langchain-ollama`)
* **Model:** `qwen2.5:3b` (or any Ollama-supported LLM)
* **Structure:** Clean Architecture (`apis/`, `core/`, `schemas/`, `services/`)

## Features & Core LangChain Concepts

This project acts as a modular, executable curriculum for LangChain, exposing each major concept as a distinct API endpoint:

* **Models & Messages:** Fundamental integration with ChatModels and standardizing inputs via System and Human messages.
* **Prompts, Parsers & LCEL:** Utilizing `ChatPromptTemplate`, output parsers, and chaining them together using LangChain Expression Language (LCEL).
* **Structured Output:** Forcing the LLM to return strictly typed JSON matching Pydantic schemas using the `.with_structured_output()` method.
* **RAG Pipeline (Retrieval-Augmented Generation):** Implements the full data lifecycle: loading documents, chunking with `RecursiveCharacterTextSplitter`, generating embeddings, indexing into Milvus, and retrieving contextual answers.
* **Tools & Agents:** Demonstrates the `create_tool_calling_agent` and `AgentExecutor` workflows, allowing the LLM to autonomously decide when and how to execute custom Python functions (e.g., weather fetching, string calculators).
* **Streaming Responses:** Implements asynchronous token streaming (`astream`) for real-time text generation to the client.
* **Observability & Middleware:** Features custom LangChain `BaseCallbackHandler` implementations to track LLM reasoning in the terminal, paired with FastAPI middleware to measure request latency.

## Directory Structure

```text
.
├── apis/
│   └── router.py             # FastAPI endpoints mapping to each LangChain lesson
├── core/
│   ├── callbacks.py          # Custom logging handlers for LangChain observability
│   └── config.py             # Pydantic BaseSettings environment configuration
├── schemas/
│   └── common.py             # Pydantic models for structured output and API requests
├── services/
│   ├── agent_service.py      # Tools registry and AgentExecutor logic
│   ├── llm_service.py        # Core LCEL chains, parsers, and streaming logic
│   └── rag_service.py        # Text splitting, embeddings, and Milvus vector search
├── main.py                   # Application entry point, lifespan, and middleware
├── requirements.txt          # Python dependencies
└── .env                      # Environment configuration

```

## Setup & Installation

**1. Clone and navigate to the project directory**

**2. Create and activate a virtual environment**

```cmd
python -m venv venv
venv\Scripts\activate

```

*(On macOS/Linux use `source venv/bin/activate`)*

**3. Install dependencies**

```cmd
pip install -r requirements.txt

```

**4. Start Background Services**
Ensure you have Ollama running locally. (Milvus Lite runs natively in Python, so no Docker container is needed for the database!).

```cmd
# Start Ollama and ensure the model is pulled
ollama run qwen2.5:3b

```

**5. Environment Variables**
Create a `.env` file in the root directory:

```env
HOST=0.0.0.0
PORT=8000
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:3b
MILVUS_URI=./milvus_local.db

```

**6. Start the Server**

```cmd
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```

## API Documentation

FastAPI automatically generates interactive API documentation based on standard OpenAPI specifications. Once the server is running, you can explore the endpoints, view request/response schemas, and test the LangChain concepts directly from your browser.

* **Swagger UI (Interactive docs):** http://localhost:8000/docs
* **ReDoc (Alternative docs):** http://localhost:8000/redoc
