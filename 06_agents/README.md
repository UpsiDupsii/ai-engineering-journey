# Advanced Local ReAct Agent API

A production-ready FastAPI backend designed to implement an advanced ReAct (Reason + Act) AI Agent pipeline entirely locally. This project demonstrates core autonomous agent engineering concepts, moving from a basic reasoning loop to dynamic tool execution, human-in-the-loop (HITL) authorization, long-term vector memory, and hierarchical multi-agent delegation.

## Architecture & Tech Stack

* **Framework:** FastAPI
* **Validation:** Pydantic (v2)
* **Vector Database:** `pymilvus` (compatible with Milvus Standalone or Lite)
* **LLM & Embedding Engine:** Ollama (Local execution)
* **Generation Model:** `qwen2.5:3b` (or any Ollama-supported LLM)
* **Embedding Model:** `nomic-embed-text` (or any Ollama-supported embedding model)
* **Structure:** Clean Architecture (`api/`, `core/`, `schemas/`, `services/`, `tools/`)

## Features & Core Agent Concepts

This project physically implements the fundamental concepts necessary for enterprise-grade autonomous applications, bypassing heavy frameworks to provide highly controllable, native Python logic:

* **Core ReAct Loop:** Implements the Reason-Act-Observe cycle with dynamic state and prompt context management, handling iterative problem solving without relying on external orchestration wrappers.
* **Dynamic Tool Execution:** Features a centralized tool registry allowing the LLM to select and execute Python functions dynamically, seamlessly handling both synchronous and asynchronous operations.
* **Long-Term Vector Memory:** Integrates Milvus to provide the agent with semantic search capabilities, enabling it to recall facts, past context, and project information using vector embeddings.
* **Human-in-the-Loop (HITL):** Implements an asynchronous pause-and-resume state architecture, requiring explicit human authorization via a dedicated endpoint before the agent can execute sensitive tools (e.g., sending emails).
* **Security & Robustness:** Applies strict guardrails to prevent prompt and tool injection attacks, alongside resilient retry strategies and iteration limits to recover smoothly from LLM formatting hallucinations.
* **Hierarchical Multi-Agent System:** Demonstrates task delegation by providing the main orchestrator agent with a tool to spin up specialized, persona-driven subagents (e.g., a Senior Coding Expert) to handle complex sub-tasks.

## Directory Structure

```text
.
├── api/
│   └── v1/
│       └── routes.py         # Endpoints for asking the agent and resuming HITL sessions
├── core/
│   └── config.py             # Pydantic BaseSettings environment configuration
├── schemas/
│   └── agent.py              # Pydantic models for request/response and session management
├── services/
│   ├── agent.py              # Core ReAct loop, state management, and HITL handling
│   ├── guardrails.py         # Prompt and tool injection security validations
│   ├── llm.py                # Asynchronous communication with the local Ollama engine
│   └── memory.py             # Vector database initialization and semantic search logic
├── tools/
│   ├── calculator.py         # AST-based mathematical evaluation tool
│   ├── email.py              # Mock email dispatch tool (sensitive tool example)
│   ├── registry.py           # Centralized tool mapping and prompt builder
│   ├── retriever.py          # Vector memory search tool wrapper
│   └── subagent.py           # Specialized persona-driven subagent delegation
└── main.py                   # Application entry point and lifespan resource management

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
pip install fastapi uvicorn pydantic pydantic-settings pymilvus ollama python-dotenv

```

**4. Start Background Services**
Ensure you have Ollama and Milvus running locally:

```cmd
# Start Ollama and ensure the models are pulled
ollama run qwen2.5:3b
ollama pull nomic-embed-text

# Start Milvus (via Docker Compose or local instance)
docker compose up -d

```

**5. Environment Variables**
Create a `.env` file in the root directory:

```env
APP_NAME="Agentic FastAPI"
APP_VERSION="0.1.0"

# Ollama Configuration
OLLAMA_HOST="http://localhost:11434"
MODEL_NAME="qwen2.5:3b"

# Milvus Vector DB Configuration
MILVUS_URI="http://localhost:19530"
MILVUS_DB_NAME="rag_database"
MILVUS_COLLECTION_NAME="rag_collection"
EMBEDDING_DIM=768

```

**6. Start the Server**

```cmd
uvicorn main:app --reload --host 0.0.0.0 --port 8000

```

## API Documentation

FastAPI automatically generates interactive API documentation based on standard OpenAPI specifications. Once the server is running, you can explore the endpoints, view request/response schemas, and test the agent loop directly from your browser.

* **Swagger UI (Interactive docs):** http://localhost:8000/docs
* **ReDoc (Alternative docs):**