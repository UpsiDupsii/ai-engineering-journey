# n8n Agent Simulator API

A production-ready FastAPI backend designed to simulate core n8n automation concepts. This project demonstrates how to build a dynamic workflow execution engine from scratch, integrating robust node-based logic with local AI reasoning and vector memory.

## Architecture & Tech Stack

* **Framework:** FastAPI
* **Validation:** Pydantic (v2)
* **LLM & AI Engine:** Ollama (`qwen2.5:3b`) for local inference
* **Vector Database Client:** `pymilvus` (for agent memory and RAG workflows)
* **HTTP Client:** `httpx` (for simulating external REST API integrations)
* **Structure:** Clean Architecture (`app/core`, `app/schemas`, `app/services`, `app/api`)

## Features & Core Automation Concepts

This project physically implements fundamental workflow automation and AI agent concepts necessary for custom orchestration:

* **Workflow Execution Engine:** Simulates n8n node sequences (Triggers, Conditions, Loops) using a sequential processing loop that passes execution context from one node to the next.
* **AI Agent Nodes:** Directly integrates with local Ollama models to execute complex reasoning tasks, data extraction, and summarization autonomously within a workflow.
* **Webhook Triggers:** Exposes API endpoints to catch external payloads and initiate workflows on demand.
* **REST API & Tool Integration:** Simulates outgoing HTTP requests to interact with third-party services and feed external data into the workflow state.
* **Vector Memory Integration:** Utilizes Milvus to give AI agents persistent memory, allowing nodes to retrieve semantic context during execution.
* **Resilience & Error Handling:** Implements node-level retry mechanisms and graceful failure handling to prevent full workflow crashes during transient errors.

## Directory Structure

```text
app/
├── core/
│   ├── config.py             # Pydantic BaseSettings environment configuration
│   └── db.py                 # MilvusClient singleton initialization
├── schemas/
│   └── n8n.py                # Pydantic models for workflows, nodes, and triggers
├── services/
│   ├── milvus_service.py     # Collection setup, indexes, and vector operations
│   ├── agent_service.py      # Local Ollama LLM generation logic
│   └── workflow_service.py   # Node execution (conditions, HTTP, AI tasks)
├── api/
│   └── router.py             # APIRouter for webhooks and workflow execution endpoints
└── main.py                   # Application entry point, router inclusion, and lifespan

```

## Setup & Installation

**1. Clone and navigate to the project directory**
*(Assuming you have your project folder created)*

**2. Create and activate a virtual environment**

```cmd
python -m venv venv
venv\Scripts\activate

```

**3. Install dependencies**

```cmd
pip install fastapi uvicorn pymilvus pydantic-settings httpx

```

**4. Environment Variables**
Create a `.env` file in the root directory:

```env
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="qwen2.5:3b"
MILVUS_URI="http://localhost:19530"
MILVUS_COLLECTION="n8n_agent_memory"

```

*(Note: Ensure you have Ollama running locally with the qwen2.5:3b model pulled, and a Milvus instance available either via Docker or Milvus Lite).*

**5. Start the Server**

```cmd
uvicorn app.main:app --reload

```

## API Documentation

FastAPI automatically generates interactive API documentation based on standard OpenAPI specifications. Once the server is running, you can explore the endpoints, view request/response schemas, and test the API directly from your browser.

* **Swagger UI (Interactive docs):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc (Alternative docs):** [http://localhost:8000/redoc](http://localhost:8000/redoc)
