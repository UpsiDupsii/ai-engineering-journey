# LangGraph Advanced Agents API

A production-ready FastAPI backend designed to implement core LangGraph AI Agent concepts in a single, unified architecture. This project bridges the gap between agentic theory and practical backend engineering, demonstrating how to build cyclic graphs, multi-agent handoffs, human-in-the-loop workflows, and time-travel debugging.

## Architecture & Tech Stack

* **Framework:** FastAPI
* **Orchestration:** LangGraph & LangChain
* **Validation:** Pydantic (v2) with `SettingsConfigDict`
* **Vector Database:** Milvus (via `pymilvus`)
* **LLM Engine:** Ollama (Local execution via `langchain-ollama`)
* **Model:** `qwen2.5:3b` (or any Ollama-supported LLM)
* **Structure:** Clean Architecture (`api/`, `core/`, `services/`)

## Features & Core LangGraph Concepts

This project acts as an executable implementation of advanced LangGraph features, exposed via REST APIs:

* **State, Nodes & Edges:** The foundational state-machine architecture (`StateGraph`) defining shared memory (`AgentState`) and functional nodes.
* **Control Flow & Loops:** Utilizing conditional edges to route logic dynamically (e.g., looping back to memory or requesting human review based on agent confidence).
* **Persistence & Checkpoints (Short-term Memory):** Using LangGraph's `MemorySaver` to track thread states, enabling durable execution and conversational memory.
* **Vector Memory (Long-term Memory):** Integrating Milvus to simulate semantic storage and retrieval across different agent sessions.
* **Human-in-the-Loop (HITL) & Interrupts:** Pausing graph execution (`interrupt_before`) to await human approval or feedback via a dedicated API endpoint before resuming.
* **Multi-Agent Workflows:** Routing tasks between specialized agents (e.g., handing off output from a Researcher Agent to a Reviewer Agent).
* **Fault Tolerance:** Implementing native LangGraph `RetryPolicy` for resilient LLM and API calls.
* **Time Travel & Debugging:** Inspecting state history (`get_state_history`) and fetching exact checkpoints to allow front-ends to rewind or fork graph execution.
* **Streaming Execution:** Real-time event streaming to the client using LangGraph's asynchronous `astream` generator.

## Directory Structure

```text
my_agent_project/
├── .env                      # Environment configuration
├── requirements.txt          # Python dependencies
└── app/
    ├── __init__.py
    ├── main.py               # FastAPI application entry point
    ├── core/
    │   ├── __init__.py
    │   └── config.py         # Pydantic V2 environment configuration
    ├── api/
    │   ├── __init__.py
    │   └── routes.py         # Endpoints for chat, HITL approval, and time-travel
    └── services/
        ├── __init__.py
        └── graph.py          # Core LangGraph logic, agents, edges, and memory

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
Ensure you have Ollama and Milvus running locally.

```cmd
# Start Ollama and ensure the model is pulled
ollama run qwen2.5:3b

```

**5. Environment Variables**
Create a `.env` file in the root directory:

```env
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="qwen2.5:3b"
MILVUS_URI="http://localhost:19530"

```

**6. Start the Server**
We use the new FastAPI CLI for a streamlined dev experience:

```cmd
fastapi dev app/main.py

```

## API Documentation

FastAPI automatically generates interactive API documentation based on standard OpenAPI specifications. Once the server is running, you can explore the endpoints and test the LangGraph streams and interrupts directly from your browser.

* **Swagger UI (Interactive docs):** http://localhost:8000/docs
* **ReDoc (Alternative docs):** http://localhost:8000/redoc
