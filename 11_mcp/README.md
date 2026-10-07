# MCP Agent Simulator API

A production-ready FastAPI backend designed to physically implement the Model Context Protocol (MCP) architecture. This project demonstrates how to build an MCP Server, an MCP Client, and an AI Agent that natively understands tool discovery, resource resolution, and prompt templates using standardized JSON-RPC 2.0 communication.

## Architecture & Tech Stack

* **Framework:** FastAPI
* **Validation & Protocol Schemas:** Pydantic (v2) implementing MCP JSON-RPC 2.0 specifications
* **LLM & Agent Engine:** Ollama (`qwen2.5:3b`) for local inference and tool execution
* **Vector Database Client:** `pymilvus` (serving as an MCP Resource and Tool)
* **Security:** HTTPBearer token authorization for MCP Server endpoints
* **Structure:** Clean Architecture (`app/core`, `app/schemas`, `app/services`, `app/api`)

## Features & Core MCP Concepts

This project physically implements fundamental Model Context Protocol primitives necessary for standardized AI interactions:

* **MCP Server & Client Segregation:** Implements a strict boundary between the server exposing capabilities and the client executing them, bridging them with JSON-RPC 2.0 message routing.
* **Tool Discovery & Execution:** The server registers available tools (e.g., `search_vector_memory`) with JSON Schema inputs. The agent dynamically discovers these tools and requests execution to perform real-world actions.
* **Resources:** Exposes contextual, read-only data (e.g., Milvus metadata state) via standardized URIs (`milvus://system/metadata`) that the agent can read to ground its responses without altering system state.
* **Prompts:** Provides pre-configured prompt templates with defined arguments to structure complex LLM interactions predictably.
* **Agent Integration Loop:** Features a complete ReAct-style workflow where the `qwen2.5:3b` model discovers MCP tools, reasons about the user query, formats a tool call, and synthesizes the tool's result into a final answer.
* **Permissions & Security:** Enforces Bearer token authentication on the server and role-based scope checking (e.g., `read` vs `admin`) before a tool is executed by the client.

## Directory Structure

```text
app/
├── core/
│   ├── config.py             # Pydantic BaseSettings environment configuration
│   ├── security.py           # Permission validation & token authorization
│   └── db.py                 # MilvusClient singleton initialization
├── schemas/
│   └── mcp.py                # JSON-RPC 2.0 & MCP protocol schemas
├── services/
│   ├── milvus_service.py     # Milvus setup exposed as an MCP Resource & Tool
│   ├── mcp_server.py         # MCP Server: Registry for Tools, Resources, and Prompts
│   ├── mcp_client.py         # MCP Client: Tool discovery and execution interface
│   └── agent_service.py      # Ollama (qwen2.5:3b) Agent driving MCP workflows
├── api/
│   ├── mcp_routes.py         # Standard JSON-RPC 2.0 endpoints for the MCP Server
│   └── agent_routes.py       # Endpoints to query the MCP-powered Agent
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
MILVUS_COLLECTION="mcp_knowledge_base"
MCP_SERVER_TOKEN="secret-mcp-token-xyz"


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