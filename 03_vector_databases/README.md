# Multimodal Milvus Search API

A production-ready FastAPI backend designed to interface with Milvus (or Milvus Lite). This project demonstrates core vector database engineering concepts, from generating dense and sparse embeddings locally to executing hybrid searches and handling multimodal (audio) ingestion.

## Architecture & Tech Stack

* **Framework:** FastAPI
* **Validation:** Pydantic (v2)
* **Vector Database Client:** `pymilvus` (compatible with Milvus Standalone, Cluster, or Lite)
* **Machine Learning:** Hugging Face `transformers` & `torch`
* **Embedding Models:** `sentence-transformers/all-mpnet-base-v2` (Text) & `openai/whisper-tiny` (Audio)
* **Structure:** Clean Architecture (`app/core`, `app/schemas`, `app/services`)

## Features & Core Vector DB Concepts

This project physically implements fundamental vector database and semantic search concepts necessary for production AI engineering:

* **Dense & Sparse Vectors:** Combines deep semantic meaning (768-dimensional dense vectors) with exact lexical keyword matching (token-frequency sparse vectors).
* **Hybrid Search & Reranking:** Executes dual vector queries simultaneously and merges the results using Reciprocal Rank Fusion (`RRFRanker`) for perfectly balanced retrieval.
* **Advanced ANN Indexing:** Implements Hierarchical Navigable Small World (`HNSW`) graphs for high-speed dense retrieval and `SPARSE_INVERTED_INDEX` for exact token lookups.
* **Dynamic Metadata & Pre-Filtering:** Utilizes `enable_dynamic_field=True` to store arbitrary JSON payloads and executes boolean filter expressions *before/during* graph traversal to guarantee accurate Top-K results.
* **Multimodal Audio Ingestion:** Automatically transcribes `.mp3`/`.wav` file uploads via Whisper, embeds the extracted text, and indexes it alongside standard text documents.
* **Production Scaling:** Implements physical data distribution via `num_shards` and logical, multi-tenant data isolation using Partitions.

## Directory Structure

```text
app/
├── core/
│   ├── config.py             # Pydantic BaseSettings environment configuration
│   └── db.py                 # MilvusClient singleton initialization
├── schemas/
│   └── search.py             # Pydantic models for ingestion and search requests
├── services/
│   ├── milvus_service.py     # Collection setup, indexes, and search logic
│   ├── embedding_service.py  # Local Hugging Face text embedding generation
│   └── audio_service.py      # Audio transcription using Whisper
└── main.py                   # Application entry point, endpoints, and lifespan


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
pip install fastapi uvicorn pymilvus pydantic-settings python-multipart transformers torch torchaudio

```

**4. Environment Variables**
Create a `.env` file in the root directory:

```env
MILVUS_URI="http://localhost:19530" 
MILVUS_COLLECTION="vector_databases_collection"
MILVUS_DB_NAME="vector_project_db"
DENSE_DIM=768

```

*(Note: If you do not have Docker running, change `MILVUS_URI` to `milvus_local.db` to automatically use Milvus Lite in your project folder).*

**5. Start the Server**

```cmd
uvicorn app.main:app --reload

```

## API Documentation

FastAPI automatically generates interactive API documentation based on standard OpenAPI specifications. Once the server is running, you can explore the endpoints, view request/response schemas, and test the API directly from your browser.

* **Swagger UI (Interactive docs):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc (Alternative docs):** [http://localhost:8000/redoc](http://localhost:8000/redoc)