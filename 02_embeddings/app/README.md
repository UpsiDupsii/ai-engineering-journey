# Embeddings & Vector Math API

A production-ready FastAPI backend designed to explore and demonstrate core AI embedding concepts, vector mathematical operations, multimodal representations, and cross-modal retrieval. This project grounds theoretical vector space principles into modular, executable Python endpoints using Clean Architecture principles.

## Architecture & Tech Stack

* **Framework:** FastAPI
* **Validation:** Pydantic (v2)
* **Embeddings / ML:** `sentence-transformers` (`clip-ViT-B-32`), `scikit-learn` (`TfidfVectorizer`), `numpy`
* **Structure:** Clean Architecture (`app/core`, `app/api`, `app/schemas`, `app/services`)

## Features & Core AI Concepts

This project physically implements fundamental embedding and vector math concepts necessary for production AI engineering:

* **Vector Math Engine:** Pure NumPy implementation of fundamental distance metrics: Dot Product, Cosine Similarity, Euclidean Distance ($L_2$), and $L_2$ Normalization verification.
* **Dense Text Embeddings:** Generates continuous 512-dimensional dense vector representations for natural language queries.
* **Image Embeddings:** Processes public image URLs through a Vision Transformer (ViT) to project visual features into a 512-dimensional vector space.
* **Multimodal Alignment & Cross-Modal Retrieval:** Evaluates semantic similarity directly between text queries and images within a shared CLIP joint embedding space.
* **Sparse Vectors (TF-IDF):** Generates high-dimensional keyword representations, outputting non-zero term weight dictionaries to eliminate dense zero-padding.
* **Sparse vs. Dense Comparison:** Side-by-side evaluation endpoint contrasting exact keyword overlap (Sparse) against underlying semantic meaning (Dense).

## Directory Structure

```text
.
├── app/
│   ├── api/
│   │   └── routes.py           # FastAPI router with math, dense, image, cross-modal, & sparse endpoints
│   ├── core/
│   │   ├── config.py           # Pydantic BaseSettings environment configuration
│   │   └── embeddings.py       # NumPy vector math engine (dot product, cosine, euclidean, normalize)
│   ├── schemas/
│   │   └── embedding_schemas.py # Pydantic request and response validation models
│   └── services/
│       ├── embedding_service.py # Dense & Multimodal CLIP logic
│       └── sparse_service.py    # TF-IDF Sparse vector logic & comparison service
├── main.py                     # Application entry point, lifespan manager, and health routes
└── .env                        # Environment variable configuration

```

## Setup & Installation

**1. Clone and navigate to the project directory**

```bash
cd ai-engineering-journey/02_embeddings

```

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
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings sentence-transformers scikit-learn numpy pillow requests

```

**4. Environment Variables**

Create a `.env` file in the root directory:

```env
APP_NAME="Embeddings & Vector Math API"
DEBUG=True
DEFAULT_EMBEDDING_MODEL="clip-ViT-B-32"

```

**5. Start the Server**

```bash
uvicorn main:app --reload

```

## API Documentation

FastAPI automatically generates interactive API documentation based on standard OpenAPI specifications. Once the server is running, you can explore the endpoints, view request/response schemas, and test vector operations directly from your browser.

* **Swagger UI (Interactive docs):** [http://127.0.0.1:8000/docs](https://www.google.com/search?q=http://127.0.0.1:8000/docs)
* **ReDoc (Alternative docs):** [http://127.0.0.1:8000/redoc](https://www.google.com/search?q=http://127.0.0.1:8000/redoc)