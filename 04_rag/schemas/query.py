# schemas/query.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

# ==========================================
# Chat & Memory (Topic 15)
# ==========================================
class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text content")

# ==========================================
# Core Retrieval Models (Topics 8 - 12, 18, 19)
# ==========================================
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's latest natural language question.")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of document chunks to retrieve.")
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional metadata filtering constraints (e.g. {'file_type': 'pdf'})."
    )
    use_hybrid: bool = Field(default=False, description="Enable Hybrid Search (Dense + BM25 with RRF).")
    use_reranker: bool = Field(default=False, description="Enable Cross-Encoder Reranking.")
    use_hyde: bool = Field(default=False, description="Enable Hypothetical Document Embeddings (HyDE).")
    chat_history: List[ChatMessage] = Field(
        default_factory=list, 
        description="List of prior chat turns for conversational contextualization."
    )

class RetrievedChunk(BaseModel):
    id: int = Field(..., description="Milvus entity ID")
    text: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata dictionary")
    distance: float = Field(..., description="Dense vector distance score")
    rrf_score: Optional[float] = Field(default=None, description="Reciprocal Rank Fusion score")
    cross_encoder_score: Optional[float] = Field(default=None, description="Cross-Encoder reranking score")

class QueryResponse(BaseModel):
    query: str = Field(..., description="The original query submitted")
    results: List[RetrievedChunk] = Field(..., description="Retrieved chunks ordered by relevance")

# ==========================================
# Master RAG Output Models (Topics 13 - 17, 20 - 23)
# ==========================================
class RagResponse(BaseModel):
    query: str = Field(..., description="The original input query")
    standalone_query: Optional[str] = Field(
        default=None, 
        description="Rewritten self-contained query used for vector retrieval"
    )
    answer: str = Field(..., description="The generated response from the LLM")
    sources: List[RetrievedChunk] = Field(
        default_factory=list, 
        description="Retrieved document chunks referenced in context"
    )
    is_cached: bool = Field(
        default=False, 
        description="True if response was served from the Semantic Cache"
    )
    route_taken: str = Field(
        default="rag", 
        description="Router decision path: 'rag' or 'chitchat'"
    )

# ==========================================
# Evaluation Models (Topic 24)
# ==========================================
class EvaluationRequest(BaseModel):
    query: str = Field(..., description="The test question to evaluate.")
    top_k: int = Field(default=3, ge=1, le=20)
    use_hybrid: bool = Field(default=False)
    use_reranker: bool = Field(default=False)
    use_hyde: bool = Field(default=False)

class EvaluationResponse(BaseModel):
    query: str = Field(..., description="Evaluated query")
    generated_answer: str = Field(..., description="Answer produced by the RAG pipeline")
    faithfulness_score: int = Field(
        ..., 
        ge=0, 
        le=10, 
        description="Score 0-10: Measures if claims are grounded strictly in context without hallucinations."
    )
    relevancy_score: int = Field(
        ..., 
        ge=0, 
        le=10, 
        description="Score 0-10: Measures how directly the generated answer addresses the question."
    )
    feedback: str = Field(..., description="Evaluation notes and reasoning from the judge model.")