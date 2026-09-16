from pydantic import BaseModel, Field
from typing import List


# --- Vector Math Schemas ---
class VectorMathRequest(BaseModel):
    vector_a: List[float] = Field(..., description="First dense vector (e.g., [1.0, 2.0, 3.0])")
    vector_b: List[float] = Field(..., description="Second dense vector")

class VectorMathResponse(BaseModel):
    dot_product: float
    cosine_similarity: float
    euclidean_distance: float
    is_normalized: bool = Field(description="True if both vectors have a magnitude of 1")


# --- Multimodal Embedding Schemas ---
class TextEmbeddingRequest(BaseModel):
    text: str = Field(..., description="Text string to embed")
    
class EmbeddingResponse(BaseModel):
    modality: str = Field(..., description="E.g., 'text' or 'image'")
    dimensions: int = Field(..., description="Size of the dense vector")
    embedding: List[float] = Field(..., description="The generated dense vector")


# --- Schemas for Sparse Vectors ---
class SparseVectorResponse(BaseModel):
    modality: str = "text"
    vocabulary_size: int = Field(..., description="Total number of known words/dimensions")
    # Instead of a list of mostly zeros, we represent sparse vectors as a dictionary of {word: weight}
    sparse_vector: dict[str, float] = Field(..., description="Non-zero values of the sparse vector")

class SparseVsDenseRequest(BaseModel):
    query: str = Field(..., description="The search query text")
    document: str = Field(..., description="The document text to compare against")

class SparseVsDenseResponse(BaseModel):
    query: str
    document: str
    dense_cosine_similarity: float = Field(..., description="Similarity based on semantic meaning (Dense)")
    sparse_cosine_similarity: float = Field(..., description="Similarity based on exact keyword overlap (Sparse)")
    