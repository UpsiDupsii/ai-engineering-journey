from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.schemas.embedding_schemas import (
    VectorMathRequest,
    VectorMathResponse,
    EmbeddingResponse,
    SparseVectorResponse,
    SparseVsDenseRequest,
    SparseVsDenseResponse,
)
from app.services.embedding_service import embedding_service
from app.services.sparse_service import sparse_service

router = APIRouter()

class CrossModalRequest(BaseModel):
    text: str = Field(..., description="Query text to compare")
    image_url: str = Field(..., description="Public image URL to compare against")

class CrossModalResponse(BaseModel):
    text: str
    image_url: str
    cosine_similarity: float
    euclidean_distance: float
    dimensions: int

@router.post("/math/compare", response_model=VectorMathResponse)
def compare_vectors(payload: VectorMathRequest):
    """Calculates vector distance and similarity metrics on two raw dense vectors."""
    if len(payload.vector_a) != len(payload.vector_b):
        raise HTTPException(
            status_code=400,
            detail=f"Dimension mismatch: Vector A has size {len(payload.vector_a)}, but Vector B has size {len(payload.vector_b)}"
        )
    
    result = embedding_service.compare_raw_vectors(payload.vector_a, payload.vector_b)
    return result

@router.post("/embed/text", response_model=EmbeddingResponse)
def get_text_embedding(text: str):
    """Generates a dense embedding for a given text string."""
    vector = embedding_service.embed_text(text)
    return EmbeddingResponse(
        modality="text",
        dimensions=len(vector),
        embedding=vector
    )

@router.post("/embed/image", response_model=EmbeddingResponse)
def get_image_embedding(image_url: str):
    """Generates a dense embedding from an image URL."""
    try:
        vector = embedding_service.embed_image_from_url(image_url)
        return EmbeddingResponse(
            modality="image",
            dimensions=len(vector),
            embedding=vector
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

@router.post("/compare/cross-modal", response_model=CrossModalResponse)
def compare_text_and_image(payload: CrossModalRequest):
    """Executes Cross-Modal Retrieval by measuring 
    semantic similarity between text and an image."""
    try:
        text_vec = embedding_service.embed_text(payload.text)
        image_vec = embedding_service.embed_image_from_url(payload.image_url)
        
        metrics = embedding_service.compare_raw_vectors(text_vec, image_vec)
        
        return CrossModalResponse(
            text=payload.text,
            image_url=payload.image_url,
            cosine_similarity=metrics["cosine_similarity"],
            euclidean_distance=metrics["euclidean_distance"],
            dimensions=len(text_vec)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in cross-modal comparison: {str(e)}")

@router.post("/embed/sparse", response_model=SparseVectorResponse)
def get_sparse_embedding(text: str):
    """Generates a sparse (TF-IDF) vector representation mapping keywords to weights."""
    result = sparse_service.generate_sparse_vector(text)
    return SparseVectorResponse(
        vocabulary_size=result["vocabulary_size"],
        sparse_vector=result["sparse_vector"]
    )

@router.post("/compare/sparse-vs-dense", response_model=SparseVsDenseResponse)
def compare_sparse_and_dense(payload: SparseVsDenseRequest):
    """Compares exact keyword overlap (Sparse) vs semantic meaning (Dense) between query and document."""
    result = sparse_service.compare_sparse_vs_dense(payload.query, payload.document)
    return SparseVsDenseResponse(**result)
