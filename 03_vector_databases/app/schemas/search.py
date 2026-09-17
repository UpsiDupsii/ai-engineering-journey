from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class TextSearchRequest(BaseModel):
    query_text: str = Field(..., description="The query string to search for")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of closest results to return")
    similarity_threshold: Optional[float] = Field(
        default=None, 
        description="Minimum similarity score required for a match"
    )
    filter_expr: Optional[str] = Field(
        default=None, 
        description="Boolean filter expression on metadata payload, e.g. 'category == \"speech\"'"
    )

class SearchResultItem(BaseModel):
    id: int
    score: float
    payload: Dict[str, Any]

class SearchResponse(BaseModel):
    total: int
    results: List[SearchResultItem]
    

class IngestDocumentRequest(BaseModel):
    text: str = Field(..., description="The raw document text to embed and index")
    category: str = Field(default="general", description="Category for scalar metadata filtering")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary dynamic payload attributes")

class IngestDocumentResponse(BaseModel):
    status: str
    inserted_id: int
    message: str
    
    