# schemas/document.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List

# Topic 1: Documents
class DocumentUploadResponse(BaseModel):
    filename: str
    message: str
    path: str

# Topic 2: Loaders
class LoadedDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentLoadResponse(BaseModel):
    filename: str
    num_documents: int
    message: str

# Topics 3-6: Chunking
class ChunkingRequest(BaseModel):
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)

class ChunkingResponse(BaseModel):
    filename: str
    num_chunks: int
    message: str

# Topic 7: Embedding (New Schemas)
class EmbeddedDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: List[float] = Field(..., description="The vector representation of the text")

class IngestionResponse(BaseModel):
    filename: str
    total_chunks: int
    message: str
    