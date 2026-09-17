from fastapi import FastAPI, UploadFile, File, Form
from contextlib import asynccontextmanager
from app.core.db import client
from app.core.config import settings
from app.services.milvus_service import setup_collection, insert_document, search_dense_vectors, search_sparse_vectors, search_hybrid_vectors
from app.services.embedding_service import generate_embeddings, get_dense_embedding, get_sparse_embedding
from app.schemas.search import (
    IngestDocumentRequest, 
    IngestDocumentResponse, 
    TextSearchRequest, 
    SearchResponse, 
    SearchResultItem
)
from app.services.audio_service import process_audio_file

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Connecting to Milvus at {settings.MILVUS_URI}...")
    setup_collection()
    yield  
    print("Shutting down API...")
    client.close()

app = FastAPI(
    title="Multimodal Milvus Search API",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/documents", response_model=IngestDocumentResponse)
def ingest_document_endpoint(payload: IngestDocumentRequest):
    dense_vec, sparse_vec = generate_embeddings(payload.text)
    doc_metadata = payload.metadata.copy()
    doc_metadata["text"] = payload.text
    
    inserted_id = insert_document(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        category=payload.category,
        extra_metadata=doc_metadata
    )
    
    return IngestDocumentResponse(
        status="success",
        inserted_id=inserted_id,
        message="Document embedded and indexed into Milvus successfully!"
    )

@app.post("/search/dense", response_model=SearchResponse)
def search_dense_endpoint(payload: TextSearchRequest):
    # 1. Convert user's query string into a 768-dim dense vector
    query_dense_vector = get_dense_embedding(payload.query_text)
    
    # 2. Perform ANN Dense Search in Milvus (Top-k)
    raw_hits = search_dense_vectors(
        query_vector=query_dense_vector,
        top_k=payload.top_k,
        filter_expr=payload.filter_expr
    )
    
    # 3. Apply Similarity Threshold Filtering
    filtered_results = []
    for hit in raw_hits:
        score = float(hit["distance"])  # In Milvus COSINE metric, distance = cosine similarity
        
        # Drop results that do not meet the minimum similarity threshold
        if payload.similarity_threshold is not None and score < payload.similarity_threshold:
            continue
            
        filtered_results.append(
            SearchResultItem(
                id=hit["id"],
                score=round(score, 4),
                payload=hit["entity"]  # Contains "text", "category", etc.
            )
        )
        
    return SearchResponse(
        total=len(filtered_results),
        results=filtered_results
    )


@app.post("/search/sparse", response_model=SearchResponse)
def search_sparse_endpoint(payload: TextSearchRequest):
    # 1. Convert user's query string into a Sparse Vector (Token Dictionary)
    query_sparse_vector = get_sparse_embedding(payload.query_text)
    
    # If the text is empty or tokenizer fails, return empty
    if not query_sparse_vector:
        return SearchResponse(total=0, results=[])
        
    # 2. Perform Sparse Search in Milvus
    raw_hits = search_sparse_vectors(
        query_vector=query_sparse_vector,
        top_k=payload.top_k,
        filter_expr=payload.filter_expr
    )
    
    # 3. Format results
    filtered_results = []
    for hit in raw_hits:
        score = float(hit["distance"]) # In IP, higher distance/score = better match
        
        if payload.similarity_threshold is not None and score < payload.similarity_threshold:
            continue
            
        filtered_results.append(
            SearchResultItem(
                id=hit["id"],
                score=round(score, 4),
                payload=hit["entity"]
            )
        )
        
    return SearchResponse(
        total=len(filtered_results),
        results=filtered_results
    )
    

@app.post("/search/hybrid", response_model=SearchResponse)
def search_hybrid_endpoint(payload: TextSearchRequest):
    # 1. Generate BOTH embeddings for the query
    query_dense, query_sparse = generate_embeddings(payload.query_text)
    
    # 2. Perform Hybrid Search
    raw_hits = search_hybrid_vectors(
        query_dense_vector=query_dense,
        query_sparse_vector=query_sparse,
        top_k=payload.top_k,
        filter_expr=payload.filter_expr
    )
    
    # 3. Format results (Threshold filtering is trickier here because RRF 
    # scores are very small numbers like 0.03, so we usually rely purely on Top-K).
    filtered_results = []
    for hit in raw_hits:
        filtered_results.append(
            SearchResultItem(
                id=hit["id"],
                score=round(float(hit["distance"]), 4), # This distance is now the RRF score
                payload=hit["entity"]
            )
        )
        
    return SearchResponse(
        total=len(filtered_results),
        results=filtered_results
    )
    

@app.post("/documents/audio", response_model=IngestDocumentResponse)
def ingest_audio_endpoint(
    file: UploadFile = File(...),
    category: str = Form(default="audio_note"),
    author: str = Form(default="unknown")
):
    """
    Accepts an audio file, transcribes it, converts it to dense/sparse vectors, 
    and inserts it into Milvus.
    """
    # 1. Transcribe the audio file to text
    transcribed_text = process_audio_file(file)
    print(f"Transcription complete: '{transcribed_text}'")
    
    # If the audio was silent or unreadable
    if not transcribed_text:
        return IngestDocumentResponse(
            status="failed", 
            inserted_id=0, 
            message="Could not extract any speech from the audio file."
        )

    # 2. Convert the transcribed text into vectors
    dense_vec, sparse_vec = generate_embeddings(transcribed_text)
    
    # 3. Store in Milvus (We keep the transcription in the payload!)
    doc_metadata = {
        "text": transcribed_text,
        "author": author,
        "source_file": file.filename
    }
    
    inserted_id = insert_document(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        category=category,
        extra_metadata=doc_metadata
    )
    
    return IngestDocumentResponse(
        status="success",
        inserted_id=inserted_id,
        message=f"Audio processed successfully. Extracted text: '{transcribed_text}'"
    )
    
    