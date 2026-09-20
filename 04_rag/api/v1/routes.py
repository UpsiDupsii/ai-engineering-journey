# api/v1/routes.py
import os
import shutil
import json
from fastapi import APIRouter, UploadFile, File, HTTPException, Path, Body
from fastapi.responses import StreamingResponse
from core.config import settings

# Schemas
from schemas.document import (
    DocumentUploadResponse, 
    DocumentLoadResponse,
    ChunkingRequest,
    ChunkingResponse,
    IngestionResponse,
    EmbeddedDocument
)
from schemas.query import (
    QueryRequest, 
    QueryResponse, 
    RetrievedChunk, 
    RagResponse,
    EvaluationRequest, # NEW: Topic 24
    EvaluationResponse # NEW: Topic 24
)

# Services
from services.loaders import load_document
from services.chunking import chunk_documents, parent_child_chunk_documents
from services.embeddings import generate_embeddings_batch, generate_embedding
from services.vector_store import insert_documents, search_documents, hybrid_search_documents
from services.reranker import rerank_documents
from services.cache import check_cache, add_to_cache
from services.evaluation import evaluate_rag_output # NEW: Topic 24
from services.llm import (
    build_rag_prompt, 
    generate_answer, 
    contextualize_query, 
    route_query, 
    extract_metadata_filters, 
    generate_hypothetical_document,
    generate_answer_stream,
    check_guardrails
)

router = APIRouter()

# ==========================================
# PHASE 1 & 2: INGESTION PIPELINE
# ==========================================

@router.post("/documents/upload", tags=["Documents"], response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    file_path = os.path.join(settings.DOCUMENTS_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")
    return DocumentUploadResponse(filename=file.filename, message="Document uploaded successfully.", path=file_path)

@router.post("/documents/{filename}/load", tags=["Loaders"], response_model=DocumentLoadResponse)
async def load_uploaded_document(filename: str = Path(...)):
    file_path = os.path.join(settings.DOCUMENTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        loaded_docs = load_document(file_path)
        return DocumentLoadResponse(filename=filename, num_documents=len(loaded_docs), message="Document loaded.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading document: {str(e)}")

@router.post("/documents/{filename}/ingest", tags=["Ingestion"], response_model=IngestionResponse)
async def ingest_document(
    filename: str = Path(...), 
    request: ChunkingRequest = Body(default_factory=ChunkingRequest),
    use_parent_child: bool = Body(default=False)
):
    file_path = os.path.join(settings.DOCUMENTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    
    try:
        loaded_docs = load_document(file_path)
        
        if use_parent_child:
            chunked_docs = parent_child_chunk_documents(
                documents=loaded_docs, parent_chunk_size=1000, child_chunk_size=request.chunk_size, overlap=request.chunk_overlap
            )
        else:
            chunked_docs = chunk_documents(loaded_docs, request.chunk_size, request.chunk_overlap)
        
        texts = [doc.page_content for doc in chunked_docs]
        embeddings = generate_embeddings_batch(texts)
        
        embedded_documents = [
            EmbeddedDocument(page_content=doc.page_content, metadata=doc.metadata, embedding=emb)
            for doc, emb in zip(chunked_docs, embeddings)
        ]
            
        insert_documents(embedded_documents)
        return IngestionResponse(filename=filename, total_chunks=len(embedded_documents), message="Ingestion complete.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")

# ==========================================
# PHASE 3 & 4: RETRIEVAL & GENERATION
# ==========================================

@router.post("/query", tags=["Retrieval"], response_model=QueryResponse)
async def search_knowledge_base(request: QueryRequest = Body(...)):
    if not check_guardrails(request.query):
        raise HTTPException(status_code=400, detail="Query violates security guardrails. Please rephrase.")
        
    try:
        query_embedding = generate_embedding(request.query)
        
        if request.use_hybrid:
            raw_results = hybrid_search_documents(request.query, query_embedding, request.top_k, request.metadata_filter)
        else:
            raw_results = search_documents(query_embedding, request.top_k, request.metadata_filter)
            
        if request.use_reranker:
            raw_results = rerank_documents(request.query, raw_results)
            
        retrieved_chunks = [
            RetrievedChunk(id=res["id"], text=res["text"], metadata=res["metadata"], distance=res["distance"], rrf_score=res.get("rrf_score"), cross_encoder_score=res.get("cross_encoder_score"))
            for res in raw_results
        ]
        return QueryResponse(query=request.query, results=retrieved_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/chat", tags=["RAG"], response_model=RagResponse)
async def chat_with_knowledge_base(request: QueryRequest = Body(...)):
    if not check_guardrails(request.query):
        raise HTTPException(status_code=400, detail="Query violates security guardrails. Please rephrase.")
        
    try:
        standalone_query = contextualize_query(request.query, request.chat_history)
        
        route = route_query(standalone_query)
        if route == "chitchat":
            prompt = f"You are a helpful AI assistant. Answer conversationally.\n\nUser: {standalone_query}\nAnswer:"
            answer = generate_answer(prompt)
            return RagResponse(query=request.query, standalone_query=standalone_query, answer=answer, sources=[], is_cached=False, route_taken="chitchat")
            
        search_text = generate_hypothetical_document(standalone_query) if request.use_hyde else standalone_query
        query_embedding = generate_embedding(search_text)
        
        standalone_embedding = generate_embedding(standalone_query) 
        cached_answer = check_cache(standalone_embedding, threshold=0.15)
        if cached_answer:
            return RagResponse(query=request.query, standalone_query=standalone_query, answer=cached_answer, sources=[], is_cached=True, route_taken="rag")
            
        extracted_filters = extract_metadata_filters(standalone_query)
        final_filters = request.metadata_filter or {}
        final_filters.update(extracted_filters)
        
        if request.use_hybrid:
            raw_results = hybrid_search_documents(search_text, query_embedding, request.top_k, final_filters)
        else:
            raw_results = search_documents(query_embedding, request.top_k, final_filters)
            
        if request.use_reranker:
            raw_results = rerank_documents(standalone_query, raw_results)
            
        retrieved_chunks = []
        parent_docs_for_prompt = {} 
        
        for res in raw_results:
            metadata = res["metadata"]
            chunk = RetrievedChunk(id=res["id"], text=res["text"], metadata=metadata, distance=res["distance"], rrf_score=res.get("rrf_score"), cross_encoder_score=res.get("cross_encoder_score"))
            retrieved_chunks.append(chunk)
            
            if metadata.get("is_child") and "parent_id" in metadata:
                parent_id = metadata["parent_id"]
                if parent_id not in parent_docs_for_prompt:
                    parent_docs_for_prompt[parent_id] = {"text": metadata.get("parent_text", res["text"]), "metadata": metadata}
            else:
                parent_docs_for_prompt[res["id"]] = {"text": res["text"], "metadata": metadata}
                
        expanded_results = list(parent_docs_for_prompt.values())
        
        prompt = build_rag_prompt(standalone_query, expanded_results)
        answer = generate_answer(prompt)
        
        add_to_cache(standalone_query, standalone_embedding, answer)
        
        return RagResponse(query=request.query, standalone_query=standalone_query, answer=answer, sources=retrieved_chunks, is_cached=False, route_taken="rag")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")

@router.post("/chat/stream", tags=["RAG"])
async def chat_with_knowledge_base_stream(request: QueryRequest = Body(...)):
    if not check_guardrails(request.query):
        raise HTTPException(status_code=400, detail="Query violates security guardrails. Please rephrase.")
        
    try:
        standalone_query = contextualize_query(request.query, request.chat_history)
        
        route = route_query(standalone_query)
        if route == "chitchat":
            prompt = f"You are a helpful AI assistant. Answer conversationally.\n\nUser: {standalone_query}\nAnswer:"
            return StreamingResponse(generate_answer_stream(prompt), media_type="text/event-stream")
            
        search_text = generate_hypothetical_document(standalone_query) if request.use_hyde else standalone_query
        query_embedding = generate_embedding(search_text)
            
        extracted_filters = extract_metadata_filters(standalone_query)
        final_filters = request.metadata_filter or {}
        final_filters.update(extracted_filters)
        
        if request.use_hybrid:
            raw_results = hybrid_search_documents(search_text, query_embedding, request.top_k, final_filters)
        else:
            raw_results = search_documents(query_embedding, request.top_k, final_filters)
            
        if request.use_reranker:
            raw_results = rerank_documents(standalone_query, raw_results)
            
        parent_docs_for_prompt = {} 
        for res in raw_results:
            metadata = res["metadata"]
            if metadata.get("is_child") and "parent_id" in metadata:
                parent_id = metadata["parent_id"]
                if parent_id not in parent_docs_for_prompt:
                    parent_docs_for_prompt[parent_id] = {"text": metadata.get("parent_text", res["text"]), "metadata": metadata}
            else:
                parent_docs_for_prompt[res["id"]] = {"text": res["text"], "metadata": metadata}
                
        expanded_results = list(parent_docs_for_prompt.values())
        
        prompt = build_rag_prompt(standalone_query, expanded_results)
        
        return StreamingResponse(generate_answer_stream(prompt), media_type="text/event-stream")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream generation failed: {str(e)}")

# ==========================================
# Topic 24: Evaluation Endpoint
# ==========================================

@router.post("/evaluate", tags=["Evaluation"], response_model=EvaluationResponse)
async def evaluate_pipeline(request: EvaluationRequest = Body(...)):
    """
    Topic 24: RAG Evaluation.
    Runs a query through the pipeline, then uses the LLM as a judge to score 
    the system on Faithfulness and Answer Relevancy.
    """
    try:
        # Step 1: Run standard retrieval
        search_text = generate_hypothetical_document(request.query) if request.use_hyde else request.query
        query_embedding = generate_embedding(search_text)
        
        if request.use_hybrid:
            raw_results = hybrid_search_documents(search_text, query_embedding, request.top_k, None)
        else:
            raw_results = search_documents(query_embedding, request.top_k, None)
            
        if request.use_reranker:
            raw_results = rerank_documents(request.query, raw_results)
            
        # Step 2: Consolidate context text for the LLM
        context_blocks = []
        parent_docs = {} 
        for res in raw_results:
            metadata = res["metadata"]
            if metadata.get("is_child") and "parent_id" in metadata:
                parent_id = metadata["parent_id"]
                if parent_id not in parent_docs:
                    parent_docs[parent_id] = {"text": metadata.get("parent_text", res["text"]), "metadata": metadata}
            else:
                parent_docs[res["id"]] = {"text": res["text"], "metadata": metadata}
                
        expanded_results = list(parent_docs.values())
        for doc in expanded_results:
            context_blocks.append(doc["text"])
            
        combined_context = "\n\n".join(context_blocks)
        
        # Step 3: Generate the actual answer
        prompt = build_rag_prompt(request.query, expanded_results)
        generated_answer = generate_answer(prompt)
        
        # Step 4: Run the Ragas-style Evaluation Judge
        eval_scores = evaluate_rag_output(request.query, combined_context, generated_answer)
        
        return EvaluationResponse(
            query=request.query,
            generated_answer=generated_answer,
            faithfulness_score=eval_scores["faithfulness"],
            relevancy_score=eval_scores["relevancy"],
            feedback=eval_scores["feedback"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")