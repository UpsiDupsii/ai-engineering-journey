# services/vector_store.py
import json
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from rank_bm25 import BM25Okapi # NEW: Used for sparse keyword scoring
from core.config import settings
from schemas.document import EmbeddedDocument

def get_milvus_collection() -> Collection:
    """Connect to Milvus and retrieve (or create) the RAG collection."""
    connections.connect(alias="default", uri=settings.MILVUS_URI)
    collection_name = settings.MILVUS_COLLECTION_NAME

    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON), 
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIM)
        ]
        schema = CollectionSchema(fields=fields, description="RAG Document Collection")
        collection = Collection(name=collection_name, schema=schema)
        
        index_params = {
            "metric_type": "L2", 
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
    
    collection.load()
    return collection

def insert_documents(documents: List[EmbeddedDocument]) -> None:
    """Takes a list of EmbeddedDocuments and inserts them into Milvus."""
    if not documents:
        return
        
    collection = get_milvus_collection()
    
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    embeddings = [doc.embedding for doc in documents]
    
    collection.insert([texts, metadatas, embeddings])
    collection.flush()

def build_milvus_expr(filters: Dict[str, Any]) -> str:
    """Topic 10: Helper to convert dictionary to Milvus boolean expression."""
    if not filters:
        return ""
    
    expr_parts = []
    for key, value in filters.items():
        if isinstance(value, str):
            safe_value = value.replace("'", "\\'")
            expr_parts.append(f"metadata[\"{key}\"] == '{safe_value}'")
        else:
            expr_parts.append(f"metadata[\"{key}\"] == {value}")
            
    return " and ".join(expr_parts)

def search_documents(
    query_embedding: List[float], 
    top_k: int = 3,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Topics 8, 9 & 10: Retrieval, Top-k, and Metadata Filtering"""
    collection = get_milvus_collection()
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    expr = build_milvus_expr(metadata_filter) if metadata_filter else None
    
    results = collection.search(
        data=[query_embedding], 
        anns_field="embedding", 
        param=search_params,
        limit=top_k,            
        expr=expr,
        output_fields=["text", "metadata"] 
    )
    
    retrieved_chunks = []
    for hit in results[0]:
        retrieved_chunks.append({
            "id": hit.id,
            "distance": hit.distance,
            "text": hit.entity.get("text"),
            "metadata": hit.entity.get("metadata")
        })
        
    return retrieved_chunks

def hybrid_search_documents(
    query_text: str,
    query_embedding: List[float],
    top_k: int = 3,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Topic 11: Hybrid Retrieval
    Combines Dense (Vector) search with Sparse (BM25 Keyword) search using Reciprocal Rank Fusion.
    """
    # Step 1: Dense Retrieval (Get a wider pool of candidates, e.g., 5x the requested top_k)
    candidate_k = top_k * 5
    dense_results = search_documents(query_embedding, candidate_k, metadata_filter)
    
    if not dense_results:
        return []

    # Step 2: Sparse Retrieval (Score candidates based on exact keyword matches)
    # Tokenize the text chunks and the user query
    tokenized_corpus = [res["text"].lower().split() for res in dense_results]
    tokenized_query = query_text.lower().split()
    
    bm25 = BM25Okapi(tokenized_corpus)
    sparse_scores = bm25.get_scores(tokenized_query)
    
    # Step 3: Reciprocal Rank Fusion (RRF)
    RRF_K = 60 # Standard constant used in RRF formulas
    
    # Sort indices by sparse score to determine sparse ranks
    sparse_ranked_indices = sorted(range(len(sparse_scores)), key=lambda i: sparse_scores[i], reverse=True)
    sparse_ranks = {idx: rank + 1 for rank, idx in enumerate(sparse_ranked_indices)}
    
    # Calculate RRF scores
    for idx, res in enumerate(dense_results):
        dense_rank = idx + 1 # dense_results are already sorted by vector distance
        sparse_rank = sparse_ranks[idx]
        
        # Combine the ranks mathematically
        res["rrf_score"] = (1.0 / (RRF_K + dense_rank)) + (1.0 / (RRF_K + sparse_rank))
        
    # Step 4: Sort by the new fused score and return the top_k
    hybrid_results = sorted(dense_results, key=lambda x: x["rrf_score"], reverse=True)
    
    return hybrid_results[:top_k]
