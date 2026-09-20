# services/cache.py
from typing import List, Optional, Tuple
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from core.config import settings

CACHE_COLLECTION_NAME = "rag_semantic_cache"

def get_cache_collection() -> Collection:
    """Initialize the Milvus collection for Semantic Caching."""
    connections.connect(alias="default", uri=settings.MILVUS_URI)
    
    if utility.has_collection(CACHE_COLLECTION_NAME):
        collection = Collection(CACHE_COLLECTION_NAME)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="query_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="query_embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIM)
        ]
        schema = CollectionSchema(fields=fields, description="Semantic Cache for RAG")
        collection = Collection(name=CACHE_COLLECTION_NAME, schema=schema)
        
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="query_embedding", index_params=index_params)
        
    collection.load()
    return collection

def check_cache(query_embedding: List[float], threshold: float = 0.15) -> Optional[str]:
    """
    Topic 16: Semantic Caching
    Checks if a highly similar question was asked recently.
    Returns the cached answer if found, otherwise returns None.
    """
    collection = get_cache_collection()
    
    # Search for the most similar past query
    results = collection.search(
        data=[query_embedding],
        anns_field="query_embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=1,
        output_fields=["answer"]
    )
    
    if not results[0]:
        return None
        
    best_match = results[0][0]
    
    # If the L2 distance is below our threshold, it's a semantic match!
    # (Lower L2 distance means the vectors are closer together)
    if best_match.distance <= threshold:
        return best_match.entity.get("answer")
        
    return None

def add_to_cache(query_text: str, query_embedding: List[float], answer: str) -> None:
    """Saves a newly generated Q&A pair to the cache."""
    collection = get_cache_collection()
    collection.insert([
        [query_text],
        [answer],
        [query_embedding]
    ])
    collection.flush()