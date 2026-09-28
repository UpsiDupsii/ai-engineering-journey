from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import logging
from app.core.config import settings
from app.core.llm import get_embedding

logger = logging.getLogger(__name__)

COLLECTION_NAME = "agent_memory"

def init_milvus():
    """Initializes connection and schema for state persistence."""
    connections.connect("default", uri=settings.MILVUS_URI, user=settings.MILVUS_USER, password=settings.MILVUS_PASSWORD)
    
    if not utility.has_collection(COLLECTION_NAME):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=3072) # Adjust dim based on Qwen2.5:3b embedding size
        ]
        schema = CollectionSchema(fields, "Agent Memory Storage")
        collection = Collection(COLLECTION_NAME, schema)
        
        index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        collection.create_index(field_name="vector", index_params=index_params)
        logger.info("Milvus collection created successfully.")

async def save_memory(text: str):
    """Persists state/memory."""
    collection = Collection(COLLECTION_NAME)
    vector = await get_embedding(text)
    collection.insert([[text], [vector]])
    collection.flush()

async def search_memory(query: str, limit: int = 3) -> list[str]:
    """Retrieves past state."""
    collection = Collection(COLLECTION_NAME)
    collection.load()
    query_vector = await get_embedding(query)
    results = collection.search([query_vector], "vector", param={"metric_type": "L2", "params": {"nprobe": 10}}, limit=limit, output_fields=["text"])
    
    memories = []
    for hits in results:
        for hit in hits:
            memories.append(hit.entity.get("text"))
    return memories
