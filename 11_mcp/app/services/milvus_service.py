from pymilvus import DataType
from app.core.db import db
from app.core.config import settings

def setup_mcp_collection():
    if not db.client.has_collection(collection_name=settings.milvus_collection):
        schema = db.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        
        index_params = db.client.prepare_index_params()
        index_params.add_index(field_name="vector", metric_type="COSINE", index_type="AUTOINDEX")
        
        db.client.create_collection(
            collection_name=settings.milvus_collection,
            schema=schema,
            index_params=index_params
        )

def query_vector_memory(query_text: str) -> str:
    """Mock vector query returning semantic context from Milvus."""
    return f"Retrieved semantic match for '{query_text}': Cluster infrastructure running optimally."