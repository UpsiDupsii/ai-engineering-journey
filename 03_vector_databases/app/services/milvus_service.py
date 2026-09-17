from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from app.core.db import client
from app.core.config import settings

def setup_collection():
    """
    Creates the collection if it does not exist, defining the schema 
    and building the necessary ANN indexes.
    """
    if client.has_collection(collection_name=settings.MILVUS_COLLECTION):
        print(f"Collection '{settings.MILVUS_COLLECTION}' already exists.")
        return

    print(f"Creating collection '{settings.MILVUS_COLLECTION}'...")

    # 1. Define the Schema
    # We are setting enable_dynamic_field=True so we can insert arbitrary JSON payload data
    schema = MilvusClient.create_schema(
        auto_id=True, 
        enable_dynamic_field=True
    )

    # Add the primary key
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    
    # Add the Dense Vector field (for text/audio embeddings)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=settings.DENSE_DIM)
    
    # Add the Sparse Vector field (for keyword matching)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    
    # Add an explicit metadata field we might want to filter on (optional, but good practice)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=100)

    # 2. Define the Indexes
    index_params = client.prepare_index_params()
    
    # HNSW Index for Dense Vectors
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )
    
    # Inverted Index for Sparse Vectors
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP" # Inner Product is required for sparse vectors in Milvus
    )

    # 3. Create the collection and build indexes simultaneously
    client.create_collection(
        collection_name=settings.MILVUS_COLLECTION,
        schema=schema,
        index_params=index_params,
        num_shards=2 # Distributes data across 2 physical segments
    )
    
    # 4. Create Partitions for Logical Isolation
    client.create_partition(
        collection_name=settings.MILVUS_COLLECTION,
        partition_name="free_users"
    )
    client.create_partition(
        collection_name=settings.MILVUS_COLLECTION,
        partition_name="premium_users"
    )
    
    print(f"Successfully created collection '{settings.MILVUS_COLLECTION}' with indexes and partitions!")
    

def insert_document(dense_vector: list, sparse_vector: dict, category: str, extra_metadata: dict = None, partition_name: str = "free_users") -> int:
    """
    Inserts an entity into Milvus with both vectors and dynamic metadata payload.
    """
    entity = {
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector,
        "category": category,
    }
    # Spread any dynamic metadata into the entity (handled by enable_dynamic_field=True)
    if extra_metadata:
        entity.update(extra_metadata)

    result = client.insert(
        collection_name=settings.MILVUS_COLLECTION,
        data=[entity],
        partition_name=partition_name
    )
    # Return the generated primary ID
    return result["ids"][0]


def search_dense_vectors(
    query_vector: list, 
    top_k: int = 5, 
    filter_expr: str = None,
    partition_name: str = "free_users"
):
    """
    Performs pure dense ANN search on the HNSW index.
    """
    # milvus.search() expects a list of query vectors (batching)
    results = client.search(
        collection_name=settings.MILVUS_COLLECTION,
        data=[query_vector],
        anns_field="dense_vector",
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        filter=filter_expr,
        output_fields=["text", "category"],  # Retrieve original text and category payload
        partition_names=[partition_name]
    )
    return results[0]  # Return hits for the single query


def search_sparse_vectors(
    query_vector: dict, 
    top_k: int = 5, 
    filter_expr: str = None
):
    """
    Performs exact keyword matching using the Sparse Inverted Index.
    """
    results = client.search(
        collection_name=settings.MILVUS_COLLECTION,
        data=[query_vector],
        anns_field="sparse_vector",
        # Sparse vectors MUST use IP (Inner Product).
        # We can also add "drop_ratio_search" to ignore tiny token weights and speed up search.
        search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0.1}},
        limit=top_k,
        filter=filter_expr,
        output_fields=["text", "category"]
    )
    return results[0]


def search_hybrid_vectors(
    query_dense_vector: list, 
    query_sparse_vector: dict, 
    top_k: int = 5, 
    filter_expr: str = None
):
    """
    Combines dense and sparse searches and reranks them using RRF.
    """
    # 1. Prepare the Dense Search Request
    req_dense = AnnSearchRequest(
        data=[query_dense_vector],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        expr=filter_expr
    )

    # 2. Prepare the Sparse Search Request
    req_sparse = AnnSearchRequest(
        data=[query_sparse_vector],
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.1}},
        limit=top_k,
        expr=filter_expr
    )

    # 3. Define the Reranking Strategy (RRF)
    reranker = RRFRanker()

    # 4. Execute the Hybrid Search with ranker=reranker
    results = client.hybrid_search(
        collection_name=settings.MILVUS_COLLECTION,
        reqs=[req_dense, req_sparse],
        ranker=reranker,  # <-- FIXED: changed from rerank to ranker
        limit=top_k,
        output_fields=["text", "category"]
    )
    
    return results[0]

