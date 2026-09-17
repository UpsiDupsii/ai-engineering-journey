from pymilvus import MilvusClient
from app.core.config import settings


# 1. First, connect to the default database to check if our custom one exists
setup_client = MilvusClient(uri=settings.MILVUS_URI)

# 2. Create the custom database if it's missing
if settings.MILVUS_DB_NAME not in setup_client.list_databases():
    print(f"Creating custom database: {settings.MILVUS_DB_NAME}")
    setup_client.create_database(settings.MILVUS_DB_NAME)

# We no longer need the setup client
setup_client.close()

# 3. Create the main client connected directly to our custom database
client = MilvusClient(
    uri=settings.MILVUS_URI, 
    db_name=settings.MILVUS_DB_NAME
)
