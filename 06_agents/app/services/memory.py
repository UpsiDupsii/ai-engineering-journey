from pymilvus import MilvusClient
from core.config import settings
from services.llm import llm_engine

class MemoryEngine:
    """Manages the agent's long-term vector memory using Milvus."""
    
    def __init__(self):
        self.uri = settings.milvus_uri
        self.db_name = settings.milvus_db_name
        self.collection_name = settings.milvus_collection_name
        self.dim = settings.embedding_dim
        self.client = MilvusClient(uri=self.uri)
        self.embed_model = "nomic-embed-text"

    def setup_collection(self):
        """Initializes the vector database and collection if they do not exist."""
        existing_dbs = self.client.list_databases()
        if self.db_name not in existing_dbs:
            self.client.create_database(self.db_name)
        
        # Re-initialize client to target the specific database
        self.client = MilvusClient(uri=self.uri, db_name=self.db_name)

        if not self.client.has_collection(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dim,
                auto_id=True
            )
            print(f"Collection '{self.collection_name}' initialized.")

    async def store_fact(self, text: str) -> str:
        """Generates an embedding for the input text and stores it in Milvus."""
        response = await llm_engine.client.embeddings(
            model=self.embed_model, 
            prompt=text
        )
        vector = response["embedding"]
        
        self.client.insert(
            collection_name=self.collection_name,
            data=[{"vector": vector, "text": text}]
        )
        return "Fact stored successfully."

    async def search_memory(self, query: str, limit: int = 2) -> str:
        """Performs a similarity search to retrieve relevant context for the agent."""
        response = await llm_engine.client.embeddings(
            model=self.embed_model, 
            prompt=query
        )
        query_vector = response["embedding"]
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["text"]
        )
        
        if not results or not results[0]:
            return "No relevant information found in memory."
            
        retrieved_texts = [hit["entity"]["text"] for hit in results[0]]
        return "\n".join(retrieved_texts)

memory_engine = MemoryEngine()