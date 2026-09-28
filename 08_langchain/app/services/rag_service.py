from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from core.config import settings

class RAGService:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.llm_model
        )
        self.collection_name = "langchain_course_docs"
        
        # Initialize Vector Store
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={"uri": settings.milvus_uri},
            collection_name=self.collection_name,
            auto_id=True,
            drop_old=False
        )

    # --- Concepts: Document Loaders, Text Splitters, Vector Stores ---
    def load_and_index_documents(self, raw_text: str):
        # 1. Document Loaders (Simulated with raw text mapping to Document objects)
        docs = [Document(page_content=raw_text, metadata={"source": "api_upload"})]
        
        # 2. Text Splitters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        split_docs = text_splitter.split_documents(docs)
        
        # 3. Vector Stores (Milvus Add Documents)
        self.vector_store.add_documents(split_docs)
        return {"chunks_indexed": len(split_docs)}

    # --- Concepts: Retrievers ---
    def retrieve_and_answer(self, query: str) -> list[str]:
        # Convert vector store to retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        retrieved_docs = retriever.invoke(query)
        
        return [doc.page_content for doc in retrieved_docs]

rag_service = RAGService()