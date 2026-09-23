from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""
    app_name: str = "Agentic FastAPI"
    app_version: str = "0.1.0"
    
    # LLM Engine Configuration
    ollama_host: str
    model_name: str
    
    # Vector Database Configuration
    milvus_uri: str
    milvus_db_name: str
    milvus_collection_name: str
    embedding_dim: int

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
