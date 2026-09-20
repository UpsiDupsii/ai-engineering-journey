# core/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Project Configuration
    PROJECT_NAME: str
    DOCUMENTS_DIR: str
    
    # Ollama LLM Configuration
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str
    
    # Milvus Vector DB Configuration
    MILVUS_URI: str
    MILVUS_DB_NAME: str
    MILVUS_COLLECTION_NAME: str
    EMBEDDING_DIM: int

    # Pydantic v2 recommended way to configure settings
    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=True, 
        extra="ignore"
    )

# Instantiate the settings to be imported across the app
settings = Settings()

# Ensure the documents directory exists when the app starts
doc_path = os.path.abspath(settings.DOCUMENTS_DIR)
os.makedirs(doc_path, exist_ok=True)
settings.DOCUMENTS_DIR = doc_path
