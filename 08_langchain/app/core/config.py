from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    
    milvus_uri: str = "./milvus_local.db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()