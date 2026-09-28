from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Agentic AI API"
    
    # Ollama Settings
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    
    # Milvus Settings
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_USER: str = "root"
    MILVUS_PASSWORD: str = ""

    # Pydantic v2 way of configuring the settings model
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" # Ignore extra variables in .env that aren't defined here
    )

settings = Settings()
