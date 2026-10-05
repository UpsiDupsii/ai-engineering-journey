from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    MILVUS_URI: str = "http://localhost:19530"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()