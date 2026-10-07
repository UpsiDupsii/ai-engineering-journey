from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "mcp_knowledge_base"
    mcp_server_token: str = "secret-mcp-token-xyz"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()