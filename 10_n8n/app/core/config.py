from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ollama_base_url: str
    ollama_model: str
    milvus_uri: str
    milvus_collection: str

    # Using SettingsConfigDict to load environment variables
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()