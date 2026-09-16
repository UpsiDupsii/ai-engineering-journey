from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "Embeddings Theory to Practice"
    DEBUG: bool = False
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

