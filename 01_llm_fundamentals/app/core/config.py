from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables.
    """
    PROJECT_NAME: str = "LLM Fundamentals API"
    VERSION: str = "0.1.0"
    
    LLM_API_BASE: str
    LLM_MODEL_NAME: str
    LLM_CONTEXT_WINDOW: int = 8192
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
