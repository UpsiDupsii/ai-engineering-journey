from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MILVUS_URI: str
    MILVUS_COLLECTION: str
    MILVUS_DB_NAME: str
    DENSE_DIM: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Create a global instance of our settings to use throughout the app
settings = Settings()
