from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str

class RAGSetupRequest(BaseModel):
    text_content: str

class StructuredOutputSchema(BaseModel):
    name: str = Field(description="The name of the entity")
    category: str = Field(description="The category it belongs to")
    summary: str = Field(description="A brief summary")