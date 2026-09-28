from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from core.config import settings
from core.callbacks import LoggingCallbackHandler
from schemas.common import StructuredOutputSchema

class LLMService:
    def __init__(self):
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            callbacks=[LoggingCallbackHandler()]
        )

    # --- Concept: Models & Messages ---
    def generate_basic_response(self, text: str) -> str:
        messages = [
            SystemMessage(content="You are a precise and helpful assistant."),
            HumanMessage(content=text)
        ]
        response = self.llm.invoke(messages)
        return response.content

    # --- Concept: Prompts & Output Parsers ---
    def generate_with_prompt_and_parser(self, topic: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert on {topic}."),
            ("human", "Give me one interesting fact about this topic.")
        ])
        parser = StrOutputParser()
        
        # --- Concept: Chains (LCEL - LangChain Expression Language) ---
        chain = prompt | self.llm | parser
        return chain.invoke({"topic": topic})

    # --- Concept: Structured Output ---
    def extract_structured_data(self, text: str) -> dict:
        structured_llm = self.llm.with_structured_output(StructuredOutputSchema)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract information from the following text into the structured format."),
            ("human", "{input}")
        ])
        chain = prompt | structured_llm
        result = chain.invoke({"input": text})
        return result.model_dump()

    # --- Concept: Streaming ---
    async def stream_response(self, query: str):
        prompt = ChatPromptTemplate.from_template("Explain {query} in detail.")
        chain = prompt | self.llm | StrOutputParser()
        
        async for chunk in chain.astream({"query": query}):
            yield chunk

llm_service = LLMService()