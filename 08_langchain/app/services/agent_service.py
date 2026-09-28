from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from core.config import settings
from core.callbacks import LoggingCallbackHandler

# --- Concept: Tools ---
@tool
def calculate_string_length(text: str) -> int:
    """Returns the length of a given string. Use this when asked to count characters."""
    return len(text)

@tool
def get_current_weather(location: str) -> str:
    """Returns the current weather for a specified location."""
    return f"The weather in {location} is currently 72 degrees and sunny."

class AgentService:
    def __init__(self):
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            callbacks=[LoggingCallbackHandler()]
        )
        self.tools = [calculate_string_length, get_current_weather]
        
        # --- Concept: Agents ---
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use tools when necessary."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def run_agent(self, query: str) -> str:
        response = self.agent_executor.invoke({"input": query})
        return response.get("output")

agent_service = AgentService()