import operator
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy  # Native fault tolerance
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from pymilvus import MilvusClient
from app.core.config import settings

# 1. State (Short-term memory shared across nodes)
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    loop_count: int
    needs_human: bool
    status: str

# Initialize connections
llm = ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL)

# Initialize Milvus (Long-term memory)
milvus_client = MilvusClient(uri=settings.MILVUS_URI)
if not milvus_client.has_collection("agent_memory"):
    milvus_client.create_collection("agent_memory", dimension=384)

# 2. Nodes
def memory_node(state: AgentState):
    """Simulates checking long-term memory (Milvus)."""
    # Standard DB fault tolerance check
    try:
        # In a real app, embed last_msg and search Milvus here
        milvus_context = "Found relevant past interactions in Milvus."
    except Exception:
        milvus_context = "Memory DB unavailable."
    
    return {"status": milvus_context, "loop_count": state.get("loop_count", 0) + 1}

def researcher_agent(state: AgentState):
    """Agent 1: Primary LLM that generates the initial response."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def reviewer_agent(state: AgentState):
    """Agent 2: Evaluates Agent 1's work (Multi-agent handoff)."""
    last_message = state["messages"][-1].content
    
    # Simple logic: if the researcher is unsure, flag it for human review
    needs_human = "unsure" in last_message.lower()
    
    # Adding a reviewer message to the state
    review_msg = AIMessage(content="Review complete. Proceeding with output.")
    return {"messages": [review_msg], "needs_human": needs_human}

def human_approval_node(state: AgentState):
    """Dummy node. LangGraph pauses BEFORE this node for HITL."""
    return {"status": "Human reviewed and approved."}

# 3. Conditional Routing
def route_after_review(state: AgentState):
    """Conditional Edge logic."""
    if state["needs_human"]:
        return "human_approval_node"
    return END

# 4. Graph Construction
builder = StateGraph(AgentState)

builder.add_node("memory_node", memory_node)
builder.add_node("researcher_agent", researcher_agent)

# Add Native Fault Tolerance (retries up to 3 times if the LLM crashes)
builder.add_node("reviewer_agent", reviewer_agent, retry=RetryPolicy(max_attempts=3))

builder.add_node("human_approval_node", human_approval_node)

# 5. Edges & Flow
builder.add_edge(START, "memory_node")
builder.add_edge("memory_node", "researcher_agent")
builder.add_edge("researcher_agent", "reviewer_agent") # Multi-agent handoff
builder.add_conditional_edges("reviewer_agent", route_after_review)
builder.add_edge("human_approval_node", END)

# 6. Persistence & Compilation (Durable execution, Checkpoints)
memory_saver = MemorySaver()
agent_app = builder.compile(
    checkpointer=memory_saver,
    interrupt_before=["human_approval_node"] # Graph pauses here for Human-in-the-loop
)