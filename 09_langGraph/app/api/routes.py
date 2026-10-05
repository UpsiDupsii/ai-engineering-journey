from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from app.services.graph import agent_app

router = APIRouter(prefix="/agent", tags=["Advanced LangGraph Agent"])

class ChatRequest(BaseModel):
    thread_id: str
    query: str

class ApprovalRequest(BaseModel):
    thread_id: str
    feedback: str

@router.post("/chat")
async def chat(request: ChatRequest):
    """Standard execution with Streaming."""
    config = {"configurable": {"thread_id": request.thread_id}}
    initial_state = {"messages": [HumanMessage(content=request.query)]}
    
    async def stream_generator():
        # Streaming chunks as they execute through nodes
        async for event in agent_app.astream(initial_state, config=config, stream_mode="values"):
            if "messages" in event:
                yield f"data: {event['messages'][-1].content}\n\n"
            
        # Check if paused for human
        state = agent_app.get_state(config)
        if state.next and "human_approval_node" in state.next:
            yield "data: [SYSTEM: Execution paused. Human input required.]\n\n"
            
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@router.post("/approve")
async def approve(request: ApprovalRequest):
    """Human-in-the-loop (HITL) resume."""
    config = {"configurable": {"thread_id": request.thread_id}}
    state = agent_app.get_state(config)
    
    if not state.next:
        raise HTTPException(status_code=400, detail="Graph is not waiting for approval.")
    
    # Inject human feedback and resume
    agent_app.update_state(config, {"messages": [HumanMessage(content=request.feedback)], "needs_human": False})
    final_state = agent_app.invoke(None, config) # Pass None to resume from checkpoint
    
    return {"status": "Resumed", "final_response": final_state["messages"][-1].content}

@router.get("/time-travel/{thread_id}")
async def time_travel(thread_id: str):
    """Debugging: Fetch history and fork past states."""
    config = {"configurable": {"thread_id": thread_id}}
    history = list(agent_app.get_state_history(config))
    
    if not history:
        return {"message": "No history found."}
        
    # Return checkpoint IDs to allow front-end to "rewind" to exact graph states
    checkpoints = [{"checkpoint_id": h.config["configurable"]["checkpoint_id"], "state": h.values} for h in history]
    return {"thread_id": thread_id, "checkpoints": checkpoints}