from fastapi import APIRouter
from schemas.agent import AgentRequest, AgentResponse, ResumeRequest
from services.agent import agent

router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])

@router.post("/ask", response_model=AgentResponse)
async def ask_agent(request: AgentRequest):
    """Initiates a new reasoning loop for the agent to solve the given question."""
    result = await agent.run(
        question=request.question, 
        max_iterations=request.max_iterations
    )
    return AgentResponse(**result)

@router.post("/resume", response_model=AgentResponse)
async def resume_agent(request: ResumeRequest):
    """Resumes a suspended agent session following a Human-in-the-Loop decision."""
    result = await agent.resume(
        session_id=request.session_id,
        is_approved=request.is_approved
    )
    return AgentResponse(**result)