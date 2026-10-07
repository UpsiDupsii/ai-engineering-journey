from fastapi import APIRouter
from app.schemas.mcp import AgentQueryRequest
from app.services.agent_service import run_mcp_agent

router = APIRouter(prefix="/agent", tags=["MCP Agent Controller"])

@router.post("/query")
async def execute_agent_query(request: AgentQueryRequest):
    """Entry point for users to query the local Agent backed by MCP capabilities."""
    return await run_mcp_agent(user_query=request.query, scopes=request.client_scopes)