from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

security = HTTPBearer()

def verify_mcp_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Enforces authentication for MCP server communications."""
    if credentials.credentials != settings.mcp_server_token:
        raise HTTPException(status_code=401, detail="Invalid or missing MCP authorization token")
    return credentials.credentials

def check_tool_permission(tool_name: str, granted_scopes: list[str]) -> bool:
    """Validates if the client has permissions to execute the requested tool."""
    risk_matrix = {
        "search_vector_memory": "read",
        "calculate_metrics": "read",
        "execute_system_command": "admin"
    }
    required_scope = risk_matrix.get(tool_name, "read")
    return required_scope in granted_scopes