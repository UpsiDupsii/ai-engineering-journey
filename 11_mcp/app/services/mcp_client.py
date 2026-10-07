from typing import Any, Dict
from app.services.mcp_server import mcp_server
from app.core.security import check_tool_permission

class MCPClient:
    """Client interface interacting with local or remote MCP servers."""
    
    def __init__(self, server=mcp_server):
        self.server = server

    def discover_tools(self) -> list[dict]:
        """Tool Discovery: queries server capabilities dynamically."""
        return self.server.list_tools()

    def discover_resources(self) -> list[dict]:
        return self.server.list_resources()

    def discover_prompts(self) -> list[dict]:
        return self.server.list_prompts()

    def call_tool(self, name: str, arguments: Dict[str, Any], scopes: list[str]) -> Dict[str, Any]:
        """Enforces security boundaries before invoking a tool."""
        if not check_tool_permission(name, scopes):
            raise PermissionError(f"Client lacks required permission scope to execute tool: '{name}'")
        
        result = self.server.call_tool(name, arguments)
        return {"tool": name, "result": result}

    def read_resource(self, uri: str) -> Any:
        return self.server.read_resource(uri)

    def get_prompt(self, name: str, arguments: Dict[str, Any]) -> str:
        return self.server.get_prompt(name, arguments)

mcp_client = MCPClient()