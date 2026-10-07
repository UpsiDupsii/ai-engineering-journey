from fastapi import APIRouter, Depends
from app.schemas.mcp import JsonRpcRequest, JsonRpcResponse
from app.services.mcp_server import mcp_server
from app.core.security import verify_mcp_token

router = APIRouter(prefix="/mcp", tags=["Model Context Protocol Server"])

@router.post("/rpc", response_model=JsonRpcResponse)
def handle_json_rpc(request: JsonRpcRequest, token: str = Depends(verify_mcp_token)):
    """Standard JSON-RPC 2.0 endpoint for MCP servers handling discovery and execution."""
    try:
        if request.method == "tools/list":
            return JsonRpcResponse(id=request.id, result={"tools": mcp_server.list_tools()})
            
        elif request.method == "tools/call":
            params = request.params or {}
            result = mcp_server.call_tool(params.get("name"), params.get("arguments", {}))
            return JsonRpcResponse(id=request.id, result={"content": [{"type": "text", "text": str(result)}]})
            
        elif request.method == "resources/list":
            return JsonRpcResponse(id=request.id, result={"resources": mcp_server.list_resources()})
            
        elif request.method == "resources/read":
            params = request.params or {}
            result = mcp_server.read_resource(params.get("uri"))
            return JsonRpcResponse(id=request.id, result={"contents": [{"uri": params.get("uri"), "text": str(result)}]})
            
        elif request.method == "prompts/list":
            return JsonRpcResponse(id=request.id, result={"prompts": mcp_server.list_prompts()})
            
        elif request.method == "prompts/get":
            params = request.params or {}
            result = mcp_server.get_prompt(params.get("name"), params.get("arguments", {}))
            return JsonRpcResponse(id=request.id, result={"description": result})
            
        else:
            return JsonRpcResponse(
                id=request.id, 
                error={"code": -32601, "message": f"Method '{request.method}' not found"}
            )
            
    except Exception as e:
        return JsonRpcResponse(
            id=request.id, 
            error={"code": -32000, "message": str(e)}
        )