from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union

# JSON-RPC 2.0 Base Primitives
class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int]
    method: str
    params: Optional[Dict[str, Any]] = None

class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int]
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

# MCP Protocol Specifications
class ToolParameterSchema(BaseModel):
    type: str = "object"
    properties: Dict[str, Any]
    required: List[str] = []

class ToolDefinition(BaseModel):
    name: str
    description: str
    inputSchema: ToolParameterSchema

class ResourceDefinition(BaseModel):
    uri: str
    name: str
    description: str
    mimeType: Optional[str] = "text/plain"

class PromptArgument(BaseModel):
    name: str
    description: str
    required: bool = True

class PromptDefinition(BaseModel):
    name: str
    description: str
    arguments: List[PromptArgument] = []

# Agent Interaction Request
class AgentQueryRequest(BaseModel):
    query: str
    client_scopes: List[str] = Field(default=["read"])