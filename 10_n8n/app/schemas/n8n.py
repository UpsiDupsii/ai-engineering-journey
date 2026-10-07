from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class NodeData(BaseModel):
    id: str
    type: str  # Options: trigger, ai_agent, http_request, condition, loop
    parameters: Dict[str, Any]
    retries: Optional[int] = 0

class WorkflowRequest(BaseModel):
    name: str
    nodes: List[NodeData]