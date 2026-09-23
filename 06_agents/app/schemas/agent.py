from pydantic import BaseModel, Field
from typing import Optional

class AgentRequest(BaseModel):
    """Payload schema for initiating an agent task."""
    question: str = Field(..., description="The question or task for the agent.")
    max_iterations: int = Field(default=5, ge=1, le=15, description="Maximum allowed ReAct loop iterations.")

class AgentResponse(BaseModel):
    """Schema for the agent's final or paused state response."""
    status: str
    answer: str
    iterations: int
    
    # Optional fields populated only during a Human-in-the-Loop suspension
    session_id: Optional[str] = Field(default=None, description="UUID for resuming paused sessions.")
    action: Optional[str] = Field(default=None, description="The tool requested before pausing.")
    action_input: Optional[str] = Field(default=None, description="The input for the requested tool.")

class ResumeRequest(BaseModel):
    """Payload schema for resuming a Human-in-the-Loop suspended session."""
    session_id: str = Field(..., description="The UUID of the paused session.")
    is_approved: bool = Field(..., description="True to execute the pending tool, False to deny.")