from fastapi import APIRouter
from app.schemas.n8n import WorkflowRequest
from app.services.workflow_service import execute_workflow

router = APIRouter()

@router.post("/webhook")
async def trigger_webhook(payload: dict):
    """Simulates an n8n Webhook Trigger Node"""
    return {
        "message": "Webhook received successfully",
        "authentication": "Verified",
        "payload": payload
    }

@router.post("/execute-workflow")
async def run_workflow(workflow: WorkflowRequest):
    """Executes a defined sequence of nodes"""
    results = await execute_workflow(workflow)
    return {
        "workflow_name": workflow.name,
        "execution_results": results
    }