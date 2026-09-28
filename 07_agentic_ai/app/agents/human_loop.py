from typing import Dict, Any
import uuid

# In-memory state dictionary to simulate database tracking
TASK_DB: Dict[str, Dict[str, Any]] = {}

def start_long_running_task(task_name: str) -> str:
    """Initiates a task that requires human approval."""
    task_id = str(uuid.uuid4())
    TASK_DB[task_id] = {
        "name": task_name,
        "status": "PENDING_APPROVAL",
        "result": None
    }
    return task_id

def approve_task(task_id: str) -> dict:
    """Human approval triggers the continuation of the task."""
    if task_id not in TASK_DB:
        return {"error": "Task not found"}
        
    if TASK_DB[task_id]["status"] == "PENDING_APPROVAL":
        TASK_DB[task_id]["status"] = "COMPLETED"
        TASK_DB[task_id]["result"] = f"Action '{TASK_DB[task_id]['name']}' executed successfully."
        return TASK_DB[task_id]
        
    return {"error": "Task already processed"}
