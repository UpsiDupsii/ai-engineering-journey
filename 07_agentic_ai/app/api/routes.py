from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from agents.routing import route_query
from agents.workflows import sequential_workflow, parallel_workflow
from agents.executor import evaluator_optimizer_loop
from agents.human_loop import start_long_running_task, approve_task
from core.memory import save_memory, search_memory

router = APIRouter()

class QueryModel(BaseModel):
    query: str

@router.post("/route")
async def api_route_query(req: QueryModel):
    return {"routing_decision": await route_query(req.query)}

@router.post("/workflow/sequential")
async def api_sequential(req: QueryModel):
    return await sequential_workflow(req.query)

@router.post("/workflow/parallel")
async def api_parallel(req: QueryModel):
    return await parallel_workflow(req.query)

@router.post("/agent/optimize")
async def api_optimize(req: QueryModel):
    result = await evaluator_optimizer_loop(req.query)
    return {"final_output": result}

@router.post("/memory/save")
async def api_save_memory(req: QueryModel):
    await save_memory(req.query)
    return {"status": "Memory saved"}

@router.post("/memory/search")
async def api_search_memory(req: QueryModel):
    results = await search_memory(req.query)
    return {"memories": results}

@router.post("/task/start")
async def api_start_task(req: QueryModel):
    task_id = start_long_running_task(req.query)
    return {"task_id": task_id, "status": "PENDING_APPROVAL"}

@router.post("/task/{task_id}/approve")
async def api_approve_task(task_id: str):
    return approve_task(task_id)