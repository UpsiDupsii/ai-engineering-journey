from app.schemas.n8n import WorkflowRequest, NodeData
from app.services.agent_service import invoke_ai_agent

async def process_node(node: NodeData, context: dict):
    """Simulates n8n node execution logic including error handling and retries"""
    attempts = 0
    max_attempts = (node.retries or 0) + 1
    
    while attempts < max_attempts:
        try:
            if node.type == "ai_agent":
                prompt = node.parameters.get("prompt", "")
                result = await invoke_ai_agent(prompt)
                return {"status": "success", "data": result}
                
            elif node.type == "condition":
                expr = node.parameters.get("expression", "False")
                # Basic string eval for simulation purposes
                result = eval(expr, {}, context)
                return {"status": "success", "data": result}
                
            elif node.type == "http_request":
                # Simulate external REST API call
                url = node.parameters.get("url")
                return {"status": "success", "data": f"Mocked response from {url}"}
                
            else:
                return {"status": "skipped", "data": f"Unknown node type: {node.type}"}
                
        except Exception as e:
            attempts += 1
            if attempts >= max_attempts:
                return {"status": "error", "error": str(e)}

async def execute_workflow(workflow: WorkflowRequest):
    execution_context = {}
    node_results = []
    
    for node in workflow.nodes:
        # Simulate passing data from previous nodes
        result = await process_node(node, execution_context)
        execution_context[node.id] = result
        node_results.append({
            "node_id": node.id,
            "type": node.type,
            "output": result
        })
        
        # Stop workflow if a node fails completely
        if result.get("status") == "error":
            break
            
    return node_results