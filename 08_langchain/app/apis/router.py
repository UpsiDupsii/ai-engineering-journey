from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from schemas.common import QueryRequest, RAGSetupRequest
from services.llm_service import llm_service
from services.rag_service import rag_service
from services.agent_service import agent_service

api_router = APIRouter()

@api_router.post("/lesson-01-models-messages")
async def lesson_models_messages(request: QueryRequest):
    result = llm_service.generate_basic_response(request.query)
    return {"result": result}

@api_router.post("/lesson-02-prompts-parsers-chains")
async def lesson_prompts_parsers_chains(request: QueryRequest):
    result = llm_service.generate_with_prompt_and_parser(request.query)
    return {"result": result}

@api_router.post("/lesson-03-structured-output")
async def lesson_structured_output(request: QueryRequest):
    result = llm_service.extract_structured_data(request.query)
    return result

@api_router.post("/lesson-04-rag-setup")
async def lesson_rag_setup(request: RAGSetupRequest):
    # Covers Document Loaders, Text Splitters, Vector Stores
    result = rag_service.load_and_index_documents(request.text_content)
    return result

@api_router.post("/lesson-05-rag-retrieve")
async def lesson_rag_retrieve(request: QueryRequest):
    # Covers Retrievers
    result = rag_service.retrieve_and_answer(request.query)
    return {"retrieved_documents": result}

@api_router.post("/lesson-06-tools-agents")
async def lesson_tools_agents(request: QueryRequest):
    # Covers Tools, Agents, AgentExecutor
    result = agent_service.run_agent(request.query)
    return {"agent_response": result}

@api_router.post("/lesson-07-streaming")
async def lesson_streaming(request: QueryRequest):
    async def generator():
        async for chunk in llm_service.stream_response(request.query):
            yield chunk
    return StreamingResponse(generator(), media_type="text/plain")

@api_router.get("/lesson-08-observability-middleware")
async def lesson_observability_middleware():
    # Middleware adds execution time headers.
    # Callbacks are demonstrated in the logs.
    return {
        "message": "Check the X-Process-Time header in the response, and look at the terminal for LangChain callback logs."
    }