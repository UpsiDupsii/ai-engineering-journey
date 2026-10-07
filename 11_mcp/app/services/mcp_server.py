from typing import Dict, Any, Callable
from app.schemas.mcp import ToolDefinition, ResourceDefinition, PromptDefinition
from app.services.milvus_service import query_vector_memory

class MCPServer:
    def __init__(self, name: str):
        self.name = name
        self.tools: Dict[str, dict] = {}
        self.resources: Dict[str, dict] = {}
        self.prompts: Dict[str, dict] = {}
        self._register_default_primitives()

    def register_tool(self, definition: ToolDefinition, handler: Callable):
        self.tools[definition.name] = {
            "definition": definition,
            "handler": handler
        }

    def register_resource(self, definition: ResourceDefinition, content_resolver: Callable):
        self.resources[definition.uri] = {
            "definition": definition,
            "resolver": content_resolver
        }

    def register_prompt(self, definition: PromptDefinition, template_generator: Callable):
        self.prompts[definition.name] = {
            "definition": definition,
            "generator": template_generator
        }

    def _register_default_primitives(self):
        # 1. Register Tool
        self.register_tool(
            definition=ToolDefinition(
                name="search_vector_memory",
                description="Query the Milvus vector database for contextual memory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Semantic search query"}
                    },
                    "required": ["query"]
                }
            ),
            handler=lambda params: query_vector_memory(params.get("query", ""))
        )

        # 2. Register Resource
        self.register_resource(
            definition=ResourceDefinition(
                uri="milvus://system/metadata",
                name="Milvus Metadata State",
                description="Physical database connection details and active collections.",
                mimeType="application/json"
            ),
            content_resolver=lambda: {"database": "default", "status": "connected", "active_partitions": 1}
        )

        # 3. Register Prompt
        self.register_prompt(
            definition=PromptDefinition(
                name="system_summary_prompt",
                description="Standard prompt template for generating architectural summaries.",
                arguments=[{"name": "system_name", "description": "Name of the target architecture", "required": True}]
            ),
            template_generator=lambda args: f"Summarize current health and operational parameters for system: {args.get('system_name')}"
        )

    # Discovery & Invocation Handlers
    def list_tools(self) -> list[dict]:
        return [tool["definition"].model_dump() for tool in self.tools.values()]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found on server.")
        return self.tools[name]["handler"](arguments)

    def list_resources(self) -> list[dict]:
        return [res["definition"].model_dump() for res in self.resources.values()]

    def read_resource(self, uri: str) -> Any:
        if uri not in self.resources:
            raise ValueError(f"Resource with URI '{uri}' not found.")
        return self.resources[uri]["resolver"]()

    def list_prompts(self) -> list[dict]:
        return [prompt["definition"].model_dump() for prompt in self.prompts.values()]

    def get_prompt(self, name: str, arguments: Dict[str, Any]) -> str:
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not registered.")
        return self.prompts[name]["generator"](arguments)

mcp_server = MCPServer(name="FastAPI-Local-MCP-Server")