# AI Engineering Journey

A comprehensive, production-oriented repository capturing my daily hands-on implementation of Core AI, Advanced Retrieval, and Multi-Agent Orchestration frameworks.

---

## The Architecture & Engineering Stack
- **API Engine:** FastAPI (Production-grade, asynchronous, type-safe API schemas)
- **Local Runtime:** Ollama (Local developer inference & iteration)
- **Production Concept Engine:** vLLM (High-throughput, continuous batching optimization paradigms)
- **Vector Compute:** Milvus (Enterprise-scale vector database engine)
- **Orc

---

## Master Curriculum & Roadmap

### 01. LLM Fundamentals
- **Token Mechanics:** Tokens, Tokenization, Input vs. Output Tokens.
- **Context Management:** Context Window, Context Overflow, Context Compression.
- **Under the Hood:** Transformer Basics, Attention Mechanisms, Inference Execution.
- **Execution Tuning:** Parameters, Temperature, Top-p, Sampling Strategies, Deterministic Generation.
- **Behavioral Control:** Hallucination mitigation, Knowledge Cutoff, System Prompts, User/Assistant Message Patterns.
- **Prompt Architectures:** Prompt Engineering, Few-shot Prompting, Chain-of-Thought Concepts.
- **Data Serialization:** Structured Output, Native JSON Mode.
- **Performance Evaluation:** Streaming Response Optimization, Latency Analysis, Throughput Squeezing.
- **Infrastructure Constraints:** Model Quantization, Model Size Scaling, CPU/GPU Inference Profiles.

### 02. Embeddings & Semantic Search
- **Vector Representations:** Mathematical Meanings, Embedding Dimensions, Dense vs. Sparse Vectors.
- **Distance Metrics:** Cosine Similarity, Dot Product, Euclidean Distance, Array Normalization.
- **Retrieval Types:** Semantic Similarity, Text/Image Embeddings, Multimodal Embeddings, Cross-modal Retrieval.

### 03. Vector Databases (Engine: Milvus)
- **Database Architecture:** Collections, Vector Arrays, Metadata Storage, Payloads.
- **Search Optimization:** Approximate Nearest Neighbor (ANN), HNSW (Hierarchical Navigable Small World), IVF (Inverted File Index).
- **Query Processing:** Similarity Threshold Tuning, Top-k Document Retrieval, Strict Metadata Filtering.
- **Hybrid Search Pipelines:** Composing Dense Search, Sparse Search, and Structural Reranking at Scale.

### 04. Retrieval-Augmented Generation (RAG)
- **Ingestion Pipelines:** Document Loaders, Parsing, Chunking Strategy (Chunk Size vs. Chunk Overlap), Array Embeddings.
- **Advanced Retrieval:** Top-k Optimization, Complex Metadata Filtering, Hybrid Retrieval Networks.
- **Context Refining:** Reranking, Query Rewriting, Query Decomposition, Context Construction.
- **Safety & Quality:** Grounding, Native Citations, Context Window Limitation Engineering.
- **Failure Analysis:** RAG Hallucination Detection, Retrieval Failures, Generation Breakdowns, Systematic RAG Evaluation.
- **Evolutionary Topologies:** Agentic RAG, Multimodal RAG Systems.

### 05. Native Tool Calling
- **Interface Definitions:** Function Declarations, JSON Schema Compliance, Tool Descriptions.
- **Argument Resolution:** Argument Extraction, Local Execution, Tool Result Plumbing.
- **Advanced Execution Flow:** Multi-tool Routing, Automated Tool Selection, Schema Validation.
- **Resilience Engineering:** Error Handling, Automated Retry Policies, Network Timeouts, Permissions, and Execution Sandbox Security.

### 06. Sovereign Agents
- **Core Loop:** Reason → Act → Observe Execution Archetypes.
- **Runtime Autonomy:** Dynamic Planning, Tool Selection, Multi-tool Iteration.
- **State Management:** Short-term Working Memory, Persistent Context Windows.
- **Boundary Controls:** Deterministic Stopping Conditions, Max Iteration Gates, Resilient Failure Recovery Strategies.
- **Threat Vector Control:** Prompt Injection Shielding, Tool Injection Auditing, Guardrail Architectures, Human-in-the-Loop Intercepts.
- **Architectural Scaling:** Multi-Agent Networks, Controlled Subagent Delegations.

### 07. Agentic AI & Advanced Workflows
- **Topological Layouts:** Fixed Linear Workflows vs. Fully Autonomous Dynamic Routing.
- **Optimization Patterns:** Evaluator-Optimizer Pipelines, Strategic Planning Modules, Self-Reflection Loops.
- **Execution Parallelism:** Parallel Tool Execution, Sequential Step Dependencies.
- **Production Operations:** Long-running Task Management, Human Approval Gatekeepers, State Persistence Engines, and Native Fault Tolerance.

### 08. Enterprise Orchestration (LangChain)
- **Component Primitives:** Models, System/User Messages, Dynamic Prompts, Output Parsers.
- **Structured Contracts:** Type-safe Structured Output Drivers.
- **Data Connections:** Tool Integrations, Custom Retrievers, Document Loaders, Strategic Text Splitters, Vector Store Wrappers.
- **Execution Chains:** LangChain Expression Language (LCEL) Chains, Agent Runtimes.
- **System Observability:** Custom Middleware, Real-time Streaming, Callbacks, Tracing Integrations.

### 09. State Graph Engineering (LangGraph)
- **Graph Foundations:** Graph State Schemas, Computational Nodes, Directed Edges, Conditional Edges, Loops.
- **Engine Execution:** Graph Run-loop Orchestration.
- **Persistence Operations:** In-memory & Database Checkpoints, State Time-Travel, Short-term vs. Long-term Memory Networks.
- **Production Patterns:** Human-in-the-Loop Interruption Gates, Durable Graph Execution, Core Fault Tolerance, Multi-Agent Inter-Graph Routing.
- **Monitoring:** Multi-stage Graph Streaming, Step-by-step Debugging.

### 10. Visual Workflow Automation (n8n)
- **Infrastructure Primitives:** Event Triggers, Execution Nodes, Asynchronous Workflows.
- **Network Interfaces:** Webhooks, Native HTTP Request Blocks, REST API Consumption, Multi-spec Authentication.
- **Logic Processing:** JavaScript-based Data Expressions, Conditions, Loops, System Schedulers.
- **Resiliency:** Granular Error Handling, Advanced Step Retry Topologies.
- **GenAI Extension:** Advanced AI Nodes, Specialized Agent Nodes, Native Tool Integration Blueprints.

### 11. Model Context Protocol (MCP)
- **Protocol Foundations:** Client-Server Architecture specifications.
- **Standardized Interfaces:** Unified Tool schemas, External Resource access, Standard Prompts.
- **Discovery Operations:** Native Tool Discovery Protocols, Execution Permissions, Boundary Security.
- **Deployments:** Building Custom Local MCP Servers, Integrating MCP Tool Providers with Autonomous Graph Agents.