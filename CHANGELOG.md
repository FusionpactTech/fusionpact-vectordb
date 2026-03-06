# Changelog

All notable changes to FusionPact will be documented in this file.

## [2.0.0] - 2026-03-06

### 🚀 Major Release — The Agent-Native Retrieval Engine

This release transforms FusionPact from a vector database into a full **hybrid retrieval engine** combining vector search, reasoning-based tree retrieval, and agent memory.

### Added

- **TreeIndex** — Hierarchical document indexing with LLM-powered reasoning-based retrieval. Inspired by the insight that similarity ≠ relevance. Supports Markdown, HTML, and plain text.
- **HybridRetriever** — Multi-strategy retrieval combining vector (HNSW), tree reasoning, and keyword (BM25) search with Reciprocal Rank Fusion.
- **Adaptive Retrieval** — System learns which strategy works best for different query patterns.
- **Multi-Agent Orchestration** — `AgentOrchestrator` for coordinating multiple AI agents with message routing, capability-based task delegation, and collaborative retrieval.
- **Shared Memory** — Cross-agent knowledge sharing pool with access control.
- **Conversation Memory** — Thread-aware chat history management per agent.
- **LLM Provider** — Unified interface for Ollama, OpenAI, and Anthropic for tree reasoning.
- **HTTP API Server** — RESTful API alongside MCP server.
- **Comprehensive Documentation** — Full API reference, architecture docs, integration guides.
- **ATTRIBUTION.md** — Clear attribution guidelines for users.

### Changed

- Repositioned from "Agent-Native Vector Database" to "Agent-Native Retrieval Engine"
- Enhanced MCP server with hybrid retrieval and orchestration tools
- Improved RAG pipeline with hybrid retrieval support
- Better CLI with demo, serve, mcp, and bench commands

### Architecture

```
FusionPact 2.0 Architecture:
├── Core: FusionEngine + HNSWIndex
├── Index: TreeIndex (reasoning-based)
├── Retrieval: HybridRetriever (vector + tree + keyword fusion)
├── Memory: AgentMemory (episodic + semantic + procedural + shared)
├── Orchestration: AgentOrchestrator (multi-agent)
├── RAG: RAGPipeline (one-click ingestion)
├── MCP: MCPServer (stdio + HTTP)
└── Embedders: Ollama, OpenAI, Mock
```

## [1.0.0] - 2025-01-15

### Initial Release

- HNSW vector indexing
- Agent Memory (episodic, semantic, procedural)
- MCP server
- RAG pipeline
- Multi-tenancy
- Ollama + OpenAI embedders

---

Built by [FusionPact Technologies Inc.](https://fusionpact.com)
