# FusionPact Architecture

> Technical architecture of the Agent-Native Retrieval Engine
> Built by FusionPact Technologies Inc.

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     FusionPact v2.0                               │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Application Layer                         │ │
│  │  CLI  │  HTTP API  │  MCP Server  │  SDK (Node.js)          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   Orchestration Layer                        │ │
│  │  AgentOrchestrator: Multi-agent messaging, delegation       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Retrieval Layer                           │ │
│  │  HybridRetriever: Vector + Tree + Keyword → RRF Fusion     │ │
│  │  RAGPipeline: Chunk → Embed → Index → Context              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│  ┌──────────────┐ ┌────────────────┐ ┌───────────────────────┐  │
│  │ HNSW Index   │ │ Tree Index     │ │ Agent Memory          │  │
│  │ (Vector)     │ │ (Reasoning)    │ │ (Epi/Sem/Pro/Shared)  │  │
│  └──────────────┘ └────────────────┘ └───────────────────────┘  │
│                              │                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Storage Layer                             │ │
│  │  FusionEngine: Collections, Multi-tenancy, Persistence      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Provider Layer                            │ │
│  │  Ollama │ OpenAI │ Anthropic │ Mock (Embedders + LLMs)     │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## HNSW Algorithm

FusionPact implements the Hierarchical Navigable Small World graph for approximate nearest neighbor search:

- **Build Phase**: Insert vectors with probabilistic level assignment. Each level is a navigable small world graph with bidirectional edges.
- **Search Phase**: Greedy descent from top layer to base layer, then expand search at layer 0 using beam search with `efSearch` candidates.
- **Complexity**: O(log N) search, O(N log N) build.
- **Tuning**: `M` controls connectivity (recall vs memory), `efConstruction` controls build quality, `efSearch` controls search quality.

## Tree Index (Reasoning-Based Retrieval)

Inspired by the insight that similarity ≠ relevance:

1. **Parse**: Document → sections based on headings/structure
2. **Build**: Sections → hierarchical tree (parent-child relationships)
3. **Summarize**: Bottom-up LLM summarization of each node
4. **Search**: Top-down LLM reasoning — at each level, the LLM evaluates which children are most relevant to the query and descends into the most promising branches

This mimics how a human expert navigates a document: read the table of contents, reason about which section likely contains the answer, drill down.

## Hybrid Retrieval

The HybridRetriever orchestrates multiple strategies:

1. **Vector search** over the HNSW index (fast, broad)
2. **Tree reasoning** over the TreeIndex (precise, structured)
3. **Keyword matching** via BM25-style scoring (exact match)
4. **Reciprocal Rank Fusion** combines ranked lists into a single ranking

RRF formula: `score(d) = Σ weight_i / (k + rank_i(d))`

## Multi-Tenancy

Soft-isolation using `_tenant_id` metadata tags:
- Inserts are auto-tagged with the tenant's ID
- Searches are auto-filtered to only return the tenant's data
- No separate indexes needed — single index, automatic filtering

## Design Principles

1. **Zero Dependencies**: Core modules have no external npm dependencies
2. **Local-First**: Works entirely offline with Ollama
3. **Agent-Native**: Every feature designed for AI agent use cases
4. **Composable**: Use individual modules or the full stack
5. **Progressive Enhancement**: Works without LLM (keyword fallback), better with LLM

---

*Built by [FusionPact Technologies Inc.](https://fusionpact.com)*
