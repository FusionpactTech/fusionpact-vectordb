# ⚡ FusionPact

### The Agent-Native Retrieval Engine

**Hybrid Vector + Reasoning + Memory for AI Agents**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Node](https://img.shields.io/badge/node-%3E%3D18-green.svg)](https://nodejs.org)
[![npm](https://img.shields.io/npm/v/fusionpact.svg)](https://www.npmjs.com/package/fusionpact)

> **Similarity ≠ Relevance.** FusionPact is the first retrieval engine that combines HNSW vector search, reasoning-based tree retrieval, and agent memory in a single platform — purpose-built for AI agents and multi-agent systems.

[Quickstart](#-quickstart) · [Hybrid Retrieval](#-hybrid-retrieval-engine) · [Agent Memory](#-agent-memory) · [Multi-Agent](#-multi-agent-orchestration) · [MCP Server](#-mcp-server) · [Tree Index](#-tree-index) · [RAG Pipeline](#-rag-pipeline) · [API Reference](#-api-reference) · [Benchmarks](#-benchmarks) · [Contributing](#-contributing)

---

## Why FusionPact?

Traditional vector databases retrieve what's **similar**. But similar ≠ relevant. Ask a vector DB for "Q3 2024 revenue" and you might get Q2 or Q4 data — semantically similar, but the **wrong answer**.

FusionPact solves this by combining **three retrieval paradigms**:

| Strategy | How It Works | Best For |
|---|---|---|
| **Vector Search** (HNSW) | Embedding similarity, O(log N) | Broad search across large collections |
| **Tree Reasoning** | LLM navigates document structure | Precise retrieval in structured documents |
| **Keyword Search** (BM25) | Term frequency matching | Exact match requirements |

Plus purpose-built **agent memory**, **multi-agent orchestration**, and **MCP server** — all zero-dependency, local-first, and free.

```
┌──────────────────────────────────────────────────────────┐
│             FusionPact Retrieval Engine                   │
│                                                          │
│  ┌────────────┐  ┌─────────────┐  ┌────────────────┐   │
│  │ Vector     │  │ Tree        │  │ Keyword        │   │
│  │ (HNSW)     │  │ (Reasoning) │  │ (BM25)         │   │
│  └─────┬──────┘  └──────┬──────┘  └───────┬────────┘   │
│        └────────────┬────┴─────────────────┘            │
│                     ▼                                    │
│           Reciprocal Rank Fusion                         │
│                     ▼                                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │        Agent Memory (Multi-Agent)                │   │
│  │  Episodic │ Semantic │ Procedural │ Shared       │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │        MCP Server (Claude, Cursor, etc.)         │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## ⚡ Quickstart

```bash
# Install
npm install fusionpact

# Run the demo
npx fusionpact demo

# Start HTTP + MCP server
npx fusionpact serve --port 8080

# Start MCP server for Claude Desktop
npx fusionpact mcp
```

### 10 Lines of Code

```javascript
const { create } = require('fusionpact');

const fp = create({ embedder: 'ollama' }); // or 'mock' for zero-config

// Ingest a document — auto-chunks, embeds, indexes
await fp.rag.ingest('Your document text here...', { source: 'doc.pdf' });

// Hybrid search — vector + reasoning + keyword, fused automatically
const results = await fp.retriever.retrieve('What safety protocols exist?', {
  collection: 'default',
  strategy: 'hybrid'
});

// Or build LLM-ready context directly
const context = await fp.rag.buildContext('What safety protocols exist?');
console.log(context.prompt); // Ready to paste into any LLM
```

---

## 🔀 Hybrid Retrieval Engine

The core differentiator: a single API that intelligently routes queries through multiple retrieval strategies and fuses results using Reciprocal Rank Fusion.

```javascript
const { create } = require('fusionpact');

const fp = create({
  embedder: 'ollama',        // Local, free, private
  llmProvider: 'ollama',     // For tree reasoning
  enableHybrid: true
});

// Index a structured document with tree structure
await fp.treeIndex.indexDocument('annual-report', reportText, {
  format: 'markdown'
});

// Hybrid retrieval — automatically uses the best strategy
const results = await fp.retriever.retrieve(
  'What were the total deferred tax assets in Q3?',
  {
    collection: 'documents',       // Vector search here
    docId: 'annual-report',        // Tree reasoning here
    topK: 5,
    strategy: 'hybrid'            // Fuse all strategies
  }
);

// Each result includes:
// - score: Fused relevance score
// - content: Retrieved text
// - sources: Which strategies contributed { vector: 0.8, tree: 0.9, keyword: 0.3 }
// - citation: "Section 3 > Financial Data > Table 3.2.1"
// - reasoning: Full tree traversal reasoning trace
```

### Strategy Weights

```javascript
const retriever = new HybridRetriever({
  engine, treeIndex, embedder,
  weights: {
    vector: 0.4,   // 40% weight to vector similarity
    tree: 0.4,     // 40% weight to reasoning-based retrieval
    keyword: 0.2   // 20% weight to keyword matching
  }
});
```

### Adaptive Learning

FusionPact learns which retrieval strategy works best for different query patterns:

```javascript
// Record feedback on result quality
retriever.recordFeedback('financial query', 'tree', 0.95);
retriever.recordFeedback('general search', 'vector', 0.85);

// Get recommended weights for a new query
const weights = retriever.getAdaptiveWeights('new financial query');
// → { vector: 0.25, tree: 0.6, keyword: 0.15 }
```

---

## 🌲 Tree Index

Reasoning-based retrieval for structured documents. Builds a hierarchical tree (like an intelligent table of contents) and uses LLM reasoning to navigate to the most relevant sections.

```javascript
const { TreeIndex, LLMProvider } = require('fusionpact');

const llm = new LLMProvider({ provider: 'ollama' }); // Free, local
const tree = new TreeIndex({ llmProvider: llm });

// Index a document
await tree.indexDocument('sec-filing', filingText, {
  format: 'markdown',
  metadata: { source: '10-K', year: 2024 }
});

// Reasoning-based search
const results = await tree.search('sec-filing', 'Total deferred tax assets', {
  maxResults: 3,
  includeReasoning: true
});

// results[0]:
// {
//   content: "Table 5.2: Deferred Tax Assets...",
//   relevanceScore: 0.95,
//   citation: "Financial Statements > Note 5 > Tax Assets > Table 5.2",
//   reasoningPath: [
//     { title: "Financial Statements", reasoning: "Deferred tax assets are in financial notes", action: "explore" },
//     { title: "Note 5: Income Taxes", reasoning: "This note covers tax-related assets", action: "explore" },
//     { title: "Table 5.2", reasoning: "Contains the deferred tax asset breakdown", action: "retrieve" }
//   ]
// }
```

### Works Without LLM Too

If no LLM provider is configured, TreeIndex falls back to keyword-based tree traversal — still useful, just without the reasoning path:

```javascript
const tree = new TreeIndex(); // No LLM — keyword fallback
await tree.indexDocument('doc', text, { format: 'markdown' });
const results = await tree.search('doc', 'safety protocols');
```

---

## 🧠 Agent Memory

Purpose-built memory system for AI agents with four memory types:

| Memory Type | What It Stores | Example |
|---|---|---|
| **Episodic** | Events, conversations, observations | "User asked about Lab B chemical storage" |
| **Semantic** | Facts, domain knowledge, learned info | "OSHA 1910.106 covers flammable liquids" |
| **Procedural** | Tool schemas, API specs, workflows | search_incidents tool definition |
| **Shared** | Cross-agent knowledge pool | "Customer ACME prefers ISO 14001" |

```javascript
const { create } = require('fusionpact');
const fp = create({ embedder: 'ollama', enableMemory: true });

// Episodic — remember what happened
await fp.memory.remember('agent-1', {
  content: 'User prefers dark mode and concise answers',
  role: 'system',
  importance: 0.8
});

// Semantic — learn knowledge
await fp.memory.learn('agent-1',
  'OSHA 29 CFR 1910 covers general industry safety standards.',
  { source: 'regulations', category: 'compliance' }
);

// Procedural — register tools
await fp.memory.registerTool('agent-1', {
  name: 'search_incidents',
  description: 'Search EHS incident reports by category and severity',
  schema: { type: 'object', properties: { severity: { type: 'string' } } }
});

// Recall — cross-memory search
const memories = await fp.memory.recall('agent-1', 'safety compliance');
// → { episodic: [...], semantic: [...], procedural: [...], shared: [...] }

// Conversation memory
fp.memory.addMessage('agent-1', 'thread-001', { role: 'user', content: 'What are the PPE requirements?' });
fp.memory.addMessage('agent-1', 'thread-001', { role: 'assistant', content: 'PPE requirements include...' });
const history = fp.memory.getConversation('agent-1', 'thread-001');

// GDPR-friendly forget
fp.memory.forget('agent-1', { type: 'all' });
```

---

## 🤖 Multi-Agent Orchestration

Coordinate multiple AI agents with isolated memory, shared knowledge, and message routing:

```javascript
const { create, AgentOrchestrator } = require('fusionpact');

const fp = create({ embedder: 'ollama', enableMemory: true });
const orchestrator = new AgentOrchestrator({
  engine: fp.engine,
  memory: fp.memory,
  retriever: fp.retriever
});

// Register agents
orchestrator.registerAgent({
  agentId: 'researcher',
  name: 'Research Agent',
  role: 'Find and analyze information',
  capabilities: ['search', 'analysis', 'summarization']
});

orchestrator.registerAgent({
  agentId: 'writer',
  name: 'Writing Agent',
  role: 'Generate reports and documentation',
  capabilities: ['writing', 'formatting', 'editing']
});

// Agent-to-agent communication
await orchestrator.send({
  from: 'researcher',
  to: 'writer',
  type: 'result',
  payload: { findings: 'Safety incidents decreased 12% YoY...' }
});

// Capability-based task delegation
await orchestrator.delegate('coordinator', 'Write a safety summary report', {
  requiredCapabilities: ['writing', 'formatting']
});
// → Automatically routes to 'writer' agent

// Collaborative retrieval across all agents
const results = await orchestrator.collaborativeRecall('safety compliance');
// → Returns memories from all agents, plus shared knowledge

// Message handling
orchestrator.onMessage('writer', async (msg) => {
  console.log(`Writer received: ${msg.type} from ${msg.from}`);
  // Process task...
});
```

---

## 🔌 MCP Server

FusionPact ships as an MCP (Model Context Protocol) server. Any AI agent (Claude, Cursor, Windsurf) can use it as persistent memory — no custom integration needed.

### Claude Desktop Setup

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fusionpact": {
      "command": "npx",
      "args": ["fusionpact", "mcp"],
      "env": {
        "EMBEDDING_PROVIDER": "ollama"
      }
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|---|---|
| `fusionpact_create_collection` | Create HNSW-indexed vector collection |
| `fusionpact_search` | Semantic vector search |
| `fusionpact_hybrid_search` | Hybrid retrieval (vector + tree + keyword) |
| `fusionpact_rag_ingest` | One-click RAG ingestion |
| `fusionpact_rag_query` | Build LLM-ready context |
| `fusionpact_memory_remember` | Store episodic memory |
| `fusionpact_memory_recall` | Recall relevant memories |
| `fusionpact_memory_learn` | Add semantic knowledge |
| `fusionpact_memory_share` | Share cross-agent knowledge |
| `fusionpact_memory_forget` | GDPR-style memory erasure |
| `fusionpact_memory_conversation` | Manage conversation threads |

---

## 📄 RAG Pipeline

End-to-end RAG in one call:

```javascript
const fp = require('fusionpact').create({ embedder: 'ollama' });

// Ingest — auto-chunks, embeds, indexes
await fp.rag.ingest(documentText, {
  source: 'safety-manual.pdf',
  title: 'Safety Manual 2024'
});

// Build context for any LLM
const ctx = await fp.rag.buildContext('What PPE is required?', {
  topK: 5,
  maxTokens: 4000,
  strategy: 'hybrid'  // Uses HybridRetriever if available
});

// ctx.prompt → Ready for any LLM
// ctx.sources → Source citations
// ctx.chunks → Number of chunks used
```

### Chunking Strategies

```javascript
const rag = new RAGPipeline(engine, {
  chunkStrategy: 'recursive',  // 'recursive' | 'sentence' | 'paragraph'
  chunkSize: 512,
  chunkOverlap: 50
});
```

---

## 🔒 Multi-Tenancy

Zero-trust soft-isolation — tenants can never see each other's data:

```javascript
const tenantA = engine.tenant('shared-collection', 'acme_corp');
const tenantB = engine.tenant('shared-collection', 'globex_inc');

tenantA.insert([{ id: 'doc-1', vector: [...], metadata: { doc: 'Acme Plan' } }]);

// Tenant A queries — only sees Acme data. Always.
const results = tenantA.search(queryVec, { topK: 10 });
```

---

## 🔌 Embedding Providers

| Provider | Setup | Dimensions | Cost |
|---|---|---|---|
| **Ollama** (recommended) | `ollama pull nomic-embed-text` | 768 | Free |
| **OpenAI** | Set `OPENAI_API_KEY` | 1536 | ~$0.02/1M tokens |
| **Mock** (testing) | None | 64 | Free |

```javascript
// Ollama (local, free, private)
const fp = create({ embedder: 'ollama' });

// OpenAI
const fp = create({ embedder: 'openai', openaiConfig: { apiKey: 'sk-...' } });

// Mock (for demos/testing — no dependencies)
const fp = create({ embedder: 'mock' });
```

---

## 📊 Benchmarks

### HNSW Performance (128D vectors)

| Vectors | Insert | Search (p50) | QPS |
|---|---|---|---|
| 1,000 | 15ms | 0.2ms | ~5,000 |
| 10,000 | 180ms | 0.3ms | ~3,300 |
| 100,000 | 2.8s | 0.5ms | ~2,000 |

Run your own:

```bash
npx fusionpact bench --count 10000
```

---

## 🆚 Comparison

| Feature | FusionPact | PageIndex | Pinecone | Chroma | Qdrant |
|---|---|---|---|---|---|
| **Hybrid Retrieval (Vector+Tree+Keyword)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Reasoning-Based Tree Index** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Agent Memory Architecture** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Multi-Agent Orchestration** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **MCP Server (Agent-Native)** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **One-Click RAG** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Multi-Tenancy** | ✅ | ❌ | ✅ | ❌ | ✅ |
| **Local-First / Zero-Cost** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **HNSW Vector Index** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Zero Dependencies** | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## 📖 API Reference

Full documentation: [docs/API.md](docs/API.md)

### Core Classes

| Class | Description |
|---|---|
| `FusionEngine` | Core database engine, collection management, CRUD |
| `HNSWIndex` | HNSW approximate nearest neighbor index |
| `TreeIndex` | Hierarchical document index for reasoning retrieval |
| `HybridRetriever` | Multi-strategy retrieval with rank fusion |
| `AgentMemory` | Multi-type agent memory system |
| `AgentOrchestrator` | Multi-agent coordination layer |
| `RAGPipeline` | End-to-end RAG pipeline |
| `MCPServer` | Model Context Protocol server |
| `OllamaEmbedder` | Ollama embedding provider |
| `OpenAIEmbedder` | OpenAI embedding provider |
| `MockEmbedder` | Testing/demo embedder |
| `LLMProvider` | Multi-provider LLM interface |

---

## 🗺 Roadmap

- [x] HNSW indexing with configurable M/ef parameters
- [x] Multi-tenancy with soft-isolation
- [x] One-Click RAG pipeline
- [x] Agent Memory (episodic, semantic, procedural, shared)
- [x] Multi-agent orchestration
- [x] Tree Index (reasoning-based retrieval)
- [x] Hybrid Retriever (vector + tree + keyword fusion)
- [x] MCP server (stdio + HTTP)
- [x] HTTP API server
- [x] Ollama + OpenAI embedding providers
- [x] Adaptive retrieval learning
- [ ] SQLite/PostgreSQL persistence
- [ ] Python SDK (`pip install fusionpact`)
- [ ] LangChain integration
- [ ] LlamaIndex integration
- [ ] CrewAI / AutoGen integration
- [ ] Vision RAG (PDF page images)
- [ ] Rust core (NAPI bindings)
- [ ] FusionPact Cloud (managed hosting)
- [ ] Dashboard UI

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/FusionpactTech/fusionpact-vectordb.git
cd fusionpact-vectordb
npm install
npm test
npx fusionpact demo
```

---

## 📜 Attribution

FusionPact is built and maintained by **[FusionPact Technologies Inc.](https://fusionpact.com)**

If you use FusionPact in your project, please include attribution in one of the following ways:

- Include "Powered by FusionPact" in your application's about page or documentation
- Keep the `NOTICE` file in your distribution
- Reference FusionPact Technologies Inc. in your project's acknowledgements

See [ATTRIBUTION.md](ATTRIBUTION.md) for full details.

## License

[Apache 2.0](LICENSE) — Use freely in commercial and open-source projects.

The Apache 2.0 license requires that you:
1. Include a copy of the license in any redistribution
2. Include the NOTICE file with attribution to FusionPact Technologies Inc.
3. State any significant changes you made to the code

---

**Built with ❤️ by [FusionPact Technologies Inc.](https://fusionpact.com)**

⭐ Star this repo if you find it useful!
