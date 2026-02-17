<div align="center">

# ⚡ FusionPact

### The Agent-Native Vector Database

**HNSW Indexing · Built-in RAG · MCP Server · Multi-Tenancy · Agent Memory**

Add AI memory to any agent in 30 seconds.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Node](https://img.shields.io/badge/node-%3E%3D18-green.svg)](https://nodejs.org)
[![npm](https://img.shields.io/npm/v/fusionpact.svg)](https://www.npmjs.com/package/fusionpact)

[Quickstart](#-quickstart) · [MCP Integration](#-mcp-server-for-ai-agents) · [Documentation](#-documentation) · [Benchmarks](#-performance) · [Contributing](#-contributing)

</div>

---

## Why FusionPact?

Every vector database today is a **generic data store**. You bolt on RAG pipelines, build custom agent memory, and write MCP glue code yourself.

FusionPact is different. It's the **first vector database built specifically for AI agents**:

- 🔌 **MCP Server built-in** — Claude, Cursor, and any MCP client can use it as memory *instantly*
- 🧠 **Agent Memory Architecture** — Episodic, semantic, and procedural memory as first-class primitives
- 📄 **One-Click RAG** — Text → chunks → embeddings → searchable context in one call
- 🔒 **Multi-Tenancy** — Zero-trust soft-isolation with automatic tenant filtering
- ⚡ **HNSW Indexing** — O(log N) approximate nearest neighbor search
- 🆓 **Zero-Cost** — Local-first, runs on your machine, no API keys required

## ⚡ Quickstart

```bash
# Install
npm install fusionpact

# Run the demo
npx fusionpact demo

# Start HTTP API server
npx fusionpact serve --port 8080

# Start MCP server (for Claude Desktop)
npx fusionpact mcp
```

### 10 Lines of Code

```javascript
const { FusionEngine, RAGPipeline } = require('fusionpact');

const engine = new FusionEngine();
const rag = new RAGPipeline(engine, { embedder: 'ollama' });

// Ingest any text — auto-chunks, embeds, and indexes
await rag.ingest('Your document text here...', { source: 'doc.pdf' });

// Search with natural language
const context = await rag.buildContext('What safety protocols exist?');
console.log(context.prompt); // Ready to paste into any LLM
```

## 🔌 MCP Server for AI Agents

FusionPact ships as an MCP (Model Context Protocol) server. This means **any AI agent can use it as persistent memory** — no custom integration code needed.

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

Now Claude can:
- **Store memories** across conversations
- **Ingest documents** and search them semantically
- **Maintain context** about your preferences, projects, and history

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `fusionpact_create_collection` | Create a new HNSW-indexed vector collection |
| `fusionpact_insert` | Insert text documents (auto-embedded) |
| `fusionpact_search` | Semantic search with metadata filtering |
| `fusionpact_rag_ingest` | One-click RAG: chunk + embed + index text |
| `fusionpact_rag_query` | Build LLM-ready context from documents |
| `fusionpact_memory_remember` | Store episodic memory (events, conversations) |
| `fusionpact_memory_recall` | Recall relevant memories for a context |
| `fusionpact_memory_learn` | Add knowledge to semantic memory |

## 🧠 Agent Memory Architecture

Unlike generic vector stores, FusionPact has **purpose-built memory types** for AI agents:

```
┌─────────────────────────────────────────────────┐
│           FusionPact Agent Memory                │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Episodic │  │ Semantic │  │Procedural│      │
│  │(what     │  │(what the │  │(what the │      │
│  │happened) │  │agent     │  │agent can │      │
│  │          │  │knows)    │  │do)       │      │
│  └──────────┘  └──────────┘  └──────────┘      │
│  Conversations   Documents     Tool schemas      │
│  Events          Knowledge     API specs          │
│  User prefs      Facts         Workflows          │
└─────────────────────────────────────────────────┘
```

```javascript
const { FusionEngine, AgentMemory } = require('fusionpact');

const engine = new FusionEngine();
const memory = new AgentMemory(engine, { embedder: 'ollama' });

// Episodic — remember what happened
await memory.remember('agent-1', {
  content: 'User prefers dark mode and concise answers',
  role: 'system',
});

// Semantic — learn knowledge
await memory.learn('agent-1',
  'OSHA 29 CFR 1910 covers general industry safety standards.',
  { source: 'regulations', category: 'compliance' }
);

// Procedural — register tools
await memory.registerTool('agent-1', {
  name: 'search_incidents',
  description: 'Search EHS incident reports by category and severity',
  schema: { /* JSON Schema */ },
});

// Recall — find relevant memories
const memories = await memory.recall('agent-1', 'safety compliance requirements');

// Cross-memory search
const all = await memory.searchAll('agent-1', 'safety training');
// → { episodic: [...], semantic: [...], procedural: [...] }

// GDPR-friendly forget
memory.forget('agent-1', { type: 'all' });
```

## 🔒 Multi-Tenancy

Automatic soft-isolation — zero trust, zero leakage:

```javascript
const tenantA = engine.tenant('shared-collection', 'acme_corp');
const tenantB = engine.tenant('shared-collection', 'globex_inc');

// Inserts are auto-tagged with _tenant_id
tenantA.insert([{ vector: [...], metadata: { doc: 'Acme Safety Plan' } }]);

// Queries are auto-filtered — Tenant A CANNOT see Tenant B's data
tenantA.query(queryVec, { topK: 10 });
// → Only returns Acme documents. Always. No exceptions.
```

## 📊 Performance

### HNSW vs Brute Force (1,000 vectors, 128D)

| Metric | HNSW | Flat (Brute Force) |
|--------|------|-------------------|
| Avg Latency | ~0.3ms | ~0.5ms |
| P99 Latency | ~0.5ms | ~1.2ms |
| QPS | ~3,000 | ~2,000 |

Run your own benchmark:

```bash
npx fusionpact bench --count 10000 --dim 128
```

## 🔧 HTTP API

```bash
npx fusionpact serve --port 8080
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/collections` | GET, POST | List/create collections |
| `/api/insert` | POST | Insert documents (auto-embed text) |
| `/api/search` | POST | Semantic search |
| `/api/rag/ingest` | POST | One-click RAG ingestion |
| `/api/rag/search` | POST | RAG chunk retrieval |
| `/api/rag/context` | POST | Build LLM prompt with context |
| `/api/memory/*` | POST | Agent memory operations |

## 🆚 Comparison

| Feature | FusionPact | Pinecone | Chroma | Qdrant | Milvus |
|---------|:----------:|:--------:|:------:|:------:|:------:|
| **MCP Server (Agent-Native)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Agent Memory Architecture** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **One-Click RAG** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Local-First / Zero-Cost** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Multi-Tenancy** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **HNSW Index** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **In-Process (No Server)** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **< 30s Setup** | ✅ | ❌ | ✅ | ❌ | ❌ |

## 🔌 Embedding Providers

| Provider | Setup | Dimensions | Cost |
|----------|-------|-----------|------|
| **Ollama** (recommended) | `ollama pull nomic-embed-text` | 768 | Free |
| **OpenAI** | Set `OPENAI_API_KEY` | 1536 | $0.02/1M tokens |
| **Mock** (testing) | None | 64 | Free |

```bash
# Use Ollama (local, free, private)
EMBEDDING_PROVIDER=ollama npx fusionpact serve

# Use OpenAI
EMBEDDING_PROVIDER=openai OPENAI_API_KEY=sk-... npx fusionpact serve

# Use mock (for demos/testing)
npx fusionpact serve
```

## 📖 Documentation

- [Architecture Design](docs/ARCHITECTURE.md) — HNSW algorithm, multi-tenancy model, RAG pipeline
- [API Reference](docs/API.md) — HTTP and programmatic API
- [MCP Integration Guide](docs/MCP.md) — Claude Desktop, Cursor, custom agents
- [Examples](examples/) — Quickstart, multi-tenant, RAG pipeline

## 🗺 Roadmap

- [x] HNSW indexing with configurable M/ef parameters
- [x] Multi-tenancy with soft-isolation
- [x] One-Click RAG pipeline
- [x] Agent Memory (episodic, semantic, procedural)
- [x] MCP server
- [x] HTTP API server
- [x] OpenAI + Ollama embedding providers
- [ ] SQLite persistence layer
- [ ] LangChain integration
- [ ] LlamaIndex integration
- [ ] CrewAI integration
- [ ] Rust core (NAPI bindings)
- [ ] Python SDK
- [ ] FusionPact Cloud (managed hosting)
- [ ] Dashboard UI

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/FusionPact/fusionpact-vectordb.git
cd fusionpact-vectordb
npm install
npm test
npm run demo
```

## License

[Apache 2.0](LICENSE) — Use it freely in commercial and open-source projects.

---

<div align="center">

**Built by [FusionPact Technologies](https://fusionpact.com)**

⭐ Star this repo if you find it useful!

</div>
