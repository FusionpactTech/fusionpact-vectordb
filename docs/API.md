# FusionPact API Reference

> Complete API documentation for FusionPact v2.0.0
> Built by FusionPact Technologies Inc.

## Table of Contents

- [Factory Function](#factory-function)
- [FusionEngine](#fusionengine)
- [HNSWIndex](#hnswindex)
- [TreeIndex](#treeindex)
- [HybridRetriever](#hybridretriever)
- [AgentMemory](#agentmemory)
- [AgentOrchestrator](#agentorchestrator)
- [RAGPipeline](#ragpipeline)
- [MCPServer](#mcpserver)
- [Embedding Providers](#embedding-providers)
- [LLMProvider](#llmprovider)

---

## Factory Function

### `create(config?)`

Creates a fully-configured FusionPact instance with sensible defaults.

**Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `config.embedder` | `string` | `'mock'` | `'ollama'`, `'openai'`, or `'mock'` |
| `config.llmProvider` | `string` | `null` | `'ollama'`, `'openai'`, `'anthropic'` |
| `config.enableHybrid` | `boolean` | `true` | Enable HybridRetriever |
| `config.enableMemory` | `boolean` | `true` | Enable AgentMemory |
| `config.enableMCP` | `boolean` | `false` | Start MCP server |
| `config.collection` | `string` | `'default'` | Default RAG collection |
| `config.weights` | `object` | `{vector:0.4, tree:0.4, keyword:0.2}` | Retrieval weights |

**Returns:** `{ engine, embedder, treeIndex, rag, retriever, memory, orchestrator, mcp }`

```javascript
const fp = require('fusionpact').create({ embedder: 'ollama', enableHybrid: true });
```

---

## FusionEngine

Core database engine managing collections, CRUD, and multi-tenancy.

### `new FusionEngine(config?)`

| Config | Type | Default | Description |
|---|---|---|---|
| `dataDir` | `string` | `null` | Persistence directory |
| `autoSave` | `boolean` | `false` | Auto-persist |

### Methods

#### `createCollection(name, config?)` → `{name, config}`

Creates a named vector collection.

| Param | Type | Default |
|---|---|---|
| `dimensions` | `number` | `768` |
| `distanceMetric` | `string` | `'cosine'` |
| `M` | `number` | `16` |
| `efConstruction` | `number` | `200` |
| `efSearch` | `number` | `50` |

#### `insert(collection, entries, options?)` → `VectorEntry[]`

Insert vectors into a collection.

#### `search(collection, queryVector, options?)` → `SearchResult[]`

Search for nearest neighbors.

| Option | Type | Default |
|---|---|---|
| `topK` | `number` | `10` |
| `filter` | `object` | `null` |
| `tenantId` | `string` | `null` |
| `includeVectors` | `boolean` | `false` |

#### `tenant(collection, tenantId)` → `TenantProxy`

Create a tenant-scoped proxy with automatic isolation.

#### `listCollections()` → `Array`
#### `getCollection(name)` → `object|null`
#### `deleteCollection(name)` → `boolean`
#### `get(collection, id)` → `VectorEntry|null`
#### `delete(collection, id)` → `boolean`
#### `exportData()` → `object`
#### `importData(data, options?)` → `void`

---

## HNSWIndex

Low-level HNSW approximate nearest neighbor index.

### `new HNSWIndex(dimensions, config?)`

### Methods

| Method | Returns | Description |
|---|---|---|
| `insert(id, vector, metadata?)` | `VectorEntry` | Insert a vector |
| `insertBatch(entries)` | `VectorEntry[]` | Batch insert |
| `search(queryVector, options?)` | `SearchResult[]` | K-NN search |
| `delete(id)` | `boolean` | Delete by ID |
| `get(id)` | `VectorEntry\|null` | Get by ID |
| `has(id)` | `boolean` | Check existence |
| `clear()` | `void` | Clear all data |
| `serialize()` | `object` | Export for persistence |
| `HNSWIndex.deserialize(data)` | `HNSWIndex` | Import from data |

---

## TreeIndex

Hierarchical document index for reasoning-based retrieval.

### `new TreeIndex(config?)`

| Config | Type | Default |
|---|---|---|
| `llmProvider` | `LLMProvider` | `null` |
| `maxDepth` | `number` | `5` |
| `maxTokensPerNode` | `number` | `20000` |
| `generateSummaries` | `boolean` | `true` |

### Methods

#### `indexDocument(docId, content, options?)` → `Promise<TreeNode>`

Build a tree structure from a document.

| Option | Type | Default |
|---|---|---|
| `format` | `string` | `'text'` (`'markdown'`, `'html'`, `'text'`) |
| `metadata` | `object` | `{}` |
| `title` | `string` | auto-detected |

#### `search(docId, query, options?)` → `Promise<TreeSearchResult[]>`

Reasoning-based retrieval within a document.

#### `searchAll(query, options?)` → `Promise<TreeSearchResult[]>`

Search across all indexed documents.

#### `getTree(docId)` → `TreeNode|null`
#### `listDocuments()` → `Array`
#### `removeDocument(docId)` → `boolean`

---

## HybridRetriever

Multi-strategy retrieval with Reciprocal Rank Fusion.

### `new HybridRetriever(config)`

| Config | Type | Description |
|---|---|---|
| `engine` | `FusionEngine` | Required |
| `treeIndex` | `TreeIndex` | Optional |
| `embedder` | `BaseEmbedder` | Optional |
| `weights` | `object` | `{vector:0.4, tree:0.4, keyword:0.2}` |
| `rrfK` | `number` | RRF constant (default: 60) |

### Methods

#### `retrieve(query, options?)` → `Promise<HybridResult[]>`

| Option | Type | Default | Description |
|---|---|---|---|
| `collection` | `string` | — | Vector collection |
| `docId` | `string` | — | Document for tree search |
| `topK` | `number` | `10` | Max results |
| `strategy` | `string` | `'hybrid'` | `'hybrid'`, `'vector'`, `'tree'`, `'keyword'` |
| `filter` | `object` | `null` | Metadata filter |
| `tenantId` | `string` | `null` | Tenant filter |

#### `buildContext(results, options?)` → `string`
#### `recordFeedback(query, strategy, quality)` → `void`
#### `getAdaptiveWeights(query)` → `object`

---

## AgentMemory

Multi-type memory system for AI agents.

### Methods

| Method | Description |
|---|---|
| `remember(agentId, memory)` | Store episodic memory |
| `learn(agentId, content, metadata?)` | Add semantic knowledge |
| `registerTool(agentId, tool)` | Register procedural tool |
| `recall(agentId, query, options?)` | Cross-memory search |
| `searchAll(agentId, query, options?)` | Flat ranked search |
| `share(agentId, content, metadata?)` | Share with other agents |
| `addMessage(agentId, threadId, message)` | Add conversation message |
| `getConversation(agentId, threadId, options?)` | Get chat history |
| `forget(agentId, options?)` | Delete memories (GDPR) |
| `getStats(agentId)` | Memory statistics |

---

## AgentOrchestrator

Multi-agent coordination and communication.

### Methods

| Method | Description |
|---|---|
| `registerAgent(config)` | Register an agent |
| `unregisterAgent(agentId, options?)` | Remove an agent |
| `listAgents()` | List all agents |
| `send(message)` | Send agent-to-agent message |
| `onMessage(agentId, handler)` | Register message handler |
| `getMessages(agentId)` | Get pending messages |
| `delegate(fromAgentId, task, options?)` | Capability-based delegation |
| `collaborativeRecall(query, options?)` | Cross-agent retrieval |
| `getStats()` | Orchestration statistics |

---

## Embedding Providers

### `OllamaEmbedder` (Recommended — Free, Local, Private)

```javascript
new OllamaEmbedder({ model: 'nomic-embed-text', baseUrl: 'http://localhost:11434' })
```

### `OpenAIEmbedder`

```javascript
new OpenAIEmbedder({ apiKey: 'sk-...', model: 'text-embedding-3-small' })
```

### `MockEmbedder` (Testing)

```javascript
new MockEmbedder({ dimensions: 64 })
```

### Common Methods

| Method | Description |
|---|---|
| `embed(text)` | Embed single text → `number[]` |
| `embedBatch(texts)` | Embed multiple texts |
| `stats` | Cache and call statistics |
| `clearCache()` | Clear embedding cache |

---

## LLMProvider

Unified LLM interface for tree reasoning.

```javascript
new LLMProvider({ provider: 'ollama', model: 'llama3.2' })
new LLMProvider({ provider: 'openai', apiKey: 'sk-...' })
new LLMProvider({ provider: 'anthropic', apiKey: 'sk-ant-...' })
```

### Methods

| Method | Description |
|---|---|
| `complete(prompt, options?)` | Generate text completion |

---

*Built by [FusionPact Technologies Inc.](https://fusionpact.com)*
