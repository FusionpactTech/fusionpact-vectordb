# Multi-Agent Framework Integration Guide

> How to use FusionPact with CrewAI, AutoGen, LangGraph, and custom multi-agent systems.
> Built by FusionPact Technologies Inc. | Apache-2.0

## Why FusionPact for Multi-Agent Systems

Multi-agent architectures need three things that generic vector databases don't provide:

1. **Isolated agent memory** — Each agent needs its own knowledge store
2. **Shared knowledge pools** — Agents need to collaborate through shared facts
3. **Conversation persistence** — Agents need to remember past interactions per thread

FusionPact provides all three as first-class primitives, plus capability-based task routing and cross-agent retrieval.

---

## Architecture Pattern

```
┌─────────────────────────────────────────────────────────┐
│                    Your Agent Framework                   │
│    (CrewAI / AutoGen / LangGraph / Custom)               │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Agent A  │  │ Agent B  │  │ Agent C  │              │
│  │ Research │  │ Analysis │  │ Writing  │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │              │              │                    │
│       └──────────┬───┴──────────────┘                    │
│                  ▼                                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │           FusionPact (npm install fusionpact)     │   │
│  │                                                    │   │
│  │  AgentMemory      → Per-agent memory stores       │   │
│  │  AgentOrchestrator → Message routing + delegation │   │
│  │  HybridRetriever  → Shared document retrieval     │   │
│  │  RAGPipeline      → Document ingestion            │   │
│  │  MCPServer        → External agent integration    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Generic Integration Pattern

This pattern works with any multi-agent framework:

```javascript
const fusionpact = require('fusionpact');

// Initialize once, share across agents
const fp = fusionpact.create({
  embedder: 'ollama',     // Free, local
  llmProvider: 'ollama',  // For tree reasoning
  enableMemory: true,
  enableHybrid: true
});

// ─── Per-Agent Memory ───
// Each agent gets isolated memory via agentId
async function initAgent(agentId, role, knowledge) {
  // Store role context
  await fp.memory.remember(agentId, {
    content: `My role: ${role}`,
    importance: 1.0
  });

  // Pre-load domain knowledge
  for (const fact of knowledge) {
    await fp.memory.learn(agentId, fact.content, fact.metadata);
  }
}

// ─── Agent Retrieval ───
// Each agent can search documents + recall its own memories
async function agentRetrieve(agentId, query) {
  // Hybrid retrieval from shared documents
  const docs = await fp.retriever.retrieve(query, {
    collection: 'shared-docs',
    strategy: 'hybrid',
    topK: 5
  });

  // Agent-specific memory recall
  const memories = await fp.memory.recall(agentId, query, {
    includeShared: true  // Also search shared knowledge pool
  });

  return { docs, memories };
}

// ─── Cross-Agent Knowledge Sharing ───
async function shareKnowledge(fromAgentId, content, metadata) {
  await fp.memory.share(fromAgentId, content, metadata);
  // Now all agents can find this via recall({ includeShared: true })
}

// ─── Conversation Persistence ───
function saveConversation(agentId, threadId, role, content) {
  fp.memory.addMessage(agentId, threadId, { role, content });
}

function getConversation(agentId, threadId, limit = 50) {
  return fp.memory.getConversation(agentId, threadId, { limit });
}
```

---

## CrewAI Integration

```javascript
// fusionpact-crewai-memory.js
// Use as a custom memory backend for CrewAI agents

const fusionpact = require('fusionpact');

class FusionPactCrewMemory {
  constructor() {
    this.fp = fusionpact.create({ embedder: 'ollama', enableMemory: true });
  }

  async storeForAgent(agentRole, content, type = 'episodic') {
    const agentId = `crew_${agentRole.replace(/\s+/g, '_').toLowerCase()}`;
    if (type === 'episodic') {
      return this.fp.memory.remember(agentId, { content });
    }
    return this.fp.memory.learn(agentId, content, { source: 'crew_task' });
  }

  async recallForAgent(agentRole, query) {
    const agentId = `crew_${agentRole.replace(/\s+/g, '_').toLowerCase()}`;
    return this.fp.memory.recall(agentId, query, { includeShared: true });
  }

  async shareAcrossCrew(agentRole, content) {
    const agentId = `crew_${agentRole.replace(/\s+/g, '_').toLowerCase()}`;
    return this.fp.memory.share(agentId, content, { source: 'crew_shared' });
  }
}

module.exports = { FusionPactCrewMemory };
```

---

## AutoGen Integration

```javascript
// fusionpact-autogen-memory.js
// Use as a persistent memory layer for AutoGen agents

const fusionpact = require('fusionpact');

class FusionPactAutoGenMemory {
  constructor() {
    this.fp = fusionpact.create({ embedder: 'ollama', enableMemory: true });
    this.orch = new fusionpact.AgentOrchestrator({
      engine: this.fp.engine,
      memory: this.fp.memory,
      retriever: this.fp.retriever
    });
  }

  registerAgent(name, capabilities = []) {
    this.orch.registerAgent({
      agentId: name,
      name,
      capabilities
    });
  }

  async onMessage(fromAgent, toAgent, content) {
    // Store in conversation memory
    this.fp.memory.addMessage(fromAgent, `${fromAgent}_${toAgent}`, {
      role: 'assistant',
      content
    });

    // Route through orchestrator
    await this.orch.send({
      from: fromAgent,
      to: toAgent,
      type: 'message',
      payload: { content }
    });
  }

  async getContext(agentId, query, topK = 10) {
    const memories = await this.fp.memory.recall(agentId, query, { topK });
    const docs = await this.fp.retriever.retrieve(query, {
      collection: 'shared-docs',
      strategy: 'hybrid',
      topK: 5
    });
    return { memories, docs };
  }
}

module.exports = { FusionPactAutoGenMemory };
```

---

## Using via MCP (Framework-Agnostic)

Any agent that supports MCP can use FusionPact without any code integration:

```bash
npx fusionpact mcp
```

The agent can then call tools like:
- `fusionpact_memory_remember` — Store episodic memory
- `fusionpact_memory_recall` — Retrieve relevant memories
- `fusionpact_memory_learn` — Add semantic knowledge
- `fusionpact_memory_share` — Share with other agents
- `fusionpact_hybrid_search` — Search documents
- `fusionpact_rag_ingest` — Ingest new documents
- `fusionpact_rag_query` — Get LLM-ready context

This is the zero-integration path — works with Claude Desktop, Cursor, Windsurf, and any MCP-compatible agent.

---

## Best Practices

1. **Use consistent agent IDs** — Map your framework's agent names to FusionPact agent IDs consistently
2. **Pre-load domain knowledge** — Use `memory.learn()` at agent initialization to seed domain expertise
3. **Use shared memory for collaboration** — When one agent discovers something useful, `memory.share()` it
4. **Use hybrid retrieval for documents** — Always prefer `strategy: 'hybrid'` unless you have a specific reason to use a single strategy
5. **Persist conversations** — Use `memory.addMessage()` to maintain chat history per thread
6. **GDPR compliance** — Use `memory.forget()` when users request data deletion

---

*Built by [FusionPact Technologies Inc.](https://fusionpact.com)*
