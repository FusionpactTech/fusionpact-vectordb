# Multi-Agent Framework Integration Guide

> Use FusionPact with CrewAI, AutoGen, LangGraph, and custom multi-agent systems.
> Built by FusionPact Technologies Inc. | Apache-2.0

## Why FusionPact for Multi-Agent Systems

Multi-agent systems need: isolated per-agent memory, shared knowledge pools, and conversation persistence. FusionPact provides all three as first-class primitives.

## Generic Pattern (Works with Any Framework)

```javascript
const fp = require('fusionpact').create({ embedder: 'ollama', enableMemory: true });

// Per-agent memory
await fp.memory.learn('researcher', 'Domain fact...', { source: 'training' });
await fp.memory.learn('writer', 'Style guide...', { source: 'training' });

// Agent retrieval (memory + documents)
const memories = await fp.memory.recall('researcher', 'topic');
const docs = await fp.retriever.retrieve('topic', { collection: 'docs', strategy: 'hybrid' });

// Cross-agent sharing
await fp.memory.share('researcher', 'Important finding...', { category: 'research' });
// All agents can now find this via recall({ includeShared: true })

// Conversation persistence
fp.memory.addMessage('agent-1', 'thread-1', { role: 'user', content: 'Question...' });
```

## Via MCP (Zero-Code Integration)

```bash
npx fusionpact mcp
```

Any MCP-compatible agent can call: fusionpact_memory_remember, fusionpact_memory_recall, fusionpact_memory_learn, fusionpact_memory_share, fusionpact_hybrid_search, fusionpact_rag_ingest, fusionpact_rag_query

## Best Practices

1. Use consistent agent IDs across sessions
2. Pre-load domain knowledge with `memory.learn()` at initialization
3. Use `memory.share()` for cross-agent collaboration
4. Always prefer `strategy: 'hybrid'` for document retrieval
5. Use `memory.forget()` for GDPR compliance

Built by [FusionPact Technologies Inc.](https://fusionpact.com)
