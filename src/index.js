/**
 * FusionPact — The Agent-Native Vector Database
 *
 * @example
 *   const { FusionEngine, RAGPipeline, AgentMemory, createEmbedder } = require('fusionpact');
 *
 *   // Create engine
 *   const engine = new FusionEngine();
 *
 *   // Create a collection with HNSW indexing
 *   engine.createCollection('my-docs', { dimension: 768, metric: 'cosine' });
 *
 *   // One-click RAG
 *   const rag = new RAGPipeline(engine, { embedder: 'ollama' });
 *   await rag.ingest('Your document text here...', { source: 'doc.pdf' });
 *   const context = await rag.buildContext('What is this about?');
 *
 *   // Agent memory
 *   const memory = new AgentMemory(engine, { embedder: 'ollama' });
 *   await memory.remember('agent-1', { content: 'User prefers dark mode', role: 'system' });
 *   const memories = await memory.recall('agent-1', 'user preferences');
 *
 *   // Multi-tenancy
 *   const tenant = engine.tenant('my-docs', 'acme_corp');
 *   tenant.insert([{ vector: [...], metadata: { ... } }]);  // auto-tagged
 *   tenant.query(queryVec, { topK: 10 });                    // auto-filtered
 */

'use strict';

const { FusionEngine, TenantClient, Collection } = require('./core/engine');
const HNSWIndex = require('./core/hnsw');
const vec = require('./core/vectors');
const { RAGPipeline } = require('./core/rag');
const { AgentMemory } = require('./memory/agent-memory');
const { createEmbedder, MockEmbedder, OllamaEmbedder, OpenAIEmbedder } = require('./embeddings');
const { MCPServer } = require('./mcp/server');
const { chunkText, generateId, Timer } = require('./utils');

module.exports = {
  // Core
  FusionEngine,
  HNSWIndex,
  Collection,

  // Multi-tenancy
  TenantClient,

  // RAG
  RAGPipeline,

  // Agent Memory
  AgentMemory,

  // Embeddings
  createEmbedder,
  MockEmbedder,
  OllamaEmbedder,
  OpenAIEmbedder,

  // MCP
  MCPServer,

  // Utilities
  vec,
  chunkText,
  generateId,
  Timer,
};
