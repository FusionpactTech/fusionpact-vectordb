/**
 * @fileoverview FusionPact — The Agent-Native Retrieval Engine
 *
 * Hybrid Vector + Reasoning + Memory for AI Agents.
 *
 * ─────────────────────────────────────────────────────────────
 *  FusionPact is built and maintained by FusionPact Technologies Inc.
 *  https://fusionpact.com | https://github.com/FusionpactTech
 *
 *  Licensed under Apache 2.0
 *  Copyright (c) 2024-2026 FusionPact Technologies Inc.
 * ─────────────────────────────────────────────────────────────
 *
 * @module fusionpact
 * @version 2.1.0
 * @license Apache-2.0
 * @see https://github.com/FusionpactTech/fusionpact-vectordb
 */

'use strict';

// Core
const { FusionEngine } = require('./core/FusionEngine');
const { HNSWIndex } = require('./core/HNSWIndex');

// Index
const { TreeIndex } = require('./index/TreeIndex');

// Retrieval
const { HybridRetriever } = require('./retrieval/HybridRetriever');

// Memory
const { AgentMemory } = require('./memory/AgentMemory');

// RAG
const { RAGPipeline } = require('./rag/RAGPipeline');

// Embedders
const { OllamaEmbedder, OpenAIEmbedder, MockEmbedder, LLMProvider } = require('./embedders/providers');

// MCP
const { MCPServer } = require('./mcp/MCPServer');

// Orchestration
const { AgentOrchestrator } = require('./orchestration/AgentOrchestrator');

// Recursive Learning
const { RecursiveLearningEngine } = require('./learning/RecursiveLearningEngine');

// Integrations
const { FusionPactVectorStore, FusionPactRetriever } = require('./integrations/langchain');
const { getTools, getToolMap } = require('./integrations/ai-tools');

// ─── Convenience Factory ──────────────────────────────────

/**
 * Create a fully-configured FusionPact instance with sensible defaults.
 * 
 * @param {Object} [config={}]
 * @param {string} [config.embedder='mock'] - Embedder: 'ollama', 'openai', or 'mock'
 * @param {string} [config.llmProvider] - LLM provider for tree reasoning: 'ollama', 'openai', 'anthropic'
 * @param {boolean} [config.enableHybrid=true] - Enable hybrid retrieval
 * @param {boolean} [config.enableMemory=true] - Enable agent memory
 * @param {boolean} [config.enableMCP=false] - Start MCP server
 * @param {Object} [config.engineConfig={}] - FusionEngine config
 * @returns {{ engine, memory, rag, treeIndex, retriever, orchestrator, mcp }}
 * 
 * @example
 * // Quickstart — zero config
 * const fp = require('fusionpact').create();
 * await fp.rag.ingest('Your document text...', { source: 'doc.pdf' });
 * const ctx = await fp.rag.buildContext('What safety protocols exist?');
 * 
 * @example
 * // Full setup with Ollama
 * const fp = require('fusionpact').create({
 *   embedder: 'ollama',
 *   llmProvider: 'ollama',
 *   enableHybrid: true,
 *   enableMemory: true
 * });
 */
function create(config = {}) {
  // Engine
  const engine = new FusionEngine(config.engineConfig || {});

  // Embedder
  let embedder;
  switch (config.embedder) {
    case 'ollama':
      embedder = new OllamaEmbedder(config.ollamaConfig || {});
      break;
    case 'openai':
      embedder = new OpenAIEmbedder(config.openaiConfig || {});
      break;
    default:
      embedder = new MockEmbedder(config.mockConfig || {});
  }

  // LLM Provider (for tree reasoning)
  let llmProvider = null;
  if (config.llmProvider) {
    llmProvider = new LLMProvider({
      provider: config.llmProvider,
      ...config.llmConfig
    });
  }

  // Tree Index
  const treeIndex = new TreeIndex({ llmProvider });

  // RAG Pipeline
  const rag = new RAGPipeline(engine, {
    embedder,
    collection: config.collection || 'default',
    enableTreeIndex: !!llmProvider,
    treeIndex
  });

  // Hybrid Retriever
  let retriever = null;
  if (config.enableHybrid !== false) {
    retriever = new HybridRetriever({
      engine,
      treeIndex,
      embedder,
      weights: config.weights || { vector: 0.4, tree: 0.4, keyword: 0.2 }
    });
    rag.hybridRetriever = retriever;
  }

  // Agent Memory
  let memory = null;
  if (config.enableMemory !== false) {
    memory = new AgentMemory(engine, { embedder });
  }

  // Orchestrator
  let orchestrator = null;
  if (memory) {
    orchestrator = new AgentOrchestrator({ engine, memory, retriever });
  }

  // Recursive Learning Engine
  let learning = null;
  if (config.enableLearning !== false && memory) {
    learning = new RecursiveLearningEngine({
      memory,
      retriever,
      llmProvider,
      enableAutoConsolidation: config.enableAutoConsolidation || false,
      enableRetrievalCritique: true,
      enableKnowledgeGraph: true,
      consolidation: config.consolidation
    });
  }

  // MCP Server
  let mcp = null;
  if (config.enableMCP) {
    mcp = new MCPServer({ engine, memory, rag, retriever, ...config.mcpConfig });
  }

  return { engine, embedder, treeIndex, rag, retriever, memory, orchestrator, learning, mcp };
}

module.exports = {
  // Factory
  create,

  // Core
  FusionEngine,
  HNSWIndex,

  // Index
  TreeIndex,

  // Retrieval
  HybridRetriever,

  // Memory
  AgentMemory,

  // RAG
  RAGPipeline,

  // Embedders
  OllamaEmbedder,
  OpenAIEmbedder,
  MockEmbedder,
  LLMProvider,

  // MCP
  MCPServer,

  // Orchestration
  AgentOrchestrator,

  // Recursive Learning
  RecursiveLearningEngine,

  // Integrations
  FusionPactVectorStore,
  FusionPactRetriever,
  getTools,
  getToolMap,

  // Version
  VERSION: '2.1.0'
};
