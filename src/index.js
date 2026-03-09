/**
 * @fileoverview FusionPact — The Agent-Native Retrieval Engine
 *
 * Hybrid Vector + Reasoning + Memory for AI Agents.
 *
 * Copyright (c) 2024-2026 FusionPact Technologies Inc. All rights reserved.
 * Licensed under Apache 2.0
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

/**
 * Create a fully-configured FusionPact instance with sensible defaults.
 *
 * @param {Object} [config={}]
 * @param {string} [config.embedder='mock'] - 'ollama', 'openai', or 'mock'
 * @param {string} [config.llmProvider] - 'ollama', 'openai', 'anthropic'
 * @param {boolean} [config.enableHybrid=true] - Enable HybridRetriever
 * @param {boolean} [config.enableMemory=true] - Enable AgentMemory
 * @param {boolean} [config.enableLearning=true] - Enable RecursiveLearningEngine
 * @param {boolean} [config.enableMCP=false] - Start MCP server
 * @returns {{ engine, memory, rag, treeIndex, retriever, orchestrator, learning, mcp }}
 *
 * @example
 * const fp = require('fusionpact').create({
 *   embedder: 'ollama',
 *   llmProvider: 'ollama',
 *   enableLearning: true
 * });
 *
 * // Retrieval with auto-critique and strategy learning
 * const { results, quality } = await fp.learning.retrieveWithCritique('agent-1', 'query');
 *
 * // Memory consolidation (merge, decay, prune)
 * await fp.learning.consolidate('agent-1');
 *
 * // Learn skills from successful workflows
 * fp.learning.learnSkill('agent-1', { name: 'audit', trigger: { keywords: ['audit'] }, steps: [...] });
 *
 * // Knowledge graph extraction
 * await fp.learning.extractKnowledge('agent-1', 'OSHA requires annual training');
 *
 * // Self-reflection
 * const reflection = await fp.learning.reflect('agent-1');
 */
function create(config = {}) {
  const engine = new FusionEngine(config.engineConfig || {});

  let embedder;
  switch (config.embedder) {
    case 'ollama': embedder = new OllamaEmbedder(config.ollamaConfig || {}); break;
    case 'openai': embedder = new OpenAIEmbedder(config.openaiConfig || {}); break;
    default: embedder = new MockEmbedder(config.mockConfig || {});
  }

  let llmProvider = null;
  if (config.llmProvider) {
    llmProvider = new LLMProvider({ provider: config.llmProvider, ...config.llmConfig });
  }

  const treeIndex = new TreeIndex({ llmProvider });

  const rag = new RAGPipeline(engine, {
    embedder,
    collection: config.collection || 'default',
    enableTreeIndex: !!llmProvider,
    treeIndex
  });

  let retriever = null;
  if (config.enableHybrid !== false) {
    retriever = new HybridRetriever({
      engine, treeIndex, embedder,
      weights: config.weights || { vector: 0.4, tree: 0.4, keyword: 0.2 }
    });
    rag.hybridRetriever = retriever;
  }

  let memory = null;
  if (config.enableMemory !== false) {
    memory = new AgentMemory(engine, { embedder });
  }

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
      enableKnowledgeGraph: true
    });
  }

  let mcp = null;
  if (config.enableMCP) {
    mcp = new MCPServer({ engine, memory, rag, retriever, ...config.mcpConfig });
  }

  return { engine, embedder, treeIndex, rag, retriever, memory, orchestrator, learning, mcp };
}

module.exports = {
  create,
  FusionEngine, HNSWIndex, TreeIndex, HybridRetriever,
  AgentMemory, AgentOrchestrator, RAGPipeline, MCPServer,
  OllamaEmbedder, OpenAIEmbedder, MockEmbedder, LLMProvider,
  RecursiveLearningEngine,
  FusionPactVectorStore, FusionPactRetriever,
  getTools, getToolMap,
  VERSION: '2.1.0'
};
