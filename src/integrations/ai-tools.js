/**
 * FusionPact AI SDK Tool Definitions
 * Compatible with Vercel AI SDK, OpenAI function calling, Anthropic tools.
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 */
'use strict';

function getTools(fp) {
  const agentId = 'default-agent';
  return [
    { name: 'fusionpact_remember', definition: { name: 'fusionpact_remember', description: 'Store a memory about the conversation or user preferences for later recall.', parameters: { type: 'object', properties: { content: { type: 'string', description: 'What to remember' }, importance: { type: 'number', description: 'Importance 0-1 (default 0.5)' } }, required: ['content'] } },
      execute: async (args) => fp.memory.remember(agentId, { content: args.content, importance: args.importance || 0.5 }) },
    { name: 'fusionpact_recall', definition: { name: 'fusionpact_recall', description: 'Search memory for relevant past conversations, facts, and knowledge.', parameters: { type: 'object', properties: { query: { type: 'string', description: 'What to search for' }, topK: { type: 'number' } }, required: ['query'] } },
      execute: async (args) => fp.memory.recall(agentId, args.query, { topK: args.topK || 5 }) },
    { name: 'fusionpact_learn', definition: { name: 'fusionpact_learn', description: 'Store a fact or knowledge permanently.', parameters: { type: 'object', properties: { content: { type: 'string' }, source: { type: 'string' }, category: { type: 'string' } }, required: ['content'] } },
      execute: async (args) => fp.memory.learn(agentId, args.content, { source: args.source, category: args.category }) },
    { name: 'fusionpact_search_documents', definition: { name: 'fusionpact_search_documents', description: 'Search documents using hybrid retrieval (vector + reasoning + keyword).', parameters: { type: 'object', properties: { query: { type: 'string' }, topK: { type: 'number' }, collection: { type: 'string' } }, required: ['query'] } },
      execute: async (args) => fp.retriever ? fp.retriever.retrieve(args.query, { collection: args.collection || 'default', topK: args.topK || 5, strategy: 'hybrid' }) : fp.rag.buildContext(args.query, { topK: args.topK || 5 }) },
    { name: 'fusionpact_ingest', definition: { name: 'fusionpact_ingest', description: 'Ingest a document — auto-chunks, embeds, and indexes.', parameters: { type: 'object', properties: { text: { type: 'string' }, source: { type: 'string' }, title: { type: 'string' } }, required: ['text'] } },
      execute: async (args) => fp.rag.ingest(args.text, { source: args.source, title: args.title }) },
    { name: 'fusionpact_forget', definition: { name: 'fusionpact_forget', description: 'Delete memories (GDPR erasure).', parameters: { type: 'object', properties: { type: { type: 'string', enum: ['episodic','semantic','procedural','all'] } }, required: ['type'] } },
      execute: async (args) => fp.memory.forget(agentId, { type: args.type }) }
  ];
}

function getToolMap(fp) { const m = {}; for (const t of getTools(fp)) m[t.name] = t.execute; return m; }

module.exports = { getTools, getToolMap };
