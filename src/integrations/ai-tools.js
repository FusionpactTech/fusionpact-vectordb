/**
 * @fileoverview FusionPact AI SDK Tool Definitions
 * 
 * Pre-built tool definitions compatible with Vercel AI SDK, OpenAI function calling,
 * and any framework that uses JSON Schema tool definitions.
 * 
 * These definitions allow any LLM to use FusionPact as a tool — the LLM can
 * store memories, search documents, and manage knowledge autonomously.
 * 
 * @module integrations/ai-tools
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 */

'use strict';

/**
 * Returns FusionPact tool definitions in OpenAI function calling format.
 * Compatible with: Vercel AI SDK, OpenAI, Anthropic, LangChain tools, etc.
 * 
 * @param {Object} fp - FusionPact instance from create()
 * @returns {Object[]} Array of tool definitions with execute functions
 * 
 * @example
 * const fp = require('fusionpact').create({ embedder: 'ollama' });
 * const tools = require('fusionpact/integrations/ai-tools').getTools(fp);
 * 
 * // Use with Vercel AI SDK
 * const result = await generateText({
 *   model: openai('gpt-4o'),
 *   tools,
 *   prompt: 'Remember that the user prefers dark mode'
 * });
 * 
 * // Use with OpenAI function calling
 * const response = await openai.chat.completions.create({
 *   model: 'gpt-4o',
 *   tools: tools.map(t => ({ type: 'function', function: t.definition })),
 *   messages: [...]
 * });
 */
function getTools(fp) {
  return [
    {
      name: 'fusionpact_remember',
      definition: {
        name: 'fusionpact_remember',
        description: 'Store a memory about the current conversation, user preferences, or important events. Use this to remember things for later.',
        parameters: {
          type: 'object',
          properties: {
            content: { type: 'string', description: 'What to remember' },
            importance: { type: 'number', description: 'How important (0-1). Use 0.8+ for critical info, 0.5 for general, 0.3 for minor.' }
          },
          required: ['content']
        }
      },
      execute: async (args) => {
        const agentId = args.agentId || 'default-agent';
        return fp.memory.remember(agentId, {
          content: args.content,
          importance: args.importance || 0.5
        });
      }
    },
    {
      name: 'fusionpact_recall',
      definition: {
        name: 'fusionpact_recall',
        description: 'Search your memory for relevant past conversations, facts, and knowledge. Use this before answering questions that might benefit from past context.',
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'What to search for in memory' },
            topK: { type: 'number', description: 'Max results (default 5)' }
          },
          required: ['query']
        }
      },
      execute: async (args) => {
        const agentId = args.agentId || 'default-agent';
        return fp.memory.recall(agentId, args.query, { topK: args.topK || 5 });
      }
    },
    {
      name: 'fusionpact_learn',
      definition: {
        name: 'fusionpact_learn',
        description: 'Store a fact or piece of knowledge permanently. Use this when the user teaches you something or when you extract important facts from documents.',
        parameters: {
          type: 'object',
          properties: {
            content: { type: 'string', description: 'The fact or knowledge to store' },
            source: { type: 'string', description: 'Where this knowledge came from' },
            category: { type: 'string', description: 'Category (e.g., regulations, preferences, technical)' }
          },
          required: ['content']
        }
      },
      execute: async (args) => {
        const agentId = args.agentId || 'default-agent';
        return fp.memory.learn(agentId, args.content, {
          source: args.source,
          category: args.category
        });
      }
    },
    {
      name: 'fusionpact_search_documents',
      definition: {
        name: 'fusionpact_search_documents',
        description: 'Search ingested documents using hybrid retrieval (vector similarity + reasoning + keyword matching). Returns the most relevant passages.',
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            topK: { type: 'number', description: 'Max results (default 5)' },
            collection: { type: 'string', description: 'Collection to search (default: "default")' }
          },
          required: ['query']
        }
      },
      execute: async (args) => {
        if (fp.retriever) {
          return fp.retriever.retrieve(args.query, {
            collection: args.collection || 'default',
            topK: args.topK || 5,
            strategy: 'hybrid'
          });
        }
        return fp.rag.buildContext(args.query, { topK: args.topK || 5 });
      }
    },
    {
      name: 'fusionpact_ingest_document',
      definition: {
        name: 'fusionpact_ingest_document',
        description: 'Ingest a document into the knowledge base. Automatically chunks, embeds, and indexes the text for later retrieval.',
        parameters: {
          type: 'object',
          properties: {
            text: { type: 'string', description: 'Document text to ingest' },
            source: { type: 'string', description: 'Source identifier (filename, URL, etc.)' },
            title: { type: 'string', description: 'Document title' }
          },
          required: ['text']
        }
      },
      execute: async (args) => {
        return fp.rag.ingest(args.text, {
          source: args.source,
          title: args.title
        });
      }
    },
    {
      name: 'fusionpact_forget',
      definition: {
        name: 'fusionpact_forget',
        description: 'Delete memories. Use when the user asks you to forget something or for GDPR data erasure.',
        parameters: {
          type: 'object',
          properties: {
            type: { type: 'string', enum: ['episodic', 'semantic', 'procedural', 'all'], description: 'Which memory type to clear' }
          },
          required: ['type']
        }
      },
      execute: async (args) => {
        const agentId = args.agentId || 'default-agent';
        return fp.memory.forget(agentId, { type: args.type });
      }
    }
  ];
}

/**
 * Returns tools as a simple { name → function } map for direct use.
 * 
 * @param {Object} fp - FusionPact instance
 * @returns {Object} Map of tool name → async execute function
 */
function getToolMap(fp) {
  const tools = getTools(fp);
  const map = {};
  for (const tool of tools) {
    map[tool.name] = tool.execute;
  }
  return map;
}

module.exports = { getTools, getToolMap };
