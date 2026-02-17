/**
 * FusionPact — MCP (Model Context Protocol) Server
 *
 * Exposes FusionPact as an MCP-compatible tool server that any AI agent
 * can connect to for persistent vector memory, RAG, and search.
 *
 * Compatible with:
 *   - Claude Desktop (claude_desktop_config.json)
 *   - Cursor IDE
 *   - Any MCP-compatible client
 *
 * Usage:
 *   npx fusionpact mcp                    # Start MCP server (stdio)
 *   node src/mcp/server.js                # Direct execution
 *
 * Claude Desktop config:
 *   {
 *     "mcpServers": {
 *       "fusionpact": {
 *         "command": "npx",
 *         "args": ["fusionpact", "mcp"]
 *       }
 *     }
 *   }
 */

'use strict';

const { FusionEngine } = require('../core/engine');
const { RAGPipeline } = require('../core/rag');
const { AgentMemory } = require('../memory/agent-memory');
const { createEmbedder } = require('../embeddings');

// ─── MCP Protocol Implementation (stdio JSON-RPC) ────────────

class MCPServer {
  constructor(options = {}) {
    this.engine = new FusionEngine();
    this.embedder = createEmbedder(options.embedder || process.env.EMBEDDING_PROVIDER || 'mock');
    this.rag = null;
    this.memory = new AgentMemory(this.engine, { embedder: this.embedder });
    this.serverInfo = {
      name: 'fusionpact',
      version: '0.1.0',
    };
  }

  /**
   * Handle a JSON-RPC request
   * @param {Object} request
   * @returns {Object} response
   */
  async handleRequest(request) {
    const { method, params, id } = request;

    try {
      let result;

      switch (method) {
        case 'initialize':
          result = this._handleInitialize(params);
          break;
        case 'initialized':
          return null; // notification, no response needed
        case 'tools/list':
          result = this._listTools();
          break;
        case 'tools/call':
          result = await this._callTool(params);
          break;
        case 'resources/list':
          result = this._listResources();
          break;
        case 'resources/read':
          result = await this._readResource(params);
          break;
        case 'ping':
          result = {};
          break;
        default:
          return this._error(id, -32601, `Method not found: ${method}`);
      }

      return { jsonrpc: '2.0', id, result };
    } catch (err) {
      return this._error(id, -32000, err.message);
    }
  }

  // ─── Initialize ─────────────────────────────────────────────

  _handleInitialize(params) {
    return {
      protocolVersion: '2024-11-05',
      capabilities: {
        tools: {},
        resources: {},
      },
      serverInfo: this.serverInfo,
    };
  }

  // ─── Tools ──────────────────────────────────────────────────

  _listTools() {
    return {
      tools: [
        {
          name: 'fusionpact_create_collection',
          description: 'Create a new vector collection with HNSW indexing for storing embeddings.',
          inputSchema: {
            type: 'object',
            properties: {
              name: { type: 'string', description: 'Collection name' },
              dimension: { type: 'number', description: 'Vector dimension (default: auto from embedder)' },
              metric: { type: 'string', enum: ['cosine', 'euclidean', 'dot'], description: 'Distance metric' },
            },
            required: ['name'],
          },
        },
        {
          name: 'fusionpact_list_collections',
          description: 'List all vector collections with their stats (document count, dimension, index type).',
          inputSchema: { type: 'object', properties: {} },
        },
        {
          name: 'fusionpact_drop_collection',
          description: 'Delete a vector collection and all its data permanently.',
          inputSchema: {
            type: 'object',
            properties: { name: { type: 'string', description: 'Collection name to drop' } },
            required: ['name'],
          },
        },
        {
          name: 'fusionpact_insert',
          description: 'Insert text documents into a collection. Text is automatically embedded into vectors.',
          inputSchema: {
            type: 'object',
            properties: {
              collection: { type: 'string', description: 'Target collection name' },
              documents: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    text: { type: 'string', description: 'Text content to embed and store' },
                    id: { type: 'string', description: 'Optional document ID' },
                    metadata: { type: 'object', description: 'Optional metadata key-value pairs' },
                  },
                  required: ['text'],
                },
                description: 'Array of documents to insert',
              },
            },
            required: ['collection', 'documents'],
          },
        },
        {
          name: 'fusionpact_search',
          description: 'Semantic search: find documents most similar to a query text. Returns ranked results with relevance scores.',
          inputSchema: {
            type: 'object',
            properties: {
              collection: { type: 'string', description: 'Collection to search' },
              query: { type: 'string', description: 'Search query text' },
              topK: { type: 'number', description: 'Number of results (default: 5)' },
              filter: { type: 'object', description: 'Metadata filter (e.g., {"category": "safety"})' },
            },
            required: ['collection', 'query'],
          },
        },
        {
          name: 'fusionpact_rag_ingest',
          description: 'One-click RAG: ingest a text document by auto-chunking, embedding, and indexing it for retrieval-augmented generation.',
          inputSchema: {
            type: 'object',
            properties: {
              text: { type: 'string', description: 'Full text content to ingest' },
              source: { type: 'string', description: 'Source identifier (e.g., filename)' },
            },
            required: ['text'],
          },
        },
        {
          name: 'fusionpact_rag_query',
          description: 'RAG search: find the most relevant text chunks for a question and build LLM context.',
          inputSchema: {
            type: 'object',
            properties: {
              question: { type: 'string', description: 'Question to answer' },
              topK: { type: 'number', description: 'Number of chunks to retrieve (default: 5)' },
            },
            required: ['question'],
          },
        },
        {
          name: 'fusionpact_memory_remember',
          description: 'Store a memory (conversation event, fact, or interaction) in agent episodic memory.',
          inputSchema: {
            type: 'object',
            properties: {
              agentId: { type: 'string', description: 'Agent identifier (default: "default")' },
              content: { type: 'string', description: 'Content to remember' },
              role: { type: 'string', description: 'Role: user, assistant, system, tool' },
              sessionId: { type: 'string', description: 'Session/conversation ID' },
            },
            required: ['content'],
          },
        },
        {
          name: 'fusionpact_memory_recall',
          description: 'Recall relevant memories for a given context from agent memory.',
          inputSchema: {
            type: 'object',
            properties: {
              agentId: { type: 'string', description: 'Agent identifier (default: "default")' },
              context: { type: 'string', description: 'What to recall about' },
              topK: { type: 'number', description: 'Number of memories to recall (default: 10)' },
            },
            required: ['context'],
          },
        },
        {
          name: 'fusionpact_memory_learn',
          description: 'Add knowledge to agent semantic memory (facts, documents, reference material).',
          inputSchema: {
            type: 'object',
            properties: {
              agentId: { type: 'string', description: 'Agent identifier (default: "default")' },
              knowledge: { type: 'string', description: 'Knowledge content to store' },
              source: { type: 'string', description: 'Source of knowledge' },
              category: { type: 'string', description: 'Knowledge category' },
            },
            required: ['knowledge'],
          },
        },
      ],
    };
  }

  async _callTool(params) {
    const { name, arguments: args } = params;

    switch (name) {
      case 'fusionpact_create_collection': {
        const info = this.engine.createCollection(args.name, {
          dimension: args.dimension || this.embedder.dimension,
          metric: args.metric || 'cosine',
          indexType: 'hnsw',
        });
        return { content: [{ type: 'text', text: JSON.stringify(info, null, 2) }] };
      }

      case 'fusionpact_list_collections': {
        const cols = this.engine.listCollections();
        return { content: [{ type: 'text', text: JSON.stringify(cols, null, 2) }] };
      }

      case 'fusionpact_drop_collection': {
        const dropped = this.engine.dropCollection(args.name);
        return { content: [{ type: 'text', text: dropped ? `Dropped '${args.name}'` : `Collection '${args.name}' not found` }] };
      }

      case 'fusionpact_insert': {
        const colName = args.collection;
        // Auto-create collection if not exists
        if (!this.engine.collections.has(colName)) {
          this.engine.createCollection(colName, {
            dimension: this.embedder.dimension,
            metric: 'cosine', indexType: 'hnsw',
          });
        }

        const texts = args.documents.map(d => d.text);
        const vectors = await this.embedder.embed(texts);

        const docs = args.documents.map((d, i) => ({
          id: d.id,
          vector: vectors[i],
          metadata: { text: d.text, ...(d.metadata || {}) },
        }));

        const ids = this.engine.insert(colName, docs);
        return { content: [{ type: 'text', text: `Inserted ${ids.length} documents into '${colName}'. IDs: ${ids.join(', ')}` }] };
      }

      case 'fusionpact_search': {
        const queryVec = await this.embedder.embedOne(args.query);
        const result = this.engine.query(args.collection, queryVec, {
          topK: args.topK || 5,
          filter: args.filter || null,
        });

        const output = result.results.map((r, i) =>
          `[${i + 1}] (score: ${r.score.toFixed(4)}) ${r.metadata.text || r.id}\n    Metadata: ${JSON.stringify(r.metadata)}`
        ).join('\n\n');

        return { content: [{ type: 'text', text: `Found ${result.results.length} results in ${result.elapsed}ms (${result.method}):\n\n${output}` }] };
      }

      case 'fusionpact_rag_ingest': {
        if (!this.rag) {
          this.rag = new RAGPipeline(this.engine, { embedder: this.embedder });
        }
        const info = await this.rag.ingest(args.text, { source: args.source || 'mcp-upload' });
        return { content: [{ type: 'text', text: `Ingested ${info.chunksCreated} chunks from '${info.source}' into '${info.collection}' (${info.provider} embeddings, ${info.dimension}D)` }] };
      }

      case 'fusionpact_rag_query': {
        if (!this.rag) {
          this.rag = new RAGPipeline(this.engine, { embedder: this.embedder });
        }
        const ctx = await this.rag.buildContext(args.question, { topK: args.topK || 5 });
        return { content: [{ type: 'text', text: ctx.prompt }] };
      }

      case 'fusionpact_memory_remember': {
        const res = await this.memory.remember(args.agentId || 'default', {
          content: args.content,
          role: args.role || 'user',
          sessionId: args.sessionId || 'default',
        });
        return { content: [{ type: 'text', text: `Remembered: ${res.id}` }] };
      }

      case 'fusionpact_memory_recall': {
        const memories = await this.memory.recall(args.agentId || 'default', args.context, {
          topK: args.topK || 10,
        });
        const output = memories.map((m, i) =>
          `[${i + 1}] (score: ${m.score.toFixed(3)}, ${m.role}, ${new Date(m.timestamp).toLocaleString()})\n    ${m.content}`
        ).join('\n\n');
        return { content: [{ type: 'text', text: memories.length > 0 ? `Recalled ${memories.length} memories:\n\n${output}` : 'No relevant memories found.' }] };
      }

      case 'fusionpact_memory_learn': {
        const res = await this.memory.learn(args.agentId || 'default', args.knowledge, {
          source: args.source || 'direct',
          category: args.category || 'general',
        });
        return { content: [{ type: 'text', text: `Learned ${res.chunks} knowledge chunk(s).` }] };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  // ─── Resources ──────────────────────────────────────────────

  _listResources() {
    const resources = [
      {
        uri: 'fusionpact://collections',
        name: 'FusionPact Collections',
        description: 'List of all vector collections with stats',
        mimeType: 'application/json',
      },
    ];

    // Add per-collection resources
    for (const col of this.engine.listCollections()) {
      resources.push({
        uri: `fusionpact://collections/${col.name}`,
        name: `Collection: ${col.name}`,
        description: `${col.count} vectors, ${col.dimension}D, ${col.indexType}`,
        mimeType: 'application/json',
      });
    }

    return { resources };
  }

  async _readResource(params) {
    const { uri } = params;

    if (uri === 'fusionpact://collections') {
      return {
        contents: [{
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(this.engine.listCollections(), null, 2),
        }],
      };
    }

    const match = uri.match(/^fusionpact:\/\/collections\/(.+)$/);
    if (match) {
      const info = this.engine.getCollection(match[1]);
      if (!info) throw new Error(`Collection '${match[1]}' not found`);
      return {
        contents: [{
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(info, null, 2),
        }],
      };
    }

    throw new Error(`Unknown resource: ${uri}`);
  }

  // ─── Error Helper ───────────────────────────────────────────

  _error(id, code, message) {
    return { jsonrpc: '2.0', id, error: { code, message } };
  }

  // ─── stdio Transport ────────────────────────────────────────

  /**
   * Start the MCP server on stdio (for Claude Desktop, Cursor, etc.)
   */
  start() {
    let buffer = '';

    process.stdin.setEncoding('utf-8');
    process.stdin.on('data', async (chunk) => {
      buffer += chunk;

      // Process complete JSON-RPC messages
      while (true) {
        // Look for Content-Length header
        const headerEnd = buffer.indexOf('\r\n\r\n');
        if (headerEnd === -1) break;

        const header = buffer.slice(0, headerEnd);
        const lengthMatch = header.match(/Content-Length:\s*(\d+)/i);

        if (lengthMatch) {
          const contentLength = parseInt(lengthMatch[1]);
          const bodyStart = headerEnd + 4;

          if (buffer.length < bodyStart + contentLength) break;

          const body = buffer.slice(bodyStart, bodyStart + contentLength);
          buffer = buffer.slice(bodyStart + contentLength);

          try {
            const request = JSON.parse(body);
            const response = await this.handleRequest(request);
            if (response) this._send(response);
          } catch (err) {
            this._send(this._error(null, -32700, `Parse error: ${err.message}`));
          }
        } else {
          // Try line-delimited JSON as fallback
          const lineEnd = buffer.indexOf('\n');
          if (lineEnd === -1) break;

          const line = buffer.slice(0, lineEnd).trim();
          buffer = buffer.slice(lineEnd + 1);

          if (!line) continue;

          try {
            const request = JSON.parse(line);
            const response = await this.handleRequest(request);
            if (response) this._send(response);
          } catch (err) {
            // Skip unparseable lines
          }
        }
      }
    });

    process.stderr.write(`FusionPact MCP Server v${this.serverInfo.version} started (embedding: ${this.embedder.provider})\n`);
  }

  _send(response) {
    const body = JSON.stringify(response);
    const message = `Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`;
    process.stdout.write(message);
  }
}

// ─── Direct Execution ─────────────────────────────────────────

if (require.main === module) {
  const server = new MCPServer();
  server.start();
}

module.exports = { MCPServer };
