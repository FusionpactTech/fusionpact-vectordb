/**
 * @fileoverview MCP Server — Model Context Protocol server for AI agents
 * 
 * Exposes FusionPact as an MCP server for Claude Desktop, Cursor, Windsurf,
 * and any MCP-compatible AI agent to use as persistent memory and retrieval.
 * 
 * @module mcp/MCPServer
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 */

'use strict';

const { createServer } = require('http');
const { EventEmitter } = require('events');

class MCPServer extends EventEmitter {
  constructor(config) {
    super();
    this.engine = config.engine;
    this.memory = config.memory || null;
    this.rag = config.rag || null;
    this.retriever = config.retriever || null;
    this.transport = config.transport || 'stdio';
    this.port = config.port || 3100;
    this._tools = this._registerTools();
  }

  getToolManifest() {
    return {
      name: 'fusionpact',
      version: '2.0.0',
      description: 'FusionPact Agent-Native Retrieval Engine — hybrid vector + reasoning search, agent memory, and RAG. Built by FusionPact Technologies Inc.',
      tools: Object.values(this._tools).map(t => ({ name: t.name, description: t.description, inputSchema: t.inputSchema }))
    };
  }

  async handleToolCall(toolName, args) {
    const tool = this._tools[toolName];
    if (!tool) return { error: `Unknown tool: ${toolName}` };
    try { return { result: await tool.handler(args) }; }
    catch (err) { return { error: err.message }; }
  }

  async start() {
    if (this.transport === 'stdio') return this._startStdio();
    return this._startHTTP();
  }

  _registerTools() {
    const tools = {};
    const s = (name, desc, schema, handler) => { tools[name] = { name, description: desc, inputSchema: schema, handler }; };

    s('fusionpact_create_collection', 'Create a new HNSW-indexed vector collection.',
      { type: 'object', properties: { name: { type: 'string' }, dimensions: { type: 'number' }, distanceMetric: { type: 'string', enum: ['cosine','euclidean','dotProduct'] } }, required: ['name'] },
      async (a) => this.engine.createCollection(a.name, a));

    s('fusionpact_list_collections', 'List all vector collections.',
      { type: 'object', properties: {} }, async () => this.engine.listCollections());

    s('fusionpact_search', 'Semantic search within a collection.',
      { type: 'object', properties: { collection: { type: 'string' }, query: { type: 'string' }, topK: { type: 'number' }, filter: { type: 'object' }, tenantId: { type: 'string' } }, required: ['collection','query'] },
      async (a) => {
        const v = this.rag?.embedder ? await this.rag.embedder.embed(a.query) : this._mockVec(a.query);
        return this.engine.search(a.collection, v, { topK: a.topK || 5, filter: a.filter, tenantId: a.tenantId });
      });

    if (this.retriever) {
      s('fusionpact_hybrid_search', 'Hybrid retrieval combining vector, reasoning-based tree search, and keyword matching.',
        { type: 'object', properties: { query: { type: 'string' }, collection: { type: 'string' }, docId: { type: 'string' }, topK: { type: 'number' }, strategy: { type: 'string', enum: ['hybrid','vector','tree','keyword'] } }, required: ['query'] },
        async (a) => this.retriever.retrieve(a.query, a));
    }

    if (this.rag) {
      s('fusionpact_rag_ingest', 'One-click RAG: chunk, embed, and index text.',
        { type: 'object', properties: { text: { type: 'string' }, source: { type: 'string' }, metadata: { type: 'object' }, tenantId: { type: 'string' } }, required: ['text'] },
        async (a) => this.rag.ingest(a.text, { source: a.source, ...a.metadata }, { tenantId: a.tenantId }));

      s('fusionpact_rag_query', 'Build LLM-ready context from ingested documents.',
        { type: 'object', properties: { query: { type: 'string' }, topK: { type: 'number' }, maxTokens: { type: 'number' }, tenantId: { type: 'string' } }, required: ['query'] },
        async (a) => this.rag.buildContext(a.query, a));
    }

    if (this.memory) {
      s('fusionpact_memory_remember', 'Store an episodic memory for an AI agent.',
        { type: 'object', properties: { agentId: { type: 'string' }, content: { type: 'string' }, role: { type: 'string' }, importance: { type: 'number' }, metadata: { type: 'object' } }, required: ['agentId','content'] },
        async (a) => this.memory.remember(a.agentId, a));

      s('fusionpact_memory_recall', 'Recall relevant memories across all memory types.',
        { type: 'object', properties: { agentId: { type: 'string' }, query: { type: 'string' }, types: { type: 'array', items: { type: 'string' } }, topK: { type: 'number' }, includeShared: { type: 'boolean' } }, required: ['agentId','query'] },
        async (a) => this.memory.recall(a.agentId, a.query, a));

      s('fusionpact_memory_learn', 'Add knowledge to semantic memory.',
        { type: 'object', properties: { agentId: { type: 'string' }, content: { type: 'string' }, source: { type: 'string' }, category: { type: 'string' }, confidence: { type: 'number' } }, required: ['agentId','content'] },
        async (a) => this.memory.learn(a.agentId, a.content, a));

      s('fusionpact_memory_share', 'Share knowledge with other agents (multi-agent collaboration).',
        { type: 'object', properties: { agentId: { type: 'string' }, content: { type: 'string' }, metadata: { type: 'object' } }, required: ['agentId','content'] },
        async (a) => this.memory.share(a.agentId, a.content, a.metadata));

      s('fusionpact_memory_forget', 'Delete agent memories (GDPR-style erasure).',
        { type: 'object', properties: { agentId: { type: 'string' }, type: { type: 'string', enum: ['episodic','semantic','procedural','all'] }, ids: { type: 'array', items: { type: 'string' } } }, required: ['agentId'] },
        async (a) => this.memory.forget(a.agentId, a));
    }

    return tools;
  }

  async _startStdio() {
    process.stdin.setEncoding('utf8');
    let buf = '';
    process.stdin.on('data', async (chunk) => {
      buf += chunk;
      const lines = buf.split('\n'); buf = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const msg = JSON.parse(line);
          const res = msg.method === 'tools/list' ? { result: this.getToolManifest() }
            : msg.method === 'tools/call' ? await this.handleToolCall(msg.params?.name, msg.params?.arguments)
            : { error: `Unknown method: ${msg.method}` };
          process.stdout.write(JSON.stringify({ id: msg.id, ...res }) + '\n');
        } catch (e) { process.stdout.write(JSON.stringify({ error: e.message }) + '\n'); }
      }
    });
    this.emit('started', { transport: 'stdio' });
  }

  async _startHTTP() {
    const srv = createServer(async (req, res) => {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
      if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

      if (req.method === 'GET' && req.url === '/mcp/tools') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify(this.getToolManifest()));
      }
      if (req.method === 'POST' && req.url === '/mcp/call') {
        let body = ''; req.on('data', c => body += c);
        req.on('end', async () => {
          try {
            const { tool, args } = JSON.parse(body);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(await this.handleToolCall(tool, args)));
          } catch (e) { res.writeHead(400); res.end(JSON.stringify({ error: e.message })); }
        });
        return;
      }
      res.writeHead(404); res.end('Not found');
    });
    srv.listen(this.port, () => this.emit('started', { transport: 'http', port: this.port }));
  }

  _mockVec(text) {
    const d = 64, v = new Float32Array(d);
    for (let i = 0; i < d; i++) { let h = 5381+i; for (let j = 0; j < Math.min(text.length,50); j++) h = ((h<<5)+h+text.charCodeAt(j))|0; v[i] = ((h%2000)-1000)/1000; }
    let n = 0; for (let i = 0; i < d; i++) n += v[i]*v[i]; n = Math.sqrt(n);
    if (n > 0) for (let i = 0; i < d; i++) v[i] /= n;
    return Array.from(v);
  }
}

module.exports = { MCPServer };
