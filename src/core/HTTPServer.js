/**
 * @fileoverview HTTP API Server for FusionPact
 * 
 * RESTful HTTP server exposing all FusionPact capabilities.
 * Uses Node.js built-in http module (zero external dependencies).
 * 
 * @module core/HTTPServer
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 * @see https://github.com/FusionpactTech/fusionpact-vectordb
 */

'use strict';

const http = require('http');

class HTTPServer {
  constructor(deps, config = {}) {
    this.engine = deps.engine;
    this.memory = deps.memory;
    this.rag = deps.rag;
    this.treeIndex = deps.treeIndex || null;
    this.hybridRetriever = deps.hybridRetriever || null;
    this.embedder = deps.embedder || null;

    this.port = config.port || 8080;
    this.host = config.host || '0.0.0.0';
    this._server = null;
  }

  async start() {
    this._server = http.createServer(async (req, res) => {
      // CORS
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
      res.setHeader('X-Powered-By', 'FusionPact Technologies Inc.');

      if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
      }

      try {
        const body = await this._readBody(req);
        const result = await this._route(req.method, req.url, body);
        res.writeHead(result.status || 200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result.data));
      } catch (err) {
        res.writeHead(err.status || 500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });

    return new Promise((resolve) => {
      this._server.listen(this.port, this.host, () => {
        console.log(`\n⚡ FusionPact API Server running at http://${this.host}:${this.port}`);
        console.log(`   Powered by FusionPact Technologies Inc.\n`);
        console.log('   Endpoints:');
        console.log('   GET  /api/health              Health check');
        console.log('   GET  /api/collections          List collections');
        console.log('   POST /api/collections          Create collection');
        console.log('   POST /api/insert               Insert documents');
        console.log('   POST /api/search               Vector search');
        console.log('   POST /api/hybrid-search        Hybrid retrieval');
        console.log('   POST /api/rag/ingest           RAG ingestion');
        console.log('   POST /api/rag/query            RAG context building');
        console.log('   POST /api/tree/index           Tree indexing');
        console.log('   POST /api/tree/search          Tree search');
        console.log('   POST /api/memory/remember      Store episodic memory');
        console.log('   POST /api/memory/recall        Recall memories');
        console.log('   POST /api/memory/learn         Add semantic knowledge');
        console.log('   POST /api/memory/share         Share across agents');
        console.log('   POST /api/memory/forget        GDPR erasure');
        console.log('   POST /api/conversation/add     Add conversation message');
        console.log('   POST /api/conversation/get     Get conversation history');
        console.log('');
        resolve(this._server);
      });
    });
  }

  stop() {
    if (this._server) this._server.close();
  }

  async _route(method, url, body) {
    const path = url.split('?')[0];

    // Health
    if (path === '/api/health') {
      return { data: { status: 'ok', engine: 'FusionPact', version: '2.0.0', vendor: 'FusionPact Technologies Inc.' } };
    }

    // Collections
    if (path === '/api/collections' && method === 'GET') {
      return { data: this.engine.listCollections() };
    }
    if (path === '/api/collections' && method === 'POST') {
      return { data: this.engine.createCollection(body.name, body), status: 201 };
    }

    // Insert
    if (path === '/api/insert' && method === 'POST') {
      let vector = body.vector;
      if (!vector && body.text && this.embedder) {
        vector = await this.embedder.embed(body.text);
      }
      const result = this.engine.insert(body.collection, [{
        id: body.id || `doc_${Date.now()}`,
        vector,
        metadata: { _content: body.text, ...body.metadata }
      }], { tenantId: body.tenantId });
      return { data: result, status: 201 };
    }

    // Search
    if (path === '/api/search' && method === 'POST') {
      let queryVector = body.vector;
      if (!queryVector && body.query && this.embedder) {
        queryVector = await this.embedder.embed(body.query);
      }
      return { data: this.engine.search(body.collection, queryVector, body) };
    }

    // Hybrid Search
    if (path === '/api/hybrid-search' && method === 'POST') {
      if (!this.hybridRetriever) throw { status: 400, message: 'HybridRetriever not configured' };
      return { data: await this.hybridRetriever.retrieve(body.query, body) };
    }

    // RAG
    if (path === '/api/rag/ingest' && method === 'POST') {
      return { data: await this.rag.ingest(body.text, body.metadata, body), status: 201 };
    }
    if (path === '/api/rag/query' && method === 'POST') {
      return { data: await this.rag.buildContext(body.query, body) };
    }

    // Tree Index
    if (path === '/api/tree/index' && method === 'POST') {
      if (!this.treeIndex) throw { status: 400, message: 'TreeIndex not configured' };
      return { data: await this.treeIndex.indexDocument(body.docId, body.content, body), status: 201 };
    }
    if (path === '/api/tree/search' && method === 'POST') {
      if (!this.treeIndex) throw { status: 400, message: 'TreeIndex not configured' };
      return { data: await this.treeIndex.search(body.docId, body.query, body) };
    }

    // Memory
    if (path === '/api/memory/remember' && method === 'POST') {
      return { data: await this.memory.remember(body.agentId || 'default', body), status: 201 };
    }
    if (path === '/api/memory/recall' && method === 'POST') {
      return { data: await this.memory.recall(body.agentId || 'default', body.query, body) };
    }
    if (path === '/api/memory/learn' && method === 'POST') {
      return { data: await this.memory.learn(body.agentId || 'default', body.content, body.metadata), status: 201 };
    }
    if (path === '/api/memory/share' && method === 'POST') {
      return { data: await this.memory.share(body.agentId || 'default', body.content, body.metadata), status: 201 };
    }
    if (path === '/api/memory/forget' && method === 'POST') {
      return { data: this.memory.forget(body.agentId || 'default', body) };
    }

    // Conversation
    if (path === '/api/conversation/add' && method === 'POST') {
      return { data: this.memory.addMessage(body.agentId || 'default', body.threadId, body), status: 201 };
    }
    if (path === '/api/conversation/get' && method === 'POST') {
      return { data: this.memory.getConversation(body.agentId || 'default', body.threadId, body) };
    }

    throw { status: 404, message: `Not found: ${method} ${path}` };
  }

  _readBody(req) {
    return new Promise((resolve) => {
      if (req.method === 'GET') return resolve({});
      let data = '';
      req.on('data', chunk => data += chunk);
      req.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch { resolve({}); }
      });
    });
  }
}

module.exports = { HTTPServer };
