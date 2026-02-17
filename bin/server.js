/**
 * FusionPact — HTTP API Server
 * Zero-dependency REST API for vector operations, RAG, and agent memory.
 */

'use strict';

const http = require('http');
const { FusionEngine } = require('../src/core/engine');
const { RAGPipeline } = require('../src/core/rag');
const { AgentMemory } = require('../src/memory/agent-memory');
const { createEmbedder } = require('../src/embeddings');
const vec = require('../src/core/vectors');

function startServer(port = 8080) {
  const engine = new FusionEngine();
  const embedder = createEmbedder();
  let rag = null;
  let memory = null;

  function getRAG() {
    if (!rag) rag = new RAGPipeline(engine, { embedder });
    return rag;
  }

  function getMemory() {
    if (!memory) memory = new AgentMemory(engine, { embedder });
    return memory;
  }

  function parseBody(req) {
    return new Promise((resolve, reject) => {
      let body = '';
      req.on('data', chunk => { body += chunk; if (body.length > 50e6) reject(new Error('Body too large')); });
      req.on('end', () => {
        if (!body) return resolve({});
        try { resolve(JSON.parse(body)); } catch { reject(new Error('Invalid JSON')); }
      });
    });
  }

  function send(res, status, data) {
    res.writeHead(status, {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type,Authorization',
    });
    res.end(JSON.stringify(data));
  }

  const server = http.createServer(async (req, res) => {
    if (req.method === 'OPTIONS') {
      res.writeHead(204, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type,Authorization',
      });
      return res.end();
    }

    const url = new URL(req.url, `http://localhost:${port}`);
    const path = url.pathname;

    try {
      // ── Health ──
      if (path === '/api/health' || path === '/') {
        return send(res, 200, {
          status: 'ok',
          engine: 'FusionPact VectorDB',
          version: '0.1.0',
          collections: engine.listCollections().length,
          embedding: embedder.provider,
          uptime: Math.round(process.uptime()),
        });
      }

      // ── Collections ──
      if (path === '/api/collections' && req.method === 'GET') {
        return send(res, 200, { collections: engine.listCollections() });
      }

      if (path === '/api/collections' && req.method === 'POST') {
        const body = await parseBody(req);
        const info = engine.createCollection(body.name, {
          dimension: body.dimension || embedder.dimension,
          metric: body.metric || 'cosine',
          indexType: body.indexType || 'hnsw',
          hnswConfig: body.hnswConfig,
        });
        return send(res, 201, info);
      }

      if (path.match(/^\/api\/collections\/[^/]+$/) && req.method === 'GET') {
        const name = decodeURIComponent(path.split('/').pop());
        const info = engine.getCollection(name);
        return info ? send(res, 200, info) : send(res, 404, { error: `Collection '${name}' not found` });
      }

      if (path.match(/^\/api\/collections\/[^/]+$/) && req.method === 'DELETE') {
        const name = decodeURIComponent(path.split('/').pop());
        const ok = engine.dropCollection(name);
        return ok ? send(res, 200, { dropped: name }) : send(res, 404, { error: `Collection '${name}' not found` });
      }

      // ── Insert ──
      if (path === '/api/insert' && req.method === 'POST') {
        const body = await parseBody(req);
        if (!body.collection) return send(res, 400, { error: 'collection required' });
        if (!body.documents?.length) return send(res, 400, { error: 'documents array required' });

        // Auto-embed if documents have text instead of vectors
        let docs = body.documents;
        if (docs[0].text && !docs[0].vector) {
          const texts = docs.map(d => d.text);
          const vectors = await embedder.embed(texts);
          docs = docs.map((d, i) => ({
            ...d, vector: vectors[i],
            metadata: { text: d.text, ...(d.metadata || {}) },
          }));
        }

        const ids = engine.insert(body.collection, docs);
        return send(res, 200, { inserted: ids.length, ids });
      }

      // ── Search ──
      if (path === '/api/search' && req.method === 'POST') {
        const body = await parseBody(req);
        if (!body.collection) return send(res, 400, { error: 'collection required' });
        if (!body.query && !body.vector) return send(res, 400, { error: 'query (text) or vector required' });

        let queryVec = body.vector;
        if (!queryVec && body.query) {
          queryVec = await embedder.embedOne(body.query);
        }

        const result = engine.query(body.collection, queryVec, {
          topK: body.topK || 10,
          filter: body.filter,
          includeVectors: body.includeVectors,
        });
        return send(res, 200, result);
      }

      // ── RAG ──
      if (path === '/api/rag/ingest' && req.method === 'POST') {
        const body = await parseBody(req);
        if (!body.text) return send(res, 400, { error: 'text required' });
        const info = await getRAG().ingest(body.text, {
          source: body.source || 'api-upload',
          metadata: body.metadata,
        });
        return send(res, 200, info);
      }

      if (path === '/api/rag/search' && req.method === 'POST') {
        const body = await parseBody(req);
        if (!body.question) return send(res, 400, { error: 'question required' });
        const result = await getRAG().search(body.question, { topK: body.topK || 5, filter: body.filter });
        return send(res, 200, result);
      }

      if (path === '/api/rag/context' && req.method === 'POST') {
        const body = await parseBody(req);
        if (!body.question) return send(res, 400, { error: 'question required' });
        const ctx = await getRAG().buildContext(body.question, {
          topK: body.topK || 5,
          systemPrompt: body.systemPrompt,
        });
        return send(res, 200, ctx);
      }

      // ── Agent Memory ──
      if (path === '/api/memory/remember' && req.method === 'POST') {
        const body = await parseBody(req);
        const result = await getMemory().remember(body.agentId || 'default', {
          content: body.content,
          role: body.role,
          sessionId: body.sessionId,
          metadata: body.metadata,
        });
        return send(res, 200, result);
      }

      if (path === '/api/memory/recall' && req.method === 'POST') {
        const body = await parseBody(req);
        const memories = await getMemory().recall(body.agentId || 'default', body.context, {
          topK: body.topK || 10,
          sessionId: body.sessionId,
        });
        return send(res, 200, { memories });
      }

      if (path === '/api/memory/learn' && req.method === 'POST') {
        const body = await parseBody(req);
        const result = await getMemory().learn(body.agentId || 'default', body.knowledge, {
          source: body.source, category: body.category,
        });
        return send(res, 200, result);
      }

      if (path === '/api/memory/forget' && req.method === 'POST') {
        const body = await parseBody(req);
        const result = getMemory().forget(body.agentId || 'default', {
          type: body.type, sessionId: body.sessionId, ids: body.ids,
        });
        return send(res, 200, result);
      }

      send(res, 404, { error: 'Not found' });
    } catch (err) {
      const status = err.message.includes('not found') ? 404
        : err.message.includes('already exists') ? 409 : 500;
      send(res, status, { error: err.message });
    }
  });

  server.listen(port, () => {
    console.log(`
╔════════════════════════════════════════════════════╗
║     ⚡ FusionPact VectorDB — HTTP API Server       ║
╠════════════════════════════════════════════════════╣
║  Port:      ${String(port).padEnd(38)}║
║  Embedding: ${embedder.provider.padEnd(38)}║
║  Dimension: ${String(embedder.dimension).padEnd(38)}║
╚════════════════════════════════════════════════════╝

  Collections:  GET|POST    /api/collections
  Insert:       POST        /api/insert
  Search:       POST        /api/search
  RAG Ingest:   POST        /api/rag/ingest
  RAG Search:   POST        /api/rag/search
  RAG Context:  POST        /api/rag/context
  Memory:       POST        /api/memory/{remember,recall,learn,forget}
  Health:       GET         /api/health
`);
  });

  return server;
}

module.exports = { startServer };
