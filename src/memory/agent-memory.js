/**
 * FusionPact — Agent Memory Architecture
 *
 * Three memory types designed for AI agents:
 *   - Episodic:   What happened (conversation history, events, interactions)
 *   - Semantic:   What the agent knows (facts, documents, knowledge)
 *   - Procedural: What the agent can do (tool schemas, workflows, APIs)
 *
 * Each agent gets isolated memory with automatic tenant separation.
 */

'use strict';

const { createEmbedder } = require('../embeddings');
const { generateId, chunkText } = require('../utils');

class AgentMemory {
  /**
   * @param {import('../core/engine').FusionEngine} engine
   * @param {Object} [options]
   * @param {string} [options.collectionPrefix='memory']
   * @param {string|Object} [options.embedder='mock']
   */
  constructor(engine, options = {}) {
    this.engine = engine;
    this.prefix = options.collectionPrefix || 'memory';
    this.embedder = typeof options.embedder === 'object' && options.embedder.embed
      ? options.embedder
      : createEmbedder(options.embedder || 'mock');

    this._ensureCollections();
  }

  /** @private */
  _ensureCollections() {
    const dim = this.embedder.dimension;
    const types = ['episodic', 'semantic', 'procedural'];
    for (const type of types) {
      const name = `${this.prefix}_${type}`;
      if (!this.engine.collections.has(name)) {
        this.engine.createCollection(name, {
          dimension: dim, metric: 'cosine', indexType: 'hnsw',
          hnswConfig: { M: 16, efConstruction: 150, efSearch: 40 },
        });
      }
    }
  }

  /** @private */
  _colName(type) { return `${this.prefix}_${type}`; }

  // ─── Episodic Memory ────────────────────────────────────────
  // Stores events, conversations, interactions with timestamps

  /**
   * Remember an event or interaction
   * @param {string} agentId
   * @param {Object} event
   * @param {string} event.content — the text content to remember
   * @param {string} [event.role] — 'user', 'assistant', 'system', 'tool'
   * @param {string} [event.sessionId] — conversation/session identifier
   * @param {string|number} [event.ttl] — time to live, e.g. '24h', '7d', '30m', or ms
   * @param {Object} [event.metadata] — additional metadata
   * @returns {Promise<{id: string}>}
   */
  async remember(agentId, event) {
    const vector = await this.embedder.embedOne(event.content);
    const id = generateId('ep');

    const col = this._colName('episodic');
    // Auto-resize collection if embedding dimension changed
    this._resizeIfNeeded(col, vector.length);

    const doc = {
      id,
      vector,
      metadata: {
        _tenant_id: agentId,
        content: event.content,
        role: event.role || 'unknown',
        session_id: event.sessionId || 'default',
        timestamp: Date.now(),
        type: 'episodic',
        ...(event.metadata || {}),
      },
    };

    // TTL support — short-term vs long-term memory
    if (event.ttl) {
      doc.ttl = event.ttl;
    }

    this.engine.insert(col, [doc]);

    return { id };
  }

  /**
   * Recall relevant memories for a given context
   * @param {string} agentId
   * @param {string} context — what to recall about
   * @param {Object} [options]
   * @param {number} [options.topK=10]
   * @param {string} [options.sessionId] — filter by session
   * @param {string} [options.role] — filter by role
   * @returns {Promise<Object[]>}
   */
  async recall(agentId, context, options = {}) {
    const queryVec = await this.embedder.embedOne(context);
    const filter = { _tenant_id: { $eq: agentId } };
    if (options.sessionId) filter.session_id = { $eq: options.sessionId };
    if (options.role) filter.role = { $eq: options.role };

    const result = this.engine.query(this._colName('episodic'), queryVec, {
      topK: options.topK || 10,
      filter,
    });

    return result.results.map(r => ({
      id: r.id,
      content: r.metadata.content,
      role: r.metadata.role,
      sessionId: r.metadata.session_id,
      timestamp: r.metadata.timestamp,
      score: r.score,
      metadata: r.metadata,
    }));
  }

  // ─── Semantic Memory ────────────────────────────────────────
  // Stores facts, knowledge, documents the agent can reference

  /**
   * Add knowledge to the agent's semantic memory
   * @param {string} agentId
   * @param {string} knowledge — text content to learn
   * @param {Object} [options]
   * @param {string} [options.source='direct'] — where this knowledge came from
   * @param {string} [options.category] — knowledge category
   * @param {string|number} [options.ttl] — time to live, e.g. '24h', '7d', or ms
   * @param {Object} [options.metadata]
   * @returns {Promise<{ids: string[], chunks: number}>}
   */
  async learn(agentId, knowledge, options = {}) {
    // Chunk long knowledge into pieces
    const chunks = knowledge.length > 600
      ? chunkText(knowledge, { chunkSize: 500, overlap: 100 })
      : [{ text: knowledge, index: 0 }];

    const texts = chunks.map(c => c.text);
    const vectors = await this.embedder.embed(texts);

    const col = this._colName('semantic');
    this._resizeIfNeeded(col, vectors[0]?.length);

    const docs = chunks.map((chunk, i) => ({
      id: generateId('sem'),
      vector: vectors[i],
      ...(options.ttl ? { ttl: options.ttl } : {}),
      metadata: {
        _tenant_id: agentId,
        content: chunk.text,
        source: options.source || 'direct',
        category: options.category || 'general',
        chunk_index: chunk.index,
        timestamp: Date.now(),
        type: 'semantic',
        ...(options.metadata || {}),
      },
    }));

    const ids = this.engine.insert(col, docs);
    return { ids, chunks: ids.length };
  }

  /**
   * Search the agent's knowledge base
   * @param {string} agentId
   * @param {string} query
   * @param {Object} [options]
   * @param {number} [options.topK=5]
   * @param {string} [options.category]
   * @param {string} [options.source]
   * @returns {Promise<Object[]>}
   */
  async query(agentId, query, options = {}) {
    const queryVec = await this.embedder.embedOne(query);
    const filter = { _tenant_id: { $eq: agentId } };
    if (options.category) filter.category = { $eq: options.category };
    if (options.source) filter.source = { $eq: options.source };

    const result = this.engine.query(this._colName('semantic'), queryVec, {
      topK: options.topK || 5,
      filter,
    });

    return result.results.map(r => ({
      id: r.id,
      content: r.metadata.content,
      source: r.metadata.source,
      category: r.metadata.category,
      score: r.score,
      metadata: r.metadata,
    }));
  }

  // ─── Procedural Memory ──────────────────────────────────────
  // Stores tool schemas, API specs, workflow definitions

  /**
   * Register a tool or workflow the agent can use
   * @param {string} agentId
   * @param {Object} tool
   * @param {string} tool.name — tool name
   * @param {string} tool.description — what the tool does
   * @param {Object} [tool.schema] — JSON schema for parameters
   * @param {Object} [tool.metadata]
   * @returns {Promise<{id: string}>}
   */
  async registerTool(agentId, tool) {
    const description = `${tool.name}: ${tool.description}`;
    const vector = await this.embedder.embedOne(description);
    const id = generateId('proc');

    const col = this._colName('procedural');
    this._resizeIfNeeded(col, vector.length);

    this.engine.insert(col, [{
      id,
      vector,
      metadata: {
        _tenant_id: agentId,
        content: description,
        tool_name: tool.name,
        description: tool.description,
        schema: tool.schema ? JSON.stringify(tool.schema) : null,
        timestamp: Date.now(),
        type: 'procedural',
        ...(tool.metadata || {}),
      },
    }]);

    return { id };
  }

  /**
   * Find relevant tools for a given task
   * @param {string} agentId
   * @param {string} task — what the agent needs to do
   * @param {number} [topK=5]
   * @returns {Promise<Object[]>}
   */
  async findTools(agentId, task, topK = 5) {
    const queryVec = await this.embedder.embedOne(task);
    const result = this.engine.query(this._colName('procedural'), queryVec, {
      topK,
      filter: { _tenant_id: { $eq: agentId } },
    });

    return result.results.map(r => ({
      id: r.id,
      name: r.metadata.tool_name,
      description: r.metadata.description,
      schema: r.metadata.schema ? JSON.parse(r.metadata.schema) : null,
      score: r.score,
    }));
  }

  // ─── Cross-Memory Search ────────────────────────────────────

  /**
   * Search across all memory types for comprehensive context
   * @param {string} agentId
   * @param {string} query
   * @param {Object} [options]
   * @param {number} [options.topK=5] — per memory type
   * @returns {Promise<{episodic: Object[], semantic: Object[], procedural: Object[]}>}
   */
  async searchAll(agentId, query, options = {}) {
    const [episodic, semantic, procedural] = await Promise.all([
      this.recall(agentId, query, options),
      this.query(agentId, query, options),
      this.findTools(agentId, query, options.topK || 5),
    ]);

    return { episodic, semantic, procedural };
  }

  // ─── Forget (GDPR-friendly) ─────────────────────────────────

  /**
   * Forget memories matching criteria
   * @param {string} agentId
   * @param {Object} [options]
   * @param {string} [options.type] — 'episodic', 'semantic', 'procedural', or 'all'
   * @param {string} [options.sessionId] — forget a specific session
   * @param {string[]} [options.ids] — forget specific memory IDs
   * @returns {{deleted: number}}
   */
  forget(agentId, options = {}) {
    const types = options.type === 'all' || !options.type
      ? ['episodic', 'semantic', 'procedural']
      : [options.type];

    let totalDeleted = 0;

    for (const type of types) {
      const colName = this._colName(type);
      const col = this.engine.collections.get(colName);
      if (!col) continue;

      if (options.ids) {
        // Delete specific IDs (with tenant verification)
        const client = this.engine.tenant(colName, agentId);
        totalDeleted += client.delete(options.ids);
      } else {
        // Delete all matching documents
        const toDelete = [];
        for (const [id, doc] of col.documents) {
          if (doc.metadata._tenant_id !== agentId) continue;
          if (options.sessionId && doc.metadata.session_id !== options.sessionId) continue;
          toDelete.push(id);
        }
        totalDeleted += this.engine.delete(colName, toDelete);
      }
    }

    return { deleted: totalDeleted };
  }

  /**
   * Get memory stats for an agent
   * @param {string} agentId
   * @returns {Object}
   */
  getStats(agentId) {
    const stats = {};
    for (const type of ['episodic', 'semantic', 'procedural']) {
      const col = this.engine.collections.get(this._colName(type));
      if (!col) { stats[type] = { count: 0 }; continue; }

      let count = 0;
      for (const doc of col.documents.values()) {
        if (doc.metadata._tenant_id === agentId) count++;
      }
      stats[type] = { count };
    }
    return {
      agentId,
      ...stats,
      total: stats.episodic.count + stats.semantic.count + stats.procedural.count,
      embeddingProvider: this.embedder.provider,
    };
  }

  /** @private */
  _resizeIfNeeded(colName, expectedDim) {
    if (!expectedDim) return;
    const col = this.engine.collections.get(colName);
    if (col && col.dimension !== expectedDim) {
      this.engine.dropCollection(colName);
      this.engine.createCollection(colName, {
        dimension: expectedDim, metric: 'cosine', indexType: 'hnsw',
        hnswConfig: { M: 16, efConstruction: 150, efSearch: 40 },
      });
    }
  }
}

module.exports = { AgentMemory };
