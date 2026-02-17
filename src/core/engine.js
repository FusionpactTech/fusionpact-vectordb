/**
 * FusionPact — Core Engine
 * Collection management, query execution, multi-tenancy, persistence.
 */

'use strict';

const HNSWIndex = require('./hnsw');
const vec = require('./vectors');
const { generateId } = require('../utils');

class Collection {
  /**
   * @param {string} name
   * @param {Object} config
   * @param {number} config.dimension
   * @param {'cosine'|'euclidean'|'dot'} [config.metric='cosine']
   * @param {'hnsw'|'flat'} [config.indexType='hnsw']
   * @param {Object} [config.hnswConfig]
   */
  constructor(name, config = {}) {
    this.name = name;
    this.dimension = config.dimension || 128;
    this.metric = config.metric || 'cosine';
    this.indexType = config.indexType || 'hnsw';
    this.hnswConfig = config.hnswConfig || {};
    this.documents = new Map();
    this.hnsw = this.indexType === 'hnsw'
      ? new HNSWIndex(this.dimension, this.metric, this.hnswConfig)
      : null;
    this.createdAt = Date.now();
    this.stats = { insertions: 0, queries: 0, totalMs: 0, deletions: 0 };
  }

  get count() { return this.documents.size; }
}

class FusionEngine {
  constructor() {
    /** @type {Map<string, Collection>} */
    this.collections = new Map();
  }

  // ─── Collection Management ──────────────────────────────────

  /**
   * Create a new collection
   * @param {string} name
   * @param {Object} config
   * @returns {Object} collection info
   */
  createCollection(name, config = {}) {
    if (this.collections.has(name)) {
      throw new Error(`Collection '${name}' already exists`);
    }
    if (!name || typeof name !== 'string') {
      throw new Error('Collection name must be a non-empty string');
    }
    const col = new Collection(name, config);
    this.collections.set(name, col);
    return this._colInfo(col);
  }

  /**
   * Drop a collection
   * @param {string} name
   * @returns {boolean}
   */
  dropCollection(name) {
    return this.collections.delete(name);
  }

  /**
   * List all collections
   * @returns {Object[]}
   */
  listCollections() {
    return [...this.collections.values()].map(c => this._colInfo(c));
  }

  /**
   * Get collection info
   * @param {string} name
   * @returns {Object|null}
   */
  getCollection(name) {
    const c = this.collections.get(name);
    return c ? this._colInfo(c) : null;
  }

  /** @private */
  _colInfo(c) {
    return {
      name: c.name, dimension: c.dimension, metric: c.metric,
      indexType: c.indexType, count: c.count, createdAt: c.createdAt,
      stats: { ...c.stats, avgMs: c.stats.queries > 0 ? +(c.stats.totalMs / c.stats.queries).toFixed(3) : 0 },
      hnswStats: c.hnsw ? c.hnsw.getStats() : null,
    };
  }

  // ─── Insert ─────────────────────────────────────────────────

  /**
   * Insert documents into a collection
   * @param {string} collectionName
   * @param {Array<{id?: string, vector: number[], metadata?: Object}>} documents
   * @returns {string[]} inserted IDs
   */
  insert(collectionName, documents) {
    const c = this.collections.get(collectionName);
    if (!c) throw new Error(`Collection '${collectionName}' not found`);

    const ids = [];
    for (const doc of documents) {
      if (!doc.vector || !Array.isArray(doc.vector)) {
        throw new Error('Each document must have a vector array');
      }
      if (doc.vector.length !== c.dimension) {
        throw new Error(`Dimension mismatch: expected ${c.dimension}, got ${doc.vector.length}`);
      }

      const id = doc.id || generateId();
      const metadata = doc.metadata || {};
      const entry = { id, vector: vec.toFloat64(doc.vector), metadata };

      c.documents.set(id, entry);
      if (c.hnsw) c.hnsw.insert(id, doc.vector, metadata);
      c.stats.insertions++;
      ids.push(id);
    }

    return ids;
  }

  // ─── Delete ─────────────────────────────────────────────────

  /**
   * Delete documents by IDs
   * @param {string} collectionName
   * @param {string[]} ids
   * @returns {number} number of deleted documents
   */
  delete(collectionName, ids) {
    const c = this.collections.get(collectionName);
    if (!c) throw new Error(`Collection '${collectionName}' not found`);

    let deleted = 0;
    for (const id of ids) {
      if (c.documents.delete(id)) {
        if (c.hnsw) c.hnsw.delete(id);
        c.stats.deletions++;
        deleted++;
      }
    }
    return deleted;
  }

  // ─── Query ──────────────────────────────────────────────────

  /**
   * Query a collection for nearest neighbors
   * @param {string} collectionName
   * @param {number[]} queryVector
   * @param {Object} [options]
   * @param {number} [options.topK=10]
   * @param {Object} [options.filter] — metadata filter
   * @param {boolean} [options.forceFlat=false] — bypass HNSW, use brute force
   * @param {number} [options.efSearch] — override ef_search for this query
   * @param {boolean} [options.includeVectors=false]
   * @returns {Object}
   */
  query(collectionName, queryVector, options = {}) {
    const c = this.collections.get(collectionName);
    if (!c) throw new Error(`Collection '${collectionName}' not found`);

    const topK = options.topK || 10;
    const filter = options.filter || null;
    const forceFlat = options.forceFlat || false;
    const includeVectors = options.includeVectors || false;

    const t0 = performance.now();
    c.stats.queries++;

    let results, method, comparisons;

    if (c.hnsw && !forceFlat) {
      method = 'hnsw';
      const ef = options.efSearch || c.hnswConfig.efSearch || c.hnsw.efSearch;
      // Over-fetch when filtering, then post-filter
      const fetchK = filter ? Math.min(topK * 10, c.documents.size) : topK;
      let hnswResults = c.hnsw.search(queryVector, fetchK, Math.max(ef, fetchK));
      comparisons = c.hnsw._comparisons;

      if (filter) {
        hnswResults = hnswResults.filter(doc => this._matchFilter(doc.metadata, filter));
      }
      results = hnswResults.slice(0, topK);
    } else {
      method = 'flat';
      let candidates = [...c.documents.values()];
      if (filter) {
        candidates = candidates.filter(doc => this._matchFilter(doc.metadata, filter));
      }
      comparisons = candidates.length;
      results = candidates
        .map(doc => ({ ...doc, score: vec.score(queryVector, doc.vector, c.metric) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
    }

    const elapsed = performance.now() - t0;
    c.stats.totalMs += elapsed;

    return {
      results: results.map(r => ({
        id: r.id,
        score: r.score,
        metadata: r.metadata,
        ...(includeVectors ? { vector: [...r.vector] } : {}),
      })),
      elapsed: +elapsed.toFixed(3),
      comparisons,
      total: c.count,
      method,
    };
  }

  /**
   * Apply metadata filter to a document
   * @private
   * @param {Object} metadata
   * @param {Object} filter
   * @returns {boolean}
   */
  _matchFilter(metadata, filter) {
    for (const [key, condition] of Object.entries(filter)) {
      const value = metadata[key];
      if (value === undefined) return false;

      if (typeof condition === 'object' && condition !== null && !Array.isArray(condition)) {
        if (condition.$eq !== undefined && value !== condition.$eq) return false;
        if (condition.$ne !== undefined && value === condition.$ne) return false;
        if (condition.$gt !== undefined && value <= condition.$gt) return false;
        if (condition.$gte !== undefined && value < condition.$gte) return false;
        if (condition.$lt !== undefined && value >= condition.$lt) return false;
        if (condition.$lte !== undefined && value > condition.$lte) return false;
        if (condition.$in !== undefined && !condition.$in.includes(value)) return false;
        if (condition.$nin !== undefined && condition.$nin.includes(value)) return false;
        if (condition.$exists !== undefined) {
          if (condition.$exists && value === undefined) return false;
          if (!condition.$exists && value !== undefined) return false;
        }
      } else {
        // Direct equality
        if (value !== condition) return false;
      }
    }
    return true;
  }

  // ─── Tenant Client Factory ──────────────────────────────────

  /**
   * Create a tenant-scoped client for a collection
   * @param {string} collectionName
   * @param {string} tenantId
   * @returns {TenantClient}
   */
  tenant(collectionName, tenantId) {
    return new TenantClient(this, collectionName, tenantId);
  }
}

/**
 * TenantClient — Soft-isolation wrapper
 * Automatically injects _tenant_id on insert and filters on query.
 * There is NO code path that bypasses the tenant filter.
 */
class TenantClient {
  /**
   * @param {FusionEngine} engine
   * @param {string} collection
   * @param {string} tenantId
   */
  constructor(engine, collection, tenantId) {
    this.engine = engine;
    this.collection = collection;
    this.tenantId = tenantId;
  }

  /**
   * Insert documents with automatic tenant tagging
   * @param {Array<{id?: string, vector: number[], metadata?: Object}>} documents
   * @returns {string[]}
   */
  insert(documents) {
    const tagged = documents.map(d => ({
      ...d,
      metadata: { ...(d.metadata || {}), _tenant_id: this.tenantId },
    }));
    return this.engine.insert(this.collection, tagged);
  }

  /**
   * Query with automatic tenant isolation
   * @param {number[]} queryVector
   * @param {Object} [options]
   * @returns {Object}
   */
  query(queryVector, options = {}) {
    const filter = {
      ...(options.filter || {}),
      _tenant_id: { $eq: this.tenantId },
    };
    return this.engine.query(this.collection, queryVector, { ...options, filter });
  }

  /**
   * Delete documents (only within this tenant's scope)
   * @param {string[]} ids
   * @returns {number}
   */
  delete(ids) {
    // Verify documents belong to this tenant before deleting
    const c = this.engine.collections.get(this.collection);
    if (!c) throw new Error(`Collection '${this.collection}' not found`);

    const safeIds = ids.filter(id => {
      const doc = c.documents.get(id);
      return doc && doc.metadata._tenant_id === this.tenantId;
    });
    return this.engine.delete(this.collection, safeIds);
  }
}

module.exports = { FusionEngine, TenantClient, Collection };
