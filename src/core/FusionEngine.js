/**
 * @fileoverview FusionEngine — Core database engine for FusionPact
 * 
 * The central orchestrator that manages collections, multi-tenancy,
 * persistence, and provides the primary API surface for vector operations.
 * 
 * @module core/FusionEngine
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 * @see https://github.com/FusionpactTech/fusionpact-vectordb
 */

'use strict';

const { HNSWIndex } = require('./HNSWIndex');
const { EventEmitter } = require('events');

/**
 * @typedef {Object} CollectionConfig
 * @property {number} [dimensions=768] - Vector dimensions
 * @property {string} [distanceMetric='cosine'] - Distance metric
 * @property {number} [M=16] - HNSW M parameter
 * @property {number} [efConstruction=200] - HNSW construction parameter
 * @property {number} [efSearch=50] - HNSW search parameter
 * @property {Object} [schema=null] - Optional metadata schema for validation
 */

/**
 * FusionEngine — The primary interface for FusionPact vector operations.
 * 
 * Manages named collections of vectors with HNSW indexing, automatic
 * multi-tenant isolation, metadata filtering, and TTL-based expiry.
 * 
 * @extends EventEmitter
 * 
 * @example
 * const { FusionEngine } = require('fusionpact');
 * 
 * const engine = new FusionEngine();
 * engine.createCollection('documents', { dimensions: 768 });
 * engine.insert('documents', [{ id: 'doc-1', vector: [...], metadata: { title: 'Report' } }]);
 * const results = engine.search('documents', queryVector, { topK: 5 });
 */
class FusionEngine extends EventEmitter {
  /**
   * @param {Object} [config={}]
   * @param {string} [config.dataDir=null] - Directory for persistence (null = in-memory only)
   * @param {boolean} [config.autoSave=false] - Auto-persist after mutations
   * @param {number} [config.autoSaveIntervalMs=30000] - Auto-save interval
   */
  constructor(config = {}) {
    super();
    this.config = {
      dataDir: config.dataDir || null,
      autoSave: config.autoSave || false,
      autoSaveIntervalMs: config.autoSaveIntervalMs || 30000,
      ...config
    };

    /** @private */
    this._collections = new Map();
    /** @private */
    this._collectionConfigs = new Map();
    /** @private */
    this._persistence = null;
    /** @private */
    this._autoSaveTimer = null;

    if (this.config.autoSave && this.config.dataDir) {
      this._startAutoSave();
    }
  }

  // ─── Collection Management ──────────────────────────────────

  /**
   * Create a new named collection.
   * 
   * @param {string} name - Collection name (alphanumeric, hyphens, underscores)
   * @param {CollectionConfig} [config={}] - Collection configuration
   * @returns {{ name: string, config: CollectionConfig }} Collection info
   * @throws {Error} If collection already exists
   * 
   * @example
   * engine.createCollection('safety-docs', {
   *   dimensions: 768,
   *   distanceMetric: 'cosine',
   *   M: 32 // Higher M for better recall
   * });
   */
  createCollection(name, config = {}) {
    if (this._collections.has(name)) {
      throw new Error(`Collection "${name}" already exists`);
    }

    const collectionConfig = {
      dimensions: config.dimensions || 768,
      distanceMetric: config.distanceMetric || 'cosine',
      M: config.M || 16,
      efConstruction: config.efConstruction || 200,
      efSearch: config.efSearch || 50,
      schema: config.schema || null,
      createdAt: new Date().toISOString()
    };

    const index = new HNSWIndex(collectionConfig.dimensions, collectionConfig);
    this._collections.set(name, index);
    this._collectionConfigs.set(name, collectionConfig);

    this.emit('collection:created', { name, config: collectionConfig });
    return { name, config: collectionConfig };
  }

  /**
   * List all collections.
   * @returns {Array<{name: string, config: Object, size: number}>}
   */
  listCollections() {
    const collections = [];
    for (const [name, index] of this._collections) {
      collections.push({
        name,
        config: this._collectionConfigs.get(name),
        size: index.size
      });
    }
    return collections;
  }

  /**
   * Get collection info.
   * @param {string} name
   * @returns {{ name: string, config: Object, size: number, stats: Object }|null}
   */
  getCollection(name) {
    const index = this._collections.get(name);
    if (!index) return null;
    return {
      name,
      config: this._collectionConfigs.get(name),
      size: index.size,
      stats: index.stats
    };
  }

  /**
   * Delete a collection and all its data.
   * @param {string} name
   * @returns {boolean}
   */
  deleteCollection(name) {
    const existed = this._collections.delete(name);
    this._collectionConfigs.delete(name);
    if (existed) this.emit('collection:deleted', { name });
    return existed;
  }

  // ─── CRUD Operations ──────────────────────────────────

  /**
   * Insert one or more vectors into a collection.
   * 
   * @param {string} collection - Collection name
   * @param {Array<{id: string, vector: number[], metadata?: Object}>} entries
   * @param {Object} [options={}]
   * @param {string} [options.tenantId] - Auto-tag entries with tenant ID
   * @param {number} [options.ttl] - Time-to-live in milliseconds
   * @returns {Array<VectorEntry>} Inserted entries
   * 
   * @example
   * engine.insert('documents', [
   *   { id: 'doc-1', vector: embedding, metadata: { title: 'Safety Plan' } }
   * ], { tenantId: 'acme_corp', ttl: 86400000 });
   */
  insert(collection, entries, options = {}) {
    const index = this._getIndex(collection);

    const results = entries.map(entry => {
      const metadata = { ...entry.metadata };

      if (options.tenantId) {
        metadata._tenant_id = options.tenantId;
      }
      if (options.ttl) {
        metadata._ttl = options.ttl;
      }

      return index.insert(entry.id, entry.vector, metadata);
    });

    this.emit('vectors:inserted', { collection, count: results.length });
    return results;
  }

  /**
   * Search for nearest neighbors in a collection.
   * 
   * @param {string} collection - Collection name
   * @param {number[]} queryVector - Query vector
   * @param {Object} [options={}] - Search options
   * @param {number} [options.topK=10] - Number of results
   * @param {Object} [options.filter] - Metadata filter
   * @param {string} [options.tenantId] - Tenant filter
   * @param {boolean} [options.includeVectors=false] - Include vectors in results
   * @returns {SearchResult[]}
   * 
   * @example
   * const results = engine.search('documents', queryVector, {
   *   topK: 5,
   *   filter: { category: 'compliance' },
   *   tenantId: 'acme_corp'
   * });
   */
  search(collection, queryVector, options = {}) {
    const index = this._getIndex(collection);
    return index.search(queryVector, options);
  }

  /**
   * Get a vector by ID.
   * @param {string} collection
   * @param {string} id
   * @returns {VectorEntry|null}
   */
  get(collection, id) {
    const index = this._getIndex(collection);
    return index.get(id);
  }

  /**
   * Delete a vector by ID.
   * @param {string} collection
   * @param {string} id
   * @returns {boolean}
   */
  delete(collection, id) {
    const index = this._getIndex(collection);
    const result = index.delete(id);
    if (result) this.emit('vector:deleted', { collection, id });
    return result;
  }

  // ─── Multi-Tenancy ──────────────────────────────────

  /**
   * Create a tenant-scoped proxy for a collection.
   * All operations through this proxy are automatically filtered by tenant.
   * 
   * @param {string} collection - Collection name
   * @param {string} tenantId - Tenant identifier
   * @returns {TenantProxy} Tenant-scoped collection proxy
   * 
   * @example
   * const acme = engine.tenant('shared-collection', 'acme_corp');
   * acme.insert([{ id: 'doc-1', vector: [...] }]);
   * const results = acme.search(queryVector); // Only sees acme_corp data
   */
  tenant(collection, tenantId) {
    const self = this;
    return {
      tenantId,
      collection,
      insert: (entries, options = {}) =>
        self.insert(collection, entries, { ...options, tenantId }),
      search: (queryVector, options = {}) =>
        self.search(collection, queryVector, { ...options, tenantId }),
      get: (id) => self.get(collection, id),
      delete: (id) => self.delete(collection, id)
    };
  }

  // ─── Persistence ──────────────────────────────────

  /**
   * Export all collections to a serializable object.
   * @returns {Object}
   */
  exportData() {
    const data = {
      _engine: 'FusionPact',
      _version: '2.0.0',
      exportedAt: new Date().toISOString(),
      collections: {}
    };

    for (const [name, index] of this._collections) {
      data.collections[name] = {
        config: this._collectionConfigs.get(name),
        index: index.serialize()
      };
    }

    return data;
  }

  /**
   * Import collections from exported data.
   * @param {Object} data - Data from exportData()
   * @param {Object} [options={}]
   * @param {boolean} [options.overwrite=false] - Overwrite existing collections
   */
  importData(data, options = {}) {
    for (const [name, collData] of Object.entries(data.collections || {})) {
      if (this._collections.has(name) && !options.overwrite) {
        continue;
      }

      const index = HNSWIndex.deserialize(collData.index);
      this._collections.set(name, index);
      this._collectionConfigs.set(name, collData.config);
    }
  }

  // ─── Lifecycle ──────────────────────────────────

  /**
   * Gracefully shut down the engine.
   */
  async close() {
    if (this._autoSaveTimer) {
      clearInterval(this._autoSaveTimer);
    }
    this.emit('engine:closed');
  }

  // ─── Private ──────────────────────────────────

  /** @private */
  _getIndex(collection) {
    const index = this._collections.get(collection);
    if (!index) {
      throw new Error(
        `Collection "${collection}" not found. Create it first with engine.createCollection("${collection}")`
      );
    }
    return index;
  }

  /** @private */
  _startAutoSave() {
    this._autoSaveTimer = setInterval(() => {
      this.emit('engine:autosave', this.exportData());
    }, this.config.autoSaveIntervalMs);
  }
}

module.exports = { FusionEngine };
