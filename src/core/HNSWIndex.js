/**
 * @fileoverview HNSW (Hierarchical Navigable Small World) Vector Index
 * 
 * High-performance approximate nearest neighbor search using the HNSW algorithm.
 * Provides O(log N) search complexity with configurable accuracy/speed tradeoffs.
 * 
 * @module core/HNSWIndex
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 * @see https://github.com/FusionpactTech/fusionpact-vectordb
 * 
 * Part of the FusionPact Agent-Native Retrieval Engine.
 * Copyright (c) 2024-2026 FusionPact Technologies Inc. All rights reserved.
 */

'use strict';

/**
 * @typedef {Object} HNSWConfig
 * @property {number} [M=16] - Number of bi-directional links per node. Higher = better recall, more memory.
 * @property {number} [efConstruction=200] - Size of dynamic candidate list during construction. Higher = better index quality, slower build.
 * @property {number} [efSearch=50] - Size of dynamic candidate list during search. Higher = better recall, slower search.
 * @property {string} [distanceMetric='cosine'] - Distance metric: 'cosine', 'euclidean', or 'dotProduct'.
 * @property {number} [maxLevel=null] - Maximum number of layers. Auto-calculated if null.
 */

/**
 * @typedef {Object} VectorEntry
 * @property {string} id - Unique identifier for this vector
 * @property {Float32Array|number[]} vector - The embedding vector
 * @property {Object} [metadata={}] - Arbitrary metadata attached to this vector
 * @property {string} [_tenant_id] - Tenant identifier for multi-tenant isolation
 * @property {number} [_timestamp] - Insertion timestamp (epoch ms)
 * @property {number} [_ttl] - Time-to-live in milliseconds (0 = no expiry)
 */

/**
 * @typedef {Object} SearchResult
 * @property {string} id - Vector entry ID
 * @property {number} score - Similarity/distance score (higher = more similar for cosine)
 * @property {Object} metadata - Entry metadata
 * @property {Float32Array} [vector] - Original vector (if includeVectors=true)
 */

class HNSWIndex {
  /**
   * Creates a new HNSW index.
   * 
   * @param {number} dimensions - Dimensionality of vectors
   * @param {HNSWConfig} [config={}] - Index configuration
   * 
   * @example
   * const index = new HNSWIndex(768, {
   *   M: 16,
   *   efConstruction: 200,
   *   efSearch: 50,
   *   distanceMetric: 'cosine'
   * });
   */
  constructor(dimensions, config = {}) {
    this.dimensions = dimensions;
    this.M = config.M || 16;
    this.efConstruction = config.efConstruction || 200;
    this.efSearch = config.efSearch || 50;
    this.distanceMetric = config.distanceMetric || 'cosine';
    this.mL = 1 / Math.log(this.M);

    // Storage
    this._nodes = new Map();          // id -> { vector, metadata, level, neighbors }
    this._entryPoint = null;          // Entry point node ID
    this._maxLevel = 0;               // Current max level in the graph
    this._size = 0;

    // Distance function
    this._distance = this._getDistanceFunction(this.distanceMetric);

    // Statistics
    this._stats = {
      insertions: 0,
      searches: 0,
      deletions: 0,
      totalSearchTimeMs: 0,
      totalInsertTimeMs: 0
    };
  }

  /**
   * Returns the number of vectors in the index.
   * @returns {number}
   */
  get size() {
    return this._size;
  }

  /**
   * Returns index statistics.
   * @returns {Object} Performance and usage statistics
   */
  get stats() {
    return {
      ...this._stats,
      size: this._size,
      dimensions: this.dimensions,
      maxLevel: this._maxLevel,
      avgSearchTimeMs: this._stats.searches > 0
        ? (this._stats.totalSearchTimeMs / this._stats.searches).toFixed(3)
        : 0,
      avgInsertTimeMs: this._stats.insertions > 0
        ? (this._stats.totalInsertTimeMs / this._stats.insertions).toFixed(3)
        : 0
    };
  }

  /**
   * Inserts a vector into the index.
   * 
   * @param {string} id - Unique identifier
   * @param {Float32Array|number[]} vector - The embedding vector
   * @param {Object} [metadata={}] - Metadata to store with the vector
   * @returns {VectorEntry} The inserted entry
   * @throws {Error} If vector dimensions don't match index dimensions
   * 
   * @example
   * index.insert('doc-1', [0.1, 0.2, ...], { title: 'Safety Manual', page: 42 });
   */
  insert(id, vector, metadata = {}) {
    const start = performance.now();

    if (vector.length !== this.dimensions) {
      throw new Error(
        `Vector dimension mismatch: expected ${this.dimensions}, got ${vector.length}`
      );
    }

    const vec = vector instanceof Float32Array ? vector : new Float32Array(vector);

    // Normalize for cosine similarity
    if (this.distanceMetric === 'cosine') {
      this._normalize(vec);
    }

    const level = this._randomLevel();
    const node = {
      id,
      vector: vec,
      metadata: { ...metadata, _timestamp: Date.now() },
      level,
      neighbors: new Array(level + 1).fill(null).map(() => [])
    };

    if (this._size === 0) {
      this._nodes.set(id, node);
      this._entryPoint = id;
      this._maxLevel = level;
      this._size++;
      this._stats.insertions++;
      this._stats.totalInsertTimeMs += performance.now() - start;
      return { id, vector: vec, metadata: node.metadata };
    }

    let currentNodeId = this._entryPoint;

    // Phase 1: Greedy traverse from top to insertion level
    for (let l = this._maxLevel; l > level; l--) {
      currentNodeId = this._greedySearch(vec, currentNodeId, l);
    }

    // Phase 2: Insert at each level from insertion level down to 0
    for (let l = Math.min(level, this._maxLevel); l >= 0; l--) {
      const neighbors = this._searchLayer(vec, currentNodeId, this.efConstruction, l);

      // Select M best neighbors
      const selected = neighbors.slice(0, this.M);
      node.neighbors[l] = selected.map(n => n.id);

      // Add bidirectional connections
      for (const neighbor of selected) {
        const neighborNode = this._nodes.get(neighbor.id);
        if (neighborNode && neighborNode.neighbors[l]) {
          neighborNode.neighbors[l].push(id);

          // Prune if too many connections
          if (neighborNode.neighbors[l].length > this.M * 2) {
            neighborNode.neighbors[l] = this._pruneConnections(
              neighborNode, l, this.M * 2
            );
          }
        }
      }

      if (selected.length > 0) {
        currentNodeId = selected[0].id;
      }
    }

    this._nodes.set(id, node);

    if (level > this._maxLevel) {
      this._maxLevel = level;
      this._entryPoint = id;
    }

    this._size++;
    this._stats.insertions++;
    this._stats.totalInsertTimeMs += performance.now() - start;

    return { id, vector: vec, metadata: node.metadata };
  }

  /**
   * Batch insert multiple vectors.
   * 
   * @param {Array<{id: string, vector: number[], metadata?: Object}>} entries
   * @returns {Array<VectorEntry>} Inserted entries
   * 
   * @example
   * index.insertBatch([
   *   { id: 'doc-1', vector: [...], metadata: { title: 'Report A' } },
   *   { id: 'doc-2', vector: [...], metadata: { title: 'Report B' } }
   * ]);
   */
  insertBatch(entries) {
    return entries.map(e => this.insert(e.id, e.vector, e.metadata || {}));
  }

  /**
   * Search for the nearest neighbors of a query vector.
   * 
   * @param {Float32Array|number[]} queryVector - The query vector
   * @param {Object} [options={}] - Search options
   * @param {number} [options.topK=10] - Number of results to return
   * @param {number} [options.efSearch] - Override efSearch for this query
   * @param {Object} [options.filter] - Metadata filter (key-value match)
   * @param {string} [options.tenantId] - Tenant filter for multi-tenant queries
   * @param {boolean} [options.includeVectors=false] - Include original vectors in results
   * @returns {SearchResult[]} Sorted results (best match first)
   * 
   * @example
   * const results = index.search(queryVector, {
   *   topK: 5,
   *   filter: { category: 'safety' },
   *   tenantId: 'acme_corp'
   * });
   */
  search(queryVector, options = {}) {
    const start = performance.now();
    const {
      topK = 10,
      efSearch = this.efSearch,
      filter = null,
      tenantId = null,
      includeVectors = false
    } = options;

    if (this._size === 0) return [];

    const vec = queryVector instanceof Float32Array
      ? queryVector
      : new Float32Array(queryVector);

    if (this.distanceMetric === 'cosine') {
      this._normalize(vec);
    }

    // Phase 1: Greedy descent to layer 0
    let currentNodeId = this._entryPoint;
    for (let l = this._maxLevel; l > 0; l--) {
      currentNodeId = this._greedySearch(vec, currentNodeId, l);
    }

    // Phase 2: Search at layer 0 with efSearch
    const ef = Math.max(efSearch, topK);
    let candidates = this._searchLayer(vec, currentNodeId, ef, 0);

    // Apply filters
    if (tenantId) {
      candidates = candidates.filter(c => {
        const node = this._nodes.get(c.id);
        return node && node.metadata._tenant_id === tenantId;
      });
    }

    if (filter) {
      candidates = candidates.filter(c => {
        const node = this._nodes.get(c.id);
        if (!node) return false;
        return Object.entries(filter).every(([key, value]) => {
          if (Array.isArray(value)) return value.includes(node.metadata[key]);
          return node.metadata[key] === value;
        });
      });
    }

    // Filter expired TTL entries
    const now = Date.now();
    candidates = candidates.filter(c => {
      const node = this._nodes.get(c.id);
      if (!node || !node.metadata._ttl) return true;
      return (now - node.metadata._timestamp) < node.metadata._ttl;
    });

    // Return top-K
    const results = candidates.slice(0, topK).map(c => {
      const node = this._nodes.get(c.id);
      const result = {
        id: c.id,
        score: 1 - c.distance,  // Convert distance to similarity
        metadata: { ...node.metadata }
      };
      if (includeVectors) {
        result.vector = node.vector;
      }
      return result;
    });

    this._stats.searches++;
    this._stats.totalSearchTimeMs += performance.now() - start;

    return results;
  }

  /**
   * Delete a vector from the index.
   * 
   * @param {string} id - Vector ID to delete
   * @returns {boolean} True if deleted, false if not found
   */
  delete(id) {
    const node = this._nodes.get(id);
    if (!node) return false;

    // Remove bidirectional connections
    for (let l = 0; l <= node.level; l++) {
      for (const neighborId of node.neighbors[l]) {
        const neighbor = this._nodes.get(neighborId);
        if (neighbor && neighbor.neighbors[l]) {
          neighbor.neighbors[l] = neighbor.neighbors[l].filter(n => n !== id);
        }
      }
    }

    this._nodes.delete(id);
    this._size--;
    this._stats.deletions++;

    // Update entry point if needed
    if (id === this._entryPoint && this._size > 0) {
      this._entryPoint = this._nodes.keys().next().value;
      this._maxLevel = this._nodes.get(this._entryPoint).level;
    }

    return true;
  }

  /**
   * Get a single vector entry by ID.
   * 
   * @param {string} id - Vector ID
   * @returns {VectorEntry|null}
   */
  get(id) {
    const node = this._nodes.get(id);
    if (!node) return null;
    return { id: node.id, vector: node.vector, metadata: { ...node.metadata } };
  }

  /**
   * Check if a vector ID exists in the index.
   * @param {string} id
   * @returns {boolean}
   */
  has(id) {
    return this._nodes.has(id);
  }

  /**
   * Clear all vectors from the index.
   */
  clear() {
    this._nodes.clear();
    this._entryPoint = null;
    this._maxLevel = 0;
    this._size = 0;
  }

  /**
   * Export index data for persistence.
   * @returns {Object} Serializable index state
   */
  serialize() {
    const nodes = {};
    for (const [id, node] of this._nodes) {
      nodes[id] = {
        id: node.id,
        vector: Array.from(node.vector),
        metadata: node.metadata,
        level: node.level,
        neighbors: node.neighbors
      };
    }
    return {
      _version: 2,
      _engine: 'FusionPact',
      dimensions: this.dimensions,
      config: {
        M: this.M,
        efConstruction: this.efConstruction,
        efSearch: this.efSearch,
        distanceMetric: this.distanceMetric
      },
      entryPoint: this._entryPoint,
      maxLevel: this._maxLevel,
      size: this._size,
      nodes,
      stats: this._stats
    };
  }

  /**
   * Load index data from a serialized state.
   * @param {Object} data - Serialized index state from serialize()
   * @returns {HNSWIndex} The loaded index instance
   */
  static deserialize(data) {
    const index = new HNSWIndex(data.dimensions, data.config);
    index._entryPoint = data.entryPoint;
    index._maxLevel = data.maxLevel;
    index._size = data.size;
    index._stats = data.stats || index._stats;

    for (const [id, nodeData] of Object.entries(data.nodes)) {
      index._nodes.set(id, {
        ...nodeData,
        vector: new Float32Array(nodeData.vector)
      });
    }

    return index;
  }

  // --- Internal Methods ---

  _getDistanceFunction(metric) {
    switch (metric) {
      case 'cosine':
        // After normalization, cosine distance = 1 - dot product
        return (a, b) => {
          let dot = 0;
          for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
          return 1 - dot;
        };
      case 'euclidean':
        return (a, b) => {
          let sum = 0;
          for (let i = 0; i < a.length; i++) {
            const d = a[i] - b[i];
            sum += d * d;
          }
          return Math.sqrt(sum);
        };
      case 'dotProduct':
        return (a, b) => {
          let dot = 0;
          for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
          return -dot; // Negate so lower = better
        };
      default:
        throw new Error(`Unknown distance metric: ${metric}`);
    }
  }

  _normalize(vec) {
    let norm = 0;
    for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let i = 0; i < vec.length; i++) vec[i] /= norm;
    }
  }

  _randomLevel() {
    let level = 0;
    while (Math.random() < (1 / this.M) && level < 32) level++;
    return level;
  }

  _greedySearch(queryVec, startNodeId, level) {
    let currentId = startNodeId;
    let currentDist = this._distance(queryVec, this._nodes.get(currentId).vector);

    let improved = true;
    while (improved) {
      improved = false;
      const node = this._nodes.get(currentId);
      if (!node || !node.neighbors[level]) break;

      for (const neighborId of node.neighbors[level]) {
        const neighbor = this._nodes.get(neighborId);
        if (!neighbor) continue;
        const dist = this._distance(queryVec, neighbor.vector);
        if (dist < currentDist) {
          currentDist = dist;
          currentId = neighborId;
          improved = true;
        }
      }
    }

    return currentId;
  }

  _searchLayer(queryVec, entryNodeId, ef, level) {
    const visited = new Set([entryNodeId]);
    const entryNode = this._nodes.get(entryNodeId);
    const entryDist = this._distance(queryVec, entryNode.vector);

    const candidates = [{ id: entryNodeId, distance: entryDist }];
    const results = [{ id: entryNodeId, distance: entryDist }];

    while (candidates.length > 0) {
      // Get nearest candidate
      candidates.sort((a, b) => a.distance - b.distance);
      const nearest = candidates.shift();

      // Get farthest result
      const farthest = results[results.length - 1];
      if (nearest.distance > farthest.distance) break;

      const node = this._nodes.get(nearest.id);
      if (!node || !node.neighbors[level]) continue;

      for (const neighborId of node.neighbors[level]) {
        if (visited.has(neighborId)) continue;
        visited.add(neighborId);

        const neighbor = this._nodes.get(neighborId);
        if (!neighbor) continue;

        const dist = this._distance(queryVec, neighbor.vector);
        const entry = { id: neighborId, distance: dist };

        if (results.length < ef || dist < results[results.length - 1].distance) {
          candidates.push(entry);
          results.push(entry);
          results.sort((a, b) => a.distance - b.distance);
          if (results.length > ef) results.pop();
        }
      }
    }

    return results;
  }

  _pruneConnections(node, level, maxConnections) {
    const scored = node.neighbors[level]
      .map(neighborId => {
        const neighbor = this._nodes.get(neighborId);
        if (!neighbor) return null;
        return {
          id: neighborId,
          distance: this._distance(node.vector, neighbor.vector)
        };
      })
      .filter(Boolean)
      .sort((a, b) => a.distance - b.distance);

    return scored.slice(0, maxConnections).map(s => s.id);
  }
}

module.exports = { HNSWIndex };
