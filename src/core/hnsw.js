/**
 * FusionPact — HNSW Index
 * Hierarchical Navigable Small World graph for Approximate Nearest Neighbor search.
 *
 * Algorithm: Malkov & Yashunin, 2016 (https://arxiv.org/abs/1603.09320)
 *
 * Configurable parameters:
 *   M             — max bidirectional connections per node per layer (default: 16)
 *   M0            — max connections at layer 0 (default: 2 * M)
 *   efConstruction — beam width during build (higher = better recall, slower build)
 *   efSearch       — beam width during query (higher = better recall, slower query)
 */

'use strict';

const vec = require('./vectors');

class HNSWIndex {
  /**
   * @param {number} dimension
   * @param {'cosine'|'euclidean'|'dot'} metric
   * @param {Object} [config]
   * @param {number} [config.M=16]
   * @param {number} [config.M0]
   * @param {number} [config.efConstruction=200]
   * @param {number} [config.efSearch=50]
   */
  constructor(dimension, metric = 'cosine', config = {}) {
    this.dimension = dimension;
    this.metric = metric;
    this.M = config.M || 16;
    this.M0 = config.M0 || this.M * 2;
    this.efConstruction = config.efConstruction || 200;
    this.efSearch = config.efSearch || 50;
    this.mL = 1 / Math.log(this.M);

    // Storage
    this.nodes = new Map();    // id → { id, vector, metadata, level, neighbors: Map<layer, Set<id>> }
    this.entryPoint = null;
    this.maxLevel = -1;

    // Stats
    this._comparisons = 0;
  }

  get size() { return this.nodes.size; }

  /**
   * Compute similarity score between two vectors
   * @private
   */
  _score(a, b) {
    this._comparisons++;
    return vec.score(a, b, this.metric);
  }

  /**
   * Generate a random insertion level using exponential distribution
   * @private
   * @returns {number}
   */
  _randomLevel() {
    return Math.floor(-Math.log(Math.random()) * this.mL);
  }

  /**
   * Beam search at a single layer. Returns up to `ef` nearest candidates.
   * @private
   * @param {number[]|Float64Array} query
   * @param {string} entryId
   * @param {number} ef
   * @param {number} layer
   * @returns {Array<{id: string, score: number}>}
   */
  _searchLayer(query, entryId, ef, layer) {
    const visited = new Set([entryId]);
    const epNode = this.nodes.get(entryId);
    if (!epNode) return [];

    const epScore = this._score(query, epNode.vector);
    const candidates = [{ id: entryId, score: epScore }];
    const results = [{ id: entryId, score: epScore }];

    while (candidates.length > 0) {
      // Pick best unprocessed candidate
      candidates.sort((a, b) => b.score - a.score);
      const current = candidates.shift();

      // Early termination: if best candidate is worse than worst result, stop
      const worstResult = results.length >= ef
        ? results[results.length - 1].score
        : -Infinity;
      if (current.score < worstResult && results.length >= ef) break;

      // Explore neighbors of current node at this layer
      const node = this.nodes.get(current.id);
      const neighbors = node.neighbors.get(layer);
      if (!neighbors) continue;

      for (const nid of neighbors) {
        if (visited.has(nid)) continue;
        visited.add(nid);

        const nnode = this.nodes.get(nid);
        if (!nnode) continue;

        const nScore = this._score(query, nnode.vector);

        if (results.length < ef || nScore > worstResult) {
          candidates.push({ id: nid, score: nScore });
          results.push({ id: nid, score: nScore });
          results.sort((a, b) => b.score - a.score);
          if (results.length > ef) results.pop();
        }
      }
    }

    return results;
  }

  /**
   * Heuristic neighbor selection — prefer diverse directions (Algorithm 4 from the paper)
   * @private
   * @param {number[]|Float64Array} query
   * @param {Array<{id: string, score: number}>} candidates
   * @param {number} maxConnections
   * @returns {Array<{id: string, score: number}>}
   */
  _selectNeighbors(query, candidates, maxConnections) {
    candidates.sort((a, b) => b.score - a.score);
    const selected = [];

    for (const c of candidates) {
      if (selected.length >= maxConnections) break;

      // Check if this candidate is dominated by an already-selected neighbor
      let dominated = false;
      for (const s of selected) {
        const sNode = this.nodes.get(s.id);
        const cNode = this.nodes.get(c.id);
        if (!sNode || !cNode) continue;
        const interScore = this._score(cNode.vector, sNode.vector);
        if (interScore > c.score) {
          dominated = true;
          break;
        }
      }

      // Always fill at least half the slots; after that, apply heuristic
      if (!dominated || selected.length < maxConnections / 2) {
        selected.push(c);
      }
    }

    return selected;
  }

  /**
   * Insert a vector into the index
   * @param {string} id — unique document identifier
   * @param {number[]|Float64Array} vector
   * @param {Object} [metadata={}]
   */
  insert(id, vector, metadata = {}) {
    if (vector.length !== this.dimension) {
      throw new Error(`Dimension mismatch: expected ${this.dimension}, got ${vector.length}`);
    }

    const level = this._randomLevel();
    const node = {
      id,
      vector: vec.toFloat64(vector),
      metadata,
      level,
      neighbors: new Map(),
    };
    for (let l = 0; l <= level; l++) node.neighbors.set(l, new Set());
    this.nodes.set(id, node);

    // First node — set as entry point
    if (this.entryPoint === null) {
      this.entryPoint = id;
      this.maxLevel = level;
      return;
    }

    let ep = this.entryPoint;

    // Phase 1: Greedy descent from top layer down to (level + 1)
    for (let l = this.maxLevel; l > level; l--) {
      const res = this._searchLayer(vector, ep, 1, l);
      if (res.length > 0) ep = res[0].id;
    }

    // Phase 2: Insert at layers min(level, maxLevel) → 0
    for (let l = Math.min(level, this.maxLevel); l >= 0; l--) {
      const maxM = l === 0 ? this.M0 : this.M;
      const candidates = this._searchLayer(vector, ep, this.efConstruction, l);
      const neighbors = this._selectNeighbors(vector, candidates, maxM);

      for (const n of neighbors) {
        // Add bidirectional edges
        node.neighbors.get(l).add(n.id);

        const nnode = this.nodes.get(n.id);
        if (!nnode.neighbors.has(l)) nnode.neighbors.set(l, new Set());
        nnode.neighbors.get(l).add(id);

        // Prune if neighbor exceeds max connections
        if (nnode.neighbors.get(l).size > maxM) {
          const nCands = [...nnode.neighbors.get(l)].map(nid => ({
            id: nid,
            score: this._score(nnode.vector, this.nodes.get(nid).vector),
          }));
          const kept = this._selectNeighbors(nnode.vector, nCands, maxM);
          nnode.neighbors.set(l, new Set(kept.map(k => k.id)));
        }
      }

      if (candidates.length > 0) ep = candidates[0].id;
    }

    // Update entry point if new node has higher level
    if (level > this.maxLevel) {
      this.entryPoint = id;
      this.maxLevel = level;
    }
  }

  /**
   * Search for top-K nearest neighbors
   * @param {number[]|Float64Array} query
   * @param {number} [topK=10]
   * @param {number} [ef] — override ef_search for this query
   * @returns {Array<{id: string, vector: Float64Array, metadata: Object, score: number}>}
   */
  search(query, topK = 10, ef = null) {
    ef = ef || this.efSearch;
    if (!this.entryPoint || this.nodes.size === 0) return [];

    this._comparisons = 0;
    let ep = this.entryPoint;

    // Greedy descent from top layer to layer 1
    for (let l = this.maxLevel; l > 0; l--) {
      const res = this._searchLayer(query, ep, 1, l);
      if (res.length > 0) ep = res[0].id;
    }

    // Beam search at layer 0
    const results = this._searchLayer(query, ep, Math.max(ef, topK), 0);

    return results.slice(0, topK).map(r => {
      const node = this.nodes.get(r.id);
      return {
        id: node.id,
        vector: node.vector,
        metadata: node.metadata,
        score: r.score,
      };
    });
  }

  /**
   * Delete a node from the index
   * @param {string} id
   * @returns {boolean}
   */
  delete(id) {
    const node = this.nodes.get(id);
    if (!node) return false;

    // Remove all edges pointing to this node
    for (let l = 0; l <= node.level; l++) {
      const neighbors = node.neighbors.get(l);
      if (!neighbors) continue;
      for (const nid of neighbors) {
        const nnode = this.nodes.get(nid);
        if (nnode && nnode.neighbors.has(l)) {
          nnode.neighbors.get(l).delete(id);
        }
      }
    }

    this.nodes.delete(id);

    // Update entry point if deleted
    if (this.entryPoint === id) {
      if (this.nodes.size === 0) {
        this.entryPoint = null;
        this.maxLevel = -1;
      } else {
        // Pick new entry point with highest level
        let bestId = null, bestLevel = -1;
        for (const [nid, nnode] of this.nodes) {
          if (nnode.level > bestLevel) { bestLevel = nnode.level; bestId = nid; }
        }
        this.entryPoint = bestId;
        this.maxLevel = bestLevel;
      }
    }

    return true;
  }

  /**
   * Get graph statistics
   * @returns {Object}
   */
  getStats() {
    let totalEdges = 0, maxEdges = 0;
    const levelCounts = {};
    for (const [, node] of this.nodes) {
      for (let l = 0; l <= node.level; l++) {
        levelCounts[l] = (levelCounts[l] || 0) + 1;
        const edges = node.neighbors.get(l)?.size || 0;
        totalEdges += edges;
        maxEdges = Math.max(maxEdges, edges);
      }
    }
    return {
      nodes: this.nodes.size,
      totalEdges: Math.round(totalEdges / 2),
      maxLevel: this.maxLevel,
      maxEdgesPerNode: maxEdges,
      levelDistribution: levelCounts,
      lastQueryComparisons: this._comparisons,
      config: { M: this.M, M0: this.M0, efConstruction: this.efConstruction, efSearch: this.efSearch },
    };
  }

  /**
   * Serialize index to a plain object (for persistence)
   * @returns {Object}
   */
  serialize() {
    const nodesArr = [];
    for (const [id, node] of this.nodes) {
      const neighbors = {};
      for (const [layer, set] of node.neighbors) {
        neighbors[layer] = [...set];
      }
      nodesArr.push({
        id, vector: [...node.vector], metadata: node.metadata,
        level: node.level, neighbors,
      });
    }
    return {
      dimension: this.dimension, metric: this.metric,
      M: this.M, M0: this.M0, efConstruction: this.efConstruction, efSearch: this.efSearch,
      entryPoint: this.entryPoint, maxLevel: this.maxLevel,
      nodes: nodesArr,
    };
  }

  /**
   * Deserialize index from a plain object
   * @param {Object} data
   * @returns {HNSWIndex}
   */
  static deserialize(data) {
    const idx = new HNSWIndex(data.dimension, data.metric, {
      M: data.M, M0: data.M0, efConstruction: data.efConstruction, efSearch: data.efSearch,
    });
    idx.entryPoint = data.entryPoint;
    idx.maxLevel = data.maxLevel;
    for (const n of data.nodes) {
      const neighbors = new Map();
      for (const [layer, ids] of Object.entries(n.neighbors)) {
        neighbors.set(parseInt(layer), new Set(ids));
      }
      idx.nodes.set(n.id, {
        id: n.id, vector: vec.toFloat64(n.vector),
        metadata: n.metadata, level: n.level, neighbors,
      });
    }
    return idx;
  }
}

module.exports = HNSWIndex;
