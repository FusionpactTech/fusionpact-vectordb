/**
 * @fileoverview HybridRetriever — Combines vector search, reasoning-based tree retrieval, and keyword search
 * 
 * The core differentiator of FusionPact: a single API that intelligently routes
 * queries through multiple retrieval strategies and fuses results using
 * Reciprocal Rank Fusion (RRF). This solves the fundamental limitation of
 * pure vector search (similarity ≠ relevance) without the latency penalty
 * of pure reasoning-based retrieval.
 * 
 * Retrieval Strategies:
 * - **vector**: HNSW-based approximate nearest neighbor search (fast, broad)
 * - **tree**: LLM reasoning-based tree traversal (precise, structured documents)
 * - **keyword**: BM25-style term frequency matching (exact match)
 * - **hybrid**: Automatic fusion of all applicable strategies
 * 
 * @module retrieval/HybridRetriever
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 * @see https://github.com/FusionpactTech/fusionpact-vectordb
 */

'use strict';

const { EventEmitter } = require('events');

/**
 * @typedef {Object} HybridResult
 * @property {string} id - Result identifier
 * @property {number} score - Fused relevance score (0-1)
 * @property {string} content - Retrieved content
 * @property {Object} metadata - Result metadata
 * @property {Object} sources - Which strategies contributed this result
 * @property {string} [citation] - Citation path (from tree retrieval)
 * @property {Object} [reasoning] - Reasoning trace (from tree retrieval)
 */

/**
 * @typedef {Object} RetrievalStrategy
 * @property {string} name - Strategy identifier
 * @property {number} weight - Weight in fusion (0-1)
 * @property {boolean} enabled - Whether this strategy is active
 */

class HybridRetriever extends EventEmitter {
  /**
   * Create a HybridRetriever.
   * 
   * @param {Object} config
   * @param {import('../core/FusionEngine')} config.engine - FusionEngine instance
   * @param {import('../index/TreeIndex')} [config.treeIndex] - TreeIndex for reasoning retrieval
   * @param {import('../embedders/BaseEmbedder')} [config.embedder] - Embedding provider
   * @param {Object} [config.weights] - Strategy weights
   * @param {number} [config.weights.vector=0.4] - Vector search weight
   * @param {number} [config.weights.tree=0.4] - Tree reasoning weight
   * @param {number} [config.weights.keyword=0.2] - Keyword search weight
   * @param {number} [config.rrfK=60] - RRF constant (higher = more even fusion)
   * 
   * @example
   * const retriever = new HybridRetriever({
   *   engine,
   *   treeIndex,
   *   embedder: ollamaEmbedder,
   *   weights: { vector: 0.4, tree: 0.4, keyword: 0.2 }
   * });
   */
  constructor(config) {
    super();
    this.engine = config.engine;
    this.treeIndex = config.treeIndex || null;
    this.embedder = config.embedder || null;

    this.weights = {
      vector: config.weights?.vector ?? 0.4,
      tree: config.weights?.tree ?? 0.4,
      keyword: config.weights?.keyword ?? 0.2
    };

    this.rrfK = config.rrfK || 60;

    // Adaptive learning (tracks which strategies work best per query pattern)
    this._strategyPerformance = new Map();
  }

  /**
   * Retrieve relevant content using hybrid strategy fusion.
   * 
   * Automatically determines which retrieval strategies to use based on
   * available indexes and configuration, then fuses results using
   * Reciprocal Rank Fusion.
   * 
   * @param {string} query - Natural language query
   * @param {Object} [options={}]
   * @param {string} [options.collection] - Vector collection to search
   * @param {string} [options.docId] - Specific document for tree search
   * @param {string[]} [options.docIds] - Multiple documents for tree search
   * @param {number} [options.topK=10] - Number of results
   * @param {string} [options.strategy='hybrid'] - Force strategy: 'vector', 'tree', 'keyword', or 'hybrid'
   * @param {Object} [options.filter] - Metadata filter (vector search)
   * @param {string} [options.tenantId] - Tenant filter
   * @param {boolean} [options.includeReasoning=false] - Include tree reasoning traces
   * @returns {Promise<HybridResult[]>} Fused and ranked results
   * 
   * @example
   * // Hybrid search across vectors and document trees
   * const results = await retriever.retrieve(
   *   'What safety protocols apply to chemical handling?',
   *   {
   *     collection: 'safety-docs',
   *     docId: 'osha-manual-2024',
   *     topK: 5,
   *     strategy: 'hybrid'
   *   }
   * );
   */
  async retrieve(query, options = {}) {
    const {
      collection,
      docId,
      docIds,
      topK = 10,
      strategy = 'hybrid',
      filter = null,
      tenantId = null,
      includeReasoning = false
    } = options;

    const start = performance.now();
    const allResults = new Map(); // id -> { sources, scores }
    const strategyResults = {};

    // ─── Vector Search ───
    if ((strategy === 'hybrid' || strategy === 'vector') && collection && this.embedder) {
      try {
        const queryVector = await this.embedder.embed(query);
        const vectorResults = this.engine.search(collection, queryVector, {
          topK: topK * 2,  // Over-fetch for better fusion
          filter,
          tenantId
        });

        strategyResults.vector = vectorResults;
        this._mergeResults(allResults, vectorResults.map((r, i) => ({
          id: r.id,
          rank: i + 1,
          content: r.metadata?._content || '',
          metadata: r.metadata,
          score: r.score,
          strategy: 'vector'
        })));
      } catch (err) {
        this.emit('strategy:error', { strategy: 'vector', error: err.message });
      }
    }

    // ─── Tree Reasoning Search ───
    if ((strategy === 'hybrid' || strategy === 'tree') && this.treeIndex) {
      try {
        const targetDocs = docIds || (docId ? [docId] : []);
        let treeResults;

        if (targetDocs.length > 0) {
          treeResults = await this.treeIndex.searchAll(query, {
            maxResults: topK * 2,
            docIds: targetDocs
          });
        } else {
          treeResults = await this.treeIndex.searchAll(query, {
            maxResults: topK * 2
          });
        }

        strategyResults.tree = treeResults;
        this._mergeResults(allResults, treeResults.map((r, i) => ({
          id: r.node?.nodeId || `tree_${i}`,
          rank: i + 1,
          content: r.content,
          metadata: { ...r.node, docId: r.docId },
          score: r.relevanceScore,
          citation: r.citation,
          reasoning: includeReasoning ? r.reasoningPath : undefined,
          strategy: 'tree'
        })));
      } catch (err) {
        this.emit('strategy:error', { strategy: 'tree', error: err.message });
      }
    }

    // ─── Keyword Search ───
    if ((strategy === 'hybrid' || strategy === 'keyword') && collection) {
      try {
        const keywordResults = this._bm25Search(collection, query, topK * 2);
        strategyResults.keyword = keywordResults;
        this._mergeResults(allResults, keywordResults.map((r, i) => ({
          id: r.id,
          rank: i + 1,
          content: r.content || '',
          metadata: r.metadata,
          score: r.score,
          strategy: 'keyword'
        })));
      } catch (err) {
        this.emit('strategy:error', { strategy: 'keyword', error: err.message });
      }
    }

    // ─── Reciprocal Rank Fusion ───
    const fused = this._reciprocalRankFusion(allResults, topK);

    const elapsed = performance.now() - start;
    this.emit('retrieve:complete', {
      query,
      strategy,
      resultCount: fused.length,
      elapsedMs: elapsed.toFixed(2),
      strategiesUsed: Object.keys(strategyResults)
    });

    return fused;
  }

  /**
   * Build LLM-ready context from retrieval results.
   * 
   * @param {HybridResult[]} results - Results from retrieve()
   * @param {Object} [options={}]
   * @param {number} [options.maxTokens=4000] - Approximate max context tokens
   * @param {boolean} [options.includeCitations=true] - Include citation info
   * @returns {string} Formatted context ready for LLM prompt
   * 
   * @example
   * const results = await retriever.retrieve('safety protocols');
   * const context = retriever.buildContext(results, { maxTokens: 3000 });
   * // Use context in your LLM prompt
   */
  buildContext(results, options = {}) {
    const { maxTokens = 4000, includeCitations = true } = options;

    let context = '';
    let approxTokens = 0;

    for (const result of results) {
      const entry = includeCitations && result.citation
        ? `[Source: ${result.citation}]\n${result.content}\n\n`
        : `${result.content}\n\n`;

      const entryTokens = Math.ceil(entry.length / 4); // rough estimate
      if (approxTokens + entryTokens > maxTokens) break;

      context += entry;
      approxTokens += entryTokens;
    }

    return context.trim();
  }

  /**
   * Record feedback on retrieval quality for adaptive learning.
   * 
   * @param {string} query - The original query
   * @param {string} strategy - Which strategy produced the best result
   * @param {number} quality - Quality score (0-1)
   */
  recordFeedback(query, strategy, quality) {
    const pattern = this._queryPattern(query);
    if (!this._strategyPerformance.has(pattern)) {
      this._strategyPerformance.set(pattern, { vector: [], tree: [], keyword: [] });
    }
    const perf = this._strategyPerformance.get(pattern);
    if (perf[strategy]) {
      perf[strategy].push(quality);
      // Keep last 100 scores
      if (perf[strategy].length > 100) perf[strategy].shift();
    }
  }

  /**
   * Get the recommended strategy weights based on learned performance.
   * 
   * @param {string} query - Query to get recommendations for
   * @returns {Object} Recommended weights
   */
  getAdaptiveWeights(query) {
    const pattern = this._queryPattern(query);
    const perf = this._strategyPerformance.get(pattern);
    if (!perf) return this.weights;

    const avgScores = {};
    for (const [strategy, scores] of Object.entries(perf)) {
      avgScores[strategy] = scores.length > 0
        ? scores.reduce((a, b) => a + b, 0) / scores.length
        : this.weights[strategy] || 0;
    }

    // Normalize
    const total = Object.values(avgScores).reduce((a, b) => a + b, 0);
    if (total === 0) return this.weights;

    const adaptive = {};
    for (const [k, v] of Object.entries(avgScores)) {
      adaptive[k] = v / total;
    }
    return adaptive;
  }

  // ─── Internal Methods ──────────────────────────────────

  /** @private - Merge results from a strategy into the combined result map */
  _mergeResults(allResults, strategyResults) {
    for (const result of strategyResults) {
      if (!allResults.has(result.id)) {
        allResults.set(result.id, {
          id: result.id,
          content: result.content,
          metadata: result.metadata,
          citation: result.citation,
          reasoning: result.reasoning,
          sources: {},
          ranks: {}
        });
      }

      const entry = allResults.get(result.id);
      entry.sources[result.strategy] = result.score;
      entry.ranks[result.strategy] = result.rank;

      // Prefer content from tree search (more structured)
      if (result.strategy === 'tree' && result.content) {
        entry.content = result.content;
      }
      if (result.citation) entry.citation = result.citation;
      if (result.reasoning) entry.reasoning = result.reasoning;
    }
  }

  /** @private - Reciprocal Rank Fusion */
  _reciprocalRankFusion(allResults, topK) {
    const scored = [];

    for (const [id, entry] of allResults) {
      let fusedScore = 0;
      const activeWeights = {};

      for (const [strategy, rank] of Object.entries(entry.ranks)) {
        const weight = this.weights[strategy] || 0;
        activeWeights[strategy] = weight;
        fusedScore += weight * (1 / (this.rrfK + rank));
      }

      scored.push({
        id: entry.id,
        score: fusedScore,
        content: entry.content,
        metadata: entry.metadata,
        sources: entry.sources,
        citation: entry.citation,
        reasoning: entry.reasoning
      });
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  /** @private - Simple BM25-style keyword search over indexed vectors' metadata */
  _bm25Search(collection, query, topK) {
    const terms = query.toLowerCase()
      .split(/\s+/)
      .filter(t => t.length > 2 && !this._isStopWord(t));

    if (terms.length === 0) return [];

    const collInfo = this.engine.getCollection(collection);
    if (!collInfo) return [];

    // Search through vector metadata for keyword matches
    const results = [];
    const index = this.engine._collections.get(collection);

    for (const [id, node] of index._nodes) {
      const text = JSON.stringify(node.metadata).toLowerCase();
      let score = 0;
      for (const term of terms) {
        const count = (text.match(new RegExp(term, 'g')) || []).length;
        // BM25-ish scoring: TF saturation
        score += (count * 2.2) / (count + 1.2);
      }
      if (score > 0) {
        results.push({
          id,
          score: score / terms.length,
          content: node.metadata._content || '',
          metadata: node.metadata
        });
      }
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  /** @private */
  _queryPattern(query) {
    // Extract a generalized pattern from queries for adaptive learning
    const words = query.toLowerCase().split(/\s+/).sort();
    return words.slice(0, 5).join('_');
  }

  /** @private */
  _isStopWord(word) {
    const stopWords = new Set([
      'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
      'in', 'with', 'to', 'for', 'of', 'not', 'no', 'can', 'had', 'has',
      'have', 'this', 'that', 'was', 'are', 'were', 'been', 'be', 'do',
      'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
      'what', 'how', 'when', 'where', 'who', 'why'
    ]);
    return stopWords.has(word);
  }
}

module.exports = { HybridRetriever };
