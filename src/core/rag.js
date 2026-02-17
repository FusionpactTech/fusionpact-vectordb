/**
 * FusionPact — RAG Pipeline
 * One-Click: text → chunks → embeddings → HNSW index → searchable context
 */

'use strict';

const { chunkText, generateId } = require('../utils');
const { createEmbedder } = require('../embeddings');

class RAGPipeline {
  /**
   * @param {import('../core/engine').FusionEngine} engine
   * @param {Object} [options]
   * @param {string} [options.collection='rag-sandbox']
   * @param {string|Object} [options.embedder='mock'] — embedder config
   * @param {number} [options.chunkSize=500]
   * @param {number} [options.chunkOverlap=100]
   */
  constructor(engine, options = {}) {
    this.engine = engine;
    this.collectionName = options.collection || 'rag-sandbox';
    this.chunkSize = options.chunkSize || 500;
    this.chunkOverlap = options.chunkOverlap || 100;

    // Create embedder
    this.embedder = typeof options.embedder === 'object' && options.embedder.embed
      ? options.embedder  // Already an embedder instance
      : createEmbedder(options.embedder || 'mock');

    // Ensure collection exists
    this._ensureCollection();

    // Track ingested sources
    this.sources = new Map();
  }

  /** @private */
  _ensureCollection() {
    if (!this.engine.collections.has(this.collectionName)) {
      this.engine.createCollection(this.collectionName, {
        dimension: this.embedder.dimension,
        metric: 'cosine',
        indexType: 'hnsw',
        hnswConfig: { M: 16, efConstruction: 200, efSearch: 50 },
      });
    }
  }

  /**
   * Ingest text: chunk → embed → index
   *
   * @param {string} text — raw text to ingest
   * @param {Object} [options]
   * @param {string} [options.source='document'] — source identifier
   * @param {Object} [options.metadata={}] — additional metadata for all chunks
   * @returns {Promise<{chunksCreated: number, collection: string, source: string, chunkIds: string[]}>}
   */
  async ingest(text, options = {}) {
    const source = options.source || 'document';
    const extraMeta = options.metadata || {};

    // 1. Chunk the text
    const chunks = chunkText(text, {
      chunkSize: this.chunkSize,
      overlap: this.chunkOverlap,
    });

    if (chunks.length === 0) {
      return { chunksCreated: 0, collection: this.collectionName, source, chunkIds: [] };
    }

    // 2. Embed all chunks
    const texts = chunks.map(c => c.text);
    const vectors = await this.embedder.embed(texts);

    // Check dimension match (embedder may have auto-detected dimension)
    const col = this.engine.collections.get(this.collectionName);
    if (col && vectors[0] && vectors[0].length !== col.dimension) {
      // Recreate collection with correct dimension
      this.engine.dropCollection(this.collectionName);
      this.engine.createCollection(this.collectionName, {
        dimension: vectors[0].length,
        metric: 'cosine',
        indexType: 'hnsw',
        hnswConfig: { M: 16, efConstruction: 200, efSearch: 50 },
      });
    }

    // 3. Build documents with metadata
    const sourceSlug = source.replace(/[^a-z0-9]/gi, '_').toLowerCase();
    const docs = chunks.map((chunk, i) => ({
      id: `${sourceSlug}_${i}_${generateId('rag')}`,
      vector: vectors[i],
      metadata: {
        text: chunk.text,
        source,
        chunk_index: chunk.index,
        char_start: chunk.charStart,
        char_end: chunk.charEnd,
        char_count: chunk.charCount,
        word_count: chunk.wordCount,
        type: 'rag_chunk',
        ...extraMeta,
      },
    }));

    // 4. Upsert into collection
    const ids = this.engine.insert(this.collectionName, docs);

    // Track source
    this.sources.set(source, {
      chunks: ids.length,
      ingestedAt: Date.now(),
      charCount: text.length,
      wordCount: text.split(/\s+/).length,
    });

    return {
      chunksCreated: ids.length,
      collection: this.collectionName,
      source,
      chunkIds: ids,
      dimension: vectors[0]?.length,
      provider: this.embedder.provider,
    };
  }

  /**
   * Search for relevant chunks given a question
   *
   * @param {string} question
   * @param {Object} [options]
   * @param {number} [options.topK=5]
   * @param {Object} [options.filter] — additional metadata filters
   * @returns {Promise<{chunks: Object[], elapsed: number, method: string}>}
   */
  async search(question, options = {}) {
    const topK = options.topK || 5;
    const filter = options.filter || null;

    // Embed the question
    const queryVec = await this.embedder.embedOne(question);

    // Query the collection
    const result = this.engine.query(this.collectionName, queryVec, {
      topK,
      filter,
    });

    return {
      chunks: result.results.map((r, i) => ({
        rank: i + 1,
        text: r.metadata.text,
        source: r.metadata.source,
        score: r.score,
        chunkIndex: r.metadata.chunk_index,
        metadata: r.metadata,
      })),
      elapsed: result.elapsed,
      method: result.method,
      total: result.total,
    };
  }

  /**
   * Build an LLM-ready prompt with retrieved context
   *
   * @param {string} question
   * @param {Object} [options]
   * @param {number} [options.topK=5]
   * @param {string} [options.systemPrompt]
   * @param {Object} [options.filter]
   * @returns {Promise<{prompt: string, chunks: Object[], sources: string[]}>}
   */
  async buildContext(question, options = {}) {
    const searchResult = await this.search(question, options);

    const defaultSystem = 'Use the following context to answer the question. If the context does not contain relevant information, say so.';
    const systemPrompt = options.systemPrompt || defaultSystem;

    const contextBlock = searchResult.chunks.map((c, i) =>
      `[${i + 1}] (relevance: ${c.score.toFixed(3)}, source: ${c.source})\n${c.text}`
    ).join('\n\n');

    const prompt = `${systemPrompt}\n\nContext:\n${contextBlock}\n\nQuestion: ${question}\nAnswer:`;

    const sources = [...new Set(searchResult.chunks.map(c => c.source))];

    return {
      prompt,
      chunks: searchResult.chunks,
      sources,
      elapsed: searchResult.elapsed,
      method: searchResult.method,
    };
  }

  /**
   * Get stats about ingested sources
   * @returns {Object}
   */
  getStats() {
    const col = this.engine.getCollection(this.collectionName);
    return {
      collection: this.collectionName,
      totalChunks: col ? col.count : 0,
      sources: Object.fromEntries(this.sources),
      embeddingProvider: this.embedder.provider,
      embeddingDimension: this.embedder.dimension,
    };
  }
}

module.exports = { RAGPipeline };
