/**
 * @fileoverview RAGPipeline — One-Click Retrieval-Augmented Generation
 * 
 * End-to-end RAG pipeline: Text → Chunks → Embeddings → Index → Context
 * Supports both vector-based and hybrid (vector + tree) retrieval.
 * 
 * @module rag/RAGPipeline
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 * @see https://github.com/FusionpactTech/fusionpact-vectordb
 */

'use strict';

const { EventEmitter } = require('events');

/**
 * @typedef {Object} RAGConfig
 * @property {string} [collection='rag_default'] - Collection name
 * @property {import('../embedders/BaseEmbedder')} [embedder] - Embedding provider
 * @property {number} [chunkSize=512] - Characters per chunk
 * @property {number} [chunkOverlap=50] - Overlap between chunks
 * @property {string} [chunkStrategy='recursive'] - Chunking strategy
 * @property {boolean} [enableTreeIndex=false] - Also build tree index for hybrid retrieval
 */

class RAGPipeline extends EventEmitter {
  /**
   * @param {import('../core/FusionEngine')} engine
   * @param {RAGConfig} [config={}]
   * 
   * @example
   * const rag = new RAGPipeline(engine, {
   *   embedder: ollamaEmbedder,
   *   chunkSize: 512,
   *   enableTreeIndex: true
   * });
   */
  constructor(engine, config = {}) {
    super();
    this.engine = engine;
    this.embedder = config.embedder || null;
    this.collection = config.collection || 'rag_default';
    this.chunkSize = config.chunkSize || 512;
    this.chunkOverlap = config.chunkOverlap || 50;
    this.chunkStrategy = config.chunkStrategy || 'recursive';
    this.enableTreeIndex = config.enableTreeIndex || false;
    this.treeIndex = config.treeIndex || null;
    this.hybridRetriever = config.hybridRetriever || null;

    this._initialized = false;
  }

  /**
   * Initialize the pipeline (creates collection if needed).
   * @returns {RAGPipeline}
   */
  init() {
    if (!this._initialized) {
      if (!this.engine.getCollection(this.collection)) {
        const dim = this.embedder?.dimensions || 64;
        this.engine.createCollection(this.collection, {
          dimensions: dim,
          distanceMetric: 'cosine'
        });
      }
      this._initialized = true;
    }
    return this;
  }

  /**
   * Ingest text into the RAG pipeline.
   * 
   * Automatically chunks, embeds, and indexes text in one call.
   * 
   * @param {string} text - Text to ingest
   * @param {Object} [metadata={}] - Metadata (source, title, etc.)
   * @param {Object} [options={}]
   * @param {string} [options.tenantId] - Tenant for multi-tenant ingestion
   * @param {string} [options.format='text'] - Input format for tree indexing
   * @returns {Promise<{chunks: number, indexed: number, docId?: string}>}
   * 
   * @example
   * await rag.ingest(documentText, {
   *   source: 'annual-report-2024.pdf',
   *   title: 'Annual Report 2024',
   *   category: 'financial'
   * });
   */
  async ingest(text, metadata = {}, options = {}) {
    this.init();
    this.emit('ingest:start', { source: metadata.source });

    // Step 1: Chunk the text
    const chunks = this._chunk(text);

    // Step 2: Embed and index each chunk
    let indexed = 0;
    for (let i = 0; i < chunks.length; i++) {
      const chunkId = `${metadata.source || 'doc'}_chunk_${i}`;

      let vector;
      if (this.embedder) {
        vector = await this.embedder.embed(chunks[i]);
      } else {
        // Mock embedding for testing
        vector = this._mockEmbed(chunks[i]);
      }

      this.engine.insert(this.collection, [{
        id: chunkId,
        vector,
        metadata: {
          _content: chunks[i],
          _chunk_index: i,
          _total_chunks: chunks.length,
          ...metadata
        }
      }], { tenantId: options.tenantId });

      indexed++;
    }

    // Step 3: Optionally build tree index for hybrid retrieval
    let docId = null;
    if (this.enableTreeIndex && this.treeIndex) {
      docId = metadata.source || `doc_${Date.now()}`;
      await this.treeIndex.indexDocument(docId, text, {
        format: options.format || 'text',
        metadata
      });
    }

    this.emit('ingest:complete', { chunks: chunks.length, indexed, docId });
    return { chunks: chunks.length, indexed, docId };
  }

  /**
   * Ingest multiple documents.
   * 
   * @param {Array<{text: string, metadata?: Object}>} documents
   * @param {Object} [options={}]
   * @returns {Promise<{totalChunks: number, totalIndexed: number}>}
   */
  async ingestBatch(documents, options = {}) {
    let totalChunks = 0;
    let totalIndexed = 0;

    for (const doc of documents) {
      const result = await this.ingest(doc.text, doc.metadata || {}, options);
      totalChunks += result.chunks;
      totalIndexed += result.indexed;
    }

    return { totalChunks, totalIndexed };
  }

  /**
   * Search the RAG index and build LLM-ready context.
   * 
   * @param {string} query - Natural language query
   * @param {Object} [options={}]
   * @param {number} [options.topK=5] - Number of chunks to retrieve
   * @param {number} [options.maxTokens=4000] - Max context tokens
   * @param {string} [options.tenantId] - Tenant filter
   * @param {string} [options.strategy='auto'] - 'vector', 'tree', 'hybrid', or 'auto'
   * @returns {Promise<{prompt: string, sources: Object[], chunks: number}>}
   * 
   * @example
   * const context = await rag.buildContext('What safety protocols exist?');
   * // context.prompt is ready to paste into any LLM
   */
  async buildContext(query, options = {}) {
    const {
      topK = 5,
      maxTokens = 4000,
      tenantId = null,
      strategy = 'auto'
    } = options;

    this.init();

    let results;

    // Use hybrid retriever if available and strategy allows
    if (this.hybridRetriever && (strategy === 'hybrid' || strategy === 'auto')) {
      results = await this.hybridRetriever.retrieve(query, {
        collection: this.collection,
        topK,
        tenantId,
        strategy: 'hybrid'
      });
    } else {
      // Vector-only search
      let queryVector;
      if (this.embedder) {
        queryVector = await this.embedder.embed(query);
      } else {
        queryVector = this._mockEmbed(query);
      }

      results = this.engine.search(this.collection, queryVector, {
        topK,
        tenantId
      }).map(r => ({
        id: r.id,
        content: r.metadata._content || '',
        score: r.score,
        metadata: r.metadata
      }));
    }

    // Build context string
    let prompt = '';
    let approxTokens = 0;
    const sources = [];

    for (const result of results) {
      const content = result.content || '';
      const tokens = Math.ceil(content.length / 4);

      if (approxTokens + tokens > maxTokens) break;

      prompt += content + '\n\n';
      approxTokens += tokens;
      sources.push({
        id: result.id,
        score: result.score,
        source: result.metadata?.source,
        citation: result.citation
      });
    }

    return {
      prompt: prompt.trim(),
      sources,
      chunks: sources.length,
      query
    };
  }

  // ─── Chunking Strategies ──────────────────────────────────

  /** @private */
  _chunk(text) {
    switch (this.chunkStrategy) {
      case 'recursive':
        return this._recursiveChunk(text);
      case 'sentence':
        return this._sentenceChunk(text);
      case 'paragraph':
        return this._paragraphChunk(text);
      default:
        return this._recursiveChunk(text);
    }
  }

  /** @private */
  _recursiveChunk(text) {
    const separators = ['\n\n', '\n', '. ', ' '];
    return this._splitRecursive(text, separators);
  }

  /** @private */
  _splitRecursive(text, separators) {
    if (text.length <= this.chunkSize) return [text];

    const separator = separators[0];
    const parts = text.split(separator);
    const chunks = [];
    let current = '';

    for (const part of parts) {
      const candidate = current ? current + separator + part : part;

      if (candidate.length > this.chunkSize && current) {
        chunks.push(current);
        // Overlap: keep the tail of the current chunk
        const overlapText = current.slice(-this.chunkOverlap);
        current = overlapText + separator + part;
      } else {
        current = candidate;
      }
    }

    if (current) chunks.push(current);

    // If any chunk is still too large and we have more separators, recurse
    if (separators.length > 1) {
      const refined = [];
      for (const chunk of chunks) {
        if (chunk.length > this.chunkSize * 1.5) {
          refined.push(...this._splitRecursive(chunk, separators.slice(1)));
        } else {
          refined.push(chunk);
        }
      }
      return refined;
    }

    return chunks;
  }

  /** @private */
  _sentenceChunk(text) {
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    const chunks = [];
    let current = '';

    for (const sentence of sentences) {
      if ((current + sentence).length > this.chunkSize && current) {
        chunks.push(current.trim());
        current = current.slice(-this.chunkOverlap) + sentence;
      } else {
        current += sentence;
      }
    }
    if (current.trim()) chunks.push(current.trim());
    return chunks;
  }

  /** @private */
  _paragraphChunk(text) {
    const paragraphs = text.split(/\n\n+/);
    const chunks = [];
    let current = '';

    for (const para of paragraphs) {
      if ((current + '\n\n' + para).length > this.chunkSize && current) {
        chunks.push(current.trim());
        current = para;
      } else {
        current = current ? current + '\n\n' + para : para;
      }
    }
    if (current.trim()) chunks.push(current.trim());
    return chunks;
  }

  /** @private - Mock embedding for testing without an embedding provider */
  _mockEmbed(text) {
    const dim = 64;
    const vec = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      let hash = 0;
      const str = text.substring(i % text.length, Math.min(i + 10, text.length));
      for (let j = 0; j < str.length; j++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(j);
        hash |= 0;
      }
      vec[i] = (hash % 1000) / 1000;
    }
    // Normalize
    let norm = 0;
    for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < dim; i++) vec[i] /= norm;
    return Array.from(vec);
  }
}

module.exports = { RAGPipeline };
