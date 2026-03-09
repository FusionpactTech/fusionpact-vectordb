/**
 * @fileoverview FusionPact LangChain.js Integration
 * 
 * Drop-in LangChain-compatible vector store and retriever that uses
 * FusionPact's hybrid retrieval engine under the hood.
 * 
 * Usage:
 *   const { FusionPactVectorStore, FusionPactRetriever } = require('fusionpact/integrations/langchain');
 * 
 * @module integrations/langchain
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 * @see https://github.com/FusionpactTech/fusionpact-vectordb
 */

'use strict';

const { FusionEngine } = require('../core/FusionEngine');
const { HybridRetriever } = require('../retrieval/HybridRetriever');
const { TreeIndex } = require('../index/TreeIndex');
const { RAGPipeline } = require('../rag/RAGPipeline');

/**
 * LangChain-compatible vector store backed by FusionPact.
 * 
 * Implements the standard LangChain VectorStore interface so it can be
 * used as a drop-in replacement for Chroma, FAISS, Pinecone, etc.
 * 
 * @example
 * const { FusionPactVectorStore } = require('fusionpact/integrations/langchain');
 * const store = new FusionPactVectorStore({ embedder: myEmbedder, collection: 'docs' });
 * await store.addDocuments([{ pageContent: 'text', metadata: {} }]);
 * const results = await store.similaritySearch('query', 5);
 */
class FusionPactVectorStore {
  constructor(config = {}) {
    this.engine = config.engine || new FusionEngine();
    this.embedder = config.embedder;
    this.collectionName = config.collection || 'langchain_default';
    this.treeIndex = config.treeIndex || new TreeIndex(config.treeConfig || {});

    this._retriever = null;

    // Auto-create collection
    if (!this.engine.getCollection(this.collectionName)) {
      this.engine.createCollection(this.collectionName, {
        dimensions: this.embedder?.dimensions || 768,
        distanceMetric: 'cosine'
      });
    }
  }

  /**
   * Add documents to the vector store.
   * Compatible with LangChain's Document interface: { pageContent: string, metadata: object }
   * 
   * @param {Array<{pageContent: string, metadata?: object}>} documents
   * @returns {Promise<string[]>} Document IDs
   */
  async addDocuments(documents) {
    const ids = [];
    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      const id = doc.metadata?.id || `doc_${Date.now()}_${i}`;
      const vector = await this.embedder.embed(doc.pageContent);

      this.engine.insert(this.collectionName, [{
        id,
        vector,
        metadata: {
          _content: doc.pageContent,
          ...doc.metadata
        }
      }]);
      ids.push(id);
    }
    return ids;
  }

  /**
   * Similarity search — standard LangChain interface.
   * 
   * @param {string} query - Search query
   * @param {number} [k=4] - Number of results
   * @param {object} [filter] - Metadata filter
   * @returns {Promise<Array<{pageContent: string, metadata: object}>>}
   */
  async similaritySearch(query, k = 4, filter = null) {
    const queryVector = await this.embedder.embed(query);
    const results = this.engine.search(this.collectionName, queryVector, {
      topK: k,
      filter
    });

    return results.map(r => ({
      pageContent: r.metadata._content || '',
      metadata: { ...r.metadata, score: r.score, _content: undefined }
    }));
  }

  /**
   * Similarity search with scores.
   * 
   * @param {string} query
   * @param {number} [k=4]
   * @returns {Promise<Array<[{pageContent: string, metadata: object}, number]>>}
   */
  async similaritySearchWithScore(query, k = 4) {
    const queryVector = await this.embedder.embed(query);
    const results = this.engine.search(this.collectionName, queryVector, { topK: k });

    return results.map(r => [
      { pageContent: r.metadata._content || '', metadata: r.metadata },
      r.score
    ]);
  }

  /**
   * Get a LangChain-compatible retriever from this store.
   * 
   * @param {object} [config={}]
   * @param {number} [config.k=4] - Number of documents to retrieve
   * @param {string} [config.strategy='hybrid'] - Retrieval strategy
   * @returns {FusionPactRetriever}
   */
  asRetriever(config = {}) {
    return new FusionPactRetriever({
      vectorStore: this,
      k: config.k || 4,
      strategy: config.strategy || 'vector'
    });
  }

  /**
   * Create from documents (LangChain factory pattern).
   * 
   * @param {Array<{pageContent: string, metadata?: object}>} documents
   * @param {object} embedder - Embedding provider
   * @param {object} [config={}]
   * @returns {Promise<FusionPactVectorStore>}
   */
  static async fromDocuments(documents, embedder, config = {}) {
    const store = new FusionPactVectorStore({ embedder, ...config });
    await store.addDocuments(documents);
    return store;
  }

  /**
   * Create from texts (LangChain factory pattern).
   * 
   * @param {string[]} texts
   * @param {object[]} metadatas
   * @param {object} embedder
   * @param {object} [config={}]
   * @returns {Promise<FusionPactVectorStore>}
   */
  static async fromTexts(texts, metadatas, embedder, config = {}) {
    const documents = texts.map((text, i) => ({
      pageContent: text,
      metadata: metadatas[i] || {}
    }));
    return FusionPactVectorStore.fromDocuments(documents, embedder, config);
  }
}

/**
 * LangChain-compatible retriever backed by FusionPact's hybrid engine.
 * 
 * @example
 * const retriever = store.asRetriever({ k: 5, strategy: 'hybrid' });
 * const docs = await retriever.getRelevantDocuments('safety protocols');
 */
class FusionPactRetriever {
  constructor(config) {
    this.vectorStore = config.vectorStore;
    this.k = config.k || 4;
    this.strategy = config.strategy || 'vector';
  }

  /**
   * Retrieve relevant documents — standard LangChain Retriever interface.
   * 
   * @param {string} query
   * @returns {Promise<Array<{pageContent: string, metadata: object}>>}
   */
  async getRelevantDocuments(query) {
    return this.vectorStore.similaritySearch(query, this.k);
  }

  /**
   * Alias for getRelevantDocuments (LangChain compatibility).
   */
  async invoke(query) {
    return this.getRelevantDocuments(query);
  }
}

module.exports = { FusionPactVectorStore, FusionPactRetriever };
