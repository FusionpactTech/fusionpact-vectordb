/**
 * FusionPact LangChain.js Integration
 * Drop-in LangChain-compatible vector store and retriever.
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 */
'use strict';
const { FusionEngine } = require('../core/FusionEngine');
const { TreeIndex } = require('../index/TreeIndex');

class FusionPactVectorStore {
  constructor(config = {}) {
    this.engine = config.engine || new FusionEngine();
    this.embedder = config.embedder;
    this.collectionName = config.collection || 'langchain_default';
    if (!this.engine.getCollection(this.collectionName)) {
      this.engine.createCollection(this.collectionName, { dimensions: this.embedder?.dimensions || 768 });
    }
  }

  async addDocuments(documents) {
    const ids = [];
    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      const id = doc.metadata?.id || `doc_${Date.now()}_${i}`;
      const vector = await this.embedder.embed(doc.pageContent);
      this.engine.insert(this.collectionName, [{ id, vector, metadata: { _content: doc.pageContent, ...doc.metadata } }]);
      ids.push(id);
    }
    return ids;
  }

  async similaritySearch(query, k = 4, filter = null) {
    const qv = await this.embedder.embed(query);
    return this.engine.search(this.collectionName, qv, { topK: k, filter }).map(r => ({
      pageContent: r.metadata._content || '', metadata: { ...r.metadata, score: r.score }
    }));
  }

  async similaritySearchWithScore(query, k = 4) {
    const qv = await this.embedder.embed(query);
    return this.engine.search(this.collectionName, qv, { topK: k }).map(r => [
      { pageContent: r.metadata._content || '', metadata: r.metadata }, r.score
    ]);
  }

  asRetriever(config = {}) { return new FusionPactRetriever({ vectorStore: this, k: config.k || 4 }); }

  static async fromDocuments(documents, embedder, config = {}) {
    const store = new FusionPactVectorStore({ embedder, ...config });
    await store.addDocuments(documents);
    return store;
  }

  static async fromTexts(texts, metadatas, embedder, config = {}) {
    return FusionPactVectorStore.fromDocuments(
      texts.map((t, i) => ({ pageContent: t, metadata: metadatas[i] || {} })), embedder, config
    );
  }
}

class FusionPactRetriever {
  constructor(config) { this.vectorStore = config.vectorStore; this.k = config.k || 4; }
  async getRelevantDocuments(query) { return this.vectorStore.similaritySearch(query, this.k); }
  async invoke(query) { return this.getRelevantDocuments(query); }
}

module.exports = { FusionPactVectorStore, FusionPactRetriever };
