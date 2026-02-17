/**
 * FusionPact — Embedding Providers
 *
 * Supports:
 *   - 'ollama'  — Local embeddings via Ollama (free, private, recommended)
 *   - 'openai'  — OpenAI text-embedding-3-small/large
 *   - 'mock'    — Deterministic hash-based embeddings (for testing/demos)
 */

'use strict';

const vec = require('../core/vectors');

// ─── Base Class ───────────────────────────────────────────────

class Embedder {
  constructor(dimension) {
    /** @type {number} */
    this.dimension = dimension;
    /** @type {string} */
    this.provider = 'base';
  }

  /**
   * Embed one or more texts
   * @param {string|string[]} texts
   * @returns {Promise<number[][]>}
   */
  async embed(texts) {
    throw new Error('embed() not implemented');
  }

  /**
   * Embed a single text
   * @param {string} text
   * @returns {Promise<number[]>}
   */
  async embedOne(text) {
    const results = await this.embed([text]);
    return results[0];
  }
}

// ─── Mock Embedder ────────────────────────────────────────────

class MockEmbedder extends Embedder {
  /**
   * Deterministic hash-based embeddings for testing.
   * Same text always produces the same vector — useful for demos.
   * @param {number} [dimension=64]
   */
  constructor(dimension = 64) {
    super(dimension);
    this.provider = 'mock';
  }

  async embed(texts) {
    if (typeof texts === 'string') texts = [texts];
    return texts.map(text => {
      const v = new Array(this.dimension).fill(0);
      for (let i = 0; i < text.length; i++) {
        const idx = i % this.dimension;
        v[idx] += text.charCodeAt(i) * (i + 1) * 0.001;
        // Mix in adjacent dimensions for richer representation
        v[(idx + 7) % this.dimension] += text.charCodeAt(i) * 0.0003;
        v[(idx + 13) % this.dimension] -= text.charCodeAt(i) * 0.0001;
      }
      return vec.normalize(v);
    });
  }
}

// ─── Ollama Embedder ──────────────────────────────────────────

class OllamaEmbedder extends Embedder {
  /**
   * Local embeddings via Ollama. Recommended models:
   *   - nomic-embed-text (768D, fast, good quality)
   *   - mxbai-embed-large (1024D, higher quality)
   *   - all-minilm (384D, very fast)
   *
   * @param {Object} [options]
   * @param {string} [options.model='nomic-embed-text']
   * @param {string} [options.baseUrl='http://localhost:11434']
   * @param {number} [options.dimension=768]
   */
  constructor(options = {}) {
    const dim = options.dimension || 768;
    super(dim);
    this.provider = 'ollama';
    this.model = options.model || 'nomic-embed-text';
    this.baseUrl = (options.baseUrl || 'http://localhost:11434').replace(/\/$/, '');
  }

  async embed(texts) {
    if (typeof texts === 'string') texts = [texts];
    const results = [];

    for (const text of texts) {
      const res = await fetch(`${this.baseUrl}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: this.model, prompt: text }),
      });

      if (!res.ok) {
        const body = await res.text().catch(() => '');
        throw new Error(`Ollama error (${res.status}): ${body}. Is Ollama running? Try: ollama pull ${this.model}`);
      }

      const data = await res.json();
      if (!data.embedding) {
        throw new Error(`Ollama returned no embedding. Model '${this.model}' may not support embeddings.`);
      }

      results.push(data.embedding);
      // Update dimension from first response
      if (results.length === 1 && data.embedding.length !== this.dimension) {
        this.dimension = data.embedding.length;
      }
    }

    return results;
  }
}

// ─── OpenAI Embedder ──────────────────────────────────────────

class OpenAIEmbedder extends Embedder {
  /**
   * OpenAI embeddings via API.
   *
   * @param {Object} options
   * @param {string} options.apiKey — OpenAI API key
   * @param {string} [options.model='text-embedding-3-small']
   * @param {number} [options.dimension=1536]
   * @param {string} [options.baseUrl='https://api.openai.com/v1']
   */
  constructor(options = {}) {
    const dim = options.dimension || 1536;
    super(dim);
    this.provider = 'openai';
    this.apiKey = options.apiKey;
    this.model = options.model || 'text-embedding-3-small';
    this.baseUrl = (options.baseUrl || 'https://api.openai.com/v1').replace(/\/$/, '');

    if (!this.apiKey) {
      throw new Error('OpenAI API key required. Set OPENAI_API_KEY or pass apiKey option.');
    }
  }

  async embed(texts) {
    if (typeof texts === 'string') texts = [texts];

    const allVectors = [];

    // Batch in chunks of 100 (OpenAI limit)
    for (let i = 0; i < texts.length; i += 100) {
      const batch = texts.slice(i, i + 100);
      const res = await fetch(`${this.baseUrl}/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({ input: batch, model: this.model }),
      });

      if (!res.ok) {
        const body = await res.text().catch(() => '');
        throw new Error(`OpenAI API error (${res.status}): ${body}`);
      }

      const data = await res.json();
      allVectors.push(...data.data.sort((a, b) => a.index - b.index).map(d => d.embedding));
    }

    return allVectors;
  }
}

// ─── Factory ──────────────────────────────────────────────────

/**
 * Create an embedder from configuration or environment variables.
 *
 * @param {Object|string} [config] — provider name or config object
 * @returns {Embedder}
 *
 * @example
 *   createEmbedder('mock')
 *   createEmbedder('ollama')
 *   createEmbedder({ provider: 'openai', apiKey: 'sk-...' })
 */
function createEmbedder(config) {
  if (typeof config === 'string') config = { provider: config };
  config = config || {};

  const provider = config.provider
    || process.env.FUSIONPACT_EMBEDDING_PROVIDER
    || process.env.EMBEDDING_PROVIDER
    || 'mock';

  switch (provider) {
    case 'ollama':
      return new OllamaEmbedder({
        model: config.model || process.env.OLLAMA_MODEL || 'nomic-embed-text',
        baseUrl: config.baseUrl || process.env.OLLAMA_BASE_URL || 'http://localhost:11434',
        dimension: config.dimension,
      });

    case 'openai':
      return new OpenAIEmbedder({
        apiKey: config.apiKey || process.env.OPENAI_API_KEY,
        model: config.model || process.env.OPENAI_EMBEDDING_MODEL || 'text-embedding-3-small',
        dimension: config.dimension || 1536,
        baseUrl: config.baseUrl || process.env.OPENAI_BASE_URL,
      });

    case 'mock':
    default:
      return new MockEmbedder(config.dimension || 64);
  }
}

module.exports = { Embedder, MockEmbedder, OllamaEmbedder, OpenAIEmbedder, createEmbedder };
