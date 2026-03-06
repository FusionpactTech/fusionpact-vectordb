/**
 * @fileoverview Embedding Providers — Pluggable embedding backends
 * 
 * Supports multiple providers with a unified interface.
 * Ollama is the recommended default (free, local, private).
 * 
 * @module embedders
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 */

'use strict';

class BaseEmbedder {
  constructor(config = {}) {
    this.model = config.model || 'default';
    this.dimensions = config.dimensions || 768;
    this.batchSize = config.batchSize || 32;
    this._cache = new Map();
    this._stats = { calls: 0, cached: 0 };
  }

  async embed(text) {
    const key = `${this.model}:${text.length}:${this._hash(text)}`;
    if (this._cache.has(key)) { this._stats.cached++; return this._cache.get(key); }
    const vec = await this._embed(text);
    this._cache.set(key, vec);
    this._stats.calls++;
    return vec;
  }

  async embedBatch(texts) {
    const results = [];
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      results.push(...await Promise.all(batch.map(t => this.embed(t))));
    }
    return results;
  }

  async _embed(text) { throw new Error('Implement _embed()'); }
  get stats() { return { ...this._stats, cacheSize: this._cache.size }; }
  clearCache() { this._cache.clear(); }

  _hash(text) {
    let h = 0;
    for (let i = 0; i < Math.min(text.length, 500); i++) {
      h = ((h << 5) - h + text.charCodeAt(i)) | 0;
    }
    return h;
  }
}

class OllamaEmbedder extends BaseEmbedder {
  constructor(config = {}) {
    super({ model: config.model || 'nomic-embed-text', dimensions: config.dimensions || 768, ...config });
    this.baseUrl = config.baseUrl || 'http://localhost:11434';
  }
  async _embed(text) {
    const res = await fetch(`${this.baseUrl}/api/embeddings`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.model, prompt: text })
    });
    if (!res.ok) throw new Error(`Ollama embedding failed: ${res.status}. Is Ollama running? Try: ollama pull ${this.model}`);
    return (await res.json()).embedding;
  }
}

class OpenAIEmbedder extends BaseEmbedder {
  constructor(config = {}) {
    super({ model: config.model || 'text-embedding-3-small', dimensions: config.dimensions || 1536, ...config });
    this.apiKey = config.apiKey || process.env.OPENAI_API_KEY;
    this.baseUrl = config.baseUrl || 'https://api.openai.com/v1';
  }
  async _embed(text) {
    if (!this.apiKey) throw new Error('OpenAI API key required. Set OPENAI_API_KEY or pass apiKey.');
    const res = await fetch(`${this.baseUrl}/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${this.apiKey}` },
      body: JSON.stringify({ model: this.model, input: text })
    });
    if (!res.ok) throw new Error(`OpenAI embedding failed: ${res.status}`);
    return (await res.json()).data[0].embedding;
  }
}

class MockEmbedder extends BaseEmbedder {
  constructor(config = {}) {
    super({ model: 'mock', dimensions: config.dimensions || 64, ...config });
  }
  async _embed(text) {
    const vec = new Float32Array(this.dimensions);
    for (let i = 0; i < this.dimensions; i++) {
      let h = 5381 + i;
      const w = text.substring(Math.floor((i / this.dimensions) * text.length), Math.floor((i / this.dimensions) * text.length) + 20);
      for (let j = 0; j < w.length; j++) h = ((h << 5) + h + w.charCodeAt(j)) | 0;
      vec[i] = ((h % 2000) - 1000) / 1000;
    }
    let norm = 0;
    for (let i = 0; i < this.dimensions; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < this.dimensions; i++) vec[i] /= norm;
    return Array.from(vec);
  }
}

class LLMProvider {
  constructor(config = {}) {
    this.provider = config.provider || 'ollama';
    this.model = config.model || ({ ollama: 'llama3.2', openai: 'gpt-4o-mini', anthropic: 'claude-sonnet-4-20250514' }[this.provider] || 'llama3.2');
    this.baseUrl = config.baseUrl || ({ ollama: 'http://localhost:11434', openai: 'https://api.openai.com/v1', anthropic: 'https://api.anthropic.com' }[this.provider]);
    this.apiKey = config.apiKey || process.env[{ openai: 'OPENAI_API_KEY', anthropic: 'ANTHROPIC_API_KEY' }[this.provider]];
    this.name = this.provider;
  }

  async complete(prompt, options = {}) {
    const { maxTokens = 500, temperature = 0.1 } = options;
    if (this.provider === 'ollama') {
      const res = await fetch(`${this.baseUrl}/api/generate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: this.model, prompt, stream: false, options: { num_predict: maxTokens, temperature } })
      });
      if (!res.ok) throw new Error(`Ollama failed: ${res.status}`);
      return (await res.json()).response;
    }
    if (this.provider === 'openai') {
      const res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${this.apiKey}` },
        body: JSON.stringify({ model: this.model, messages: [{ role: 'user', content: prompt }], max_tokens: maxTokens, temperature })
      });
      if (!res.ok) throw new Error(`OpenAI failed: ${res.status}`);
      return (await res.json()).choices[0].message.content;
    }
    if (this.provider === 'anthropic') {
      const res = await fetch(`${this.baseUrl}/v1/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-api-key': this.apiKey, 'anthropic-version': '2023-06-01' },
        body: JSON.stringify({ model: this.model, messages: [{ role: 'user', content: prompt }], max_tokens: maxTokens, temperature })
      });
      if (!res.ok) throw new Error(`Anthropic failed: ${res.status}`);
      return (await res.json()).content[0].text;
    }
    throw new Error(`Unknown provider: ${this.provider}`);
  }
}

module.exports = { BaseEmbedder, OllamaEmbedder, OpenAIEmbedder, MockEmbedder, LLMProvider };
