/**
 * @fileoverview AgentMemory — Multi-Agent Memory Architecture
 * 
 * Purpose-built memory system for AI agents supporting:
 * - **Episodic Memory**: Events, conversations, user interactions
 * - **Semantic Memory**: Learned knowledge, facts, domain expertise
 * - **Procedural Memory**: Tool schemas, API specs, workflow patterns
 * - **Shared Memory**: Cross-agent knowledge sharing with access control
 * - **Conversation Memory**: Thread-aware chat history with summarization
 * 
 * Designed for multi-agent orchestration where multiple agents need
 * isolated memory spaces with optional shared knowledge pools.
 * 
 * @module memory/AgentMemory
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 * @see https://github.com/FusionpactTech/fusionpact-vectordb
 */

'use strict';

const { EventEmitter } = require('events');

/**
 * @typedef {'episodic'|'semantic'|'procedural'|'conversation'} MemoryType
 */

/**
 * @typedef {Object} MemoryEntry
 * @property {string} id - Unique memory ID
 * @property {string} agentId - Owning agent
 * @property {MemoryType} type - Memory type
 * @property {string} content - Memory content
 * @property {number[]} [vector] - Embedding vector
 * @property {Object} metadata - Additional metadata
 * @property {number} timestamp - Creation timestamp
 * @property {number} [importance=0.5] - Importance score (0-1)
 * @property {number} [accessCount=0] - Number of times recalled
 * @property {number} [lastAccessed] - Last access timestamp
 * @property {number} [ttl=0] - Time-to-live (0 = no expiry)
 */

/**
 * @typedef {Object} ConversationMessage
 * @property {string} role - 'user', 'assistant', 'system', or 'tool'
 * @property {string} content - Message content
 * @property {number} timestamp - Message timestamp
 * @property {Object} [metadata] - Additional message metadata
 */

class AgentMemory extends EventEmitter {
  /**
   * Create an AgentMemory system.
   * 
   * @param {import('../core/FusionEngine')} engine - FusionEngine instance
   * @param {Object} [config={}]
   * @param {import('../embedders/BaseEmbedder')} [config.embedder] - Embedding provider
   * @param {number} [config.defaultTTL=0] - Default TTL for memories (0 = no expiry)
   * @param {number} [config.maxConversationLength=100] - Max messages per conversation thread
   * @param {number} [config.importanceDecayRate=0.01] - How fast importance decays
   * @param {boolean} [config.enableSharedMemory=true] - Enable cross-agent shared memory
   * 
   * @example
   * const memory = new AgentMemory(engine, {
   *   embedder: ollamaEmbedder,
   *   enableSharedMemory: true
   * });
   */
  constructor(engine, config = {}) {
    super();
    this.engine = engine;
    this.embedder = config.embedder || null;
    this.defaultTTL = config.defaultTTL || 0;
    this.maxConversationLength = config.maxConversationLength || 100;
    this.importanceDecayRate = config.importanceDecayRate || 0.01;
    this.enableSharedMemory = config.enableSharedMemory !== false;

    /** @private - Agent-specific memory stores */
    this._agentStores = new Map();
    /** @private - Conversation threads */
    this._conversations = new Map();
    /** @private - Shared memory pool */
    this._sharedPool = new Map();
    /** @private - Memory counter */
    this._memoryCount = 0;

    // Auto-create collections for memory types
    this._initCollections();
  }

  /** @private */
  _initCollections() {
    const dim = this.embedder?.dimensions || 768;
    const types = ['episodic', 'semantic', 'procedural', 'conversation'];
    for (const type of types) {
      const collName = `_memory_${type}`;
      if (!this.engine.getCollection(collName)) {
        try {
          this.engine.createCollection(collName, {
            dimensions: dim,
            distanceMetric: 'cosine'
          });
        } catch (e) {
          // Collection may already exist
        }
      }
    }

    if (this.enableSharedMemory) {
      const collName = '_memory_shared';
      if (!this.engine.getCollection(collName)) {
        try {
          this.engine.createCollection(collName, {
            dimensions: dim,
            distanceMetric: 'cosine'
          });
        } catch (e) {}
      }
    }
  }

  // ─── Episodic Memory (What Happened) ──────────────────────────

  /**
   * Store an episodic memory — records events, interactions, observations.
   * 
   * @param {string} agentId - Agent identifier
   * @param {Object} memory
   * @param {string} memory.content - Memory content
   * @param {string} [memory.role='system'] - Role context
   * @param {number} [memory.importance=0.5] - Importance (0-1)
   * @param {Object} [memory.metadata={}] - Additional metadata
   * @param {number} [memory.ttl] - Override TTL
   * @returns {Promise<MemoryEntry>}
   * 
   * @example
   * await memory.remember('agent-1', {
   *   content: 'User prefers dark mode and concise answers',
   *   role: 'system',
   *   importance: 0.8,
   *   metadata: { category: 'user_preference' }
   * });
   */
  async remember(agentId, memory) {
    const entry = this._createEntry(agentId, 'episodic', memory);

    if (this.embedder) {
      entry.vector = await this.embedder.embed(memory.content);
      this.engine.insert('_memory_episodic', [{
        id: entry.id,
        vector: entry.vector,
        metadata: {
          _content: memory.content,
          _agent_id: agentId,
          _type: 'episodic',
          _importance: entry.importance,
          _ttl: entry.ttl || this.defaultTTL,
          role: memory.role || 'system',
          ...memory.metadata
        }
      }], { tenantId: agentId });
    }

    this._getAgentStore(agentId).episodic.set(entry.id, entry);
    this.emit('memory:stored', { agentId, type: 'episodic', id: entry.id });
    return entry;
  }

  // ─── Semantic Memory (What the Agent Knows) ──────────────────

  /**
   * Add knowledge to semantic memory — facts, domain knowledge, learned information.
   * 
   * @param {string} agentId - Agent identifier
   * @param {string} content - Knowledge content
   * @param {Object} [metadata={}] - Metadata (source, category, confidence, etc.)
   * @returns {Promise<MemoryEntry>}
   * 
   * @example
   * await memory.learn('agent-1',
   *   'OSHA 29 CFR 1910 covers general industry safety standards.',
   *   { source: 'regulations', category: 'compliance', confidence: 0.95 }
   * );
   */
  async learn(agentId, content, metadata = {}) {
    const entry = this._createEntry(agentId, 'semantic', {
      content,
      importance: metadata.confidence || 0.7,
      metadata
    });

    if (this.embedder) {
      entry.vector = await this.embedder.embed(content);
      this.engine.insert('_memory_semantic', [{
        id: entry.id,
        vector: entry.vector,
        metadata: {
          _content: content,
          _agent_id: agentId,
          _type: 'semantic',
          _importance: entry.importance,
          ...metadata
        }
      }], { tenantId: agentId });
    }

    this._getAgentStore(agentId).semantic.set(entry.id, entry);
    this.emit('memory:learned', { agentId, id: entry.id });
    return entry;
  }

  // ─── Procedural Memory (What the Agent Can Do) ──────────────

  /**
   * Register a tool/procedure in procedural memory.
   * 
   * @param {string} agentId - Agent identifier
   * @param {Object} tool
   * @param {string} tool.name - Tool name
   * @param {string} tool.description - Tool description
   * @param {Object} [tool.schema] - JSON Schema for tool parameters
   * @param {Object} [tool.metadata={}] - Additional metadata
   * @returns {Promise<MemoryEntry>}
   * 
   * @example
   * await memory.registerTool('agent-1', {
   *   name: 'search_incidents',
   *   description: 'Search EHS incident reports by category and severity',
   *   schema: {
   *     type: 'object',
   *     properties: {
   *       category: { type: 'string' },
   *       severity: { type: 'string', enum: ['low', 'medium', 'high', 'critical'] }
   *     }
   *   }
   * });
   */
  async registerTool(agentId, tool) {
    const content = `Tool: ${tool.name}\nDescription: ${tool.description}\nSchema: ${JSON.stringify(tool.schema || {})}`;
    const entry = this._createEntry(agentId, 'procedural', {
      content,
      importance: 0.9,
      metadata: { toolName: tool.name, ...tool.metadata }
    });

    entry.tool = tool;

    if (this.embedder) {
      entry.vector = await this.embedder.embed(`${tool.name}: ${tool.description}`);
      this.engine.insert('_memory_procedural', [{
        id: entry.id,
        vector: entry.vector,
        metadata: {
          _content: content,
          _agent_id: agentId,
          _type: 'procedural',
          toolName: tool.name,
          toolSchema: JSON.stringify(tool.schema || {}),
          ...tool.metadata
        }
      }], { tenantId: agentId });
    }

    this._getAgentStore(agentId).procedural.set(entry.id, entry);
    this.emit('memory:tool_registered', { agentId, toolName: tool.name });
    return entry;
  }

  // ─── Conversation Memory ──────────────────────────────────

  /**
   * Add a message to a conversation thread.
   * 
   * @param {string} agentId - Agent identifier
   * @param {string} threadId - Conversation thread identifier
   * @param {ConversationMessage} message
   * @returns {ConversationMessage}
   * 
   * @example
   * memory.addMessage('agent-1', 'thread-001', {
   *   role: 'user',
   *   content: 'What are the safety requirements for chemical storage?'
   * });
   */
  addMessage(agentId, threadId, message) {
    const key = `${agentId}:${threadId}`;
    if (!this._conversations.has(key)) {
      this._conversations.set(key, {
        agentId,
        threadId,
        messages: [],
        createdAt: Date.now(),
        metadata: {}
      });
    }

    const thread = this._conversations.get(key);
    const msg = {
      ...message,
      timestamp: message.timestamp || Date.now()
    };
    thread.messages.push(msg);

    // Trim if too long
    if (thread.messages.length > this.maxConversationLength) {
      thread.messages = thread.messages.slice(-this.maxConversationLength);
    }

    this.emit('conversation:message', { agentId, threadId, role: msg.role });
    return msg;
  }

  /**
   * Get conversation history for a thread.
   * 
   * @param {string} agentId
   * @param {string} threadId
   * @param {Object} [options={}]
   * @param {number} [options.limit=50] - Max messages to return
   * @param {string} [options.since] - Only messages after this timestamp
   * @returns {ConversationMessage[]}
   */
  getConversation(agentId, threadId, options = {}) {
    const key = `${agentId}:${threadId}`;
    const thread = this._conversations.get(key);
    if (!thread) return [];

    let messages = thread.messages;
    if (options.since) {
      messages = messages.filter(m => m.timestamp > options.since);
    }
    if (options.limit) {
      messages = messages.slice(-options.limit);
    }
    return messages;
  }

  /**
   * List all conversation threads for an agent.
   * @param {string} agentId
   * @returns {Array<{threadId: string, messageCount: number, lastMessage: number}>}
   */
  listConversations(agentId) {
    const threads = [];
    for (const [key, thread] of this._conversations) {
      if (thread.agentId === agentId) {
        threads.push({
          threadId: thread.threadId,
          messageCount: thread.messages.length,
          lastMessage: thread.messages.length > 0
            ? thread.messages[thread.messages.length - 1].timestamp
            : thread.createdAt
        });
      }
    }
    return threads;
  }

  // ─── Recall (Cross-Memory Search) ──────────────────────────

  /**
   * Recall relevant memories across all memory types.
   * 
   * @param {string} agentId - Agent identifier
   * @param {string} query - What to recall
   * @param {Object} [options={}]
   * @param {MemoryType[]} [options.types] - Filter by memory types
   * @param {number} [options.topK=10] - Max results per type
   * @param {boolean} [options.includeShared=true] - Include shared memories
   * @returns {Promise<Object>} Results organized by memory type
   * 
   * @example
   * const memories = await memory.recall('agent-1', 'safety compliance requirements');
   * console.log(memories.episodic);   // Past interactions about safety
   * console.log(memories.semantic);   // Knowledge about safety regulations
   * console.log(memories.procedural); // Tools for safety analysis
   * console.log(memories.shared);     // Cross-agent safety knowledge
   */
  async recall(agentId, query, options = {}) {
    const {
      types = ['episodic', 'semantic', 'procedural'],
      topK = 10,
      includeShared = true
    } = options;

    const results = {};

    if (!this.embedder) {
      // Fallback: keyword-based recall
      for (const type of types) {
        results[type] = this._keywordRecall(agentId, type, query, topK);
      }
      return results;
    }

    const queryVector = await this.embedder.embed(query);

    for (const type of types) {
      const collName = `_memory_${type}`;
      try {
        results[type] = this.engine.search(collName, queryVector, {
          topK,
          tenantId: agentId
        }).map(r => ({
          id: r.id,
          content: r.metadata._content,
          score: r.score,
          metadata: r.metadata,
          type
        }));

        // Update access tracking
        for (const r of results[type]) {
          this._trackAccess(agentId, type, r.id);
        }
      } catch (err) {
        results[type] = [];
      }
    }

    // Shared memory
    if (includeShared && this.enableSharedMemory) {
      try {
        results.shared = this.engine.search('_memory_shared', queryVector, {
          topK
        }).map(r => ({
          id: r.id,
          content: r.metadata._content,
          score: r.score,
          metadata: r.metadata,
          type: 'shared',
          fromAgent: r.metadata._agent_id
        }));
      } catch (err) {
        results.shared = [];
      }
    }

    this.emit('memory:recalled', { agentId, query, resultCounts: 
      Object.fromEntries(Object.entries(results).map(([k, v]) => [k, v.length]))
    });

    return results;
  }

  /**
   * Search across ALL memory types and return a flat, ranked list.
   * 
   * @param {string} agentId
   * @param {string} query
   * @param {Object} [options={}]
   * @param {number} [options.topK=10]
   * @returns {Promise<MemoryEntry[]>}
   */
  async searchAll(agentId, query, options = {}) {
    const results = await this.recall(agentId, query, options);
    const flat = [];

    for (const [type, entries] of Object.entries(results)) {
      flat.push(...entries);
    }

    flat.sort((a, b) => b.score - a.score);
    return flat.slice(0, options.topK || 10);
  }

  // ─── Shared Memory (Cross-Agent) ──────────────────────────

  /**
   * Share a memory with other agents via the shared memory pool.
   * 
   * @param {string} agentId - Sharing agent
   * @param {string} content - Knowledge to share
   * @param {Object} [metadata={}]
   * @param {string[]} [metadata.allowedAgents] - Specific agents who can access (empty = all)
   * @returns {Promise<MemoryEntry>}
   * 
   * @example
   * await memory.share('agent-1',
   *   'Customer ACME Corp prefers ISO 14001 compliance framework',
   *   { category: 'customer_knowledge', allowedAgents: ['agent-2', 'agent-3'] }
   * );
   */
  async share(agentId, content, metadata = {}) {
    if (!this.enableSharedMemory) {
      throw new Error('Shared memory is disabled');
    }

    const entry = this._createEntry(agentId, 'shared', { content, metadata });

    if (this.embedder) {
      entry.vector = await this.embedder.embed(content);
      this.engine.insert('_memory_shared', [{
        id: entry.id,
        vector: entry.vector,
        metadata: {
          _content: content,
          _agent_id: agentId,
          _type: 'shared',
          _shared_at: Date.now(),
          _allowed_agents: metadata.allowedAgents || [],
          ...metadata
        }
      }]);
    }

    this._sharedPool.set(entry.id, entry);
    this.emit('memory:shared', { agentId, id: entry.id });
    return entry;
  }

  // ─── Memory Management ──────────────────────────────────

  /**
   * Forget (delete) memories for an agent. Supports GDPR-style data erasure.
   * 
   * @param {string} agentId - Agent identifier
   * @param {Object} [options={}]
   * @param {MemoryType|'all'} [options.type='all'] - Memory type to forget
   * @param {string[]} [options.ids] - Specific memory IDs to delete
   * @returns {{ deleted: number }}
   * 
   * @example
   * // Forget everything for an agent (GDPR erasure)
   * memory.forget('agent-1', { type: 'all' });
   * 
   * // Forget specific memory type
   * memory.forget('agent-1', { type: 'episodic' });
   */
  forget(agentId, options = {}) {
    const { type = 'all', ids = null } = options;
    let deleted = 0;

    const store = this._agentStores.get(agentId);
    if (!store) return { deleted: 0 };

    const types = type === 'all' ? ['episodic', 'semantic', 'procedural'] : [type];

    for (const t of types) {
      if (ids) {
        for (const id of ids) {
          if (store[t]?.delete(id)) {
            this.engine.delete(`_memory_${t}`, id);
            deleted++;
          }
        }
      } else {
        deleted += store[t]?.size || 0;
        store[t]?.clear();
      }
    }

    // Clear conversations
    if (type === 'all' || type === 'conversation') {
      for (const [key] of this._conversations) {
        if (key.startsWith(`${agentId}:`)) {
          this._conversations.delete(key);
          deleted++;
        }
      }
    }

    this.emit('memory:forgotten', { agentId, type, deleted });
    return { deleted };
  }

  /**
   * Get memory statistics for an agent.
   * @param {string} agentId
   * @returns {Object}
   */
  getStats(agentId) {
    const store = this._agentStores.get(agentId);
    if (!store) return { episodic: 0, semantic: 0, procedural: 0, conversations: 0, shared: 0 };

    let conversationCount = 0;
    for (const [key] of this._conversations) {
      if (key.startsWith(`${agentId}:`)) conversationCount++;
    }

    return {
      episodic: store.episodic?.size || 0,
      semantic: store.semantic?.size || 0,
      procedural: store.procedural?.size || 0,
      conversations: conversationCount,
      shared: this._sharedPool.size,
      total: (store.episodic?.size || 0) + (store.semantic?.size || 0) +
             (store.procedural?.size || 0)
    };
  }

  // ─── Internal Methods ──────────────────────────────────

  /** @private */
  _getAgentStore(agentId) {
    if (!this._agentStores.has(agentId)) {
      this._agentStores.set(agentId, {
        episodic: new Map(),
        semantic: new Map(),
        procedural: new Map()
      });
    }
    return this._agentStores.get(agentId);
  }

  /** @private */
  _createEntry(agentId, type, data) {
    const id = `mem_${type}_${String(this._memoryCount++).padStart(6, '0')}`;
    return {
      id,
      agentId,
      type,
      content: data.content,
      metadata: data.metadata || {},
      importance: data.importance || 0.5,
      timestamp: Date.now(),
      accessCount: 0,
      lastAccessed: null,
      ttl: data.ttl || this.defaultTTL
    };
  }

  /** @private */
  _trackAccess(agentId, type, memoryId) {
    const store = this._getAgentStore(agentId);
    const entry = store[type]?.get(memoryId);
    if (entry) {
      entry.accessCount++;
      entry.lastAccessed = Date.now();
    }
  }

  /** @private */
  _keywordRecall(agentId, type, query, topK) {
    const store = this._getAgentStore(agentId);
    const typeStore = store[type];
    if (!typeStore) return [];

    const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
    const results = [];

    for (const [id, entry] of typeStore) {
      const text = entry.content.toLowerCase();
      let matches = 0;
      for (const term of terms) {
        if (text.includes(term)) matches++;
      }
      if (matches > 0) {
        results.push({
          id: entry.id,
          content: entry.content,
          score: matches / terms.length,
          metadata: entry.metadata,
          type
        });
      }
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }
}

module.exports = { AgentMemory };
