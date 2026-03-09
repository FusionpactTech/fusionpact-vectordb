/**
 * @fileoverview RecursiveLearningEngine — Self-Improving Memory & Retrieval
 * 
 * The next frontier in agent memory: memory that doesn't just store — it evolves.
 * 
 * Implements five recursive learning loops:
 * 
 * 1. **Memory Consolidation** — Periodically re-encodes memories, merging related
 *    facts, strengthening important ones, and decaying irrelevant ones (mimics
 *    human sleep-based memory consolidation)
 * 
 * 2. **Retrieval Self-Critique** — After each retrieval, evaluates whether the
 *    results actually answered the query. Logs failures and adjusts strategy
 *    weights for future similar queries (Self-RAG pattern)
 * 
 * 3. **Experience Replay** — Stores successful agent workflows as "skills" in
 *    procedural memory, so agents learn from past successes (Voyager pattern)
 * 
 * 4. **Knowledge Graph Extraction** — Automatically extracts entity-relationship
 *    triples from memories and documents, building a navigable knowledge graph
 *    that improves retrieval precision over time
 * 
 * 5. **Reflection & Meta-Learning** — Agents periodically reflect on their own
 *    performance, generating self-improvement insights stored as meta-memories
 *    (Reflexion pattern)
 * 
 * @module learning/RecursiveLearningEngine
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 * @see https://github.com/FusionpactTech/fusionpact-vectordb
 */

'use strict';

const { EventEmitter } = require('events');

/**
 * @typedef {Object} ConsolidationResult
 * @property {number} merged - Number of memories merged
 * @property {number} strengthened - Number of memories with increased importance
 * @property {number} decayed - Number of memories with decreased importance
 * @property {number} pruned - Number of memories removed (below threshold)
 */

/**
 * @typedef {Object} RetrievalFeedback
 * @property {string} query - The original query
 * @property {string} strategy - Strategy used ('vector', 'tree', 'hybrid', 'keyword')
 * @property {number} quality - 0-1 quality score
 * @property {string[]} resultIds - IDs of returned results
 * @property {string} [correction] - What the correct answer should have been
 * @property {number} timestamp
 */

/**
 * @typedef {Object} Skill
 * @property {string} id - Skill identifier
 * @property {string} name - Skill name
 * @property {string} description - What this skill does
 * @property {Object} trigger - When to activate this skill
 * @property {Object[]} steps - Sequence of actions
 * @property {number} successRate - Historical success rate
 * @property {number} useCount - Times this skill was used
 */

/**
 * @typedef {Object} KnowledgeTriple
 * @property {string} subject - Entity
 * @property {string} predicate - Relationship
 * @property {string} object - Related entity
 * @property {number} confidence - 0-1 confidence score
 * @property {string} source - Where this was extracted from
 */

class RecursiveLearningEngine extends EventEmitter {
  /**
   * @param {Object} config
   * @param {import('../memory/AgentMemory')} config.memory - AgentMemory instance
   * @param {import('../retrieval/HybridRetriever')} [config.retriever] - HybridRetriever
   * @param {import('../embedders/providers').LLMProvider} [config.llmProvider] - LLM for reasoning
   * @param {Object} [config.consolidation={}] - Consolidation settings
   * @param {number} [config.consolidation.intervalMs=3600000] - Consolidation interval (default: 1 hour)
   * @param {number} [config.consolidation.decayRate=0.02] - Importance decay per cycle
   * @param {number} [config.consolidation.pruneThreshold=0.05] - Remove memories below this importance
   * @param {number} [config.consolidation.mergeThreshold=0.85] - Merge memories above this similarity
   * @param {boolean} [config.enableAutoConsolidation=false] - Run consolidation automatically
   * @param {boolean} [config.enableRetrievalCritique=true] - Auto-critique retrievals
   * @param {boolean} [config.enableKnowledgeGraph=true] - Build knowledge graph
   */
  constructor(config) {
    super();
    this.memory = config.memory;
    this.retriever = config.retriever || null;
    this.llmProvider = config.llmProvider || null;

    // Consolidation config
    this.consolidation = {
      intervalMs: config.consolidation?.intervalMs || 3600000,
      decayRate: config.consolidation?.decayRate || 0.02,
      pruneThreshold: config.consolidation?.pruneThreshold || 0.05,
      mergeThreshold: config.consolidation?.mergeThreshold || 0.85
    };

    this.enableAutoConsolidation = config.enableAutoConsolidation || false;
    this.enableRetrievalCritique = config.enableRetrievalCritique !== false;
    this.enableKnowledgeGraph = config.enableKnowledgeGraph !== false;

    // ─── Internal State ───

    /** @private - Retrieval feedback log */
    this._feedbackLog = new Map(); // agentId -> RetrievalFeedback[]

    /** @private - Learned strategy weights per query pattern */
    this._learnedWeights = new Map(); // pattern -> { vector, tree, keyword }

    /** @private - Skill library */
    this._skills = new Map(); // agentId -> Map<skillId, Skill>

    /** @private - Knowledge graph triples */
    this._knowledgeGraph = new Map(); // agentId -> KnowledgeTriple[]

    /** @private - Reflection log */
    this._reflections = new Map(); // agentId -> string[]

    /** @private - Consolidation timer */
    this._consolidationTimer = null;

    /** @private - Stats */
    this._stats = {
      consolidations: 0,
      critiques: 0,
      skillsLearned: 0,
      triplesExtracted: 0,
      reflections: 0,
      strategyAdjustments: 0
    };

    if (this.enableAutoConsolidation) {
      this._startAutoConsolidation();
    }
  }

  // ═══════════════════════════════════════════════════════
  // 1. MEMORY CONSOLIDATION
  //    Mimics human sleep-based memory consolidation:
  //    - Merge similar memories into stronger, unified entries
  //    - Strengthen frequently accessed memories
  //    - Decay rarely accessed memories
  //    - Prune memories below importance threshold
  // ═══════════════════════════════════════════════════════

  /**
   * Run memory consolidation for an agent.
   * 
   * This is the core recursive loop: each consolidation cycle makes the
   * memory system more efficient and accurate, which improves future
   * retrievals, which generates better feedback, which guides the next
   * consolidation cycle.
   * 
   * @param {string} agentId
   * @returns {Promise<ConsolidationResult>}
   * 
   * @example
   * // Run manually
   * const result = await learning.consolidate('agent-1');
   * console.log(`Merged: ${result.merged}, Pruned: ${result.pruned}`);
   * 
   * // Or enable auto-consolidation in constructor
   */
  async consolidate(agentId) {
    this.emit('consolidation:start', { agentId });
    const result = { merged: 0, strengthened: 0, decayed: 0, pruned: 0 };

    const store = this.memory._getAgentStore(agentId);
    if (!store) return result;

    for (const type of ['episodic', 'semantic', 'procedural']) {
      const typeStore = store[type];
      if (!typeStore || typeStore.size === 0) continue;

      const entries = Array.from(typeStore.values());

      // ─── Step 1: Decay importance based on recency and access ───
      const now = Date.now();
      for (const entry of entries) {
        const ageHours = (now - entry.timestamp) / 3600000;
        const accessRecency = entry.lastAccessed
          ? (now - entry.lastAccessed) / 3600000
          : ageHours;

        // Decay: memories that haven't been accessed lose importance over time
        // But frequently accessed memories resist decay
        const accessBoost = Math.min(entry.accessCount * 0.02, 0.3);
        const decay = this.consolidation.decayRate * (accessRecency / 24);
        const netDecay = Math.max(0, decay - accessBoost);

        entry.importance = Math.max(0, entry.importance - netDecay);

        if (netDecay > 0) result.decayed++;
        if (accessBoost > 0 && entry.accessCount > 3) {
          entry.importance = Math.min(1, entry.importance + 0.01);
          result.strengthened++;
        }
      }

      // ─── Step 2: Merge highly similar memories ───
      if (this.memory.embedder && entries.length > 1) {
        const merged = await this._mergeSimilarMemories(agentId, type, entries);
        result.merged += merged;
      }

      // ─── Step 3: Prune memories below threshold ───
      for (const entry of entries) {
        if (entry.importance < this.consolidation.pruneThreshold) {
          typeStore.delete(entry.id);
          result.pruned++;
        }
      }
    }

    this._stats.consolidations++;
    this.emit('consolidation:complete', { agentId, result });
    return result;
  }

  /** @private */
  async _mergeSimilarMemories(agentId, type, entries) {
    let merged = 0;

    // Compare pairs and merge highly similar ones
    const toRemove = new Set();

    for (let i = 0; i < entries.length && i < 100; i++) {
      if (toRemove.has(entries[i].id)) continue;

      for (let j = i + 1; j < entries.length && j < 100; j++) {
        if (toRemove.has(entries[j].id)) continue;

        // Quick text similarity check
        const similarity = this._textSimilarity(entries[i].content, entries[j].content);

        if (similarity > this.consolidation.mergeThreshold) {
          // Merge: keep the more important one, absorb the other's content
          const keeper = entries[i].importance >= entries[j].importance ? entries[i] : entries[j];
          const absorbed = keeper === entries[i] ? entries[j] : entries[i];

          // Combine content if LLM available, otherwise concatenate
          if (this.llmProvider) {
            try {
              keeper.content = await this.llmProvider.complete(
                `Merge these two related memories into a single, concise memory:\n\nMemory 1: ${keeper.content}\n\nMemory 2: ${absorbed.content}\n\nMerged memory (1-2 sentences):`,
                { maxTokens: 150 }
              );
            } catch {
              keeper.content = `${keeper.content} | ${absorbed.content}`;
            }
          } else {
            keeper.content = `${keeper.content} | ${absorbed.content}`;
          }

          // Boost importance of merged memory
          keeper.importance = Math.min(1, keeper.importance + 0.1);
          keeper.accessCount += absorbed.accessCount;
          keeper.metadata._merged = true;
          keeper.metadata._mergedFrom = absorbed.id;

          toRemove.add(absorbed.id);
          merged++;
        }
      }
    }

    // Remove absorbed memories
    const store = this.memory._getAgentStore(agentId);
    for (const id of toRemove) {
      store[type]?.delete(id);
    }

    return merged;
  }

  // ═══════════════════════════════════════════════════════
  // 2. RETRIEVAL SELF-CRITIQUE (Self-RAG Pattern)
  //    After each retrieval, evaluate quality and adjust
  //    strategy weights for future similar queries
  // ═══════════════════════════════════════════════════════

  /**
   * Record retrieval feedback and adjust strategy weights.
   * 
   * Call this after each retrieval with a quality score.
   * The system learns which strategies work best for which
   * types of queries and auto-adjusts weights over time.
   * 
   * @param {string} agentId
   * @param {RetrievalFeedback} feedback
   * @returns {{ adjusted: boolean, newWeights: Object }}
   * 
   * @example
   * const results = await retriever.retrieve('Q3 revenue');
   * // ... user or system evaluates quality ...
   * learning.recordRetrievalFeedback('agent-1', {
   *   query: 'Q3 revenue',
   *   strategy: 'hybrid',
   *   quality: 0.9,  // Great result
   *   resultIds: results.map(r => r.id)
   * });
   */
  recordRetrievalFeedback(agentId, feedback) {
    if (!this._feedbackLog.has(agentId)) {
      this._feedbackLog.set(agentId, []);
    }

    const entry = { ...feedback, timestamp: Date.now() };
    this._feedbackLog.get(agentId).push(entry);

    // Keep last 1000 entries
    const log = this._feedbackLog.get(agentId);
    if (log.length > 1000) log.splice(0, log.length - 1000);

    // Learn from this feedback
    const pattern = this._queryPattern(feedback.query);
    const adjustment = this._adjustStrategyWeights(pattern, feedback);

    this._stats.critiques++;
    if (adjustment.adjusted) this._stats.strategyAdjustments++;

    // Store the learning as a meta-memory
    if (feedback.quality < 0.3 && feedback.correction) {
      // Bad result with correction → learn from the failure
      this.memory.learn(agentId,
        `Retrieval failure: Query "${feedback.query}" returned poor results via ${feedback.strategy}. Better answer: ${feedback.correction}`,
        { type: 'retrieval_failure', strategy: feedback.strategy }
      ).catch(() => {});
    }

    this.emit('critique:recorded', { agentId, pattern, ...adjustment });
    return adjustment;
  }

  /**
   * Get the learned optimal weights for a query.
   * 
   * @param {string} query
   * @returns {Object} Optimized weights { vector, tree, keyword }
   */
  getOptimalWeights(query) {
    const pattern = this._queryPattern(query);
    return this._learnedWeights.get(pattern) || { vector: 0.4, tree: 0.4, keyword: 0.2 };
  }

  /**
   * Perform retrieval with auto-critique.
   * 
   * Wraps the standard retriever with automatic quality evaluation
   * and strategy adjustment. If the retrieval is poor, automatically
   * retries with a different strategy.
   * 
   * @param {string} agentId
   * @param {string} query
   * @param {Object} options - Standard retriever options
   * @returns {Promise<Object>} Enhanced results with quality metadata
   */
  async retrieveWithCritique(agentId, query, options = {}) {
    if (!this.retriever) throw new Error('No retriever configured');

    // Use learned weights for this query pattern
    const optimalWeights = this.getOptimalWeights(query);

    // First attempt
    const results = await this.retriever.retrieve(query, {
      ...options,
      weights: optimalWeights
    });

    // Self-critique: evaluate quality
    let quality = this._estimateResultQuality(query, results);

    // If quality is poor and LLM is available, try a different strategy
    if (quality < 0.3 && this.llmProvider && options.strategy !== 'tree') {
      this.emit('critique:retry', { agentId, query, originalQuality: quality });

      const retryResults = await this.retriever.retrieve(query, {
        ...options,
        strategy: 'tree' // Fallback to reasoning-based retrieval
      });

      const retryQuality = this._estimateResultQuality(query, retryResults);

      if (retryQuality > quality) {
        this.recordRetrievalFeedback(agentId, {
          query, strategy: 'tree', quality: retryQuality,
          resultIds: retryResults.map(r => r.id)
        });
        return { results: retryResults, quality: retryQuality, retried: true, strategy: 'tree' };
      }
    }

    this.recordRetrievalFeedback(agentId, {
      query, strategy: options.strategy || 'hybrid', quality,
      resultIds: results.map(r => r.id)
    });

    return { results, quality, retried: false, strategy: options.strategy || 'hybrid' };
  }

  // ═══════════════════════════════════════════════════════
  // 3. EXPERIENCE REPLAY / SKILL LEARNING (Voyager Pattern)
  //    Store successful workflows as reusable skills
  // ═══════════════════════════════════════════════════════

  /**
   * Record a successful workflow as a learned skill.
   * 
   * @param {string} agentId
   * @param {Skill} skill
   * @returns {Skill}
   * 
   * @example
   * learning.learnSkill('agent-1', {
   *   name: 'safety_audit',
   *   description: 'Complete safety audit workflow for a facility',
   *   trigger: { keywords: ['safety', 'audit', 'compliance'] },
   *   steps: [
   *     { action: 'recall', params: { query: 'safety regulations' } },
   *     { action: 'search', params: { collection: 'docs', query: 'facility audit checklist' } },
   *     { action: 'generate', params: { template: 'audit_report' } }
   *   ]
   * });
   */
  learnSkill(agentId, skill) {
    if (!this._skills.has(agentId)) {
      this._skills.set(agentId, new Map());
    }

    const skillEntry = {
      id: skill.id || `skill_${Date.now()}`,
      name: skill.name,
      description: skill.description,
      trigger: skill.trigger || {},
      steps: skill.steps || [],
      successRate: skill.successRate || 1.0,
      useCount: 0,
      learnedAt: Date.now(),
      lastUsed: null
    };

    this._skills.get(agentId).set(skillEntry.id, skillEntry);

    // Also store in procedural memory for retrieval
    this.memory.registerTool(agentId, {
      name: skillEntry.name,
      description: skillEntry.description,
      schema: { trigger: skillEntry.trigger, steps: skillEntry.steps }
    }).catch(() => {});

    this._stats.skillsLearned++;
    this.emit('skill:learned', { agentId, skill: skillEntry });
    return skillEntry;
  }

  /**
   * Find applicable skills for a given context.
   * 
   * @param {string} agentId
   * @param {string} context - Current task/query context
   * @returns {Skill[]} Matching skills sorted by success rate
   */
  findApplicableSkills(agentId, context) {
    const skills = this._skills.get(agentId);
    if (!skills) return [];

    const contextLower = context.toLowerCase();
    const matches = [];

    for (const [, skill] of skills) {
      if (skill.trigger?.keywords) {
        const matchCount = skill.trigger.keywords.filter(kw =>
          contextLower.includes(kw.toLowerCase())
        ).length;

        if (matchCount > 0) {
          matches.push({
            ...skill,
            matchScore: matchCount / skill.trigger.keywords.length
          });
        }
      }
    }

    matches.sort((a, b) => (b.successRate * b.matchScore) - (a.successRate * a.matchScore));
    return matches;
  }

  /**
   * Record skill execution outcome for reinforcement learning.
   * 
   * @param {string} agentId
   * @param {string} skillId
   * @param {boolean} success
   */
  recordSkillOutcome(agentId, skillId, success) {
    const skills = this._skills.get(agentId);
    const skill = skills?.get(skillId);
    if (!skill) return;

    skill.useCount++;
    skill.lastUsed = Date.now();

    // Exponential moving average of success rate
    const alpha = 0.3;
    skill.successRate = alpha * (success ? 1 : 0) + (1 - alpha) * skill.successRate;

    this.emit('skill:outcome', { agentId, skillId, success, newSuccessRate: skill.successRate });
  }

  // ═══════════════════════════════════════════════════════
  // 4. KNOWLEDGE GRAPH EXTRACTION
  //    Auto-extract entity-relationship triples from memories
  //    to build a navigable knowledge graph
  // ═══════════════════════════════════════════════════════

  /**
   * Extract knowledge triples from text and add to the graph.
   * 
   * @param {string} agentId
   * @param {string} text - Text to extract from
   * @param {string} [source='unknown'] - Source identifier
   * @returns {Promise<KnowledgeTriple[]>} Extracted triples
   */
  async extractKnowledge(agentId, text, source = 'unknown') {
    if (!this._knowledgeGraph.has(agentId)) {
      this._knowledgeGraph.set(agentId, []);
    }

    let triples = [];

    if (this.llmProvider) {
      try {
        const response = await this.llmProvider.complete(
          `Extract entity-relationship triples from this text. Return ONLY a JSON array of objects with subject, predicate, object fields. No markdown.\n\nText: "${text.substring(0, 2000)}"\n\nTriples:`,
          { maxTokens: 500 }
        );

        triples = JSON.parse(response.replace(/```json?|```/g, '').trim());
      } catch {
        // Fallback: simple pattern extraction
        triples = this._simpleTripleExtraction(text);
      }
    } else {
      triples = this._simpleTripleExtraction(text);
    }

    // Add to graph with confidence and source
    const enriched = triples.map(t => ({
      subject: t.subject,
      predicate: t.predicate,
      object: t.object,
      confidence: 0.7,
      source,
      extractedAt: Date.now()
    }));

    this._knowledgeGraph.get(agentId).push(...enriched);
    this._stats.triplesExtracted += enriched.length;

    this.emit('knowledge:extracted', { agentId, count: enriched.length });
    return enriched;
  }

  /**
   * Query the knowledge graph.
   * 
   * @param {string} agentId
   * @param {Object} query
   * @param {string} [query.subject] - Filter by subject entity
   * @param {string} [query.predicate] - Filter by relationship
   * @param {string} [query.object] - Filter by object entity
   * @returns {KnowledgeTriple[]}
   */
  queryKnowledgeGraph(agentId, query = {}) {
    const triples = this._knowledgeGraph.get(agentId) || [];

    return triples.filter(t => {
      if (query.subject && !t.subject.toLowerCase().includes(query.subject.toLowerCase())) return false;
      if (query.predicate && !t.predicate.toLowerCase().includes(query.predicate.toLowerCase())) return false;
      if (query.object && !t.object.toLowerCase().includes(query.object.toLowerCase())) return false;
      return true;
    });
  }

  /**
   * Get all entities and their connections.
   * @param {string} agentId
   * @returns {{ entities: string[], relationships: Object[] }}
   */
  getGraphSummary(agentId) {
    const triples = this._knowledgeGraph.get(agentId) || [];
    const entities = new Set();
    const relationships = [];

    for (const t of triples) {
      entities.add(t.subject);
      entities.add(t.object);
      relationships.push({ from: t.subject, to: t.object, type: t.predicate });
    }

    return { entities: Array.from(entities), relationships, tripleCount: triples.length };
  }

  // ═══════════════════════════════════════════════════════
  // 5. REFLECTION & META-LEARNING (Reflexion Pattern)
  //    Periodic self-assessment generating improvement insights
  // ═══════════════════════════════════════════════════════

  /**
   * Generate a reflection on the agent's recent performance.
   * 
   * The agent reviews its recent retrievals, memory usage, and outcomes
   * to generate meta-insights that improve future behavior.
   * 
   * @param {string} agentId
   * @returns {Promise<string>} Reflection text
   */
  async reflect(agentId) {
    const feedbackLog = this._feedbackLog.get(agentId) || [];
    const recentFeedback = feedbackLog.slice(-20);
    const memStats = this.memory.getStats(agentId);
    const skills = this._skills.get(agentId);

    // Build context for reflection
    const context = {
      recentRetrievals: recentFeedback.length,
      avgQuality: recentFeedback.length > 0
        ? recentFeedback.reduce((s, f) => s + f.quality, 0) / recentFeedback.length
        : 0,
      failedQueries: recentFeedback.filter(f => f.quality < 0.3).map(f => f.query),
      memoryStats: memStats,
      skillCount: skills?.size || 0,
      knowledgeTriples: (this._knowledgeGraph.get(agentId) || []).length
    };

    let reflection;

    if (this.llmProvider) {
      try {
        reflection = await this.llmProvider.complete(
          `You are an AI agent reviewing your own performance. Based on this data, identify patterns, weaknesses, and specific improvements you should make.\n\nPerformance data:\n${JSON.stringify(context, null, 2)}\n\nFailed queries: ${context.failedQueries.join(', ') || 'none'}\n\nGenerate 2-3 specific, actionable insights:`,
          { maxTokens: 300 }
        );
      } catch {
        reflection = this._generateSimpleReflection(context);
      }
    } else {
      reflection = this._generateSimpleReflection(context);
    }

    // Store reflection as meta-memory
    if (!this._reflections.has(agentId)) {
      this._reflections.set(agentId, []);
    }
    this._reflections.get(agentId).push({
      reflection,
      context,
      timestamp: Date.now()
    });

    // Also store in episodic memory so it influences future retrievals
    await this.memory.remember(agentId, {
      content: `Self-reflection: ${reflection}`,
      importance: 0.7,
      metadata: { type: 'reflection', ...context }
    });

    this._stats.reflections++;
    this.emit('reflection:complete', { agentId, reflection });
    return reflection;
  }

  /**
   * Get the agent's reflection history.
   * @param {string} agentId
   * @param {number} [limit=10]
   * @returns {Array<{reflection: string, context: Object, timestamp: number}>}
   */
  getReflections(agentId, limit = 10) {
    const reflections = this._reflections.get(agentId) || [];
    return reflections.slice(-limit);
  }

  // ═══════════════════════════════════════════════════════
  // STATISTICS & LIFECYCLE
  // ═══════════════════════════════════════════════════════

  /**
   * Get comprehensive learning statistics.
   * @returns {Object}
   */
  getStats() {
    return {
      ...this._stats,
      learnedPatterns: this._learnedWeights.size,
      feedbackEntries: Array.from(this._feedbackLog.values())
        .reduce((sum, log) => sum + log.length, 0)
    };
  }

  /**
   * Stop auto-consolidation.
   */
  stop() {
    if (this._consolidationTimer) {
      clearInterval(this._consolidationTimer);
      this._consolidationTimer = null;
    }
  }

  /**
   * Export all learned state for persistence.
   * @returns {Object}
   */
  serialize() {
    const skills = {};
    for (const [agentId, skillMap] of this._skills) {
      skills[agentId] = Array.from(skillMap.values());
    }

    const graph = {};
    for (const [agentId, triples] of this._knowledgeGraph) {
      graph[agentId] = triples;
    }

    return {
      _version: 1,
      _engine: 'FusionPact',
      learnedWeights: Object.fromEntries(this._learnedWeights),
      skills,
      knowledgeGraph: graph,
      reflections: Object.fromEntries(this._reflections),
      stats: this._stats
    };
  }

  /**
   * Import learned state.
   * @param {Object} data
   */
  importState(data) {
    if (data.learnedWeights) {
      for (const [k, v] of Object.entries(data.learnedWeights)) {
        this._learnedWeights.set(k, v);
      }
    }
    if (data.skills) {
      for (const [agentId, skillList] of Object.entries(data.skills)) {
        const map = new Map();
        for (const s of skillList) map.set(s.id, s);
        this._skills.set(agentId, map);
      }
    }
    if (data.knowledgeGraph) {
      for (const [agentId, triples] of Object.entries(data.knowledgeGraph)) {
        this._knowledgeGraph.set(agentId, triples);
      }
    }
    if (data.stats) this._stats = { ...this._stats, ...data.stats };
  }

  // ─── Private Helpers ──────────────────────────────────

  /** @private */
  _startAutoConsolidation() {
    this._consolidationTimer = setInterval(async () => {
      for (const agentId of this.memory._agentStores.keys()) {
        try {
          await this.consolidate(agentId);
        } catch (e) {
          this.emit('consolidation:error', { agentId, error: e.message });
        }
      }
    }, this.consolidation.intervalMs);
  }

  /** @private */
  _adjustStrategyWeights(pattern, feedback) {
    if (!this._learnedWeights.has(pattern)) {
      this._learnedWeights.set(pattern, { vector: 0.4, tree: 0.4, keyword: 0.2 });
    }

    const weights = this._learnedWeights.get(pattern);
    const alpha = 0.1; // Learning rate

    // Reinforce the strategy that worked well, decay the rest
    if (feedback.quality > 0.7) {
      if (feedback.strategy === 'vector' || feedback.strategy === 'hybrid') {
        weights.vector = Math.min(0.8, weights.vector + alpha * feedback.quality);
      }
      if (feedback.strategy === 'tree' || feedback.strategy === 'hybrid') {
        weights.tree = Math.min(0.8, weights.tree + alpha * feedback.quality);
      }
      if (feedback.strategy === 'keyword' || feedback.strategy === 'hybrid') {
        weights.keyword = Math.min(0.5, weights.keyword + alpha * feedback.quality * 0.5);
      }
    } else if (feedback.quality < 0.3) {
      // Penalize the strategy that failed
      if (feedback.strategy === 'vector') weights.vector = Math.max(0.1, weights.vector - alpha);
      if (feedback.strategy === 'tree') weights.tree = Math.max(0.1, weights.tree - alpha);
      if (feedback.strategy === 'keyword') weights.keyword = Math.max(0.05, weights.keyword - alpha);
    }

    // Normalize
    const total = weights.vector + weights.tree + weights.keyword;
    weights.vector /= total;
    weights.tree /= total;
    weights.keyword /= total;

    return { adjusted: true, newWeights: { ...weights } };
  }

  /** @private */
  _queryPattern(query) {
    const words = query.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).sort();
    return words.filter(w => w.length > 3).slice(0, 4).join('_') || 'generic';
  }

  /** @private */
  _estimateResultQuality(query, results) {
    if (!results || results.length === 0) return 0;

    // Heuristic quality estimation based on:
    // 1. Score distribution (high top score = good)
    // 2. Content overlap with query terms
    // 3. Number of results

    const topScore = results[0]?.score || 0;
    const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
    let contentOverlap = 0;

    for (const result of results.slice(0, 3)) {
      const content = (result.content || '').toLowerCase();
      for (const term of queryTerms) {
        if (content.includes(term)) contentOverlap++;
      }
    }

    const overlapScore = queryTerms.length > 0
      ? Math.min(1, contentOverlap / (queryTerms.length * 2))
      : 0.5;

    return (topScore * 0.4 + overlapScore * 0.4 + Math.min(results.length / 5, 1) * 0.2);
  }

  /** @private */
  _textSimilarity(a, b) {
    const wordsA = new Set(a.toLowerCase().split(/\s+/));
    const wordsB = new Set(b.toLowerCase().split(/\s+/));
    const intersection = new Set([...wordsA].filter(w => wordsB.has(w)));
    const union = new Set([...wordsA, ...wordsB]);
    return union.size > 0 ? intersection.size / union.size : 0;
  }

  /** @private */
  _simpleTripleExtraction(text) {
    // Simple pattern-based extraction without LLM
    const triples = [];
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);

    for (const sentence of sentences.slice(0, 10)) {
      // Pattern: "X is/are Y"
      const isMatch = sentence.match(/(.{3,30})\s+(?:is|are|was|were)\s+(.{3,50})/i);
      if (isMatch) {
        triples.push({
          subject: isMatch[1].trim(),
          predicate: 'is',
          object: isMatch[2].trim().substring(0, 50)
        });
      }

      // Pattern: "X covers/requires/includes Y"
      const verbMatch = sentence.match(/(.{3,30})\s+(covers|requires|includes|contains|provides|manages)\s+(.{3,50})/i);
      if (verbMatch) {
        triples.push({
          subject: verbMatch[1].trim(),
          predicate: verbMatch[2].trim(),
          object: verbMatch[3].trim().substring(0, 50)
        });
      }
    }

    return triples;
  }

  /** @private */
  _generateSimpleReflection(context) {
    const insights = [];

    if (context.avgQuality < 0.5) {
      insights.push(`Average retrieval quality is low (${(context.avgQuality * 100).toFixed(0)}%). Consider switching to tree-based retrieval for structured documents.`);
    }

    if (context.failedQueries.length > 3) {
      insights.push(`${context.failedQueries.length} queries failed recently. Common patterns: ${context.failedQueries.slice(0, 3).join(', ')}. These may need additional knowledge ingestion.`);
    }

    if (context.memoryStats.semantic < 5) {
      insights.push('Semantic memory is sparse. Learning more domain knowledge would improve retrieval quality.');
    }

    if (context.skillCount === 0) {
      insights.push('No skills learned yet. Recording successful workflows as skills would enable automation.');
    }

    return insights.length > 0
      ? insights.join(' ')
      : `Performance is stable with ${context.recentRetrievals} recent retrievals at ${(context.avgQuality * 100).toFixed(0)}% quality.`;
  }
}

module.exports = { RecursiveLearningEngine };
