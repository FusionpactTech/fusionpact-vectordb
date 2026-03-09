/**
 * @fileoverview RecursiveLearningEngine — Self-Improving Memory & Retrieval
 *
 * Five recursive feedback loops that make agents improve over time:
 * 1. Memory Consolidation — merge, strengthen, decay, prune
 * 2. Retrieval Self-Critique — evaluate + auto-retry + strategy learning
 * 3. Skill Learning — store successful workflows (Voyager pattern)
 * 4. Knowledge Graph Extraction — auto-extract entity-relationship triples
 * 5. Reflection & Meta-Learning — periodic self-assessment (Reflexion pattern)
 *
 * @module learning/RecursiveLearningEngine
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 * @see https://github.com/FusionpactTech/fusionpact-vectordb
 */

'use strict';

const { EventEmitter } = require('events');

// ─── Constants ───────────────────────────────────────────
const MAX_FEEDBACK_LOG = 1000;
const MAX_SKILLS_PER_AGENT = 500;
const MAX_TRIPLES_PER_AGENT = 5000;
const MAX_REFLECTIONS_PER_AGENT = 100;
const MAX_MERGE_SCAN = 200;
const VALID_STRATEGIES = new Set(['vector', 'tree', 'keyword', 'hybrid']);
const MIN_CONTENT_LENGTH = 1;
const MAX_CONTENT_FOR_MERGE = 4000;

/**
 * Clamp a number between min and max.
 * @param {number} val
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

/**
 * Safe JSON parse with fallback.
 * @param {string} str
 * @param {*} fallback
 * @returns {*}
 */
function safeJsonParse(str, fallback) {
  try {
    return JSON.parse(str.replace(/```json?|```/g, '').trim());
  } catch {
    return fallback;
  }
}

class RecursiveLearningEngine extends EventEmitter {
  /**
   * @param {Object} config
   * @param {import('../memory/AgentMemory')} config.memory - AgentMemory instance (required)
   * @param {import('../retrieval/HybridRetriever')} [config.retriever] - HybridRetriever
   * @param {import('../embedders/providers').LLMProvider} [config.llmProvider] - LLM for reasoning
   * @param {Object} [config.consolidation={}]
   * @param {number} [config.consolidation.intervalMs=3600000] - Consolidation interval (default: 1h)
   * @param {number} [config.consolidation.decayRate=0.02] - Importance decay per cycle
   * @param {number} [config.consolidation.pruneThreshold=0.05] - Prune below this importance
   * @param {number} [config.consolidation.mergeThreshold=0.85] - Merge above this similarity
   * @param {boolean} [config.enableAutoConsolidation=false]
   * @param {boolean} [config.enableRetrievalCritique=true]
   * @param {boolean} [config.enableKnowledgeGraph=true]
   * @throws {TypeError} If config.memory is not provided
   */
  constructor(config) {
    super();

    if (!config || !config.memory) {
      throw new TypeError('RecursiveLearningEngine requires config.memory (AgentMemory instance)');
    }

    this.memory = config.memory;
    this.retriever = config.retriever || null;
    this.llmProvider = config.llmProvider || null;

    this.consolidation = Object.freeze({
      intervalMs: Math.max(10000, config.consolidation?.intervalMs || 3600000),
      decayRate: clamp(config.consolidation?.decayRate || 0.02, 0, 0.5),
      pruneThreshold: clamp(config.consolidation?.pruneThreshold || 0.05, 0, 1),
      mergeThreshold: clamp(config.consolidation?.mergeThreshold || 0.85, 0.5, 1)
    });

    this.enableAutoConsolidation = !!config.enableAutoConsolidation;
    this.enableRetrievalCritique = config.enableRetrievalCritique !== false;
    this.enableKnowledgeGraph = config.enableKnowledgeGraph !== false;

    /** @private */ this._feedbackLog = new Map();
    /** @private */ this._learnedWeights = new Map();
    /** @private */ this._skills = new Map();
    /** @private */ this._knowledgeGraph = new Map();
    /** @private */ this._reflections = new Map();
    /** @private */ this._consolidationTimer = null;
    /** @private */ this._consolidating = new Set(); // guards against concurrent consolidation
    /** @private */ this._stats = {
      consolidations: 0, critiques: 0, skillsLearned: 0,
      triplesExtracted: 0, reflections: 0, strategyAdjustments: 0
    };

    if (this.enableAutoConsolidation) {
      this._startAutoConsolidation();
    }
  }

  // ═══════════════════════════════════════════════════════
  // 1. MEMORY CONSOLIDATION
  // ═══════════════════════════════════════════════════════

  /**
   * Run memory consolidation for an agent.
   * Safe to call concurrently — uses a per-agent lock.
   *
   * @param {string} agentId - Non-empty agent identifier
   * @returns {Promise<ConsolidationResult>}
   * @throws {TypeError} If agentId is not a non-empty string
   */
  async consolidate(agentId) {
    this._validateAgentId(agentId);

    // Prevent concurrent consolidation for same agent
    if (this._consolidating.has(agentId)) {
      return { merged: 0, strengthened: 0, decayed: 0, pruned: 0, skipped: true };
    }
    this._consolidating.add(agentId);

    try {
      this.emit('consolidation:start', { agentId });
      const result = { merged: 0, strengthened: 0, decayed: 0, pruned: 0 };

      const store = this.memory._getAgentStore(agentId);
      if (!store) return result;

      for (const type of ['episodic', 'semantic', 'procedural']) {
        const typeStore = store[type];
        if (!typeStore || typeStore.size === 0) continue;

        const entries = Array.from(typeStore.values());
        const now = Date.now();

        // Step 1: Decay and strengthen based on access patterns
        for (const entry of entries) {
          const ageHours = Math.max(0, (now - (entry.timestamp || now)) / 3600000);
          const accessRecency = entry.lastAccessed
            ? Math.max(0, (now - entry.lastAccessed) / 3600000)
            : ageHours;

          const accessBoost = Math.min((entry.accessCount || 0) * 0.02, 0.3);
          const decay = this.consolidation.decayRate * (accessRecency / 24);
          const netDecay = Math.max(0, decay - accessBoost);

          if (netDecay > 0) {
            entry.importance = Math.max(0, (entry.importance || 0.5) - netDecay);
            result.decayed++;
          }
          if ((entry.accessCount || 0) > 3) {
            entry.importance = Math.min(1, (entry.importance || 0.5) + 0.01);
            result.strengthened++;
          }
        }

        // Step 2: Merge similar memories (bounded scan)
        if (this.memory.embedder && entries.length > 1) {
          result.merged += await this._mergeSimilarMemories(agentId, type, entries);
        }

        // Step 3: Prune below threshold
        for (const entry of Array.from(typeStore.values())) {
          if ((entry.importance || 0) < this.consolidation.pruneThreshold) {
            typeStore.delete(entry.id);
            result.pruned++;
          }
        }
      }

      this._stats.consolidations++;
      this.emit('consolidation:complete', { agentId, result });
      return result;
    } finally {
      this._consolidating.delete(agentId);
    }
  }

  /** @private */
  async _mergeSimilarMemories(agentId, type, entries) {
    let merged = 0;
    const toRemove = new Set();
    const scanLimit = Math.min(entries.length, MAX_MERGE_SCAN);

    for (let i = 0; i < scanLimit; i++) {
      if (toRemove.has(entries[i].id)) continue;

      for (let j = i + 1; j < scanLimit; j++) {
        if (toRemove.has(entries[j].id)) continue;

        const similarity = this._textSimilarity(
          entries[i].content || '',
          entries[j].content || ''
        );

        if (similarity > this.consolidation.mergeThreshold) {
          const keeper = (entries[i].importance || 0) >= (entries[j].importance || 0)
            ? entries[i] : entries[j];
          const absorbed = keeper === entries[i] ? entries[j] : entries[i];

          if (this.llmProvider) {
            try {
              const merged_content = await this.llmProvider.complete(
                `Merge these two related memories into one concise memory:\n\nMemory 1: ${(keeper.content || '').substring(0, MAX_CONTENT_FOR_MERGE)}\n\nMemory 2: ${(absorbed.content || '').substring(0, MAX_CONTENT_FOR_MERGE)}\n\nMerged (1-2 sentences):`,
                { maxTokens: 150 }
              );
              if (merged_content && merged_content.length > 0) {
                keeper.content = merged_content;
              }
            } catch {
              keeper.content = `${keeper.content} | ${absorbed.content}`;
            }
          } else {
            keeper.content = `${keeper.content} | ${absorbed.content}`;
          }

          keeper.importance = Math.min(1, (keeper.importance || 0.5) + 0.1);
          keeper.accessCount = (keeper.accessCount || 0) + (absorbed.accessCount || 0);
          if (!keeper.metadata) keeper.metadata = {};
          keeper.metadata._merged = true;

          toRemove.add(absorbed.id);
          merged++;
        }
      }
    }

    const store = this.memory._getAgentStore(agentId);
    for (const id of toRemove) {
      store[type]?.delete(id);
    }
    return merged;
  }

  // ═══════════════════════════════════════════════════════
  // 2. RETRIEVAL SELF-CRITIQUE
  // ═══════════════════════════════════════════════════════

  /**
   * Record retrieval feedback and adjust strategy weights.
   *
   * @param {string} agentId
   * @param {Object} feedback
   * @param {string} feedback.query - Original query
   * @param {string} feedback.strategy - Strategy used
   * @param {number} feedback.quality - 0-1 quality score
   * @param {string[]} [feedback.resultIds] - Result IDs
   * @param {string} [feedback.correction] - What correct answer was
   * @returns {{ adjusted: boolean, newWeights: Object }}
   * @throws {TypeError} On invalid inputs
   */
  recordRetrievalFeedback(agentId, feedback) {
    this._validateAgentId(agentId);
    if (!feedback || typeof feedback.query !== 'string' || feedback.query.length === 0) {
      throw new TypeError('feedback.query must be a non-empty string');
    }
    if (typeof feedback.quality !== 'number' || feedback.quality < 0 || feedback.quality > 1) {
      throw new TypeError('feedback.quality must be a number between 0 and 1');
    }
    if (feedback.strategy && !VALID_STRATEGIES.has(feedback.strategy)) {
      throw new TypeError(`feedback.strategy must be one of: ${[...VALID_STRATEGIES].join(', ')}`);
    }

    if (!this._feedbackLog.has(agentId)) {
      this._feedbackLog.set(agentId, []);
    }

    const log = this._feedbackLog.get(agentId);
    log.push({ ...feedback, timestamp: Date.now() });

    // Cap log size
    if (log.length > MAX_FEEDBACK_LOG) {
      log.splice(0, log.length - MAX_FEEDBACK_LOG);
    }

    const pattern = this._queryPattern(feedback.query);
    const adjustment = this._adjustStrategyWeights(pattern, feedback);
    this._stats.critiques++;
    if (adjustment.adjusted) this._stats.strategyAdjustments++;

    // Learn from failures
    if (feedback.quality < 0.3 && feedback.correction) {
      this.memory.learn(agentId,
        `Retrieval failure: "${feedback.query}" via ${feedback.strategy || 'unknown'}. Better: ${feedback.correction}`,
        { type: 'retrieval_failure', strategy: feedback.strategy }
      ).catch(() => {}); // fire-and-forget
    }

    this.emit('critique:recorded', { agentId, pattern, ...adjustment });
    return adjustment;
  }

  /**
   * Get learned optimal weights for a query pattern.
   * @param {string} query
   * @returns {{ vector: number, tree: number, keyword: number }}
   */
  getOptimalWeights(query) {
    if (typeof query !== 'string') return { vector: 0.4, tree: 0.4, keyword: 0.2 };
    const pattern = this._queryPattern(query);
    return this._learnedWeights.get(pattern) || { vector: 0.4, tree: 0.4, keyword: 0.2 };
  }

  /**
   * Retrieve with automatic quality critique and strategy retry.
   *
   * @param {string} agentId
   * @param {string} query
   * @param {Object} [options={}]
   * @returns {Promise<{ results: Object[], quality: number, retried: boolean, strategy: string }>}
   * @throws {Error} If no retriever is configured
   */
  async retrieveWithCritique(agentId, query, options = {}) {
    this._validateAgentId(agentId);
    if (!this.retriever) {
      throw new Error('No retriever configured. Pass config.retriever to constructor.');
    }
    if (typeof query !== 'string' || query.length === 0) {
      throw new TypeError('query must be a non-empty string');
    }

    const optimalWeights = this.getOptimalWeights(query);
    const strategy = options.strategy || 'hybrid';

    let results;
    try {
      results = await this.retriever.retrieve(query, { ...options, weights: optimalWeights });
    } catch (err) {
      this.emit('critique:error', { agentId, query, error: err.message });
      return { results: [], quality: 0, retried: false, strategy, error: err.message };
    }

    const quality = this._estimateResultQuality(query, results);

    // Auto-retry with tree strategy if quality is poor
    if (quality < 0.3 && this.llmProvider && strategy !== 'tree') {
      this.emit('critique:retry', { agentId, query, originalQuality: quality });

      try {
        const retryResults = await this.retriever.retrieve(query, { ...options, strategy: 'tree' });
        const retryQuality = this._estimateResultQuality(query, retryResults);

        if (retryQuality > quality) {
          this.recordRetrievalFeedback(agentId, {
            query, strategy: 'tree', quality: retryQuality,
            resultIds: retryResults.map(r => r.id || '').filter(Boolean)
          });
          return { results: retryResults, quality: retryQuality, retried: true, strategy: 'tree' };
        }
      } catch {
        // Retry failed; fall through to original results
      }
    }

    this.recordRetrievalFeedback(agentId, {
      query, strategy, quality,
      resultIds: results.map(r => r.id || '').filter(Boolean)
    });

    return { results, quality, retried: false, strategy };
  }

  // ═══════════════════════════════════════════════════════
  // 3. SKILL LEARNING
  // ═══════════════════════════════════════════════════════

  /**
   * Record a successful workflow as a reusable skill.
   *
   * @param {string} agentId
   * @param {Object} skill
   * @param {string} skill.name - Skill name (required)
   * @param {string} [skill.description] - What this skill does
   * @param {Object} [skill.trigger] - Activation conditions
   * @param {string[]} [skill.trigger.keywords] - Trigger keywords
   * @param {Object[]} [skill.steps] - Sequence of actions
   * @returns {Skill}
   * @throws {TypeError} If skill.name is missing
   */
  learnSkill(agentId, skill) {
    this._validateAgentId(agentId);
    if (!skill || typeof skill.name !== 'string' || skill.name.length === 0) {
      throw new TypeError('skill.name must be a non-empty string');
    }

    if (!this._skills.has(agentId)) {
      this._skills.set(agentId, new Map());
    }

    const agentSkills = this._skills.get(agentId);
    if (agentSkills.size >= MAX_SKILLS_PER_AGENT) {
      // Evict lowest success rate skill
      let worstId = null, worstRate = Infinity;
      for (const [id, s] of agentSkills) {
        if (s.successRate < worstRate) { worstRate = s.successRate; worstId = id; }
      }
      if (worstId) agentSkills.delete(worstId);
    }

    const entry = {
      id: skill.id || `skill_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`,
      name: skill.name,
      description: skill.description || '',
      trigger: skill.trigger || {},
      steps: Array.isArray(skill.steps) ? skill.steps : [],
      successRate: clamp(skill.successRate || 1.0, 0, 1),
      useCount: 0,
      learnedAt: Date.now(),
      lastUsed: null
    };

    agentSkills.set(entry.id, entry);

    // Also register in procedural memory (fire-and-forget)
    this.memory.registerTool(agentId, {
      name: entry.name,
      description: entry.description,
      schema: { trigger: entry.trigger, steps: entry.steps }
    }).catch(() => {});

    this._stats.skillsLearned++;
    this.emit('skill:learned', { agentId, skillId: entry.id, name: entry.name });
    return entry;
  }

  /**
   * Find applicable skills for a context.
   * @param {string} agentId
   * @param {string} context
   * @returns {Skill[]}
   */
  findApplicableSkills(agentId, context) {
    if (typeof context !== 'string') return [];
    const skills = this._skills.get(agentId);
    if (!skills) return [];

    const contextLower = context.toLowerCase();
    const matches = [];

    for (const [, skill] of skills) {
      const keywords = skill.trigger?.keywords;
      if (!Array.isArray(keywords) || keywords.length === 0) continue;

      const matchCount = keywords.filter(kw =>
        typeof kw === 'string' && contextLower.includes(kw.toLowerCase())
      ).length;

      if (matchCount > 0) {
        matches.push({ ...skill, matchScore: matchCount / keywords.length });
      }
    }

    return matches.sort((a, b) => (b.successRate * b.matchScore) - (a.successRate * a.matchScore));
  }

  /**
   * Record skill execution outcome.
   * @param {string} agentId
   * @param {string} skillId
   * @param {boolean} success
   */
  recordSkillOutcome(agentId, skillId, success) {
    const skill = this._skills.get(agentId)?.get(skillId);
    if (!skill) return;

    skill.useCount++;
    skill.lastUsed = Date.now();
    const alpha = 0.3;
    skill.successRate = clamp(alpha * (success ? 1 : 0) + (1 - alpha) * skill.successRate, 0, 1);

    this.emit('skill:outcome', { agentId, skillId, success, successRate: skill.successRate });
  }

  /**
   * List all skills for an agent.
   * @param {string} agentId
   * @returns {Skill[]}
   */
  listSkills(agentId) {
    const skills = this._skills.get(agentId);
    return skills ? Array.from(skills.values()) : [];
  }

  // ═══════════════════════════════════════════════════════
  // 4. KNOWLEDGE GRAPH
  // ═══════════════════════════════════════════════════════

  /**
   * Extract knowledge triples from text.
   *
   * @param {string} agentId
   * @param {string} text
   * @param {string} [source='unknown']
   * @returns {Promise<KnowledgeTriple[]>}
   */
  async extractKnowledge(agentId, text, source = 'unknown') {
    this._validateAgentId(agentId);
    if (typeof text !== 'string' || text.length < MIN_CONTENT_LENGTH) return [];

    if (!this._knowledgeGraph.has(agentId)) {
      this._knowledgeGraph.set(agentId, []);
    }

    let triples;
    if (this.llmProvider) {
      try {
        const response = await this.llmProvider.complete(
          `Extract entity-relationship triples from this text. Return ONLY a JSON array with {subject, predicate, object} objects. No markdown.\n\nText: "${text.substring(0, 2000)}"\n\nTriples:`,
          { maxTokens: 500 }
        );
        const parsed = safeJsonParse(response, null);
        triples = Array.isArray(parsed) ? parsed.filter(t =>
          t && typeof t.subject === 'string' && typeof t.predicate === 'string' && typeof t.object === 'string'
        ) : this._simpleTripleExtraction(text);
      } catch {
        triples = this._simpleTripleExtraction(text);
      }
    } else {
      triples = this._simpleTripleExtraction(text);
    }

    const graph = this._knowledgeGraph.get(agentId);

    // Cap graph size
    const spaceLeft = MAX_TRIPLES_PER_AGENT - graph.length;
    const toAdd = triples.slice(0, Math.max(0, spaceLeft));

    const enriched = toAdd.map(t => ({
      subject: String(t.subject).substring(0, 200),
      predicate: String(t.predicate).substring(0, 100),
      object: String(t.object).substring(0, 200),
      confidence: 0.7,
      source: String(source).substring(0, 200),
      extractedAt: Date.now()
    }));

    graph.push(...enriched);
    this._stats.triplesExtracted += enriched.length;
    this.emit('knowledge:extracted', { agentId, count: enriched.length });
    return enriched;
  }

  /**
   * Query the knowledge graph.
   * @param {string} agentId
   * @param {Object} [query={}] - Filter by subject, predicate, object
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
   * Get graph summary.
   * @param {string} agentId
   * @returns {{ entities: string[], relationships: Object[], tripleCount: number }}
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
  // 5. REFLECTION
  // ═══════════════════════════════════════════════════════

  /**
   * Generate a self-reflection on recent performance.
   *
   * @param {string} agentId
   * @returns {Promise<string>}
   */
  async reflect(agentId) {
    this._validateAgentId(agentId);

    const feedbackLog = this._feedbackLog.get(agentId) || [];
    const recentFeedback = feedbackLog.slice(-20);
    const memStats = this.memory.getStats(agentId);

    const context = {
      recentRetrievals: recentFeedback.length,
      avgQuality: recentFeedback.length > 0
        ? recentFeedback.reduce((s, f) => s + (f.quality || 0), 0) / recentFeedback.length
        : 0,
      failedQueries: recentFeedback.filter(f => (f.quality || 0) < 0.3).map(f => f.query),
      memoryStats: memStats,
      skillCount: this._skills.get(agentId)?.size || 0,
      knowledgeTriples: (this._knowledgeGraph.get(agentId) || []).length
    };

    let reflection;
    if (this.llmProvider) {
      try {
        reflection = await this.llmProvider.complete(
          `You are an AI agent reviewing your performance. Generate 2-3 specific, actionable improvement insights.\n\nData: ${JSON.stringify(context)}\nFailed queries: ${context.failedQueries.join(', ') || 'none'}`,
          { maxTokens: 300 }
        );
      } catch {
        reflection = this._generateSimpleReflection(context);
      }
    } else {
      reflection = this._generateSimpleReflection(context);
    }

    if (!this._reflections.has(agentId)) {
      this._reflections.set(agentId, []);
    }

    const reflectionLog = this._reflections.get(agentId);
    reflectionLog.push({ reflection, context, timestamp: Date.now() });

    // Cap reflections
    if (reflectionLog.length > MAX_REFLECTIONS_PER_AGENT) {
      reflectionLog.splice(0, reflectionLog.length - MAX_REFLECTIONS_PER_AGENT);
    }

    // Store in episodic memory (fire-and-forget)
    this.memory.remember(agentId, {
      content: `Self-reflection: ${reflection}`,
      importance: 0.7,
      metadata: { type: 'reflection' }
    }).catch(() => {});

    this._stats.reflections++;
    this.emit('reflection:complete', { agentId, reflection });
    return reflection;
  }

  /**
   * Get reflection history.
   * @param {string} agentId
   * @param {number} [limit=10]
   * @returns {Array}
   */
  getReflections(agentId, limit = 10) {
    return (this._reflections.get(agentId) || []).slice(-Math.max(1, limit));
  }

  // ═══════════════════════════════════════════════════════
  // LIFECYCLE & STATS
  // ═══════════════════════════════════════════════════════

  /** Get learning statistics. */
  getStats() {
    return {
      ...this._stats,
      learnedPatterns: this._learnedWeights.size,
      feedbackEntries: Array.from(this._feedbackLog.values()).reduce((s, l) => s + l.length, 0)
    };
  }

  /** Stop auto-consolidation timer. */
  stop() {
    if (this._consolidationTimer) {
      clearInterval(this._consolidationTimer);
      this._consolidationTimer = null;
    }
  }

  /** Export all learned state for persistence. */
  serialize() {
    const skills = {};
    for (const [agentId, m] of this._skills) skills[agentId] = Array.from(m.values());
    const graph = {};
    for (const [agentId, t] of this._knowledgeGraph) graph[agentId] = t;
    const reflections = {};
    for (const [agentId, r] of this._reflections) reflections[agentId] = r;

    return {
      _version: 1, _engine: 'FusionPact',
      learnedWeights: Object.fromEntries(this._learnedWeights),
      skills, knowledgeGraph: graph, reflections, stats: { ...this._stats }
    };
  }

  /** Import previously exported state. */
  importState(data) {
    if (!data || typeof data !== 'object') return;
    if (data.learnedWeights && typeof data.learnedWeights === 'object') {
      for (const [k, v] of Object.entries(data.learnedWeights)) {
        if (v && typeof v.vector === 'number') this._learnedWeights.set(k, v);
      }
    }
    if (data.skills && typeof data.skills === 'object') {
      for (const [agentId, list] of Object.entries(data.skills)) {
        if (!Array.isArray(list)) continue;
        const m = new Map();
        for (const s of list) { if (s && s.id) m.set(s.id, s); }
        this._skills.set(agentId, m);
      }
    }
    if (data.knowledgeGraph && typeof data.knowledgeGraph === 'object') {
      for (const [agentId, triples] of Object.entries(data.knowledgeGraph)) {
        if (Array.isArray(triples)) this._knowledgeGraph.set(agentId, triples);
      }
    }
    if (data.stats && typeof data.stats === 'object') {
      Object.assign(this._stats, data.stats);
    }
  }

  // ─── Private Helpers ──────────────────────────────────

  /** @private */
  _validateAgentId(agentId) {
    if (typeof agentId !== 'string' || agentId.length === 0) {
      throw new TypeError('agentId must be a non-empty string');
    }
  }

  /** @private */
  _startAutoConsolidation() {
    this._consolidationTimer = setInterval(async () => {
      for (const agentId of this.memory._agentStores.keys()) {
        try { await this.consolidate(agentId); }
        catch (e) { this.emit('consolidation:error', { agentId, error: e.message }); }
      }
    }, this.consolidation.intervalMs);
    // Allow process to exit even if timer is running
    if (this._consolidationTimer.unref) this._consolidationTimer.unref();
  }

  /** @private */
  _adjustStrategyWeights(pattern, feedback) {
    if (!this._learnedWeights.has(pattern)) {
      this._learnedWeights.set(pattern, { vector: 0.4, tree: 0.4, keyword: 0.2 });
    }
    const w = this._learnedWeights.get(pattern);
    const alpha = 0.1;
    const strategy = feedback.strategy || 'hybrid';
    const q = feedback.quality;

    if (q > 0.7) {
      if (strategy === 'vector' || strategy === 'hybrid') w.vector = Math.min(0.8, w.vector + alpha * q);
      if (strategy === 'tree' || strategy === 'hybrid') w.tree = Math.min(0.8, w.tree + alpha * q);
      if (strategy === 'keyword' || strategy === 'hybrid') w.keyword = Math.min(0.5, w.keyword + alpha * q * 0.5);
    } else if (q < 0.3) {
      if (strategy === 'vector') w.vector = Math.max(0.1, w.vector - alpha);
      if (strategy === 'tree') w.tree = Math.max(0.1, w.tree - alpha);
      if (strategy === 'keyword') w.keyword = Math.max(0.05, w.keyword - alpha);
    }

    const total = w.vector + w.tree + w.keyword;
    if (total > 0) { w.vector /= total; w.tree /= total; w.keyword /= total; }

    return { adjusted: true, newWeights: { ...w } };
  }

  /** @private */
  _queryPattern(query) {
    return (query || '').toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/)
      .filter(w => w.length > 3).sort().slice(0, 4).join('_') || 'generic';
  }

  /** @private */
  _estimateResultQuality(query, results) {
    if (!Array.isArray(results) || results.length === 0) return 0;
    const topScore = results[0]?.score || 0;
    const terms = (query || '').toLowerCase().split(/\s+/).filter(t => t.length > 2);
    let overlap = 0;
    for (const r of results.slice(0, 3)) {
      const c = (r.content || '').toLowerCase();
      for (const t of terms) { if (c.includes(t)) overlap++; }
    }
    const overlapScore = terms.length > 0 ? Math.min(1, overlap / (terms.length * 2)) : 0.5;
    return clamp(topScore * 0.4 + overlapScore * 0.4 + Math.min(results.length / 5, 1) * 0.2, 0, 1);
  }

  /** @private */
  _textSimilarity(a, b) {
    if (!a || !b) return 0;
    const wA = new Set(a.toLowerCase().split(/\s+/));
    const wB = new Set(b.toLowerCase().split(/\s+/));
    const inter = new Set([...wA].filter(w => wB.has(w)));
    const union = new Set([...wA, ...wB]);
    return union.size > 0 ? inter.size / union.size : 0;
  }

  /** @private */
  _simpleTripleExtraction(text) {
    const triples = [];
    const sentences = (text || '').split(/[.!?]+/).filter(s => s.trim().length > 10);
    for (const s of sentences.slice(0, 10)) {
      const m1 = s.match(/(.{3,30})\s+(?:is|are|was|were)\s+(.{3,50})/i);
      if (m1) triples.push({ subject: m1[1].trim(), predicate: 'is', object: m1[2].trim().substring(0, 50) });
      const m2 = s.match(/(.{3,30})\s+(covers|requires|includes|contains|provides|manages)\s+(.{3,50})/i);
      if (m2) triples.push({ subject: m2[1].trim(), predicate: m2[2].trim(), object: m2[3].trim().substring(0, 50) });
    }
    return triples;
  }

  /** @private */
  _generateSimpleReflection(ctx) {
    const insights = [];
    if (ctx.avgQuality < 0.5) insights.push(`Retrieval quality is low (${(ctx.avgQuality * 100).toFixed(0)}%). Consider tree-based retrieval for structured documents.`);
    if (ctx.failedQueries.length > 3) insights.push(`${ctx.failedQueries.length} recent failures. May need additional knowledge ingestion.`);
    if ((ctx.memoryStats?.semantic || 0) < 5) insights.push('Semantic memory is sparse. More domain knowledge would help.');
    if (ctx.skillCount === 0) insights.push('No skills learned. Recording successful workflows would enable automation.');
    return insights.length > 0 ? insights.join(' ') : `Stable: ${ctx.recentRetrievals} retrievals at ${(ctx.avgQuality * 100).toFixed(0)}% quality.`;
  }
}

module.exports = { RecursiveLearningEngine };
