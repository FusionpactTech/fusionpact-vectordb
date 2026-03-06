/**
 * @fileoverview AgentOrchestrator — Multi-Agent Coordination Layer
 * 
 * Manages multiple AI agents with isolated memory, shared knowledge pools,
 * message routing, task delegation, and collaborative retrieval.
 * 
 * Designed for frameworks like CrewAI, AutoGen, LangGraph, and custom
 * multi-agent architectures.
 * 
 * @module orchestration/AgentOrchestrator
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 * @see https://github.com/FusionpactTech/fusionpact-vectordb
 */

'use strict';

const { EventEmitter } = require('events');

/**
 * @typedef {Object} AgentConfig
 * @property {string} agentId - Unique agent identifier
 * @property {string} name - Human-readable agent name
 * @property {string} [role] - Agent's role description
 * @property {string[]} [capabilities=[]] - List of capabilities/skills
 * @property {Object} [memoryConfig={}] - Agent-specific memory configuration
 * @property {Object} [metadata={}] - Additional agent metadata
 */

/**
 * @typedef {Object} AgentMessage
 * @property {string} from - Sender agent ID
 * @property {string} to - Recipient agent ID (or '*' for broadcast)
 * @property {string} type - Message type: 'task', 'result', 'query', 'knowledge', 'status'
 * @property {*} payload - Message content
 * @property {number} timestamp - Message timestamp
 * @property {string} [correlationId] - For request-response tracking
 */

class AgentOrchestrator extends EventEmitter {
  /**
   * Create a multi-agent orchestrator.
   * 
   * @param {Object} config
   * @param {import('../core/FusionEngine')} config.engine - FusionEngine instance
   * @param {import('../memory/AgentMemory')} config.memory - AgentMemory instance
   * @param {import('../retrieval/HybridRetriever')} [config.retriever] - Shared retriever
   * 
   * @example
   * const orchestrator = new AgentOrchestrator({ engine, memory, retriever });
   * 
   * orchestrator.registerAgent({ agentId: 'researcher', name: 'Research Agent', role: 'Find information' });
   * orchestrator.registerAgent({ agentId: 'analyst', name: 'Analysis Agent', role: 'Analyze data' });
   * 
   * // Agents can communicate
   * await orchestrator.send({ from: 'researcher', to: 'analyst', type: 'result', payload: { findings: [...] } });
   */
  constructor(config) {
    super();
    this.engine = config.engine;
    this.memory = config.memory;
    this.retriever = config.retriever || null;

    /** @private */
    this._agents = new Map();
    /** @private */
    this._messageQueues = new Map();
    /** @private */
    this._messageHandlers = new Map();
    /** @private */
    this._messageLog = [];
    /** @private */
    this._taskCounter = 0;
  }

  /**
   * Register an agent with the orchestrator.
   * 
   * @param {AgentConfig} config
   * @returns {AgentConfig}
   */
  registerAgent(config) {
    if (this._agents.has(config.agentId)) {
      throw new Error(`Agent "${config.agentId}" already registered`);
    }

    const agent = {
      ...config,
      capabilities: config.capabilities || [],
      metadata: config.metadata || {},
      registeredAt: Date.now(),
      status: 'active',
      messageCount: 0
    };

    this._agents.set(config.agentId, agent);
    this._messageQueues.set(config.agentId, []);
    this.emit('agent:registered', { agentId: config.agentId, name: config.name });
    return agent;
  }

  /**
   * Unregister an agent.
   * @param {string} agentId
   * @param {Object} [options={}]
   * @param {boolean} [options.preserveMemory=true] - Keep agent's memories
   * @returns {boolean}
   */
  unregisterAgent(agentId, options = {}) {
    if (!options.preserveMemory) {
      this.memory.forget(agentId, { type: 'all' });
    }
    this._messageQueues.delete(agentId);
    this._messageHandlers.delete(agentId);
    const existed = this._agents.delete(agentId);
    if (existed) this.emit('agent:unregistered', { agentId });
    return existed;
  }

  /**
   * List all registered agents.
   * @returns {AgentConfig[]}
   */
  listAgents() {
    return Array.from(this._agents.values());
  }

  /**
   * Get agent info.
   * @param {string} agentId
   * @returns {AgentConfig|null}
   */
  getAgent(agentId) {
    return this._agents.get(agentId) || null;
  }

  /**
   * Send a message between agents.
   * 
   * @param {AgentMessage} message
   * @returns {AgentMessage} The sent message with timestamp and ID
   */
  async send(message) {
    const msg = {
      ...message,
      id: `msg_${Date.now()}_${this._taskCounter++}`,
      timestamp: Date.now()
    };

    this._messageLog.push(msg);

    if (msg.to === '*') {
      // Broadcast to all agents except sender
      for (const [agentId] of this._agents) {
        if (agentId !== msg.from) {
          this._deliver(agentId, msg);
        }
      }
    } else {
      this._deliver(msg.to, msg);
    }

    // Auto-store significant messages in episodic memory
    if (['task', 'result', 'knowledge'].includes(msg.type)) {
      await this.memory.remember(msg.from, {
        content: `Sent ${msg.type} to ${msg.to}: ${JSON.stringify(msg.payload).substring(0, 500)}`,
        importance: msg.type === 'result' ? 0.7 : 0.5,
        metadata: { messageType: msg.type, to: msg.to, correlationId: msg.correlationId }
      });
    }

    this.emit('message:sent', msg);
    return msg;
  }

  /**
   * Register a message handler for an agent.
   * 
   * @param {string} agentId
   * @param {function(AgentMessage): Promise<void>} handler
   */
  onMessage(agentId, handler) {
    this._messageHandlers.set(agentId, handler);
  }

  /**
   * Get pending messages for an agent.
   * @param {string} agentId
   * @returns {AgentMessage[]}
   */
  getMessages(agentId) {
    const queue = this._messageQueues.get(agentId) || [];
    this._messageQueues.set(agentId, []);
    return queue;
  }

  /**
   * Delegate a task to the best-suited agent based on capabilities.
   * 
   * @param {string} fromAgentId - Delegating agent
   * @param {string} taskDescription - What needs to be done
   * @param {Object} [options={}]
   * @param {string[]} [options.requiredCapabilities] - Required agent capabilities
   * @param {string} [options.preferredAgent] - Preferred agent ID
   * @returns {Promise<AgentMessage>} Task delegation message
   * 
   * @example
   * await orchestrator.delegate('coordinator', 'Analyze the safety report', {
   *   requiredCapabilities: ['analysis', 'safety']
   * });
   */
  async delegate(fromAgentId, taskDescription, options = {}) {
    const targetAgent = options.preferredAgent
      ? this._agents.get(options.preferredAgent)
      : this._findBestAgent(options.requiredCapabilities || []);

    if (!targetAgent) {
      throw new Error(`No suitable agent found for capabilities: ${options.requiredCapabilities?.join(', ')}`);
    }

    return this.send({
      from: fromAgentId,
      to: targetAgent.agentId,
      type: 'task',
      payload: {
        description: taskDescription,
        requiredCapabilities: options.requiredCapabilities,
        delegatedAt: Date.now()
      },
      correlationId: `task_${this._taskCounter++}`
    });
  }

  /**
   * Collaborative retrieval — search using all agents' knowledge.
   * 
   * @param {string} query - Search query
   * @param {Object} [options={}]
   * @param {string[]} [options.agentIds] - Specific agents to search
   * @param {number} [options.topK=10] - Results per agent
   * @returns {Promise<Object>} Results keyed by agent ID
   */
  async collaborativeRecall(query, options = {}) {
    const agentIds = options.agentIds || Array.from(this._agents.keys());
    const topK = options.topK || 10;

    const results = {};
    for (const agentId of agentIds) {
      results[agentId] = await this.memory.recall(agentId, query, {
        topK,
        includeShared: true
      });
    }

    // Also search shared pool
    if (this.retriever) {
      try {
        results._shared = await this.retriever.retrieve(query, {
          topK,
          strategy: 'hybrid'
        });
      } catch (e) {
        results._shared = [];
      }
    }

    return results;
  }

  /**
   * Get orchestration statistics.
   * @returns {Object}
   */
  getStats() {
    return {
      agents: this._agents.size,
      totalMessages: this._messageLog.length,
      agentStats: Array.from(this._agents.values()).map(a => ({
        agentId: a.agentId,
        name: a.name,
        status: a.status,
        messageCount: a.messageCount,
        memoryStats: this.memory.getStats(a.agentId)
      }))
    };
  }

  /** @private */
  _deliver(agentId, message) {
    const handler = this._messageHandlers.get(agentId);
    if (handler) {
      handler(message).catch(err => {
        this.emit('message:error', { agentId, error: err.message });
      });
    } else {
      const queue = this._messageQueues.get(agentId);
      if (queue) queue.push(message);
    }

    const agent = this._agents.get(agentId);
    if (agent) agent.messageCount++;
  }

  /** @private */
  _findBestAgent(requiredCapabilities) {
    let bestAgent = null;
    let bestScore = -1;

    for (const [, agent] of this._agents) {
      if (agent.status !== 'active') continue;

      const score = requiredCapabilities.filter(cap =>
        agent.capabilities.includes(cap)
      ).length;

      if (score > bestScore) {
        bestScore = score;
        bestAgent = agent;
      }
    }

    return bestAgent;
  }
}

module.exports = { AgentOrchestrator };
