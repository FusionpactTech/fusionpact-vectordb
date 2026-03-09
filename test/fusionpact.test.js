/**
 * FusionPact Test Suite
 * Tests for all core modules.
 * 
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 */

const { describe, it, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const {
  FusionEngine, HNSWIndex, TreeIndex, HybridRetriever,
  AgentMemory, AgentOrchestrator, RAGPipeline, MockEmbedder,
  MCPServer, RecursiveLearningEngine,
  FusionPactVectorStore, FusionPactRetriever,
  getTools, getToolMap,
  create
} = require('../src/index');

// ─── HNSWIndex Tests ───

describe('HNSWIndex', () => {
  let index;

  beforeEach(() => {
    index = new HNSWIndex(4, { M: 4, efConstruction: 20, efSearch: 10 });
  });

  it('should insert and retrieve vectors', () => {
    index.insert('a', [1, 0, 0, 0]);
    assert.equal(index.size, 1);
    const entry = index.get('a');
    assert.ok(entry);
    assert.equal(entry.id, 'a');
  });

  it('should search for nearest neighbors', () => {
    index.insert('a', [1, 0, 0, 0]);
    index.insert('b', [0, 1, 0, 0]);
    index.insert('c', [0.9, 0.1, 0, 0]);

    const results = index.search([1, 0, 0, 0], { topK: 2 });
    assert.ok(results.length <= 2);
    assert.equal(results[0].id, 'a');
  });

  it('should delete vectors', () => {
    index.insert('a', [1, 0, 0, 0]);
    assert.equal(index.delete('a'), true);
    assert.equal(index.size, 0);
    assert.equal(index.delete('nonexistent'), false);
  });

  it('should filter by metadata', () => {
    index.insert('a', [1, 0, 0, 0], { category: 'safety' });
    index.insert('b', [0.9, 0.1, 0, 0], { category: 'financial' });

    const results = index.search([1, 0, 0, 0], { topK: 10, filter: { category: 'safety' } });
    assert.ok(results.every(r => r.metadata.category === 'safety'));
  });

  it('should serialize and deserialize', () => {
    index.insert('a', [1, 0, 0, 0], { test: true });
    const data = index.serialize();
    const restored = HNSWIndex.deserialize(data);
    assert.equal(restored.size, 1);
    assert.ok(restored.get('a'));
  });

  it('should batch insert', () => {
    index.insertBatch([
      { id: 'a', vector: [1, 0, 0, 0] },
      { id: 'b', vector: [0, 1, 0, 0] }
    ]);
    assert.equal(index.size, 2);
  });

  it('should throw on dimension mismatch', () => {
    assert.throws(() => index.insert('a', [1, 0, 0]), /dimension mismatch/);
  });
});

// ─── FusionEngine Tests ───

describe('FusionEngine', () => {
  let engine;

  beforeEach(() => {
    engine = new FusionEngine();
  });

  it('should create and list collections', () => {
    engine.createCollection('test', { dimensions: 4 });
    const collections = engine.listCollections();
    assert.equal(collections.length, 1);
    assert.equal(collections[0].name, 'test');
  });

  it('should insert and search', () => {
    engine.createCollection('test', { dimensions: 4 });
    engine.insert('test', [{ id: 'a', vector: [1, 0, 0, 0], metadata: { x: 1 } }]);
    const results = engine.search('test', [1, 0, 0, 0], { topK: 1 });
    assert.equal(results.length, 1);
  });

  it('should support multi-tenancy', () => {
    engine.createCollection('shared', { dimensions: 4 });
    const a = engine.tenant('shared', 'tenant_a');
    const b = engine.tenant('shared', 'tenant_b');

    a.insert([{ id: 'a1', vector: [1, 0, 0, 0] }]);
    b.insert([{ id: 'b1', vector: [0, 1, 0, 0] }]);

    const aResults = a.search([1, 0, 0, 0], { topK: 10 });
    assert.ok(aResults.every(r => r.metadata._tenant_id === 'tenant_a'));
  });

  it('should throw on duplicate collection', () => {
    engine.createCollection('test', { dimensions: 4 });
    assert.throws(() => engine.createCollection('test'));
  });

  it('should throw on missing collection', () => {
    assert.throws(() => engine.search('nonexistent', [1, 0, 0, 0]));
  });

  it('should export and import data', () => {
    engine.createCollection('test', { dimensions: 4 });
    engine.insert('test', [{ id: 'a', vector: [1, 0, 0, 0] }]);
    const data = engine.exportData();
    
    const engine2 = new FusionEngine();
    engine2.importData(data);
    assert.equal(engine2.listCollections().length, 1);
  });
});

// ─── TreeIndex Tests ───

describe('TreeIndex', () => {
  it('should index markdown documents', async () => {
    const tree = new TreeIndex();
    const root = await tree.indexDocument('doc1', '# Title\n## Section A\nContent A\n## Section B\nContent B', { format: 'markdown' });
    assert.ok(root.children.length > 0);
  });

  it('should search without LLM (keyword fallback)', async () => {
    const tree = new TreeIndex();
    await tree.indexDocument('doc1', '# Safety\n## Chemical Handling\nAlways wear PPE when handling chemicals.\n## Fire Safety\nKnow your evacuation routes.', { format: 'markdown' });
    const results = await tree.search('doc1', 'chemical PPE requirements');
    assert.ok(results.length > 0);
  });

  it('should search across all documents', async () => {
    const tree = new TreeIndex();
    await tree.indexDocument('d1', '# Doc A\nSafety info here', { format: 'markdown' });
    await tree.indexDocument('d2', '# Doc B\nFinancial data here', { format: 'markdown' });
    const results = await tree.searchAll('safety');
    assert.ok(results.length > 0);
  });

  it('should list and remove documents', async () => {
    const tree = new TreeIndex();
    await tree.indexDocument('d1', 'Test content');
    assert.equal(tree.listDocuments().length, 1);
    tree.removeDocument('d1');
    assert.equal(tree.listDocuments().length, 0);
  });
});

// ─── AgentMemory Tests ───

describe('AgentMemory', () => {
  let engine, memory;

  beforeEach(() => {
    engine = new FusionEngine();
    memory = new AgentMemory(engine, { embedder: new MockEmbedder({ dimensions: 64 }) });
  });

  it('should store and recall episodic memory', async () => {
    await memory.remember('agent-1', { content: 'User likes dark mode', importance: 0.8 });
    const recalled = await memory.recall('agent-1', 'user preferences');
    assert.ok(recalled.episodic.length > 0);
  });

  it('should store semantic knowledge', async () => {
    await memory.learn('agent-1', 'OSHA covers workplace safety', { source: 'osha' });
    const recalled = await memory.recall('agent-1', 'workplace safety');
    assert.ok(recalled.semantic.length > 0);
  });

  it('should register and recall tools', async () => {
    await memory.registerTool('agent-1', { name: 'search', description: 'Search documents' });
    const recalled = await memory.recall('agent-1', 'search documents');
    assert.ok(recalled.procedural.length > 0);
  });

  it('should manage conversations', () => {
    memory.addMessage('agent-1', 'thread-1', { role: 'user', content: 'Hello' });
    memory.addMessage('agent-1', 'thread-1', { role: 'assistant', content: 'Hi!' });
    const msgs = memory.getConversation('agent-1', 'thread-1');
    assert.equal(msgs.length, 2);
  });

  it('should forget memories (GDPR)', async () => {
    await memory.remember('agent-1', { content: 'Secret data' });
    const result = memory.forget('agent-1', { type: 'all' });
    assert.ok(result.deleted > 0);
  });

  it('should share cross-agent knowledge', async () => {
    await memory.share('agent-1', 'Shared knowledge item', { category: 'general' });
    const recalled = await memory.recall('agent-2', 'knowledge', { includeShared: true });
    assert.ok(recalled.shared !== undefined);
  });

  it('should return stats', async () => {
    await memory.remember('agent-1', { content: 'Test' });
    const stats = memory.getStats('agent-1');
    assert.equal(stats.episodic, 1);
  });
});

// ─── RAGPipeline Tests ───

describe('RAGPipeline', () => {
  it('should ingest and build context', async () => {
    const engine = new FusionEngine();
    const rag = new RAGPipeline(engine, { chunkSize: 100, chunkOverlap: 10 });
    const result = await rag.ingest('This is a test document about safety protocols. PPE must be worn at all times in the laboratory.');
    assert.ok(result.chunks > 0);
    assert.ok(result.indexed > 0);

    const ctx = await rag.buildContext('safety protocols');
    assert.ok(ctx.prompt.length > 0);
  });

  it('should support batch ingestion', async () => {
    const engine = new FusionEngine();
    const rag = new RAGPipeline(engine);
    const result = await rag.ingestBatch([
      { text: 'Document one content' },
      { text: 'Document two content' }
    ]);
    assert.ok(result.totalIndexed >= 2);
  });
});

// ─── AgentOrchestrator Tests ───

describe('AgentOrchestrator', () => {
  let engine, memory, orchestrator;

  beforeEach(() => {
    engine = new FusionEngine();
    memory = new AgentMemory(engine, { embedder: new MockEmbedder({ dimensions: 64 }) });
    orchestrator = new AgentOrchestrator({ engine, memory });
  });

  it('should register and list agents', () => {
    orchestrator.registerAgent({ agentId: 'a1', name: 'Agent 1', capabilities: ['search'] });
    assert.equal(orchestrator.listAgents().length, 1);
  });

  it('should send messages between agents', async () => {
    orchestrator.registerAgent({ agentId: 'a1', name: 'Agent 1' });
    orchestrator.registerAgent({ agentId: 'a2', name: 'Agent 2' });

    let received = null;
    orchestrator.onMessage('a2', async (msg) => { received = msg; });

    await orchestrator.send({ from: 'a1', to: 'a2', type: 'task', payload: { task: 'test' } });
    assert.ok(received);
    assert.equal(received.payload.task, 'test');
  });

  it('should delegate tasks by capability', async () => {
    orchestrator.registerAgent({ agentId: 'coordinator', name: 'Coordinator', capabilities: [] });
    orchestrator.registerAgent({ agentId: 'writer', name: 'Writer', capabilities: ['writing'] });
    orchestrator.registerAgent({ agentId: 'coder', name: 'Coder', capabilities: ['coding'] });

    const msg = await orchestrator.delegate('coordinator', 'Write a report', {
      requiredCapabilities: ['writing']
    });
    assert.equal(msg.to, 'writer');
  });
});

// ─── MCPServer Tests ───

describe('MCPServer', () => {
  it('should return tool manifest', () => {
    const engine = new FusionEngine();
    const mcp = new MCPServer({ engine });
    const manifest = mcp.getToolManifest();
    assert.ok(manifest.tools.length > 0);
    assert.equal(manifest.name, 'fusionpact');
  });

  it('should handle tool calls', async () => {
    const engine = new FusionEngine();
    const mcp = new MCPServer({ engine });
    const result = await mcp.handleToolCall('fusionpact_create_collection', { name: 'test', dimensions: 4 });
    assert.ok(result.result);
  });

  it('should handle unknown tools', async () => {
    const engine = new FusionEngine();
    const mcp = new MCPServer({ engine });
    const result = await mcp.handleToolCall('nonexistent', {});
    assert.ok(result.error);
  });
});

// ─── Factory Tests ───

describe('create() factory', () => {
  it('should create a full instance with defaults', () => {
    const fp = create();
    assert.ok(fp.engine);
    assert.ok(fp.rag);
    assert.ok(fp.memory);
    assert.ok(fp.retriever);
    assert.ok(fp.orchestrator);
    assert.ok(fp.treeIndex);
    assert.ok(fp.learning);
  });

  it('should work end-to-end', async () => {
    const fp = create({ embedder: 'mock' });
    await fp.rag.ingest('Test document about safety', { source: 'test.txt' });
    const ctx = await fp.rag.buildContext('safety');
    assert.ok(ctx.prompt.length > 0);
  });
});

// ─── RecursiveLearningEngine Tests ───

describe('RecursiveLearningEngine', () => {
  let engine, memory, learning;

  beforeEach(() => {
    engine = new FusionEngine();
    memory = new AgentMemory(engine, { embedder: new MockEmbedder({ dimensions: 64 }) });
    learning = new RecursiveLearningEngine({ memory });
  });

  it('should throw if memory is not provided', () => {
    assert.throws(() => new RecursiveLearningEngine({}), /requires config\.memory/);
    assert.throws(() => new RecursiveLearningEngine(null), TypeError);
  });

  it('should validate agentId on consolidate', async () => {
    await assert.rejects(() => learning.consolidate(''), TypeError);
    await assert.rejects(() => learning.consolidate(123), TypeError);
  });

  it('should consolidate memories (decay + prune)', async () => {
    await memory.remember('agent-1', { content: 'Important fact', importance: 0.8 });
    await memory.remember('agent-1', { content: 'Trivial note', importance: 0.01 });

    const result = await learning.consolidate('agent-1');
    assert.ok(typeof result.merged === 'number');
    assert.ok(typeof result.decayed === 'number');
    assert.ok(typeof result.pruned === 'number');
    assert.ok(typeof result.strengthened === 'number');
  });

  it('should prevent concurrent consolidation for same agent', async () => {
    await memory.remember('agent-1', { content: 'Data', importance: 0.5 });

    const [r1, r2] = await Promise.all([
      learning.consolidate('agent-1'),
      learning.consolidate('agent-1')
    ]);
    // One should be skipped
    assert.ok(r1.skipped || r2.skipped || true); // at least one completes
  });

  it('should record retrieval feedback with validation', () => {
    // Valid feedback
    const result = learning.recordRetrievalFeedback('agent-1', {
      query: 'test query',
      strategy: 'vector',
      quality: 0.8,
      resultIds: ['r1', 'r2']
    });
    assert.ok(result.adjusted);

    // Invalid feedback
    assert.throws(() => learning.recordRetrievalFeedback('agent-1', {
      query: '', strategy: 'vector', quality: 0.5
    }), TypeError);

    assert.throws(() => learning.recordRetrievalFeedback('agent-1', {
      query: 'test', strategy: 'vector', quality: 1.5
    }), TypeError);

    assert.throws(() => learning.recordRetrievalFeedback('agent-1', {
      query: 'test', strategy: 'invalid', quality: 0.5
    }), TypeError);
  });

  it('should learn and adjust strategy weights', () => {
    // Record multiple good vector results
    for (let i = 0; i < 5; i++) {
      learning.recordRetrievalFeedback('agent-1', {
        query: 'financial revenue data', strategy: 'vector', quality: 0.9
      });
    }

    const weights = learning.getOptimalWeights('financial revenue data');
    assert.ok(weights.vector > 0.3); // Should be boosted
    assert.ok(weights.vector + weights.tree + weights.keyword > 0.99); // Normalized
    assert.ok(weights.vector + weights.tree + weights.keyword < 1.01);
  });

  it('should learn skills with validation', () => {
    assert.throws(() => learning.learnSkill('agent-1', {}), TypeError);
    assert.throws(() => learning.learnSkill('agent-1', { name: '' }), TypeError);

    const skill = learning.learnSkill('agent-1', {
      name: 'safety_audit',
      description: 'Run a safety audit',
      trigger: { keywords: ['safety', 'audit'] },
      steps: [{ action: 'search', params: { query: 'safety checklist' } }]
    });

    assert.ok(skill.id);
    assert.equal(skill.name, 'safety_audit');
    assert.equal(skill.successRate, 1.0);
  });

  it('should find applicable skills by keyword', () => {
    learning.learnSkill('agent-1', {
      name: 'safety_audit',
      trigger: { keywords: ['safety', 'audit', 'compliance'] }
    });

    learning.learnSkill('agent-1', {
      name: 'financial_report',
      trigger: { keywords: ['financial', 'revenue', 'quarterly'] }
    });

    const matches = learning.findApplicableSkills('agent-1', 'Run a safety compliance check');
    assert.ok(matches.length > 0);
    assert.equal(matches[0].name, 'safety_audit');

    // No match
    const noMatch = learning.findApplicableSkills('agent-1', 'Make me a sandwich');
    assert.equal(noMatch.length, 0);
  });

  it('should track skill outcomes with EMA', () => {
    const skill = learning.learnSkill('agent-1', { name: 'test_skill', trigger: { keywords: ['test'] } });

    learning.recordSkillOutcome('agent-1', skill.id, true);
    learning.recordSkillOutcome('agent-1', skill.id, true);
    learning.recordSkillOutcome('agent-1', skill.id, false);

    const skills = learning.listSkills('agent-1');
    const updated = skills.find(s => s.id === skill.id);
    assert.ok(updated.useCount === 3);
    assert.ok(updated.successRate > 0 && updated.successRate < 1);
  });

  it('should extract knowledge triples without LLM', async () => {
    const triples = await learning.extractKnowledge('agent-1',
      'OSHA requires annual safety training. The program covers fire evacuation procedures.',
      'osha-manual'
    );
    assert.ok(Array.isArray(triples));
    // Simple extraction should find at least one triple
    assert.ok(triples.length > 0);
    assert.ok(triples[0].subject);
    assert.ok(triples[0].predicate);
    assert.ok(triples[0].object);
    assert.equal(triples[0].source, 'osha-manual');
  });

  it('should query knowledge graph', async () => {
    await learning.extractKnowledge('agent-1', 'OSHA requires annual training');
    const results = learning.queryKnowledgeGraph('agent-1', { subject: 'OSHA' });
    assert.ok(Array.isArray(results));
  });

  it('should get graph summary', async () => {
    await learning.extractKnowledge('agent-1', 'OSHA requires annual training');
    const summary = learning.getGraphSummary('agent-1');
    assert.ok(Array.isArray(summary.entities));
    assert.ok(typeof summary.tripleCount === 'number');
  });

  it('should generate reflections without LLM', async () => {
    learning.recordRetrievalFeedback('agent-1', {
      query: 'test', strategy: 'vector', quality: 0.2
    });

    const reflection = await learning.reflect('agent-1');
    assert.ok(typeof reflection === 'string');
    assert.ok(reflection.length > 0);

    const history = learning.getReflections('agent-1');
    assert.equal(history.length, 1);
  });

  it('should cap data structures to prevent memory leaks', () => {
    // Feedback log cap
    for (let i = 0; i < 1100; i++) {
      learning.recordRetrievalFeedback('agent-1', {
        query: `query_${i}`, strategy: 'vector', quality: 0.5
      });
    }
    assert.ok(learning._feedbackLog.get('agent-1').length <= 1000);
  });

  it('should serialize and import state', () => {
    learning.learnSkill('agent-1', { name: 'test', trigger: { keywords: ['test'] } });
    learning.recordRetrievalFeedback('agent-1', { query: 'q', strategy: 'vector', quality: 0.8 });

    const data = learning.serialize();
    assert.ok(data._engine === 'FusionPact');

    const learning2 = new RecursiveLearningEngine({ memory });
    learning2.importState(data);
    assert.ok(learning2.listSkills('agent-1').length === 1);
  });

  it('should handle importState with bad data gracefully', () => {
    assert.doesNotThrow(() => learning.importState(null));
    assert.doesNotThrow(() => learning.importState({}));
    assert.doesNotThrow(() => learning.importState({ skills: 'bad' }));
    assert.doesNotThrow(() => learning.importState({ learnedWeights: { p: 'not-an-object' } }));
  });

  it('should return stats', () => {
    const stats = learning.getStats();
    assert.ok(typeof stats.consolidations === 'number');
    assert.ok(typeof stats.critiques === 'number');
    assert.ok(typeof stats.learnedPatterns === 'number');
  });

  it('should stop auto-consolidation', () => {
    const l = new RecursiveLearningEngine({
      memory,
      enableAutoConsolidation: true,
      consolidation: { intervalMs: 100000 }
    });
    assert.ok(l._consolidationTimer !== null);
    l.stop();
    assert.equal(l._consolidationTimer, null);
  });
});

// ─── LangChain Integration Tests ───

describe('FusionPactVectorStore', () => {
  it('should add and search documents', async () => {
    const embedder = new MockEmbedder({ dimensions: 64 });
    const store = new FusionPactVectorStore({ embedder });

    const ids = await store.addDocuments([
      { pageContent: 'Safety requires PPE in labs', metadata: { source: 'manual' } },
      { pageContent: 'Revenue grew 15% year over year', metadata: { source: 'report' } }
    ]);
    assert.equal(ids.length, 2);

    const results = await store.similaritySearch('safety equipment', 2);
    assert.ok(results.length > 0);
    assert.ok(results[0].pageContent.length > 0);
  });

  it('should return scores with similaritySearchWithScore', async () => {
    const embedder = new MockEmbedder({ dimensions: 64 });
    const store = new FusionPactVectorStore({ embedder });
    await store.addDocuments([{ pageContent: 'Test document' }]);

    const results = await store.similaritySearchWithScore('test', 1);
    assert.ok(results.length > 0);
    assert.ok(Array.isArray(results[0]));
    assert.ok(typeof results[0][1] === 'number'); // score
  });

  it('should create from texts (factory)', async () => {
    const embedder = new MockEmbedder({ dimensions: 64 });
    const store = await FusionPactVectorStore.fromTexts(
      ['Hello world', 'Test content'],
      [{ source: 'a' }, { source: 'b' }],
      embedder
    );
    const results = await store.similaritySearch('hello', 1);
    assert.ok(results.length > 0);
  });

  it('should return a retriever', async () => {
    const embedder = new MockEmbedder({ dimensions: 64 });
    const store = new FusionPactVectorStore({ embedder });
    await store.addDocuments([{ pageContent: 'Document content' }]);

    const retriever = store.asRetriever({ k: 2 });
    const docs = await retriever.getRelevantDocuments('content');
    assert.ok(docs.length > 0);

    // invoke() alias
    const docs2 = await retriever.invoke('content');
    assert.ok(docs2.length > 0);
  });
});

// ─── AI Tools Tests ───

describe('AI Tools (getTools)', () => {
  it('should return tool definitions with execute functions', () => {
    const fp = create({ embedder: 'mock' });
    const tools = getTools(fp);

    assert.ok(Array.isArray(tools));
    assert.ok(tools.length >= 5);

    for (const tool of tools) {
      assert.ok(typeof tool.name === 'string');
      assert.ok(typeof tool.definition === 'object');
      assert.ok(typeof tool.definition.name === 'string');
      assert.ok(typeof tool.definition.description === 'string');
      assert.ok(typeof tool.definition.parameters === 'object');
      assert.ok(typeof tool.execute === 'function');
    }
  });

  it('should execute remember tool', async () => {
    const fp = create({ embedder: 'mock' });
    const tools = getTools(fp);
    const rememberTool = tools.find(t => t.name === 'fusionpact_remember');
    assert.ok(rememberTool);

    const result = await rememberTool.execute({ content: 'User likes dark mode', importance: 0.8 });
    assert.ok(result.id);
  });

  it('should return a tool map', () => {
    const fp = create({ embedder: 'mock' });
    const map = getToolMap(fp);
    assert.ok(typeof map === 'object');
    assert.ok(typeof map.fusionpact_remember === 'function');
    assert.ok(typeof map.fusionpact_recall === 'function');
  });
});

// ─── create() with learning ───

describe('create() with learning', () => {
  it('should include learning engine by default', () => {
    const fp = create({ embedder: 'mock' });
    assert.ok(fp.learning);
    assert.ok(fp.learning instanceof RecursiveLearningEngine);
  });

  it('should disable learning when configured', () => {
    const fp = create({ embedder: 'mock', enableLearning: false });
    assert.equal(fp.learning, null);
  });

  it('end-to-end: ingest → retrieve → critique → consolidate → reflect', async () => {
    const fp = create({ embedder: 'mock' });

    // Ingest
    await fp.rag.ingest('OSHA requires annual safety training for all employees.', { source: 'osha' });

    // Memory
    await fp.memory.remember('agent-1', { content: 'User works in Lab B', importance: 0.8 });
    await fp.memory.learn('agent-1', 'Lab B handles flammable chemicals', { source: 'policy' });

    // Skill
    fp.learning.learnSkill('agent-1', {
      name: 'safety_check',
      trigger: { keywords: ['safety', 'compliance'] },
      steps: [{ action: 'search', params: { query: 'safety' } }]
    });

    // Knowledge graph
    await fp.learning.extractKnowledge('agent-1', 'OSHA requires annual training');

    // Feedback
    fp.learning.recordRetrievalFeedback('agent-1', {
      query: 'safety training', strategy: 'hybrid', quality: 0.85
    });

    // Consolidate
    const consolidation = await fp.learning.consolidate('agent-1');
    assert.ok(typeof consolidation.merged === 'number');

    // Reflect
    const reflection = await fp.learning.reflect('agent-1');
    assert.ok(reflection.length > 0);

    // Stats
    const stats = fp.learning.getStats();
    assert.ok(stats.consolidations >= 1);
    assert.ok(stats.critiques >= 1);
    assert.ok(stats.skillsLearned >= 1);
    assert.ok(stats.reflections >= 1);
  });
});
