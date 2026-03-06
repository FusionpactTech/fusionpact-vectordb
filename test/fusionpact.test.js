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
  MCPServer, create
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
  });

  it('should work end-to-end', async () => {
    const fp = create({ embedder: 'mock' });
    await fp.rag.ingest('Test document about safety', { source: 'test.txt' });
    const ctx = await fp.rag.buildContext('safety');
    assert.ok(ctx.prompt.length > 0);
  });
});
