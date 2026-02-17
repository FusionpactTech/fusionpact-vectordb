/**
 * FusionPact — Test Suite
 * Run: node test/run.js
 */

'use strict';

const { FusionEngine, HNSWIndex, RAGPipeline, AgentMemory, vec, createEmbedder, TenantClient } = require('../src/index');

let passed = 0, failed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ✅ ${name}`);
  } catch (err) {
    failed++;
    console.log(`  ❌ ${name}: ${err.message}`);
  }
}

function assert(condition, msg = 'Assertion failed') {
  if (!condition) throw new Error(msg);
}

async function asyncTest(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  ✅ ${name}`);
  } catch (err) {
    failed++;
    console.log(`  ❌ ${name}: ${err.message}`);
  }
}

// ─── Vector Math ──────────────────────────────────────────────
console.log('\n📐 Vector Math');

test('cosine similarity of identical vectors = 1', () => {
  const a = [1, 0, 0];
  assert(Math.abs(vec.cosine(a, a) - 1) < 0.001);
});

test('cosine similarity of opposite vectors = -1', () => {
  const a = [1, 0, 0], b = [-1, 0, 0];
  assert(Math.abs(vec.cosine(a, b) + 1) < 0.001);
});

test('normalize produces unit vector', () => {
  const n = vec.normalize([3, 4]);
  assert(Math.abs(vec.magnitude(n) - 1) < 0.001);
});

test('random vector has correct dimension', () => {
  assert(vec.random(128).length === 128);
});

// ─── HNSW Index ───────────────────────────────────────────────
console.log('\n🔗 HNSW Index');

test('insert and search basic', () => {
  const idx = new HNSWIndex(4, 'cosine', { M: 4, efConstruction: 20, efSearch: 10 });
  idx.insert('a', [1, 0, 0, 0]);
  idx.insert('b', [0, 1, 0, 0]);
  idx.insert('c', [0.9, 0.1, 0, 0]);
  const results = idx.search([1, 0, 0, 0], 2);
  assert(results.length === 2, `Expected 2 results, got ${results.length}`);
  assert(results[0].id === 'a', `Expected 'a' first, got '${results[0].id}'`);
});

test('insert 1000 vectors and search', () => {
  const idx = new HNSWIndex(32, 'cosine', { M: 16, efConstruction: 100, efSearch: 30 });
  for (let i = 0; i < 1000; i++) idx.insert(`v${i}`, vec.random(32));
  const results = idx.search(vec.random(32), 10);
  assert(results.length === 10, `Expected 10 results, got ${results.length}`);
  // Scores should be sorted descending
  for (let i = 1; i < results.length; i++) {
    assert(results[i - 1].score >= results[i].score, 'Results not sorted');
  }
});

test('delete node', () => {
  const idx = new HNSWIndex(4, 'cosine');
  idx.insert('a', [1, 0, 0, 0]);
  idx.insert('b', [0, 1, 0, 0]);
  assert(idx.delete('a') === true);
  assert(idx.size === 1);
  const results = idx.search([1, 0, 0, 0], 5);
  assert(results.every(r => r.id !== 'a'), 'Deleted node found in results');
});

test('serialize and deserialize', () => {
  const idx = new HNSWIndex(8, 'cosine');
  for (let i = 0; i < 50; i++) idx.insert(`n${i}`, vec.random(8), { i });
  const data = idx.serialize();
  const restored = HNSWIndex.deserialize(data);
  assert(restored.size === 50);
  const results = restored.search(vec.random(8), 5);
  assert(results.length === 5);
});

test('getStats returns valid structure', () => {
  const idx = new HNSWIndex(16, 'cosine');
  for (let i = 0; i < 100; i++) idx.insert(`s${i}`, vec.random(16));
  const stats = idx.getStats();
  assert(stats.nodes === 100);
  assert(stats.totalEdges > 0);
  assert(stats.maxLevel >= 0);
});

// ─── Engine ───────────────────────────────────────────────────
console.log('\n🔧 Engine');

test('create and list collections', () => {
  const e = new FusionEngine();
  e.createCollection('test', { dimension: 64 });
  assert(e.listCollections().length === 1);
  assert(e.getCollection('test').count === 0);
});

test('insert and query', () => {
  const e = new FusionEngine();
  e.createCollection('test', { dimension: 8, indexType: 'hnsw' });
  e.insert('test', [
    { id: 'd1', vector: vec.normalize([1,0,0,0,0,0,0,0]), metadata: { x: 1 } },
    { id: 'd2', vector: vec.normalize([0,1,0,0,0,0,0,0]), metadata: { x: 2 } },
  ]);
  const r = e.query('test', vec.normalize([1,0,0,0,0,0,0,0]), { topK: 1 });
  assert(r.results.length === 1);
  assert(r.results[0].id === 'd1');
});

test('metadata filter operators', () => {
  const e = new FusionEngine();
  e.createCollection('f', { dimension: 4, indexType: 'flat' });
  e.insert('f', [
    { id: 'a', vector: vec.random(4), metadata: { score: 10, tag: 'fire' } },
    { id: 'b', vector: vec.random(4), metadata: { score: 20, tag: 'flood' } },
    { id: 'c', vector: vec.random(4), metadata: { score: 30, tag: 'fire' } },
  ]);
  const r1 = e.query('f', vec.random(4), { topK: 10, filter: { tag: 'fire' } });
  assert(r1.results.length === 2, `Expected 2 fire results, got ${r1.results.length}`);

  const r2 = e.query('f', vec.random(4), { topK: 10, filter: { score: { $gte: 20 } } });
  assert(r2.results.length === 2, `Expected 2 results with score>=20, got ${r2.results.length}`);

  const r3 = e.query('f', vec.random(4), { topK: 10, filter: { tag: { $in: ['fire', 'flood'] } } });
  assert(r3.results.length === 3);
});

test('drop collection', () => {
  const e = new FusionEngine();
  e.createCollection('temp', { dimension: 4 });
  assert(e.dropCollection('temp') === true);
  assert(e.listCollections().length === 0);
});

test('duplicate collection throws', () => {
  const e = new FusionEngine();
  e.createCollection('x', { dimension: 4 });
  try { e.createCollection('x', { dimension: 4 }); assert(false); } catch (err) { assert(err.message.includes('exists')); }
});

test('dimension mismatch throws', () => {
  const e = new FusionEngine();
  e.createCollection('d', { dimension: 4 });
  try { e.insert('d', [{ vector: [1,2,3] }]); assert(false); } catch (err) { assert(err.message.includes('mismatch')); }
});

// ─── Multi-Tenancy ────────────────────────────────────────────
console.log('\n🔒 Multi-Tenancy');

test('tenant isolation', () => {
  const e = new FusionEngine();
  e.createCollection('shared', { dimension: 4, indexType: 'flat' });
  const tA = e.tenant('shared', 'alpha');
  const tB = e.tenant('shared', 'beta');
  tA.insert([{ id: 'a1', vector: vec.random(4), metadata: { x: 1 } }]);
  tB.insert([{ id: 'b1', vector: vec.random(4), metadata: { x: 2 } }]);

  const rA = tA.query(vec.random(4), { topK: 10 });
  const rB = tB.query(vec.random(4), { topK: 10 });

  assert(rA.results.length === 1 && rA.results[0].id === 'a1', 'Tenant A should only see its own docs');
  assert(rB.results.length === 1 && rB.results[0].id === 'b1', 'Tenant B should only see its own docs');
});

test('tenant delete only own docs', () => {
  const e = new FusionEngine();
  e.createCollection('td', { dimension: 4, indexType: 'flat' });
  const tA = e.tenant('td', 'a');
  const tB = e.tenant('td', 'b');
  tA.insert([{ id: 'a1', vector: vec.random(4) }]);
  tB.insert([{ id: 'b1', vector: vec.random(4) }]);

  // Tenant A tries to delete B's doc — should not work
  const deleted = tA.delete(['b1']);
  assert(deleted === 0, 'Should not delete another tenant\'s doc');
  assert(e.collections.get('td').count === 2);
});

// ─── RAG Pipeline ─────────────────────────────────────────────
console.log('\n📄 RAG Pipeline');

asyncTest('ingest and search', async () => {
  const e = new FusionEngine();
  const rag = new RAGPipeline(e, { embedder: 'mock', collection: 'rag-test' });
  const info = await rag.ingest('FusionPact is a vector database for AI agents. It supports HNSW indexing and multi-tenancy.', { source: 'test.txt' });
  assert(info.chunksCreated > 0, `Expected chunks, got ${info.chunksCreated}`);

  const result = await rag.search('vector database');
  assert(result.chunks.length > 0, 'Expected search results');
  assert(result.chunks[0].text.length > 0, 'Chunk should have text');
}).then(() => {

  asyncTest('build context', async () => {
    const e = new FusionEngine();
    const rag = new RAGPipeline(e, { embedder: 'mock', collection: 'rag-ctx' });
    await rag.ingest('OSHA requires annual safety training for all workers.', { source: 'osha.txt' });
    const ctx = await rag.buildContext('safety training requirements');
    assert(ctx.prompt.includes('Question:'), 'Prompt should contain question');
    assert(ctx.sources.length > 0, 'Should have sources');
  }).then(() => {

    // ─── Agent Memory ──────────────────────────────────────
    console.log('\n🧠 Agent Memory');

    asyncTest('episodic memory (remember + recall)', async () => {
      const e = new FusionEngine();
      const mem = new AgentMemory(e, { embedder: 'mock' });
      await mem.remember('a1', { content: 'User discussed fire safety', role: 'user' });
      await mem.remember('a1', { content: 'Agent provided OSHA guidelines', role: 'assistant' });
      const recalled = await mem.recall('a1', 'safety guidelines');
      assert(recalled.length === 2);
    }).then(() => {

      asyncTest('semantic memory (learn + query)', async () => {
        const e = new FusionEngine();
        const mem = new AgentMemory(e, { embedder: 'mock' });
        await mem.learn('a1', 'ISO 45001 is the international standard for occupational health and safety.');
        const results = await mem.query('a1', 'safety standards');
        assert(results.length > 0);
        assert(results[0].content.includes('ISO'));
      }).then(() => {

        asyncTest('procedural memory (tools)', async () => {
          const e = new FusionEngine();
          const mem = new AgentMemory(e, { embedder: 'mock' });
          await mem.registerTool('a1', { name: 'search_incidents', description: 'Search EHS incident reports' });
          await mem.registerTool('a1', { name: 'generate_report', description: 'Generate compliance PDF report' });
          const tools = await mem.findTools('a1', 'find incident data');
          assert(tools.length > 0);
        }).then(() => {

          asyncTest('forget (GDPR)', async () => {
            const e = new FusionEngine();
            const mem = new AgentMemory(e, { embedder: 'mock' });
            await mem.remember('a1', { content: 'sensitive info', role: 'user' });
            await mem.learn('a1', 'also sensitive');
            const before = mem.getStats('a1');
            assert(before.total > 0);
            mem.forget('a1', { type: 'all' });
            const after = mem.getStats('a1');
            assert(after.total === 0, `Expected 0 after forget, got ${after.total}`);
          }).then(() => {

            asyncTest('agent isolation', async () => {
              const e = new FusionEngine();
              const mem = new AgentMemory(e, { embedder: 'mock' });
              await mem.remember('agent-A', { content: 'A data', role: 'user' });
              await mem.remember('agent-B', { content: 'B data', role: 'user' });
              const aRecall = await mem.recall('agent-A', 'data');
              const bRecall = await mem.recall('agent-B', 'data');
              assert(aRecall.length === 1 && aRecall[0].content.includes('A'));
              assert(bRecall.length === 1 && bRecall[0].content.includes('B'));
            }).then(() => {

              // ─── Summary ──────────────────────────────────
              console.log(`\n${'─'.repeat(40)}`);
              console.log(`  ${passed} passed, ${failed} failed`);
              console.log(`${'─'.repeat(40)}\n`);
              process.exit(failed > 0 ? 1 : 0);
            });
          });
        });
      });
    });
  });
});
