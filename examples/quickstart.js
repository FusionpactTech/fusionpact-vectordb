/**
 * FusionPact — Quickstart Example
 * Run: node examples/quickstart.js
 */

const { FusionEngine, RAGPipeline, AgentMemory, createEmbedder, vec } = require('../src/index');

async function main() {
  console.log('⚡ FusionPact Quickstart\n');
  const engine = new FusionEngine();

  // 1. Create HNSW collection
  engine.createCollection('my-docs', { dimension: 64, metric: 'cosine', indexType: 'hnsw' });
  console.log('✅ Created HNSW-indexed collection');

  // 2. Insert vectors
  const docs = Array.from({ length: 100 }, (_, i) => ({
    id: `doc-${i}`, vector: vec.random(64),
    metadata: { category: ['safety', 'legal', 'product'][i % 3], priority: i % 4 },
  }));
  engine.insert('my-docs', docs);
  console.log(`✅ Inserted ${docs.length} documents`);

  // 3. Search with filter
  const result = engine.query('my-docs', vec.random(64), { topK: 3, filter: { category: 'safety' } });
  console.log(`\n🔍 Search: ${result.results.length} results in ${result.elapsed}ms [${result.method}]`);
  result.results.forEach((r, i) => console.log(`   [${i+1}] ${r.id} — ${r.score.toFixed(4)}`));

  // 4. Multi-tenancy
  engine.createCollection('shared', { dimension: 64 });
  const tenantA = engine.tenant('shared', 'acme');
  const tenantB = engine.tenant('shared', 'globex');
  tenantA.insert(docs.slice(0, 50));
  tenantB.insert(docs.slice(50));
  const aRes = tenantA.query(vec.random(64), { topK: 5 });
  console.log(`\n🔒 Tenant A: ${aRes.results.length} results (isolated from B)`);

  // 5. One-Click RAG
  const rag = new RAGPipeline(engine, { embedder: 'mock' });
  await rag.ingest('FusionPact is an open-source vector database for AI agents with HNSW indexing and MCP support.', { source: 'readme.md' });
  const ctx = await rag.buildContext('What is FusionPact?');
  console.log(`\n📄 RAG: ${ctx.chunks.length} chunks, prompt is ${ctx.prompt.length} chars`);

  // 6. Agent Memory
  const mem = new AgentMemory(engine, { embedder: 'mock' });
  await mem.remember('agent-1', { content: 'User asked about safety', role: 'user' });
  await mem.learn('agent-1', 'OSHA requires annual safety training for confined spaces.');
  const recalled = await mem.recall('agent-1', 'safety training');
  console.log(`\n🧠 Memory: recalled ${recalled.length} memories`);

  console.log('\n✅ Done! Next: fusionpact serve --port 8080\n');
}

main().catch(console.error);
