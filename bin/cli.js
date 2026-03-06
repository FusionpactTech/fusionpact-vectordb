#!/usr/bin/env node
/**
 * FusionPact CLI
 * Built by FusionPact Technologies Inc. | Apache-2.0
 */
'use strict';
const fp = require('../src/index');
const BANNER = `\n  ⚡ FusionPact v${fp.VERSION} — The Agent-Native Retrieval Engine\n  Built by FusionPact Technologies Inc. | https://fusionpact.com\n`;
const HELP = `${BANNER}\nUsage: fusionpact <command> [options]\n\nCommands:\n  demo              Run interactive demo\n  serve [--port N]  Start HTTP + MCP server (default: 8080)\n  mcp               Start MCP server (stdio) for Claude Desktop\n  bench [--count N] Run HNSW benchmarks\n  help              Show this help\n\nEnvironment:\n  EMBEDDING_PROVIDER  'ollama' | 'openai' | 'mock' (default: mock)\n  LLM_PROVIDER        'ollama' | 'openai' | 'anthropic' (for tree reasoning)\n  OPENAI_API_KEY      OpenAI API key\n  ANTHROPIC_API_KEY   Anthropic API key\n`;

async function demo() {
  console.log(BANNER);
  const { engine, rag, memory, treeIndex } = fp.create({ embedder: process.env.EMBEDDING_PROVIDER || 'mock' });
  const embedder = new fp.MockEmbedder({ dimensions: 64 });

  console.log('━━━ 1. Vector Search ━━━');
  engine.createCollection('demo', { dimensions: 64 });
  const docs = [
    { id: 'd1', text: 'OSHA requires chemical hazard communication including safety data sheets' },
    { id: 'd2', text: 'Personal protective equipment must be provided at no cost to employees' },
    { id: 'd3', text: 'Quarterly revenue increased by 15% driven by cloud services growth' },
    { id: 'd4', text: 'Confined space entry requires atmospheric testing before access' },
  ];
  for (const d of docs) {
    engine.insert('demo', [{ id: d.id, vector: await embedder.embed(d.text), metadata: { _content: d.text } }]);
  }
  const qv = await embedder.embed('chemical safety requirements');
  const res = engine.search('demo', qv, { topK: 3 });
  res.forEach((r, i) => console.log(`  ${i+1}. [${r.score.toFixed(3)}] ${r.metadata._content?.substring(0, 70)}...`));

  console.log('\n━━━ 2. RAG Pipeline ━━━');
  const ir = await rag.ingest('All employees must complete safety orientation within 30 days. The orientation covers fire evacuation, chemical handling, and emergency contacts. All machinery must have proper guarding. Lockout/tagout procedures must be followed.', { source: 'manual.txt' });
  console.log(`  Ingested ${ir.chunks} chunks`);
  const ctx = await rag.buildContext('safety orientation requirements');
  console.log(`  Context: ${ctx.chunks} chunks, ${ctx.prompt.length} chars`);

  console.log('\n━━━ 3. Tree Index ━━━');
  await treeIndex.indexDocument('report', '# Safety Report\n## Incidents\nTotal injuries: 47\n## Training\n98% completion rate', { format: 'markdown' });
  const tr = await treeIndex.search('report', 'How many injuries?');
  if (tr.length) console.log(`  Found: "${tr[0].content.substring(0, 60)}..." (score: ${tr[0].relevanceScore.toFixed(2)})`);

  console.log('\n━━━ 4. Agent Memory ━━━');
  await memory.remember('agent-1', { content: 'User prefers detailed safety reports', importance: 0.8 });
  await memory.learn('agent-1', 'OSHA 1910.106 covers flammable liquid storage', { source: 'osha' });
  const stats = memory.getStats('agent-1');
  console.log(`  Memories: episodic=${stats.episodic}, semantic=${stats.semantic}, procedural=${stats.procedural}`);

  console.log('\n━━━ 5. Multi-Tenancy ━━━');
  engine.createCollection('shared', { dimensions: 64 });
  const a = engine.tenant('shared', 'acme'), b = engine.tenant('shared', 'globex');
  a.insert([{ id: 'a1', vector: await embedder.embed('Acme data'), metadata: {} }]);
  b.insert([{ id: 'b1', vector: await embedder.embed('Globex data'), metadata: {} }]);
  console.log(`  Acme sees: ${a.search(await embedder.embed('data'), { topK: 5 }).length} | Globex sees: ${b.search(await embedder.embed('data'), { topK: 5 }).length}`);

  console.log('\n✅ Demo complete! Star us: https://github.com/FusionpactTech/fusionpact-vectordb');
}

async function serve(port) {
  console.log(BANNER);
  const inst = fp.create({ embedder: process.env.EMBEDDING_PROVIDER || 'mock', llmProvider: process.env.LLM_PROVIDER });
  const mcp = new fp.MCPServer({ ...inst, transport: 'http', port });
  const http = require('http');
  http.createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    if (req.method === 'OPTIONS') { res.writeHead(204); return res.end(); }
    const j = (c, d) => { res.writeHead(c, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(d)); };
    try {
      if (req.url === '/api/health') return j(200, { status: 'ok', version: fp.VERSION, engine: 'FusionPact Technologies Inc.' });
      if (req.url === '/mcp/tools') return j(200, mcp.getToolManifest());
      if (req.url === '/mcp/call' && req.method === 'POST') {
        let body = ''; req.on('data', c => body += c);
        return req.on('end', async () => { const { tool, args } = JSON.parse(body); j(200, await mcp.handleToolCall(tool, args)); });
      }
      j(404, { error: 'Not found' });
    } catch (e) { j(400, { error: e.message }); }
  }).listen(port, () => console.log(`🚀 Listening on http://localhost:${port}\n   GET  /api/health\n   GET  /mcp/tools\n   POST /mcp/call`));
}

async function bench(count) {
  console.log(BANNER); console.log(`Benchmarking ${count} vectors (128D)...\n`);
  const dim = 128, idx = new fp.HNSWIndex(dim, { M: 16, efConstruction: 200, efSearch: 50 });
  let t = performance.now();
  for (let i = 0; i < count; i++) { const v = new Float32Array(dim); for (let j = 0; j < dim; j++) v[j] = Math.random()*2-1; idx.insert(`v${i}`, v); }
  const iT = performance.now() - t; const qN = 1000; t = performance.now();
  for (let i = 0; i < qN; i++) { const q = new Float32Array(dim); for (let j = 0; j < dim; j++) q[j] = Math.random()*2-1; idx.search(q, { topK: 10 }); }
  const sT = performance.now() - t;
  console.log(`Insert: ${iT.toFixed(0)}ms (${(iT/count).toFixed(3)}ms/vec)\nSearch: ${sT.toFixed(0)}ms (${(sT/qN).toFixed(3)}ms/query)\nQPS:    ~${Math.round(qN/(sT/1000))}\nHeap:   ${(process.memoryUsage().heapUsed/1024/1024).toFixed(1)}MB`);
}

const args = process.argv.slice(2), cmd = args[0] || 'help';
(async () => {
  try {
    if (cmd === 'demo') await demo();
    else if (cmd === 'serve') await serve(parseInt(args[args.indexOf('--port')+1]) || 8080);
    else if (cmd === 'mcp') { const inst = fp.create({ embedder: 'mock' }); const m = new fp.MCPServer({ ...inst, transport: 'stdio' }); await m.start(); }
    else if (cmd === 'bench') await bench(parseInt(args[args.indexOf('--count')+1]) || 5000);
    else console.log(HELP);
  } catch (e) { console.error('Error:', e.message); process.exit(1); }
})();
