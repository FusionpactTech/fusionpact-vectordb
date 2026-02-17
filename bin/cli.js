#!/usr/bin/env node

/**
 * FusionPact CLI
 *
 * Usage:
 *   fusionpact serve [--port 8080]         Start HTTP API server
 *   fusionpact mcp                          Start MCP server (stdio)
 *   fusionpact demo                         Run quickstart demo
 *   fusionpact bench [--count 10000]        Run HNSW benchmark
 *   fusionpact version                      Show version
 */

'use strict';

const args = process.argv.slice(2);
const command = args[0] || 'help';

function getArg(name, defaultValue) {
  const idx = args.indexOf(`--${name}`);
  return idx >= 0 && args[idx + 1] ? args[idx + 1] : defaultValue;
}

async function main() {
  switch (command) {
    case 'serve': {
      const port = parseInt(getArg('port', '8080'));
      const { startServer } = require('./server');
      startServer(port);
      break;
    }

    case 'mcp': {
      const { MCPServer } = require('../src/mcp/server');
      const server = new MCPServer();
      server.start();
      break;
    }

    case 'demo': {
      const { FusionEngine, RAGPipeline, AgentMemory, createEmbedder } = require('../src/index');

      console.log('\n⚡ FusionPact — Quick Demo\n');

      const engine = new FusionEngine();
      const embedder = createEmbedder('mock');

      // 1. HNSW Collection
      console.log('1. Creating HNSW-indexed collection...');
      engine.createCollection('demo', { dimension: 64, metric: 'cosine', indexType: 'hnsw' });

      // 2. Insert vectors
      console.log('2. Inserting 1,000 vectors...');
      const { random } = require('../src/core/vectors');
      const docs = Array.from({ length: 1000 }, (_, i) => ({
        id: `doc-${i}`,
        vector: random(64),
        metadata: { category: ['safety', 'legal', 'product'][i % 3], priority: i % 4 },
      }));
      engine.insert('demo', docs);

      // 3. Search
      console.log('3. Searching...');
      const result = engine.query('demo', random(64), { topK: 5 });
      console.log(`   Found ${result.results.length} results in ${result.elapsed}ms [${result.method}]`);
      result.results.forEach((r, i) => {
        console.log(`   [${i + 1}] ${r.id} — score: ${r.score.toFixed(4)} — ${JSON.stringify(r.metadata)}`);
      });

      // 4. Multi-tenancy
      console.log('\n4. Multi-tenancy demo...');
      engine.createCollection('shared', { dimension: 64, metric: 'cosine' });
      const tenantA = engine.tenant('shared', 'acme_corp');
      const tenantB = engine.tenant('shared', 'globex_inc');
      tenantA.insert(docs.slice(0, 50));
      tenantB.insert(docs.slice(50, 100));
      const tenantResult = tenantA.query(random(64), { topK: 3 });
      console.log(`   Tenant A sees: ${tenantResult.results.length} results (isolated from B)`);

      // 5. RAG
      console.log('\n5. One-Click RAG...');
      const rag = new RAGPipeline(engine, { embedder });
      await rag.ingest(
        'FusionPact is an open-source vector database built for AI agents. ' +
        'It features HNSW indexing for fast approximate nearest neighbor search, ' +
        'built-in multi-tenancy for data isolation, and an MCP server for ' +
        'direct integration with Claude, GPT, and other AI agents. ' +
        'The one-click RAG pipeline handles text chunking, embedding, and retrieval automatically.',
        { source: 'about.txt' }
      );
      const ctx = await rag.buildContext('What is FusionPact?');
      console.log(`   Chunks retrieved: ${ctx.chunks.length}`);
      console.log(`   Prompt length: ${ctx.prompt.length} chars`);

      // 6. Agent Memory
      console.log('\n6. Agent Memory...');
      const memory = new AgentMemory(engine, { embedder });
      await memory.remember('agent-1', { content: 'User asked about safety protocols', role: 'user' });
      await memory.remember('agent-1', { content: 'Provided OSHA compliance checklist', role: 'assistant' });
      await memory.learn('agent-1', 'OSHA requires annual safety training for all workers in confined spaces.');
      const recalled = await memory.recall('agent-1', 'safety training');
      console.log(`   Recalled ${recalled.length} relevant memories`);

      const stats = memory.getStats('agent-1');
      console.log(`   Memory stats: ${JSON.stringify(stats)}`);

      // Benchmark
      console.log('\n7. HNSW Benchmark (100 queries)...');
      const times = [];
      for (let i = 0; i < 100; i++) {
        const t0 = performance.now();
        engine.query('demo', random(64), { topK: 10 });
        times.push(performance.now() - t0);
      }
      times.sort((a, b) => a - b);
      const avg = times.reduce((a, b) => a + b) / times.length;
      console.log(`   Avg: ${avg.toFixed(3)}ms | P50: ${times[50].toFixed(3)}ms | P99: ${times[99].toFixed(3)}ms | QPS: ${Math.round(1000 / avg)}`);

      console.log('\n✅ Demo complete! Try: fusionpact serve --port 8080\n');
      break;
    }

    case 'bench': {
      const count = parseInt(getArg('count', '10000'));
      const dim = parseInt(getArg('dim', '128'));
      const { FusionEngine } = require('../src/index');
      const { random } = require('../src/core/vectors');

      console.log(`\n⚡ FusionPact Benchmark — ${count.toLocaleString()} vectors, ${dim}D\n`);
      const engine = new FusionEngine();

      // Build
      console.log('Building HNSW index...');
      engine.createCollection('bench-hnsw', { dimension: dim, metric: 'cosine', indexType: 'hnsw' });
      engine.createCollection('bench-flat', { dimension: dim, metric: 'cosine', indexType: 'flat' });

      const t0 = performance.now();
      const docs = Array.from({ length: count }, (_, i) => ({
        id: `v-${i}`, vector: random(dim), metadata: { i },
      }));
      engine.insert('bench-hnsw', docs);
      engine.insert('bench-flat', docs);
      console.log(`Build time: ${((performance.now() - t0) / 1000).toFixed(2)}s`);

      // Benchmark
      function runBench(colName, n = 100) {
        const times = [];
        for (let i = 0; i < n; i++) {
          const t = performance.now();
          engine.query(colName, random(dim), { topK: 10 });
          times.push(performance.now() - t);
        }
        times.sort((a, b) => a - b);
        const avg = times.reduce((a, b) => a + b) / times.length;
        return {
          avg: avg.toFixed(3),
          p50: times[Math.floor(n * 0.5)].toFixed(3),
          p95: times[Math.floor(n * 0.95)].toFixed(3),
          p99: times[Math.floor(n * 0.99)].toFixed(3),
          qps: Math.round(1000 / avg),
        };
      }

      console.log('\nRunning 100 queries each...\n');
      const hnsw = runBench('bench-hnsw');
      const flat = runBench('bench-flat');

      console.log('┌─────────────┬───────────┬───────────┐');
      console.log('│ Metric      │ HNSW      │ Flat      │');
      console.log('├─────────────┼───────────┼───────────┤');
      console.log(`│ Avg Latency │ ${hnsw.avg.padStart(7)}ms │ ${flat.avg.padStart(7)}ms │`);
      console.log(`│ P50         │ ${hnsw.p50.padStart(7)}ms │ ${flat.p50.padStart(7)}ms │`);
      console.log(`│ P95         │ ${hnsw.p95.padStart(7)}ms │ ${flat.p95.padStart(7)}ms │`);
      console.log(`│ P99         │ ${hnsw.p99.padStart(7)}ms │ ${flat.p99.padStart(7)}ms │`);
      console.log(`│ QPS         │ ${String(hnsw.qps).padStart(9)} │ ${String(flat.qps).padStart(9)} │`);
      console.log('└─────────────┴───────────┴───────────┘');
      console.log(`\nSpeedup: ${(parseFloat(flat.avg) / parseFloat(hnsw.avg)).toFixed(1)}x faster with HNSW\n`);
      break;
    }

    case 'version':
    case '--version':
    case '-v':
      console.log('fusionpact v0.1.0');
      break;

    case 'help':
    case '--help':
    case '-h':
    default:
      console.log(`
  ⚡ FusionPact — The Agent-Native Vector Database

  Usage: fusionpact <command> [options]

  Commands:
    serve [--port 8080]       Start HTTP API server
    mcp                       Start MCP server (for Claude Desktop, Cursor)
    demo                      Run interactive quickstart demo
    bench [--count 10000]     Run HNSW vs Flat benchmark
    version                   Show version

  Environment:
    EMBEDDING_PROVIDER        mock | ollama | openai (default: mock)
    OLLAMA_MODEL              Ollama model name (default: nomic-embed-text)
    OPENAI_API_KEY            OpenAI API key (required for openai provider)

  MCP Integration (Claude Desktop):
    Add to claude_desktop_config.json:
    {
      "mcpServers": {
        "fusionpact": {
          "command": "npx",
          "args": ["fusionpact", "mcp"]
        }
      }
    }

  Docs: https://github.com/FusionPact/fusionpact-vectordb
`);
  }
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
