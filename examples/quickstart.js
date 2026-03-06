/**
 * FusionPact Quickstart Example
 * 
 * Demonstrates the core capabilities in under 30 lines.
 * Built by FusionPact Technologies Inc.
 * 
 * Run: node examples/quickstart.js
 */

const { create } = require('fusionpact');

async function main() {
  // Create a fully-configured instance (mock embedder for demo)
  const fp = create({ embedder: 'mock' });

  // 1. Ingest a document — auto-chunks, embeds, indexes
  await fp.rag.ingest(
    'All employees must complete safety orientation within 30 days of hire. ' +
    'The orientation covers fire evacuation, chemical handling, and emergency contacts. ' +
    'PPE must be worn in all laboratory areas. Hard hats required in construction zones.',
    { source: 'safety-manual.pdf', title: 'Safety Manual 2024' }
  );

  // 2. Build LLM-ready context
  const context = await fp.rag.buildContext('What PPE is required?');
  console.log('RAG Context:', context.prompt.substring(0, 200) + '...');
  console.log('Sources:', context.sources.length, 'chunks\n');

  // 3. Agent memory
  await fp.memory.remember('safety-bot', {
    content: 'User is a new hire in the chemistry department',
    importance: 0.9
  });

  await fp.memory.learn('safety-bot',
    'Chemistry lab requires safety goggles, lab coat, and closed-toe shoes',
    { source: 'lab-policy', category: 'ppe' }
  );

  const memories = await fp.memory.recall('safety-bot', 'PPE requirements for chemistry');
  console.log('Recalled memories:', {
    episodic: memories.episodic?.length || 0,
    semantic: memories.semantic?.length || 0
  });

  console.log('\n✅ Done! See https://github.com/FusionpactTech/fusionpact-vectordb for full docs.');
}

main().catch(console.error);
