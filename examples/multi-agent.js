/**
 * FusionPact Multi-Agent Example
 * 
 * Demonstrates multi-agent orchestration with shared memory,
 * message routing, and capability-based task delegation.
 * 
 * Built by FusionPact Technologies Inc.
 * Run: node examples/multi-agent.js
 */

const { create, AgentOrchestrator } = require('fusionpact');

async function main() {
  const fp = create({ embedder: 'mock', enableMemory: true });

  const orchestrator = new AgentOrchestrator({
    engine: fp.engine,
    memory: fp.memory,
    retriever: fp.retriever
  });

  // Register specialized agents
  orchestrator.registerAgent({
    agentId: 'researcher',
    name: 'Research Agent',
    role: 'Find and analyze information from documents',
    capabilities: ['search', 'analysis', 'summarization']
  });

  orchestrator.registerAgent({
    agentId: 'safety-expert',
    name: 'Safety Expert Agent',
    role: 'Evaluate safety compliance and recommend actions',
    capabilities: ['safety', 'compliance', 'risk-assessment']
  });

  orchestrator.registerAgent({
    agentId: 'writer',
    name: 'Report Writer Agent',
    role: 'Generate reports and documentation',
    capabilities: ['writing', 'formatting', 'reporting']
  });

  // Each agent learns its domain knowledge
  await fp.memory.learn('researcher', 'Company has 3 facilities: HQ, Lab A, and Warehouse B');
  await fp.memory.learn('safety-expert', 'OSHA requires annual safety training for all employees');
  await fp.memory.learn('safety-expert', 'Lab A handles hazardous chemicals requiring special PPE');

  // Share knowledge across agents
  await fp.memory.share('safety-expert', 'Lab A had 3 near-miss incidents in Q4 2024', {
    category: 'incident-data'
  });

  // Message handling
  orchestrator.onMessage('safety-expert', async (msg) => {
    console.log(`[Safety Expert] Received ${msg.type} from ${msg.from}`);
    if (msg.type === 'task') {
      // Process the task and send result back
      await orchestrator.send({
        from: 'safety-expert',
        to: msg.from,
        type: 'result',
        payload: { assessment: 'Lab A requires immediate PPE audit', risk: 'medium' },
        correlationId: msg.correlationId
      });
    }
  });

  orchestrator.onMessage('writer', async (msg) => {
    console.log(`[Writer] Received ${msg.type} from ${msg.from}`);
  });

  // Delegate a safety assessment task
  console.log('Delegating safety assessment...');
  const task = await orchestrator.delegate('researcher', 'Evaluate PPE compliance in Lab A', {
    requiredCapabilities: ['safety', 'compliance']
  });
  console.log(`Task delegated to: ${task.to}\n`);

  // Collaborative recall across all agents
  console.log('Collaborative recall: "Lab A safety"');
  const results = await orchestrator.collaborativeRecall('Lab A safety');
  for (const [agentId, memories] of Object.entries(results)) {
    const total = Object.values(memories).flat().length;
    if (total > 0) console.log(`  ${agentId}: ${total} relevant memories`);
  }

  // Stats
  console.log('\nOrchestration stats:', JSON.stringify(orchestrator.getStats(), null, 2));
}

main().catch(console.error);
