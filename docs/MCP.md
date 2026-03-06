# MCP Integration Guide

> Connect FusionPact to Claude Desktop, Cursor, Windsurf, and any MCP client
> Built by FusionPact Technologies Inc.

## Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fusionpact": {
      "command": "npx",
      "args": ["fusionpact", "mcp"],
      "env": {
        "EMBEDDING_PROVIDER": "ollama",
        "LLM_PROVIDER": "ollama"
      }
    }
  }
}
```

## Cursor

Add to Cursor settings → MCP Servers:

```json
{
  "fusionpact": {
    "command": "npx",
    "args": ["fusionpact", "mcp"]
  }
}
```

## HTTP Mode

For custom MCP clients or web applications:

```bash
npx fusionpact serve --port 8080
```

Endpoints:
- `GET /mcp/tools` — List available tools
- `POST /mcp/call` — Invoke a tool: `{ "tool": "fusionpact_search", "args": { ... } }`

## Available Tools

See README.md for the complete list of MCP tools covering:
- Collection management
- Vector search
- Hybrid retrieval
- RAG ingestion and query
- Agent memory (remember, recall, learn, share, forget)
- Conversation management

---

*Built by [FusionPact Technologies Inc.](https://fusionpact.com)*
