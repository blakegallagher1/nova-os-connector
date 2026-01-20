import 'dotenv/config';
import express from 'express';

console.log('Step 1: Express imported');

// Try importing MCP modules one by one
console.log('Step 2: About to import McpServer...');
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
console.log('Step 3: McpServer imported');

import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
console.log('Step 4: StreamableHTTPServerTransport imported');

import { z } from 'zod';
console.log('Step 5: Zod imported');

// Note: NOT importing Client or StdioClientTransport (the child process stuff)

const PORT = parseInt(process.env.PORT || '8000', 10);

const app = express();
app.use(express.json({ limit: '10mb' }));

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString(), mcpImported: true });
});

const server = new McpServer({
  name: 'nova-os-test',
  version: '1.0.0',
});

// Register a simple test tool
server.registerTool(
  'test_echo',
  {
    description: 'Echo back the input',
    inputSchema: { message: z.string() },
  },
  async ({ message }) => {
    return { content: [{ type: 'text' as const, text: `Echo: ${message}` }] };
  }
);

app.post('/mcp', async (req, res) => {
  try {
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
      enableJsonResponse: true,
    });

    await server.connect(transport);

    res.on('close', () => {
      transport.close().catch(console.error);
    });

    await transport.handleRequest(req, res, req.body);
  } catch (error) {
    console.error('MCP error:', error);
    res.status(500).json({ error: 'Internal error' });
  }
});

app.listen(PORT, () => {
  console.log(`Test server listening on port ${PORT}`);
});
