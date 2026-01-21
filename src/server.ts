import 'dotenv/config';
import express from 'express';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { z } from 'zod';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import { Vercel } from '@vercel/sdk';

const execAsync = promisify(exec);

// ============================================================================
// CONFIGURATION
// ============================================================================

const PORT = parseInt(process.env.PORT || '8000', 10);
const GITHUB_TOKEN = process.env.GITHUB_PERSONAL_ACCESS_TOKEN;
const VERCEL_TOKEN = process.env.VERCEL_TOKEN;

// Initialize Vercel client if token is available
const vercel = VERCEL_TOKEN ? new Vercel({ bearerToken: VERCEL_TOKEN }) : null;

// Resilience settings
const TOOL_TIMEOUT_MS = 50000; // 50 seconds (under ChatGPT's 60s limit)
const MAX_RETRIES = 2; // Number of retries for idempotent operations
const RETRY_DELAY_MS = 1000; // Delay between retries

// ============================================================================
// CACHING LAYER
// ============================================================================

const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes default TTL
const CACHE_MAX_ENTRIES = 500; // Max cache entries to prevent memory bloat

interface CacheEntry {
  data: any;
  timestamp: number;
  ttl: number;
  hits: number;
}

class ContentCache {
  private cache = new Map<string, CacheEntry>();
  private stats = {
    hits: 0,
    misses: 0,
    evictions: 0,
    invalidations: 0,
  };

  // Generate cache key for file content
  fileKey(owner: string, repo: string, path: string, branch?: string): string {
    return `file:${owner}/${repo}/${branch || 'default'}:${path}`;
  }

  // Generate cache key for directory listing
  dirKey(owner: string, repo: string, path: string, branch?: string): string {
    return `dir:${owner}/${repo}/${branch || 'default'}:${path}`;
  }

  get(key: string): any | null {
    const entry = this.cache.get(key);
    if (!entry) {
      this.stats.misses++;
      return null;
    }

    // Check if expired
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      this.stats.misses++;
      return null;
    }

    entry.hits++;
    this.stats.hits++;
    console.log(`[cache] HIT: ${key} (${entry.hits} hits)`);
    return entry.data;
  }

  set(key: string, data: any, ttl: number = CACHE_TTL_MS): void {
    // Evict oldest entries if at capacity
    if (this.cache.size >= CACHE_MAX_ENTRIES) {
      this.evictOldest();
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
      hits: 0,
    });
    console.log(`[cache] SET: ${key} (TTL: ${ttl / 1000}s)`);
  }

  // Invalidate cache for a specific repo (after writes)
  invalidateRepo(owner: string, repo: string): void {
    const prefix = `file:${owner}/${repo}/`;
    const dirPrefix = `dir:${owner}/${repo}/`;
    let count = 0;

    for (const key of this.cache.keys()) {
      if (key.startsWith(prefix) || key.startsWith(dirPrefix)) {
        this.cache.delete(key);
        count++;
      }
    }

    this.stats.invalidations += count;
    if (count > 0) {
      console.log(`[cache] INVALIDATED: ${count} entries for ${owner}/${repo}`);
    }
  }

  // Invalidate specific file
  invalidateFile(owner: string, repo: string, path: string, branch?: string): void {
    const key = this.fileKey(owner, repo, path, branch);
    if (this.cache.delete(key)) {
      this.stats.invalidations++;
      console.log(`[cache] INVALIDATED: ${key}`);
    }
  }

  // Clear entire cache
  clear(): number {
    const size = this.cache.size;
    this.cache.clear();
    console.log(`[cache] CLEARED: ${size} entries`);
    return size;
  }

  // Evict oldest entries
  private evictOldest(): void {
    let oldest: { key: string; timestamp: number } | null = null;

    for (const [key, entry] of this.cache.entries()) {
      if (!oldest || entry.timestamp < oldest.timestamp) {
        oldest = { key, timestamp: entry.timestamp };
      }
    }

    if (oldest) {
      this.cache.delete(oldest.key);
      this.stats.evictions++;
      console.log(`[cache] EVICTED: ${oldest.key}`);
    }
  }

  // Get cache statistics
  getStats() {
    const hitRate = this.stats.hits + this.stats.misses > 0
      ? ((this.stats.hits / (this.stats.hits + this.stats.misses)) * 100).toFixed(1)
      : '0';

    return {
      entries: this.cache.size,
      maxEntries: CACHE_MAX_ENTRIES,
      hits: this.stats.hits,
      misses: this.stats.misses,
      hitRate: `${hitRate}%`,
      evictions: this.stats.evictions,
      invalidations: this.stats.invalidations,
    };
  }
}

const cache = new ContentCache();
const HEALTH_CHECK_INTERVAL_MS = 30000; // Check child process health every 30s
const RECONNECT_DELAY_MS = 2000; // Delay before reconnecting

if (!GITHUB_TOKEN) {
  console.error('ERROR: GITHUB_PERSONAL_ACCESS_TOKEN is required');
  process.exit(1);
}

// ============================================================================
// TOOL ANNOTATIONS
// ============================================================================

interface ToolAnnotations {
  readOnlyHint?: boolean;
  destructiveHint?: boolean;
  idempotentHint?: boolean;
  openWorldHint?: boolean;
}

const READ_ONLY_TOOLS = new Set([
  'get_file_contents',
  'get_issue',
  'get_pull_request',
  'get_pull_request_comments',
  'get_pull_request_files',
  'get_pull_request_reviews',
  'get_pull_request_status',
  'list_commits',
  'list_issues',
  'list_pull_requests',
  'search_code',
  'search_issues',
  'search_repositories',
  'search_users',
]);

const DESTRUCTIVE_TOOLS = new Set([
  'merge_pull_request',
  'create_or_update_file',
  'push_files',
]);

const IDEMPOTENT_TOOLS = new Set([
  'get_file_contents',
  'get_issue',
  'get_pull_request',
  'search_code',
  'search_issues',
  'search_repositories',
  'search_users',
  'list_commits',
  'list_issues',
  'list_pull_requests',
  'get_pull_request_comments',
  'get_pull_request_files',
  'get_pull_request_reviews',
  'get_pull_request_status',
]);

function getAnnotations(toolName: string): ToolAnnotations {
  return {
    readOnlyHint: READ_ONLY_TOOLS.has(toolName),
    destructiveHint: DESTRUCTIVE_TOOLS.has(toolName),
    idempotentHint: IDEMPOTENT_TOOLS.has(toolName),
    openWorldHint: false,
  };
}

// ============================================================================
// CONNECTION STATE MANAGEMENT
// ============================================================================

interface ConnectionState {
  client: Client | null;
  transport: StdioClientTransport | null;
  tools: any[];
  isConnected: boolean;
  isConnecting: boolean;
  lastError: string | null;
  reconnectAttempts: number;
  lastToolsListAt: Date | null;
  lastToolsListCount: number | null;
}

const state: ConnectionState = {
  client: null,
  transport: null,
  tools: [],
  isConnected: false,
  isConnecting: false,
  lastError: null,
  reconnectAttempts: 0,
  lastToolsListAt: null,
  lastToolsListCount: null,
};

// ============================================================================
// RESILIENCE UTILITIES
// ============================================================================

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  operationName: string
): Promise<T> {
  let timeoutId: NodeJS.Timeout;

  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(`Operation '${operationName}' timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });

  try {
    const result = await Promise.race([promise, timeoutPromise]);
    clearTimeout(timeoutId!);
    return result;
  } catch (error) {
    clearTimeout(timeoutId!);
    throw error;
  }
}

// Rate limit state tracking
interface RateLimitState {
  remaining: number | null;
  limit: number | null;
  resetAt: Date | null;
  lastChecked: Date | null;
}

const rateLimitState: RateLimitState = {
  remaining: null,
  limit: null,
  resetAt: null,
  lastChecked: null,
};

function getTotalToolCount(): number {
  const customToolCount = 4 + (vercel ? 3 : 0); // batch_read_files, validate_build, check_github_rate_limit, clear_cache + vercel tools
  return state.tools.length + customToolCount;
}

// Check if we're likely rate limited based on error message
function isRateLimitError(error: Error): boolean {
  const msg = error.message.toLowerCase();
  return msg.includes('rate limit') ||
         msg.includes('403') ||
         msg.includes('429') ||
         msg.includes('api rate limit exceeded') ||
         msg.includes('secondary rate limit');
}

// Calculate wait time for rate limit
function getRateLimitWaitTime(): number {
  if (rateLimitState.resetAt) {
    const waitMs = rateLimitState.resetAt.getTime() - Date.now();
    if (waitMs > 0) {
      return Math.min(waitMs + 1000, 60000); // Add 1s buffer, max 60s
    }
  }
  return 60000; // Default to 60s if we don't know reset time
}

// Fetch current rate limit from GitHub API
async function fetchGitHubRateLimit(): Promise<RateLimitState> {
  try {
    const response = await fetch('https://api.github.com/rate_limit', {
      headers: {
        'Authorization': `Bearer ${GITHUB_TOKEN}`,
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
      },
    });

    if (response.ok) {
      const data = await response.json() as { resources?: { core?: any }; rate?: any };
      const core = data.resources?.core || data.rate;

      rateLimitState.remaining = core.remaining;
      rateLimitState.limit = core.limit;
      rateLimitState.resetAt = new Date(core.reset * 1000);
      rateLimitState.lastChecked = new Date();

      console.log(`[wrapper] GitHub rate limit: ${rateLimitState.remaining}/${rateLimitState.limit}, resets at ${rateLimitState.resetAt.toISOString()}`);
    }
  } catch (e) {
    console.error('[wrapper] Failed to fetch rate limit:', e);
  }
  return rateLimitState;
}

async function withRetry<T>(
  operation: () => Promise<T>,
  operationName: string,
  maxRetries: number,
  isIdempotent: boolean
): Promise<T> {
  let lastError: Error | null = null;
  const attempts = isIdempotent ? maxRetries + 1 : 1;

  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Check for rate limiting
      if (isRateLimitError(lastError)) {
        const waitTime = getRateLimitWaitTime();
        console.log(`[wrapper] Rate limited on ${operationName}, waiting ${waitTime / 1000}s before retry...`);
        await fetchGitHubRateLimit(); // Update our rate limit info
        await sleep(waitTime);
        // Don't count this as an attempt - rate limits are recoverable
        attempt--;
        continue;
      }

      if (attempt < attempts) {
        console.log(`[wrapper] ${operationName} failed (attempt ${attempt}/${attempts}), retrying in ${RETRY_DELAY_MS}ms...`);
        await sleep(RETRY_DELAY_MS);

        // Check if we need to reconnect
        if (!state.isConnected && !state.isConnecting) {
          console.log('[wrapper] Connection lost, attempting reconnect before retry...');
          await reconnectToGitHub();
        }
      }
    }
  }

  throw lastError;
}

// ============================================================================
// GITHUB MCP CONNECTION
// ============================================================================

async function connectToGitHub(): Promise<void> {
  if (state.isConnecting) {
    console.log('[wrapper] Connection already in progress, waiting...');
    while (state.isConnecting) {
      await sleep(100);
    }
    return;
  }

  state.isConnecting = true;
  state.isConnected = false;
  state.lastError = null;

  try {
    console.log('[wrapper] Connecting to GitHub MCP server...');

    // Clean up existing connection if any
    if (state.transport) {
      try {
        await state.transport.close();
      } catch (e) {
        // Ignore close errors
      }
    }

    console.log('[wrapper] Attempting to connect to GitHub MCP server...');

    // Use npx to run the MCP server
    state.transport = new StdioClientTransport({
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-github'],
      env: {
        ...process.env,
        GITHUB_PERSONAL_ACCESS_TOKEN: GITHUB_TOKEN,
      } as Record<string, string>,
    });

    state.client = new Client({
      name: 'nova-os-wrapper',
      version: '3.0.0',
    });

    console.log('[wrapper] Connecting to transport...');
    await state.client.connect(state.transport);
    console.log('[wrapper] Transport connected successfully');

    // Fetch available tools
    const toolsResult = await state.client.listTools();
    const nextTools = toolsResult.tools || [];
    if (nextTools.length > 0 || state.tools.length === 0) {
      state.tools = nextTools;
      state.lastToolsListAt = new Date();
      state.lastToolsListCount = getTotalToolCount();
    } else {
      console.warn('[wrapper] listTools returned no tools; keeping last known tool registry');
    }

    state.isConnected = true;
    state.reconnectAttempts = 0;
    console.log(`[wrapper] Connected to GitHub MCP server (${state.tools.length} tools)`);

  } catch (error) {
    state.lastError = error instanceof Error ? error.message : String(error);
    state.isConnected = false;
    console.error('[wrapper] Failed to connect to GitHub MCP server:', state.lastError);
    throw error;
  } finally {
    state.isConnecting = false;
  }
}

async function reconnectToGitHub(): Promise<void> {
  state.reconnectAttempts++;
  console.log(`[wrapper] Reconnecting to GitHub MCP server (attempt ${state.reconnectAttempts})...`);

  await sleep(RECONNECT_DELAY_MS);

  try {
    await connectToGitHub();
  } catch (error) {
    console.error('[wrapper] Reconnection failed:', error);
    // Don't throw - let the caller handle the disconnected state
  }
}

function ensureConnected(): void {
  if (!state.isConnected || !state.client) {
    throw new Error('GitHub MCP server not connected. Please try again in a moment.');
  }
}

// Health check - runs periodically
async function healthCheck(): Promise<void> {
  if (!state.isConnected || !state.client) {
    return;
  }

  try {
    // Try a simple operation to verify connection
    await withTimeout(
      state.client.listTools(),
      5000,
      'health-check'
    );
  } catch (error) {
    console.error('[wrapper] Health check failed, connection may be dead:', error);
    state.isConnected = false;

    // Trigger reconnection
    reconnectToGitHub().catch((e) => {
      console.error('[wrapper] Background reconnection failed:', e);
    });
  }
}

// ============================================================================
// TOOL CALL WRAPPER WITH RESILIENCE
// ============================================================================

async function callToolWithResilience(
  toolName: string,
  args: Record<string, unknown>
): Promise<any> {
  const isIdempotent = IDEMPOTENT_TOOLS.has(toolName);

  return withRetry(
    async () => {
      ensureConnected();

      console.log(`[wrapper] Calling tool: ${toolName}`);

      const result = await withTimeout(
        state.client!.callTool({
          name: toolName,
          arguments: args,
        }),
        TOOL_TIMEOUT_MS,
        toolName
      );

      return result;
    },
    toolName,
    MAX_RETRIES,
    isIdempotent
  );
}

// ============================================================================
// JSON SCHEMA TO ZOD CONVERSION
// ============================================================================

function convertJsonSchemaToZod(schema: any): z.ZodTypeAny {
  if (!schema) {
    return z.any();
  }

  if (Array.isArray(schema)) {
    if (schema.length > 0) {
      return convertJsonSchemaToZod(schema[0]);
    }
    return z.any();
  }

  if (schema.anyOf && Array.isArray(schema.anyOf) && schema.anyOf.length > 0) {
    return convertJsonSchemaToZod(schema.anyOf[0]);
  }
  if (schema.oneOf && Array.isArray(schema.oneOf) && schema.oneOf.length > 0) {
    return convertJsonSchemaToZod(schema.oneOf[0]);
  }

  if (!schema.type) {
    if (schema.properties) {
      const objProps: Record<string, z.ZodTypeAny> = {};
      const required = new Set(schema.required || []);
      for (const [k, v] of Object.entries(schema.properties)) {
        let prop = convertJsonSchemaToZod(v as any);
        if (!required.has(k)) {
          prop = prop.optional();
        }
        objProps[k] = prop;
      }
      return z.object(objProps);
    }
    return z.any();
  }

  let zodType: z.ZodTypeAny;

  switch (schema.type) {
    case 'string':
      zodType = z.string();
      break;
    case 'number':
    case 'integer':
      zodType = z.number();
      break;
    case 'boolean':
      zodType = z.boolean();
      break;
    case 'array':
      if (schema.items) {
        if (Array.isArray(schema.items)) {
          zodType = z.array(convertJsonSchemaToZod(schema.items[0]));
        } else {
          zodType = z.array(convertJsonSchemaToZod(schema.items));
        }
      } else {
        zodType = z.array(z.string());
      }
      break;
    case 'object':
      if (schema.properties) {
        const objProps: Record<string, z.ZodTypeAny> = {};
        const required = new Set(schema.required || []);
        for (const [k, v] of Object.entries(schema.properties)) {
          let prop = convertJsonSchemaToZod(v as any);
          if (!required.has(k)) {
            prop = prop.optional();
          }
          objProps[k] = prop;
        }
        zodType = z.object(objProps);
      } else {
        zodType = z.record(z.string(), z.any());
      }
      break;
    default:
      zodType = z.any();
  }

  if (schema.description) {
    zodType = zodType.describe(schema.description);
  }

  return zodType;
}

// ============================================================================
// MCP SERVER SETUP
// ============================================================================

const server = new McpServer({
  name: 'nova-os-github-connector',
  version: '3.0.0',
  description: 'Resilient GitHub connector for Nova OS with auto-reconnect and retry logic.',
});

function registerProxiedTools(): void {
  for (const tool of state.tools) {
    const annotations = getAnnotations(tool.name);

    const inputSchemaProps: Record<string, z.ZodTypeAny> = {};
    const properties = tool.inputSchema?.properties || {};
    const required = new Set(tool.inputSchema?.required || []);

    for (const [propName, propDef] of Object.entries(properties)) {
      const def = propDef as any;
      let zodType: z.ZodTypeAny;

      switch (def.type) {
        case 'string':
          zodType = z.string();
          if (def.description) zodType = zodType.describe(def.description);
          break;
        case 'number':
        case 'integer':
          zodType = z.number();
          if (def.description) zodType = zodType.describe(def.description);
          break;
        case 'boolean':
          zodType = z.boolean();
          if (def.description) zodType = zodType.describe(def.description);
          break;
        case 'array':
          if (def.items) {
            const itemsSchema = convertJsonSchemaToZod(def.items);
            zodType = z.array(itemsSchema);
          } else if (propName === 'files' && tool.name === 'push_files') {
            zodType = z.array(z.object({
              path: z.string().describe('File path'),
              content: z.string().describe('File content'),
            }));
          } else if (propName === 'comments' && tool.name === 'create_pull_request_review') {
            zodType = z.array(z.object({
              path: z.string().describe('File path to comment on'),
              position: z.number().optional().describe('Line position in the diff'),
              body: z.string().describe('Comment text'),
            }));
          } else {
            zodType = z.array(z.string());
          }
          if (def.description) zodType = zodType.describe(def.description);
          break;
        case 'object':
          if (def.properties) {
            const objProps: Record<string, z.ZodTypeAny> = {};
            for (const [k, v] of Object.entries(def.properties)) {
              objProps[k] = convertJsonSchemaToZod(v as any);
            }
            zodType = z.object(objProps);
          } else {
            zodType = z.record(z.string(), z.any());
          }
          if (def.description) zodType = zodType.describe(def.description);
          break;
        default:
          zodType = z.any();
          if (def.description) zodType = zodType.describe(def.description);
      }

      if (!required.has(propName)) {
        zodType = zodType.optional();
      }

      inputSchemaProps[propName] = zodType;
    }

    const toolConfig: any = {
      title: tool.name.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase()),
      description: `${tool.description || tool.name}${annotations.readOnlyHint ? ' [Read-Only]' : ''}`,
      inputSchema: inputSchemaProps,
      annotations,
    };

    server.registerTool(
      tool.name,
      toolConfig,
      async (args: Record<string, unknown>) => {
        try {
          // CACHING: Check cache for get_file_contents
          if (tool.name === 'get_file_contents') {
            const owner = args.owner as string;
            const repo = args.repo as string;
            const path = args.path as string;
            const branch = args.branch as string | undefined;

            const cacheKey = cache.fileKey(owner, repo, path, branch);
            const cached = cache.get(cacheKey);
            if (cached) {
              return {
                content: [{ type: 'text' as const, text: cached }],
                _cached: true,
              };
            }
          }

          const result = await callToolWithResilience(tool.name, args);

          const content = result.content as Array<{ type: string; text: string }>;
          let responseText: string;

          if (content && content.length === 1 && content[0].type === 'text') {
            responseText = content[0].text;
          } else {
            responseText = content.map((c) => typeof c === 'string' ? c : JSON.stringify(c)).join('\n');
          }

          // CACHING: Store result for get_file_contents
          if (tool.name === 'get_file_contents') {
            const owner = args.owner as string;
            const repo = args.repo as string;
            const path = args.path as string;
            const branch = args.branch as string | undefined;

            const cacheKey = cache.fileKey(owner, repo, path, branch);
            cache.set(cacheKey, responseText);
          }

          // CACHING: Invalidate cache on write operations
          if (tool.name === 'push_files' || tool.name === 'create_or_update_file') {
            const owner = args.owner as string;
            const repo = args.repo as string;
            cache.invalidateRepo(owner, repo);
          }

          return {
            content: [{ type: 'text' as const, text: responseText }],
          };
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          console.error(`[wrapper] Tool ${tool.name} failed:`, errorMessage);

          // Return error as content instead of throwing (prevents connection drop)
          return {
            content: [{
              type: 'text' as const,
              text: JSON.stringify({
                error: true,
                message: errorMessage,
                tool: tool.name,
                suggestion: 'The operation failed. You may retry or try a different approach.'
              })
            }],
          };
        }
      }
    );

    console.log(`[wrapper] Registered: ${tool.name} (readOnly: ${annotations.readOnlyHint}, idempotent: ${annotations.idempotentHint})`);
  }
}

function registerBatchReadFiles(): void {
  server.registerTool(
    'batch_read_files',
    {
      title: 'Batch Read Files',
      description: 'Read multiple files from a GitHub repository in a single call. Much more efficient than calling get_file_contents multiple times. [Read-Only]',
      inputSchema: {
        owner: z.string().describe('Repository owner (username or organization)'),
        repo: z.string().describe('Repository name'),
        paths: z.array(z.string()).describe('Array of file paths to read (e.g., ["src/index.ts", "README.md"])'),
        branch: z.string().optional().describe('Branch name (defaults to main branch)'),
      },
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ owner, repo, paths, branch }) => {
      console.log(`[wrapper] batch_read_files: Reading ${paths.length} files from ${owner}/${repo}`);

      const results: Array<{ path: string; content: string | null; error?: string; cached?: boolean }> = [];
      let cacheHits = 0;

      // Read files with individual error handling and caching
      const promises = paths.map(async (filePath: string) => {
        // Check cache first
        const cacheKey = cache.fileKey(owner, repo, filePath, branch);
        const cached = cache.get(cacheKey);
        if (cached) {
          cacheHits++;
          // Parse cached content to extract actual file content
          try {
            const parsed = JSON.parse(cached);
            if (parsed.content) {
              return { path: filePath, content: parsed.content, cached: true };
            }
            return { path: filePath, content: cached, cached: true };
          } catch {
            return { path: filePath, content: cached, cached: true };
          }
        }

        try {
          const args: Record<string, unknown> = { owner, repo, path: filePath };
          if (branch) args.branch = branch;

          const result = await callToolWithResilience('get_file_contents', args);

          const content = result.content as Array<{ type: string; text: string }>;
          let fileContent = '';
          let rawText = '';

          if (content && content.length > 0) {
            rawText = content[0].text;
            try {
              const parsed = JSON.parse(rawText);
              if (parsed.content) {
                fileContent = parsed.content;
              } else if (typeof parsed === 'string') {
                fileContent = parsed;
              } else {
                fileContent = rawText;
              }
            } catch {
              fileContent = rawText;
            }
          }

          // Cache the raw result
          cache.set(cacheKey, rawText);

          return { path: filePath, content: fileContent, cached: false };
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          console.error(`[wrapper] Error reading ${filePath}:`, errorMessage);
          return { path: filePath, content: null, error: errorMessage };
        }
      });

      const fileResults = await Promise.all(promises);
      results.push(...fileResults);

      const output = {
        repository: `${owner}/${repo}`,
        branch: branch || 'default',
        files_requested: paths.length,
        files_read: results.filter((r) => r.content !== null).length,
        cache_hits: cacheHits,
        files: results,
      };

      return {
        content: [{ type: 'text' as const, text: JSON.stringify(output, null, 2) }],
      };
    }
  );

  console.log('[wrapper] Registered: batch_read_files (readOnly: true, idempotent: true)');
}

function registerValidateBuild(): void {
  server.registerTool(
    'validate_build',
    {
      title: 'Validate Build',
      description: 'Clone a repository branch and run the build to verify it compiles successfully. Use this BEFORE pushing code changes to catch build errors early. Returns build output and success/failure status. [Read-Only - does not modify the repository]',
      inputSchema: {
        owner: z.string().describe('Repository owner (username or organization)'),
        repo: z.string().describe('Repository name'),
        branch: z.string().optional().describe('Branch name to validate (defaults to main)'),
      },
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ owner, repo, branch }) => {
      const branchName = branch || 'main';
      const repoUrl = `https://github.com/${owner}/${repo}.git`;
      const tempDir = path.join(os.tmpdir(), `validate-build-${Date.now()}`);

      console.log(`[wrapper] validate_build: Validating ${owner}/${repo}@${branchName}`);

      const startTime = Date.now();
      let cloneOutput = '';
      let installOutput = '';
      let buildOutput = '';
      let success = false;
      let errorSummary = '';

      try {
        // Create temp directory
        await fs.mkdir(tempDir, { recursive: true });

        // Clone the repository (shallow clone for speed)
        console.log(`[wrapper] Cloning ${repoUrl} to ${tempDir}...`);
        try {
          const cloneResult = await execAsync(
            `git clone --depth 1 --branch ${branchName} ${repoUrl} .`,
            { cwd: tempDir, timeout: 120000 }
          );
          cloneOutput = cloneResult.stdout + cloneResult.stderr;
        } catch (error: any) {
          cloneOutput = error.stdout + error.stderr;
          throw new Error(`Clone failed: ${error.message}`);
        }

        // Install dependencies
        console.log('[wrapper] Installing dependencies...');
        try {
          const installResult = await execAsync(
            'npm ci --prefer-offline 2>&1 || npm install 2>&1',
            { cwd: tempDir, timeout: 300000 }
          );
          installOutput = installResult.stdout + installResult.stderr;
        } catch (error: any) {
          installOutput = error.stdout + error.stderr;
          throw new Error(`Install failed: ${error.message}`);
        }

        // Run the build
        console.log('[wrapper] Running build...');
        try {
          const buildResult = await execAsync(
            'npm run build:web 2>&1',
            { cwd: tempDir, timeout: 300000, maxBuffer: 10 * 1024 * 1024 }
          );
          buildOutput = buildResult.stdout + buildResult.stderr;
          success = true;
        } catch (error: any) {
          buildOutput = error.stdout + error.stderr;

          // Extract meaningful error lines
          const errorLines = buildOutput.split('\n').filter((line: string) =>
            line.includes('Error:') ||
            line.includes('error TS') ||
            line.includes('Module not found') ||
            line.includes("Can't resolve") ||
            line.includes('Parsing') ||
            line.includes('Export') ||
            line.includes("doesn't exist")
          );
          errorSummary = errorLines.slice(0, 20).join('\n');

          throw new Error(`Build failed`);
        }

      } catch (error: any) {
        success = false;
        if (!errorSummary) {
          errorSummary = error.message;
        }
      } finally {
        // Cleanup temp directory
        try {
          await fs.rm(tempDir, { recursive: true, force: true });
          console.log('[wrapper] Cleaned up temp directory');
        } catch (e) {
          console.error('[wrapper] Failed to cleanup temp directory:', e);
        }
      }

      const duration = ((Date.now() - startTime) / 1000).toFixed(1);

      // Truncate outputs to avoid huge responses
      const truncate = (str: string, maxLen: number) =>
        str.length > maxLen ? str.slice(-maxLen) + '\n... (truncated)' : str;

      const output = {
        repository: `${owner}/${repo}`,
        branch: branchName,
        success,
        duration_seconds: parseFloat(duration),
        summary: success
          ? '✅ Build completed successfully! Safe to push changes.'
          : `❌ Build failed. Fix the errors below before pushing.`,
        error_summary: errorSummary || null,
        build_output: truncate(buildOutput, 8000),
      };

      console.log(`[wrapper] validate_build: ${success ? 'SUCCESS' : 'FAILED'} in ${duration}s`);

      return {
        content: [{ type: 'text' as const, text: JSON.stringify(output, null, 2) }],
      };
    }
  );

  console.log('[wrapper] Registered: validate_build (readOnly: true, idempotent: true)');
}

function registerVercelTools(): void {
  if (!vercel) {
    console.log('[wrapper] Skipping Vercel tools (VERCEL_TOKEN not set)');
    return;
  }

  // List deployments for a project
  server.registerTool(
    'list_vercel_deployments',
    {
      title: 'List Vercel Deployments',
      description: 'List recent deployments for a Vercel project. Use this to see deployment history and find deployment IDs. [Read-Only]',
      inputSchema: {
        projectId: z.string().optional().describe('Project ID or name (e.g., "nova-os"). If not provided, lists all deployments.'),
        limit: z.number().optional().describe('Number of deployments to return (default: 10, max: 100)'),
        state: z.string().optional().describe('Filter by state: BUILDING, ERROR, INITIALIZING, QUEUED, READY, CANCELED'),
      },
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ projectId, limit, state: deploymentState }) => {
      console.log(`[wrapper] list_vercel_deployments: project=${projectId || 'all'}, limit=${limit || 10}`);

      try {
        const params: any = {
          limit: Math.min(limit || 10, 100),
        };
        if (projectId) params.projectId = projectId;
        if (deploymentState) params.state = deploymentState;

        const response = await vercel.deployments.getDeployments(params);

        const deployments = (response.deployments || []).map((d: any) => ({
          id: d.uid,
          name: d.name,
          url: d.url ? `https://${d.url}` : null,
          state: d.readyState || d.state,
          createdAt: d.createdAt ? new Date(d.createdAt).toISOString() : null,
          branch: d.meta?.githubCommitRef || d.gitSource?.ref || null,
          commit: d.meta?.githubCommitSha?.slice(0, 7) || null,
          commitMessage: d.meta?.githubCommitMessage || null,
        }));

        return {
          content: [{
            type: 'text' as const,
            text: JSON.stringify({
              count: deployments.length,
              deployments,
            }, null, 2),
          }],
        };
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        console.error('[wrapper] list_vercel_deployments error:', msg);
        return {
          content: [{ type: 'text' as const, text: JSON.stringify({ error: true, message: msg }) }],
        };
      }
    }
  );

  console.log('[wrapper] Registered: list_vercel_deployments (readOnly: true)');

  // Get deployment status and details
  server.registerTool(
    'get_vercel_deployment',
    {
      title: 'Get Vercel Deployment',
      description: 'Get detailed status and information about a specific Vercel deployment. Use after list_vercel_deployments to check a specific deployment. [Read-Only]',
      inputSchema: {
        deploymentId: z.string().describe('Deployment ID (uid) or URL to check'),
      },
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ deploymentId }) => {
      console.log(`[wrapper] get_vercel_deployment: ${deploymentId}`);

      try {
        const deployment = await vercel.deployments.getDeployment({ idOrUrl: deploymentId });

        const result = {
          id: (deployment as any).uid || deployment.id,
          name: deployment.name,
          url: deployment.url ? `https://${deployment.url}` : null,
          state: deployment.readyState,
          createdAt: deployment.createdAt ? new Date(deployment.createdAt).toISOString() : null,
          buildingAt: deployment.buildingAt ? new Date(deployment.buildingAt).toISOString() : null,
          ready: deployment.ready ? new Date(deployment.ready).toISOString() : null,
          branch: (deployment as any).meta?.githubCommitRef || null,
          commit: (deployment as any).meta?.githubCommitSha?.slice(0, 7) || null,
          commitMessage: (deployment as any).meta?.githubCommitMessage || null,
          errorCode: (deployment as any).errorCode || null,
          errorMessage: (deployment as any).errorMessage || null,
        };

        return {
          content: [{ type: 'text' as const, text: JSON.stringify(result, null, 2) }],
        };
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        console.error('[wrapper] get_vercel_deployment error:', msg);
        return {
          content: [{ type: 'text' as const, text: JSON.stringify({ error: true, message: msg }) }],
        };
      }
    }
  );

  console.log('[wrapper] Registered: get_vercel_deployment (readOnly: true)');

  // Get build logs for a deployment
  server.registerTool(
    'get_vercel_build_logs',
    {
      title: 'Get Vercel Build Logs',
      description: 'Get build logs for a Vercel deployment. Use this to see why a deployment failed. Returns the most recent log entries. [Read-Only]',
      inputSchema: {
        deploymentId: z.string().describe('Deployment ID (uid) or URL'),
        limit: z.number().optional().describe('Number of log entries to return (default: 100)'),
      },
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ deploymentId, limit }) => {
      console.log(`[wrapper] get_vercel_build_logs: ${deploymentId}`);

      try {
        const events = await vercel.deployments.getDeploymentEvents({
          idOrUrl: deploymentId,
          direction: 'backward',
          limit: limit || 100,
        });

        // Extract text from events
        const logs: string[] = [];
        if (Array.isArray(events)) {
          for (const event of events) {
            if (event && typeof event === 'object' && 'text' in event) {
              logs.push(`[${(event as any).type || 'log'}] ${(event as any).text}`);
            }
          }
        }

        // Reverse to get chronological order
        logs.reverse();

        // Find error lines
        const errorLines = logs.filter(line =>
          line.toLowerCase().includes('error') ||
          line.includes('Module not found') ||
          line.includes('failed')
        );

        return {
          content: [{
            type: 'text' as const,
            text: JSON.stringify({
              deploymentId,
              totalLogs: logs.length,
              errorCount: errorLines.length,
              errors: errorLines.slice(0, 20),
              recentLogs: logs.slice(-50),
            }, null, 2),
          }],
        };
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        console.error('[wrapper] get_vercel_build_logs error:', msg);
        return {
          content: [{ type: 'text' as const, text: JSON.stringify({ error: true, message: msg }) }],
        };
      }
    }
  );

  console.log('[wrapper] Registered: get_vercel_build_logs (readOnly: true)');
}

function registerGitHubRateLimitTool(): void {
  server.registerTool(
    'check_github_rate_limit',
    {
      title: 'Check GitHub Rate Limit',
      description: 'Check current GitHub API rate limit status. Use this to see how many API calls you have remaining before hitting the rate limit. [Read-Only]',
      inputSchema: {},
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async () => {
      console.log('[wrapper] check_github_rate_limit');

      try {
        const response = await fetch('https://api.github.com/rate_limit', {
          headers: {
            'Authorization': `Bearer ${GITHUB_TOKEN}`,
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
          },
        });

        if (!response.ok) {
          throw new Error(`GitHub API error: ${response.status}`);
        }

        const data = await response.json() as { resources?: { core?: any; search?: any }; rate?: any };

        // Update our internal state
        const core = data.resources?.core || data.rate;
        rateLimitState.remaining = core.remaining;
        rateLimitState.limit = core.limit;
        rateLimitState.resetAt = new Date(core.reset * 1000);
        rateLimitState.lastChecked = new Date();

        const result = {
          core: {
            limit: core.limit,
            remaining: core.remaining,
            used: core.used,
            resetAt: rateLimitState.resetAt.toISOString(),
            minutesUntilReset: Math.ceil((rateLimitState.resetAt.getTime() - Date.now()) / 60000),
          },
          search: data.resources?.search ? {
            limit: data.resources.search.limit,
            remaining: data.resources.search.remaining,
            used: data.resources.search.used,
            resetAt: new Date(data.resources.search.reset * 1000).toISOString(),
          } : null,
          status: core.remaining > 100 ? '✅ Healthy' :
                  core.remaining > 10 ? '⚠️ Low' :
                  '🛑 Critical - wait before making more requests',
        };

        return {
          content: [{ type: 'text' as const, text: JSON.stringify(result, null, 2) }],
        };
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        console.error('[wrapper] check_github_rate_limit error:', msg);
        return {
          content: [{ type: 'text' as const, text: JSON.stringify({ error: true, message: msg }) }],
        };
      }
    }
  );

  console.log('[wrapper] Registered: check_github_rate_limit (readOnly: true)');
}

function registerCacheTool(): void {
  server.registerTool(
    'clear_cache',
    {
      title: 'Clear Cache',
      description: 'Clear the file content cache. Use this if you need fresh data after external changes to the repository. Returns cache statistics before clearing.',
      inputSchema: {
        owner: z.string().optional().describe('Optional: Only clear cache for this repo owner'),
        repo: z.string().optional().describe('Optional: Only clear cache for this repo (requires owner)'),
      },
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ owner, repo }) => {
      console.log(`[wrapper] clear_cache: owner=${owner || 'all'}, repo=${repo || 'all'}`);

      const statsBefore = cache.getStats();
      let cleared = 0;

      if (owner && repo) {
        // Clear specific repo
        cache.invalidateRepo(owner, repo);
        cleared = statsBefore.invalidations; // approximate
      } else {
        // Clear everything
        cleared = cache.clear();
      }

      const statsAfter = cache.getStats();

      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify({
            cleared: owner && repo ? `${owner}/${repo}` : 'all',
            entriesCleared: cleared,
            statsBefore,
            statsAfter,
          }, null, 2),
        }],
      };
    }
  );

  console.log('[wrapper] Registered: clear_cache');
}

// ============================================================================
// EXPRESS APP
// ============================================================================

const app = express();
app.use(express.json({ limit: '10mb' }));

app.get('/health', (_req, res) => {
  const cacheStats = cache.getStats();
  res.json({
    status: state.isConnected ? 'ok' : 'degraded',
    connected: state.isConnected,
    connecting: state.isConnecting,
    tools: getTotalToolCount(),
    lastToolsListAt: state.lastToolsListAt ? state.lastToolsListAt.toISOString() : null,
    lastToolsListCount: state.lastToolsListCount,
    vercelEnabled: !!vercel,
    cache: {
      entries: cacheStats.entries,
      hitRate: cacheStats.hitRate,
      hits: cacheStats.hits,
      misses: cacheStats.misses,
    },
    rateLimitRemaining: rateLimitState.remaining,
    rateLimitResetAt: rateLimitState.resetAt?.toISOString() || null,
    lastError: state.lastError,
    reconnectAttempts: state.reconnectAttempts,
  });
});

app.post('/mcp', async (req, res) => {
  try {
    if (req.body?.method === 'resources/list') {
      res.json({
        jsonrpc: '2.0',
        id: req.body?.id ?? null,
        result: { resources: [] },
      });
      return;
    }

    const method = req.body?.method as string | undefined;
    const isDiscoveryCall = method === 'tools/list' || method === 'initialize';

    if (method === 'tools/list') {
      state.lastToolsListAt = new Date();
      state.lastToolsListCount = getTotalToolCount();
    }

    // Check connection before handling request
    if (!isDiscoveryCall && !state.isConnected && !state.isConnecting) {
      console.log('[wrapper] Connection lost, attempting reconnect...');
      await reconnectToGitHub();
    }

    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
      enableJsonResponse: true,
    });

    await server.connect(transport);

    res.on('close', () => {
      transport.close().catch((err) => console.error('[wrapper] Transport close error:', err));
    });

    await transport.handleRequest(req, res, req.body);
  } catch (error) {
    console.error('[wrapper] MCP request error:', error);
    res.status(500).json({
      jsonrpc: '2.0',
      error: {
        code: -32603,
        message: error instanceof Error ? error.message : 'Internal server error',
      },
      id: null,
    });
  }
});

app.options('/mcp', (_req, res) => {
  res.status(204).send();
});

// ============================================================================
// STARTUP
// ============================================================================

async function main(): Promise<void> {
  console.log('========================================');
  console.log('  Nova OS GitHub Connector v3');
  console.log('  (Resilient Wrapper)');
  console.log('========================================');
  console.log('');
  console.log('Resilience settings:');
  console.log(`  - Tool timeout:     ${TOOL_TIMEOUT_MS / 1000}s`);
  console.log(`  - Max retries:      ${MAX_RETRIES}`);
  console.log(`  - Health check:     every ${HEALTH_CHECK_INTERVAL_MS / 1000}s`);
  console.log('');
  console.log('Caching settings:');
  console.log(`  - Cache TTL:        ${CACHE_TTL_MS / 1000}s`);
  console.log(`  - Max entries:      ${CACHE_MAX_ENTRIES}`);
  console.log('');

  // Start server FIRST so health checks work even if GitHub connection fails
  await new Promise<void>((resolve) => {
    app.listen(PORT, () => {
      console.log(`Server listening on port ${PORT}`);
      console.log('');
      console.log('Endpoints:');
      console.log(`  - MCP:    http://localhost:${PORT}/mcp`);
      console.log(`  - Health: http://localhost:${PORT}/health`);
      console.log('');
      resolve();
    });
  });

  // Register custom tools (no GitHub MCP dependency)
  // These tools work independently and don't require the MCP child process
  registerValidateBuild();
  registerVercelTools();
  registerGitHubRateLimitTool();
  registerCacheTool();

  // Try to connect to GitHub MCP server for additional tools (optional)
  // Set SKIP_GITHUB_MCP=1 to skip this and only use custom tools
  const skipGitHubMCP = process.env.SKIP_GITHUB_MCP === '1';

  if (skipGitHubMCP) {
    console.log('[wrapper] SKIP_GITHUB_MCP=1, skipping GitHub MCP connection');
    console.log('[wrapper] Available tools: validate_build, check_github_rate_limit, clear_cache');
    if (vercel) {
      console.log('[wrapper] + Vercel tools: list_vercel_deployments, get_vercel_deployment, get_vercel_build_logs');
    }
  } else {
    try {
      console.log('[wrapper] Attempting to connect to GitHub MCP server...');
      await connectToGitHub();

      // Register GitHub MCP tools after successful connection
      registerProxiedTools();
      registerBatchReadFiles();

      const totalTools = state.tools.length + 4 + (vercel ? 3 : 0);
      console.log('');
      console.log(`Total tools registered: ${totalTools}`);
      console.log(`  - GitHub MCP tools: ${state.tools.length}`);
      console.log(`  - Custom tools: ${4 + (vercel ? 3 : 0)}`);
      console.log('');
    } catch (error) {
      console.error('[wrapper] GitHub MCP connection failed (server will continue with custom tools only):', error);
      console.log('[wrapper] Available tools: validate_build, check_github_rate_limit, clear_cache');
      if (vercel) {
        console.log('[wrapper] + Vercel tools: list_vercel_deployments, get_vercel_deployment, get_vercel_build_logs');
      }
    }
  }

  // Start health check interval
  setInterval(() => {
    healthCheck().catch((e) => console.error('[wrapper] Health check error:', e));
  }, HEALTH_CHECK_INTERVAL_MS);

  console.log('Render URL:');
  console.log('  https://nova-os-connector.onrender.com/mcp');
  console.log('');
}

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n[wrapper] Shutting down...');
  if (state.transport) {
    await state.transport.close();
  }
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\n[wrapper] Shutting down...');
  if (state.transport) {
    await state.transport.close();
  }
  process.exit(0);
});

// Handle uncaught errors gracefully
process.on('uncaughtException', (error) => {
  console.error('[wrapper] Uncaught exception:', error);
  // Don't exit - try to keep running
});

process.on('unhandledRejection', (reason) => {
  console.error('[wrapper] Unhandled rejection:', reason);
  // Don't exit - try to keep running
});

main().catch((error) => {
  console.error('[wrapper] Fatal error:', error);
  process.exit(1);
});
