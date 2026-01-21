import 'dotenv/config';
import express from 'express';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { z } from 'zod';
import * as fs from 'fs/promises';
import crypto from 'crypto';

// ============================================================================
// CONFIGURATION
// ============================================================================

const PORT = parseInt(process.env.PORT || '8000', 10);
const GITHUB_TOKEN = process.env.GITHUB_PERSONAL_ACCESS_TOKEN;
const VERCEL_TOKEN = process.env.VERCEL_TOKEN;
const VERCEL_TEAM_ID = process.env.VERCEL_TEAM_ID;

// Configurable build command (defaults to 'npm run build')
const BUILD_COMMAND = process.env.BUILD_COMMAND || 'npm run build';

// Request queue configuration
const MAX_CONCURRENT_REQUESTS = parseInt(process.env.MAX_CONCURRENT_REQUESTS || '5', 10);

// File size guardrails
const MAX_FILE_BYTES = parseInt(process.env.MAX_FILE_BYTES || '4194304', 10); // 4MB default
const ALLOW_BINARY_FILES = process.env.ALLOW_BINARY_FILES === 'true';

// Validate build offloading
const VALIDATE_BUILD_MODE = (process.env.VALIDATE_BUILD_MODE || 'worker').toLowerCase();
const VALIDATE_BUILD_WORKER_URL = process.env.VALIDATE_BUILD_WORKER_URL;
const VALIDATE_BUILD_WORKFLOW = process.env.VALIDATE_BUILD_WORKFLOW;
const VALIDATE_BUILD_WORKFLOW_REF = process.env.VALIDATE_BUILD_WORKFLOW_REF || 'main';

// Default enabled toolsets (can be overridden via env)
const DEFAULT_ENABLED_TOOLSETS = (process.env.ENABLED_TOOLSETS || 'repos,issues,pulls,search,utility,vercel,toolsets').split(',').map(s => s.trim());
const ALLOWED_REPOS = (process.env.ALLOWED_REPOS || 'blakegallagher1/nova-os')
  .split(',')
  .map((repo) => repo.trim().toLowerCase())
  .filter(Boolean);

if (!GITHUB_TOKEN) {
  console.error('ERROR: GITHUB_PERSONAL_ACCESS_TOKEN is required');
  process.exit(1);
}

// ============================================================================
// TOOLSET SYSTEM
// ============================================================================

/**
 * A toolset groups related tools together and provides metadata about them.
 */
interface Toolset {
  name: string;
  description: string;
  enabled: boolean;
  readOnly: boolean;
  tools: ToolDefinition[];
}

/**
 * Definition of a tool within a toolset.
 */
interface ToolDefinition {
  name: string;
  title: string;
  description: string;
  readOnly: boolean;
  destructive: boolean;
}

/**
 * Manages toolsets - enables/disables groups of tools dynamically.
 */
class ToolsetManager {
  private toolsets = new Map<string, Toolset>();
  private registeredTools = new Set<string>();

  /**
   * Register a new toolset.
   */
  registerToolset(
    name: string,
    description: string,
    readOnly: boolean,
    tools: ToolDefinition[],
    enabled: boolean = true
  ): void {
    this.toolsets.set(name, {
      name,
      description,
      enabled,
      readOnly,
      tools,
    });

    // Track all tool names
    for (const tool of tools) {
      this.registeredTools.add(tool.name);
    }
  }

  /**
   * Enable a toolset.
   */
  enableToolset(name: string): boolean {
    const toolset = this.toolsets.get(name);
    if (!toolset) return false;
    if (name === 'toolsets') return false; // Can't disable toolsets meta-toolset
    toolset.enabled = true;
    return true;
  }

  /**
   * Disable a toolset.
   */
  disableToolset(name: string): boolean {
    const toolset = this.toolsets.get(name);
    if (!toolset) return false;
    if (name === 'toolsets') return false; // Can't disable toolsets meta-toolset
    toolset.enabled = false;
    return true;
  }

  /**
   * Check if a toolset is enabled.
   */
  isToolsetEnabled(name: string): boolean {
    return this.toolsets.get(name)?.enabled ?? false;
  }

  /**
   * Check if a specific tool is available (its toolset is enabled).
   */
  isToolEnabled(toolName: string): boolean {
    for (const toolset of this.toolsets.values()) {
      if (toolset.tools.some(t => t.name === toolName)) {
        return toolset.enabled;
      }
    }
    return false;
  }

  /**
   * Check if a specific tool is destructive.
   */
  isToolDestructive(toolName: string): boolean {
    for (const toolset of this.toolsets.values()) {
      const tool = toolset.tools.find(t => t.name === toolName);
      if (tool) return tool.destructive;
    }
    return false;
  }

  /**
   * Get all toolsets with their status.
   */
  getAllToolsets(): Array<{
    name: string;
    description: string;
    enabled: boolean;
    readOnly: boolean;
    toolCount: number;
    tools: string[];
  }> {
    return Array.from(this.toolsets.values()).map(ts => ({
      name: ts.name,
      description: ts.description,
      enabled: ts.enabled,
      readOnly: ts.readOnly,
      toolCount: ts.tools.length,
      tools: ts.tools.map(t => t.name),
    }));
  }

  /**
   * Get tools in a specific toolset.
   */
  getToolsetTools(name: string): ToolDefinition[] | null {
    const toolset = this.toolsets.get(name);
    return toolset ? toolset.tools : null;
  }

  /**
   * Get all enabled tools across all enabled toolsets.
   */
  getEnabledTools(): ToolDefinition[] {
    const tools: ToolDefinition[] = [];
    for (const toolset of this.toolsets.values()) {
      if (toolset.enabled) {
        tools.push(...toolset.tools);
      }
    }
    return tools;
  }

  /**
   * Get summary statistics.
   */
  getStats(): {
    totalToolsets: number;
    enabledToolsets: number;
    totalTools: number;
    enabledTools: number;
  } {
    let enabledToolsets = 0;
    let totalTools = 0;
    let enabledTools = 0;

    for (const toolset of this.toolsets.values()) {
      totalTools += toolset.tools.length;
      if (toolset.enabled) {
        enabledToolsets++;
        enabledTools += toolset.tools.length;
      }
    }

    return {
      totalToolsets: this.toolsets.size,
      enabledToolsets,
      totalTools,
      enabledTools,
    };
  }
}

// Create global toolset manager instance
const toolsetManager = new ToolsetManager();

// ============================================================================
// SECURITY: PATH VALIDATION
// ============================================================================

/**
 * Validates a file path to prevent path traversal attacks.
 * Throws an error if the path is potentially malicious.
 */
function validateFilePath(filePath: string): void {
  // Check for path traversal attempts
  if (filePath.includes('..')) {
    throw new Error('Invalid path: Path traversal detected (contains "..")');
  }

  // Check for absolute paths (should be relative to repo root)
  if (filePath.startsWith('/')) {
    throw new Error('Invalid path: Absolute paths not allowed (starts with "/")');
  }

  // Check for backslash path traversal (Windows-style)
  if (filePath.includes('\\..') || filePath.includes('..\\')) {
    throw new Error('Invalid path: Path traversal detected (contains backslash traversal)');
  }

  // Check for null bytes (can bypass path checks in some systems)
  if (filePath.includes('\x00')) {
    throw new Error('Invalid path: Null byte detected');
  }

  // Check for URL-encoded traversal
  if (filePath.includes('%2e%2e') || filePath.includes('%2E%2E')) {
    throw new Error('Invalid path: URL-encoded path traversal detected');
  }
}

/**
 * Validates multiple file paths.
 */
function validateFilePaths(paths: string[]): void {
  for (const p of paths) {
    validateFilePath(p);
  }
}

// ============================================================================
// SECURITY: ERROR SANITIZATION
// ============================================================================

/**
 * Patterns that indicate sensitive information in error messages.
 */
const SENSITIVE_PATTERNS = [
  /ghp_[a-zA-Z0-9]{36}/g,                    // GitHub PAT
  /ghs_[a-zA-Z0-9]{36}/g,                    // GitHub App token
  /github_pat_[a-zA-Z0-9_]{82}/g,            // Fine-grained PAT
  /x-access-token:[^@]+@/gi,                 // Git HTTPS auth URL
  /sk-[a-zA-Z0-9]{48}/g,                     // OpenAI API key
  /Bearer\s+[a-zA-Z0-9._-]+/gi,              // Bearer tokens
  /Authorization:\s*[^\s]+/gi,               // Authorization headers
  /password[=:]\s*[^\s&]+/gi,                // Password in query strings
  /token[=:]\s*[^\s&]+/gi,                   // Token in query strings
  /\/Users\/[^\/]+/g,                        // MacOS user paths
  /\/home\/[^\/]+/g,                         // Linux user paths
  /C:\\Users\\[^\\]+/gi,                     // Windows user paths
  /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,  // Email addresses
];

/**
 * Sanitizes an error message by removing sensitive information.
 */
function sanitizeErrorMessage(message: string): string {
  let sanitized = message;

  for (const pattern of SENSITIVE_PATTERNS) {
    sanitized = sanitized.replace(pattern, '[REDACTED]');
  }

  // Truncate very long error messages
  if (sanitized.length > 1000) {
    sanitized = sanitized.substring(0, 1000) + '... (truncated)';
  }

  return sanitized;
}

/**
 * Creates a safe error response for returning to clients.
 */
function createSafeErrorResponse(error: unknown, toolName: string): DualOutput {
  const rawMessage = error instanceof Error ? error.message : String(error);
  const sanitizedMessage = sanitizeErrorMessage(rawMessage);

  // Extract additional context from domain-specific errors
  let errorCode: string | undefined;
  let retryable = false;
  let retryAfter: number | undefined;

  if (error instanceof GitHubRateLimitError) {
    errorCode = 'RATE_LIMIT_EXCEEDED';
    retryable = true;
    retryAfter = error.retryAfter;
  } else if (error instanceof GitHubNotFoundError) {
    errorCode = 'NOT_FOUND';
  } else if (error instanceof GitHubAuthError) {
    errorCode = 'AUTH_ERROR';
  } else if (error instanceof GitHubValidationError) {
    errorCode = 'VALIDATION_ERROR';
  } else if (error instanceof GitHubConflictError) {
    errorCode = 'CONFLICT';
    retryable = true;
  } else if (error instanceof GitHubTimeoutError) {
    errorCode = 'TIMEOUT';
    retryable = true;
  } else if (error instanceof ToolRateLimitError) {
    errorCode = 'TOOL_RATE_LIMIT';
    retryable = true;
    retryAfter = Math.ceil(error.retryAfterMs / 1000);
  }

  const payload = {
    success: false,
    error: true,
    code: errorCode,
    message: sanitizedMessage,
    tool: toolName,
    retryable,
    ...(retryAfter !== undefined ? { retry_after_seconds: retryAfter } : {}),
    suggestion: retryable
      ? `The operation failed but may succeed if retried${retryAfter ? ` after ${retryAfter} seconds` : ''}.`
      : 'The operation failed. You may need to try a different approach.',
  };

  return {
    content: [{ type: 'text' as const, text: JSON.stringify(payload, null, 2) }],
    structuredContent: payload,
  };
}

// ============================================================================
// DOMAIN-SPECIFIC EXCEPTIONS
// ============================================================================

/**
 * Base class for GitHub API errors with structured error data.
 */
class GitHubAPIError extends Error {
  constructor(
    message: string,
    public readonly statusCode: number,
    public readonly endpoint: string,
    public readonly rawResponse?: string
  ) {
    super(message);
    this.name = 'GitHubAPIError';
  }
}

/**
 * Thrown when GitHub API rate limit is exceeded.
 * Contains retry information from X-RateLimit headers.
 */
class GitHubRateLimitError extends GitHubAPIError {
  constructor(
    public readonly limit: number,
    public readonly remaining: number,
    public readonly resetTimestamp: number,
    public readonly retryAfter: number,
    endpoint: string
  ) {
    super(
      `GitHub API rate limit exceeded. Limit: ${limit}, Remaining: ${remaining}. Resets at ${new Date(resetTimestamp * 1000).toISOString()}`,
      429,
      endpoint
    );
    this.name = 'GitHubRateLimitError';
  }
}

/**
 * Thrown when a requested resource is not found (404).
 */
class GitHubNotFoundError extends GitHubAPIError {
  constructor(
    public readonly resourceType: string,
    public readonly resourcePath: string,
    endpoint: string,
    rawResponse?: string
  ) {
    super(`${resourceType} not found: ${resourcePath}`, 404, endpoint, rawResponse);
    this.name = 'GitHubNotFoundError';
  }
}

/**
 * Thrown when authentication fails (401/403).
 */
class GitHubAuthError extends GitHubAPIError {
  constructor(
    message: string,
    statusCode: 401 | 403,
    endpoint: string,
    public readonly requiredScopes?: string[]
  ) {
    super(message, statusCode, endpoint);
    this.name = 'GitHubAuthError';
  }
}

/**
 * Thrown when request validation fails (422).
 */
class GitHubValidationError extends GitHubAPIError {
  constructor(
    message: string,
    endpoint: string,
    public readonly errors?: Array<{ resource: string; field: string; code: string; message?: string }>
  ) {
    super(message, 422, endpoint);
    this.name = 'GitHubValidationError';
  }
}

/**
 * Thrown when there's a conflict (409), e.g., merge conflicts.
 */
class GitHubConflictError extends GitHubAPIError {
  constructor(
    message: string,
    endpoint: string,
    rawResponse?: string
  ) {
    super(message, 409, endpoint, rawResponse);
    this.name = 'GitHubConflictError';
  }
}

/**
 * Thrown when a request times out.
 */
class GitHubTimeoutError extends GitHubAPIError {
  constructor(
    endpoint: string,
    public readonly timeoutMs: number
  ) {
    super(`Request to ${endpoint} timed out after ${timeoutMs}ms`, 408, endpoint);
    this.name = 'GitHubTimeoutError';
  }
}

class ToolRateLimitError extends Error {
  constructor(
    public readonly toolName: string,
    public readonly limit: number,
    public readonly windowMs: number,
    public readonly retryAfterMs: number
  ) {
    super(`Rate limit exceeded for ${toolName}. Limit: ${limit} per ${Math.round(windowMs / 1000)}s.`);
    this.name = 'ToolRateLimitError';
  }
}

/**
 * Parse GitHub API error response and throw appropriate domain-specific error.
 */
function parseGitHubError(
  response: Response,
  endpoint: string,
  responseText: string
): never {
  const status = response.status;

  // Parse rate limit headers
  const rateLimit = parseInt(response.headers.get('X-RateLimit-Limit') || '0', 10);
  const rateRemaining = parseInt(response.headers.get('X-RateLimit-Remaining') || '0', 10);
  const rateReset = parseInt(response.headers.get('X-RateLimit-Reset') || '0', 10);
  const retryAfter = parseInt(response.headers.get('Retry-After') || '0', 10);

  // Try to parse error body
  let errorBody: any = {};
  try {
    errorBody = JSON.parse(responseText);
  } catch {
    // Not JSON, use raw text
  }

  switch (status) {
    case 401:
    case 403:
      throw new GitHubAuthError(
        errorBody.message || `Authentication failed (${status})`,
        status as 401 | 403,
        endpoint,
        errorBody.required_scopes
      );

    case 404:
      // Try to extract resource type from endpoint
      const resourceMatch = endpoint.match(/\/repos\/[^/]+\/[^/]+\/(\w+)/);
      const resourceType = resourceMatch ? resourceMatch[1] : 'Resource';
      throw new GitHubNotFoundError(resourceType, endpoint, endpoint, responseText);

    case 409:
      throw new GitHubConflictError(
        errorBody.message || 'Conflict detected',
        endpoint,
        responseText
      );

    case 422:
      throw new GitHubValidationError(
        errorBody.message || 'Validation failed',
        endpoint,
        errorBody.errors
      );

    case 429:
      const retrySeconds = retryAfter || Math.max(1, rateReset - Math.floor(Date.now() / 1000));
      throw new GitHubRateLimitError(rateLimit, rateRemaining, rateReset, retrySeconds, endpoint);

    default:
      throw new GitHubAPIError(
        errorBody.message || `GitHub API error (${status}): ${responseText}`,
        status,
        endpoint,
        responseText
      );
  }
}

// ============================================================================
// REQUEST QUEUE (Concurrency Limiter)
// ============================================================================

/**
 * Simple semaphore for limiting concurrent requests.
 */
class RequestQueue {
  private running = 0;
  private queue: Array<() => void> = [];
  private stats = { queued: 0, completed: 0, maxConcurrent: 0 };

  constructor(private maxConcurrent: number) {}

  async acquire(): Promise<void> {
    if (this.running < this.maxConcurrent) {
      this.running++;
      if (this.running > this.stats.maxConcurrent) {
        this.stats.maxConcurrent = this.running;
      }
      return;
    }

    this.stats.queued++;
    return new Promise<void>((resolve) => {
      this.queue.push(() => {
        this.running++;
        if (this.running > this.stats.maxConcurrent) {
          this.stats.maxConcurrent = this.running;
        }
        resolve();
      });
    });
  }

  release(): void {
    this.running--;
    this.stats.completed++;
    const next = this.queue.shift();
    if (next) next();
  }

  getStats() {
    return {
      running: this.running,
      queued: this.queue.length,
      totalQueued: this.stats.queued,
      completed: this.stats.completed,
      maxConcurrent: this.stats.maxConcurrent,
    };
  }
}

const requestQueue = new RequestQueue(MAX_CONCURRENT_REQUESTS);

/**
 * Executes an async function with concurrency limiting.
 */
async function withConcurrencyLimit<T>(fn: () => Promise<T>): Promise<T> {
  await requestQueue.acquire();
  try {
    return await fn();
  } finally {
    requestQueue.release();
  }
}

// ============================================================================
// GITHUB API HELPER (with Retry, Timeout, and Domain-Specific Errors)
// ============================================================================

// Default configuration
const DEFAULT_TIMEOUT_MS = 30000;
const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_RETRY_DELAY_MS = 1000;

/**
 * Options for GitHub API requests.
 */
interface GitHubAPIOptions extends RequestInit {
  /** Request timeout in milliseconds (default: 30000) */
  timeoutMs?: number;
  /** Maximum retry attempts (default: 3) */
  maxRetries?: number;
  /** Base delay for exponential backoff in ms (default: 1000) */
  retryDelayMs?: number;
  /** Whether to skip retry logic (default: false) */
  noRetry?: boolean;
}

/**
 * Sleep for a specified number of milliseconds.
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Fetch with timeout support.
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new GitHubTimeoutError(url, timeoutMs);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Determines if an error is retryable.
 */
function isRetryableError(error: unknown): boolean {
  if (error instanceof GitHubRateLimitError) return true;
  if (error instanceof GitHubTimeoutError) return true;
  if (error instanceof GitHubConflictError) return true;
  if (error instanceof GitHubAPIError) {
    // Retry on server errors (5xx)
    return error.statusCode >= 500 && error.statusCode < 600;
  }
  // Retry on network errors
  if (error instanceof TypeError && error.message.includes('fetch')) return true;
  return false;
}

/**
 * Calculate delay for exponential backoff with jitter.
 */
function calculateBackoffDelay(attempt: number, baseDelayMs: number, error?: unknown): number {
  // If it's a rate limit error, use the retry-after value
  if (error instanceof GitHubRateLimitError && error.retryAfter > 0) {
    return error.retryAfter * 1000;
  }

  // Exponential backoff with jitter: baseDelay * 2^attempt * (0.5 to 1.5)
  const exponentialDelay = baseDelayMs * Math.pow(2, attempt);
  const jitter = 0.5 + Math.random();
  return Math.min(exponentialDelay * jitter, 60000); // Cap at 60 seconds
}

/**
 * Enhanced GitHub API caller with retry, timeout, and domain-specific errors.
 */
async function githubAPI(
  endpoint: string,
  options: GitHubAPIOptions = {}
): Promise<any> {
  const {
    timeoutMs = DEFAULT_TIMEOUT_MS,
    maxRetries = DEFAULT_MAX_RETRIES,
    retryDelayMs = DEFAULT_RETRY_DELAY_MS,
    noRetry = false,
    ...fetchOptions
  } = options;

  const url = endpoint.startsWith('http') ? endpoint : `https://api.github.com${endpoint}`;
  const effectiveMaxRetries = noRetry ? 1 : maxRetries;

  let lastError: unknown;

  for (let attempt = 0; attempt < effectiveMaxRetries; attempt++) {
    try {
      const response = await fetchWithTimeout(
        url,
        {
          ...fetchOptions,
          headers: {
            'Authorization': `Bearer ${GITHUB_TOKEN}`,
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
            'Content-Type': 'application/json',
            ...fetchOptions.headers,
          },
        },
        timeoutMs
      );

      if (!response.ok) {
        const responseText = await response.text();
        parseGitHubError(response, endpoint, responseText); // Always throws
      }

      return response.json();
    } catch (error) {
      lastError = error;

      // Check if we should retry
      const isLastAttempt = attempt === effectiveMaxRetries - 1;
      if (isLastAttempt || !isRetryableError(error)) {
        throw error;
      }

      // Calculate backoff delay
      const delay = calculateBackoffDelay(attempt, retryDelayMs, error);
      console.log(`[Retry] Attempt ${attempt + 1}/${effectiveMaxRetries} failed for ${endpoint}. Retrying in ${Math.round(delay)}ms...`);
      await sleep(delay);
    }
  }

  throw lastError;
}

// ============================================================================
// CACHING
// ============================================================================

const CACHE_TTL_MS = 5 * 60 * 1000;
const CACHE_MAX_ENTRIES = 500;
const CACHE_MAX_ENTRY_BYTES = parseInt(process.env.CACHE_MAX_ENTRY_BYTES || '1048576', 10); // 1MB
const CACHE_MAX_TOTAL_BYTES = parseInt(process.env.CACHE_MAX_TOTAL_BYTES || '268435456', 10); // 256MB
const CACHE_TTL_FILE_MS = parseInt(process.env.CACHE_TTL_FILE_MS || String(CACHE_TTL_MS), 10);
const CACHE_TTL_BATCH_MS = parseInt(process.env.CACHE_TTL_BATCH_MS || String(CACHE_TTL_MS), 10);

interface CacheEntry {
  data: any;
  timestamp: number;
  ttl: number;
  hits: number;
  sizeBytes: number;
}

class ContentCache {
  private cache = new Map<string, CacheEntry>();
  private stats = { hits: 0, misses: 0, evictions: 0, invalidations: 0 };
  private totalBytes = 0;

  fileKey(owner: string, repo: string, path: string, branch?: string): string {
    return `file:${owner}/${repo}/${branch || 'default'}:${path}`;
  }

  get(key: string): any | null {
    const entry = this.cache.get(key);
    if (!entry) { this.stats.misses++; return null; }
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      this.stats.misses++;
      return null;
    }
    entry.hits++;
    this.stats.hits++;
    return entry.data;
  }

  set(key: string, data: any, ttl: number = CACHE_TTL_MS): boolean {
    const sizeBytes = this.estimateSize(data);
    if (sizeBytes > CACHE_MAX_ENTRY_BYTES) {
      return false;
    }

    while (this.cache.size >= CACHE_MAX_ENTRIES || this.totalBytes + sizeBytes > CACHE_MAX_TOTAL_BYTES) {
      if (!this.evictOldest()) break;
    }

    this.cache.set(key, { data, timestamp: Date.now(), ttl, hits: 0, sizeBytes });
    this.totalBytes += sizeBytes;
    return true;
  }

  invalidateRepo(owner: string, repo: string): void {
    const prefix = `file:${owner}/${repo}/`;
    let count = 0;
    for (const key of this.cache.keys()) {
      if (key.startsWith(prefix)) {
        const entry = this.cache.get(key);
        if (entry) this.totalBytes -= entry.sizeBytes;
        this.cache.delete(key);
        count++;
      }
    }
    this.stats.invalidations += count;
  }

  clear(): number {
    const size = this.cache.size;
    this.cache.clear();
    this.totalBytes = 0;
    return size;
  }

  private evictOldest(): boolean {
    let oldest: { key: string; timestamp: number } | null = null;
    for (const [key, entry] of this.cache.entries()) {
      if (!oldest || entry.timestamp < oldest.timestamp) oldest = { key, timestamp: entry.timestamp };
    }
    if (oldest) {
      const entry = this.cache.get(oldest.key);
      if (entry) this.totalBytes -= entry.sizeBytes;
      this.cache.delete(oldest.key);
      this.stats.evictions++;
      return true;
    }
    return false;
  }

  getStats() {
    const hitRate = this.stats.hits + this.stats.misses > 0
      ? ((this.stats.hits / (this.stats.hits + this.stats.misses)) * 100).toFixed(1)
      : '0';
    return {
      entries: this.cache.size,
      hitRate: `${hitRate}%`,
      totalBytes: this.totalBytes,
      maxEntryBytes: CACHE_MAX_ENTRY_BYTES,
      maxTotalBytes: CACHE_MAX_TOTAL_BYTES,
      ...this.stats,
    };
  }

  private estimateSize(data: any): number {
    if (typeof data === 'string') {
      return Buffer.byteLength(data, 'utf8');
    }
    try {
      return Buffer.byteLength(JSON.stringify(data), 'utf8');
    } catch {
      return 0;
    }
  }
}

const cache = new ContentCache();

// ============================================================================
// TOOL RATE LIMITING
// ============================================================================

interface ToolRateLimitConfig {
  limit: number;
  windowMs: number;
}

class ToolRateLimiter {
  private windowStart = Date.now();
  private count = 0;

  constructor(private readonly config: ToolRateLimitConfig) {}

  check(): { allowed: boolean; retryAfterMs: number } {
    const now = Date.now();
    if (now - this.windowStart >= this.config.windowMs) {
      this.windowStart = now;
      this.count = 0;
    }

    this.count += 1;
    if (this.count > this.config.limit) {
      const retryAfterMs = this.config.windowMs - (now - this.windowStart);
      return { allowed: false, retryAfterMs: Math.max(0, retryAfterMs) };
    }

    return { allowed: true, retryAfterMs: 0 };
  }
}

const DEFAULT_TOOL_RATE_LIMITS: Record<string, ToolRateLimitConfig> = {
  batch_read_files: { limit: 30, windowMs: 60_000 },
  push_files: { limit: 20, windowMs: 60_000 },
  validate_build: { limit: 10, windowMs: 10 * 60_000 },
  apply_patch: { limit: 50, windowMs: 60_000 },
};

const TOOL_RATE_LIMITS: Record<string, ToolRateLimitConfig> = (() => {
  if (!process.env.TOOL_RATE_LIMITS) return DEFAULT_TOOL_RATE_LIMITS;
  try {
    const parsed = JSON.parse(process.env.TOOL_RATE_LIMITS) as Record<string, ToolRateLimitConfig>;
    return { ...DEFAULT_TOOL_RATE_LIMITS, ...parsed };
  } catch {
    return DEFAULT_TOOL_RATE_LIMITS;
  }
})();

const toolRateLimiters = new Map<string, ToolRateLimiter>();

function getToolRateLimiter(toolName: string): ToolRateLimiter | null {
  const config = TOOL_RATE_LIMITS[toolName];
  if (!config) return null;
  if (!toolRateLimiters.has(toolName)) {
    toolRateLimiters.set(toolName, new ToolRateLimiter(config));
  }
  return toolRateLimiters.get(toolName)!;
}

// ============================================================================
// AUDIT LOGGING
// ============================================================================

const AUDIT_LOG_MAX_ENTRIES = parseInt(process.env.AUDIT_LOG_MAX_ENTRIES || '1000', 10);
const AUDIT_LOG_PATH = process.env.AUDIT_LOG_PATH;
const AUDIT_ALERT_DESTRUCTIVE = process.env.AUDIT_ALERT_DESTRUCTIVE === 'true';

interface AuditLogEntry {
  id: string;
  tool: string;
  timestamp: string;
  durationMs?: number;
  success?: boolean;
  destructive?: boolean;
  argsHash?: string;
  repo?: string;
  owner?: string;
  error?: string;
}

const auditLog: AuditLogEntry[] = [];

function hashArgs(args: Record<string, unknown>): string {
  try {
    return crypto.createHash('sha256').update(JSON.stringify(args)).digest('hex');
  } catch {
    return 'unhashable';
  }
}

async function appendAuditLog(entry: AuditLogEntry): Promise<void> {
  auditLog.push(entry);
  if (auditLog.length > AUDIT_LOG_MAX_ENTRIES) {
    auditLog.shift();
  }

  if (AUDIT_LOG_PATH) {
    const line = JSON.stringify(entry) + '\n';
    await fs.appendFile(AUDIT_LOG_PATH, line).catch(() => {});
  }
}

// ============================================================================
// DUAL OUTPUT FORMAT (Text + Structured)
// ============================================================================

/**
 * Structured content that can be returned alongside text content.
 * This allows AI assistants to get both human-readable and machine-parseable output.
 */
interface DualOutput {
  [key: string]: unknown;
  /** Human-readable text content for display */
  content: Array<{ type: 'text'; text: string }>;
  /** Structured data for programmatic access (optional) */
  structuredContent?: Record<string, unknown>;
  /** Whether the result was served from cache */
  _cached?: boolean;
  /** Progress information if applicable */
  _progress?: { completed: number; total: number; message?: string };
}

/**
 * Creates a dual output response with both text and structured data.
 * The text is human-readable, while structuredContent is for programmatic use.
 */
function createDualOutput(
  textSummary: string,
  structuredData: Record<string, unknown>,
  options?: { cached?: boolean; progress?: { completed: number; total: number; message?: string } }
): DualOutput {
  return {
    content: [{
      type: 'text' as const,
      text: `${textSummary}\n\n---\n\n**Structured Data:**\n\`\`\`json\n${JSON.stringify(structuredData, null, 2)}\n\`\`\``,
    }],
    structuredContent: structuredData,
    ...(options?.cached !== undefined ? { _cached: options.cached } : {}),
    ...(options?.progress !== undefined ? { _progress: options.progress } : {}),
  };
}

/**
 * Creates a success response with dual output format.
 */
function createSuccessOutput(
  operation: string,
  details: Record<string, unknown>,
  additionalText?: string
): DualOutput {
  const emoji = '✅';
  const textSummary = additionalText
    ? `${emoji} ${operation}\n\n${additionalText}`
    : `${emoji} ${operation}`;

  return createDualOutput(textSummary, {
    success: true,
    operation,
    ...details,
    timestamp: new Date().toISOString(),
  });
}

// ============================================================================
// PROGRESS NOTIFICATIONS
// ============================================================================

/**
 * Progress tracker for bulk operations.
 * Tracks progress and provides formatted updates.
 */
class ProgressTracker {
  private completed = 0;
  private errors: Array<{ item: string; error: string }> = [];
  private startTime = Date.now();

  constructor(
    private total: number,
    private operationName: string
  ) {}

  /**
   * Mark an item as completed successfully.
   */
  success(item?: string): void {
    this.completed++;
  }

  /**
   * Mark an item as failed.
   */
  failure(item: string, error: string): void {
    this.completed++;
    this.errors.push({ item, error: sanitizeErrorMessage(error) });
  }

  /**
   * Get current progress percentage.
   */
  getPercentage(): number {
    return this.total > 0 ? Math.round((this.completed / this.total) * 100) : 0;
  }

  /**
   * Get progress summary for logging or display.
   */
  getProgressMessage(): string {
    return `[${this.operationName}] ${this.completed}/${this.total} (${this.getPercentage()}%)`;
  }

  /**
   * Get final summary with timing and error details.
   */
  getSummary(): {
    operation: string;
    total: number;
    successful: number;
    failed: number;
    errors: Array<{ item: string; error: string }>;
    durationMs: number;
    itemsPerSecond: number;
  } {
    const durationMs = Date.now() - this.startTime;
    const successful = this.completed - this.errors.length;

    return {
      operation: this.operationName,
      total: this.total,
      successful,
      failed: this.errors.length,
      errors: this.errors.slice(0, 10), // Limit to first 10 errors
      durationMs,
      itemsPerSecond: durationMs > 0 ? Math.round((this.completed / durationMs) * 1000 * 10) / 10 : 0,
    };
  }

  /**
   * Get progress object for DualOutput.
   */
  getProgress(): { completed: number; total: number; message: string } {
    return {
      completed: this.completed,
      total: this.total,
      message: this.getProgressMessage(),
    };
  }
}

// ============================================================================
// PRE-PARSE JSON INPUTS
// ============================================================================

/**
 * Pre-parse inputs that may be double-encoded JSON.
 * Claude Desktop and some clients send JSON as stringified values.
 * This function attempts to parse string values that look like JSON.
 */
function preParseInputs<T extends Record<string, unknown>>(
  inputs: T,
  jsonFields?: string[]
): T {
  const result = { ...inputs } as Record<string, unknown>;

  for (const [key, value] of Object.entries(result)) {
    // Skip if not a string
    if (typeof value !== 'string') continue;

    // Skip if explicitly not a JSON field
    if (jsonFields && !jsonFields.includes(key)) continue;

    // Try to parse if it looks like JSON
    const trimmed = value.trim();
    if (
      (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
      (trimmed.startsWith('[') && trimmed.endsWith(']'))
    ) {
      try {
        result[key] = JSON.parse(trimmed);
      } catch {
        // Not valid JSON, keep original string
      }
    }
  }

  return result as T;
}

/**
 * Pre-parse a single value that may be double-encoded JSON.
 */
function preParseValue<T>(value: unknown): T {
  if (typeof value !== 'string') return value as T;

  const trimmed = value.trim();
  if (
    (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
    (trimmed.startsWith('[') && trimmed.endsWith(']'))
  ) {
    try {
      return JSON.parse(trimmed) as T;
    } catch {
      // Not valid JSON, return original
    }
  }

  return value as T;
}

/**
 * Validate and pre-parse files array input.
 * Handles both properly typed arrays and stringified JSON arrays.
 */
function parseFilesInput(
  files: unknown
): Array<{ path: string; content: string }> {
  const parsed = preParseValue<Array<{ path: string; content: string }>>(files);

  if (!Array.isArray(parsed)) {
    throw new Error('Files must be an array of {path, content} objects');
  }

  // Validate each file entry
  for (const file of parsed) {
    if (typeof file.path !== 'string' || typeof file.content !== 'string') {
      throw new Error('Each file must have "path" (string) and "content" (string) properties');
    }
  }

  return parsed;
}

function isBinaryBuffer(buffer: Buffer): boolean {
  const maxCheck = Math.min(buffer.length, 8192);
  for (let i = 0; i < maxCheck; i += 1) {
    if (buffer[i] === 0) return true;
  }
  return false;
}

function applyUnifiedDiff(original: string, patch: string): string {
  const originalLines = original.split('\n');
  const patchLines = patch.split('\n');
  const result: string[] = [];
  let cursor = 0;

  let i = 0;
  while (i < patchLines.length) {
    const line = patchLines[i];
    if (line.startsWith('---') || line.startsWith('+++')) {
      i += 1;
      continue;
    }
    if (!line.startsWith('@@')) {
      i += 1;
      continue;
    }

    const match = /@@\s+-([0-9]+)(?:,([0-9]+))?\s+\+([0-9]+)(?:,([0-9]+))?\s+@@/.exec(line);
    if (!match) {
      throw new Error('Invalid patch hunk header');
    }

    const startOld = parseInt(match[1], 10) - 1;
    if (startOld < cursor) {
      throw new Error('Patch hunk overlaps previous hunk');
    }

    result.push(...originalLines.slice(cursor, startOld));
    cursor = startOld;
    i += 1;

    while (i < patchLines.length && !patchLines[i].startsWith('@@')) {
      const hunkLine = patchLines[i];
      if (hunkLine.startsWith('\\')) {
        i += 1;
        continue;
      }
      const marker = hunkLine[0];
      const text = hunkLine.slice(1);
      if (marker === ' ') {
        if (originalLines[cursor] !== text) {
          throw new Error(`Patch context mismatch at line ${cursor + 1}`);
        }
        result.push(originalLines[cursor]);
        cursor += 1;
      } else if (marker === '-') {
        if (originalLines[cursor] !== text) {
          throw new Error(`Patch removal mismatch at line ${cursor + 1}`);
        }
        cursor += 1;
      } else if (marker === '+') {
        result.push(text);
      }
      i += 1;
    }
  }

  result.push(...originalLines.slice(cursor));
  return result.join('\n');
}

// ============================================================================
// MCP SERVER
// ============================================================================

const server = new McpServer({
  name: 'nova-os-github-connector',
  version: '4.3.2',
  description: 'Direct GitHub API connector for Nova OS with dynamic toolsets',
});

function extractTextContent(content: Array<{ type: string; text?: string }>): string | null {
  const texts = content
    .filter((item) => item.type === 'text' && typeof item.text === 'string')
    .map((item) => item.text as string);
  if (texts.length === 0) return null;
  return texts.join('\n');
}

function normalizeToolResult(toolName: string, result: DualOutput, durationMs: number): DualOutput {
  const text = extractTextContent(result.content || []);
  const structured = result.structuredContent ?? { text };
  const normalizedStructured = {
    success: structured.success ?? true,
    tool: structured.tool ?? toolName,
    ...structured,
  };
  return {
    ...result,
    structuredContent: { ...normalizedStructured, duration_ms: durationMs },
  };
}

function getAuditBase(toolName: string, args: any): AuditLogEntry {
  const owner = typeof args.owner === 'string' ? args.owner : undefined;
  const repo = typeof args.repo === 'string' ? args.repo : undefined;
  return {
    id: crypto.randomUUID(),
    tool: toolName,
    timestamp: new Date().toISOString(),
    destructive: toolsetManager.isToolDestructive(toolName),
    argsHash: hashArgs(args),
    owner,
    repo,
  };
}

function enforceAllowedRepo(args: any): void {
  const owner = typeof args.owner === 'string' ? args.owner : undefined;
  const repo = typeof args.repo === 'string' ? args.repo : undefined;
  if (!owner || !repo) return;
  const target = `${owner}/${repo}`.toLowerCase();
  if (!ALLOWED_REPOS.includes(target)) {
    throw new Error(`Repository not allowed: ${target}. Allowed: ${ALLOWED_REPOS.join(', ')}`);
  }
}

function registerTool(
  name: string,
  config: any,
  handler: (args: any) => Promise<DualOutput>
): void {
  server.registerTool(name, config, async (args: any) => {
    const auditEntry = getAuditBase(name, args);
    const limiter = getToolRateLimiter(name);
    if (limiter) {
      const decision = limiter.check();
      if (!decision.allowed) {
        const error = new ToolRateLimitError(name, TOOL_RATE_LIMITS[name].limit, TOOL_RATE_LIMITS[name].windowMs, decision.retryAfterMs);
        const errorResult = createSafeErrorResponse(error, name);
        auditEntry.success = false;
        auditEntry.error = error.message;
        await appendAuditLog(auditEntry);
        return normalizeToolResult(name, errorResult, 0);
      }
    }

    const start = Date.now();
    try {
      enforceAllowedRepo(args);
      const result = await handler(args);
      const normalized = normalizeToolResult(name, result, Date.now() - start);
      auditEntry.success = true;
      auditEntry.durationMs = Date.now() - start;
      await appendAuditLog(auditEntry);
      if (auditEntry.destructive && AUDIT_ALERT_DESTRUCTIVE) {
        console.warn(`[audit] Destructive tool executed: ${name} (${auditEntry.owner || 'unknown'}/${auditEntry.repo || 'unknown'})`);
      }
      return normalized;
    } catch (error) {
      const errorResult = createSafeErrorResponse(error, name);
      auditEntry.success = false;
      auditEntry.durationMs = Date.now() - start;
      auditEntry.error = error instanceof Error ? sanitizeErrorMessage(error.message) : 'Unknown error';
      await appendAuditLog(auditEntry);
      return normalizeToolResult(name, errorResult, auditEntry.durationMs);
    }
  });
}

// ============================================================================
// REGISTER TOOLSETS
// ============================================================================

// Repos toolset - file and branch operations
toolsetManager.registerToolset(
  'repos',
  'Repository operations: read/write files, branches, commits',
  false, // Not read-only (has write tools)
  [
    { name: 'get_file_contents', title: 'Get File Contents', description: 'Read a file from a repository', readOnly: true, destructive: false },
    { name: 'batch_read_files', title: 'Batch Read Files', description: 'Read multiple files in one call', readOnly: true, destructive: false },
    { name: 'get_diff', title: 'Get Diff', description: 'Get diff between two refs', readOnly: true, destructive: false },
    { name: 'suggest_changes', title: 'Suggest Changes', description: 'Summarize or suggest changes for a diff', readOnly: true, destructive: false },
    { name: 'create_or_update_file', title: 'Create or Update File', description: 'Create or update a single file', readOnly: false, destructive: true },
    { name: 'apply_patch', title: 'Apply Patch', description: 'Apply a unified diff patch to a file', readOnly: false, destructive: true },
    { name: 'push_files', title: 'Push Files', description: 'Commit multiple files in one commit', readOnly: false, destructive: true },
    { name: 'create_branch', title: 'Create Branch', description: 'Create a new branch', readOnly: false, destructive: false },
    { name: 'list_commits', title: 'List Commits', description: 'List recent commits', readOnly: true, destructive: false },
  ],
  DEFAULT_ENABLED_TOOLSETS.includes('repos')
);

// Issues toolset
toolsetManager.registerToolset(
  'issues',
  'Issue tracking: list, filter, and manage issues',
  false,
  [
    { name: 'list_issues', title: 'List Issues', description: 'List issues in a repository', readOnly: true, destructive: false },
    { name: 'get_issue', title: 'Get Issue', description: 'Get issue details', readOnly: true, destructive: false },
    { name: 'search_issues', title: 'Search Issues', description: 'Search issues and PRs', readOnly: true, destructive: false },
    { name: 'list_issue_comments', title: 'List Issue Comments', description: 'List comments on an issue', readOnly: true, destructive: false },
    { name: 'create_issue', title: 'Create Issue', description: 'Create an issue', readOnly: false, destructive: true },
    { name: 'update_issue', title: 'Update Issue', description: 'Update an issue', readOnly: false, destructive: true },
    { name: 'add_issue_comment', title: 'Add Issue Comment', description: 'Comment on an issue', readOnly: false, destructive: true },
  ],
  DEFAULT_ENABLED_TOOLSETS.includes('issues')
);

// Pull requests toolset
toolsetManager.registerToolset(
  'pulls',
  'Pull request operations: create, review, merge',
  false, // Has write operations
  [
    { name: 'create_pull_request', title: 'Create Pull Request', description: 'Create a new PR', readOnly: false, destructive: true },
    { name: 'update_pull_request', title: 'Update Pull Request', description: 'Update a PR', readOnly: false, destructive: true },
    { name: 'get_pull_request', title: 'Get Pull Request', description: 'Get PR details', readOnly: true, destructive: false },
    { name: 'list_pull_requests', title: 'List Pull Requests', description: 'List PRs in a repository', readOnly: true, destructive: false },
    { name: 'get_pull_request_files', title: 'Get PR Files', description: 'List files in a PR', readOnly: true, destructive: false },
    { name: 'get_pull_request_comments', title: 'Get PR Comments', description: 'List issue comments on a PR', readOnly: true, destructive: false },
    { name: 'get_pull_request_reviews', title: 'Get PR Reviews', description: 'List reviews on a PR', readOnly: true, destructive: false },
    { name: 'create_pull_request_review', title: 'Create PR Review', description: 'Create a PR review', readOnly: false, destructive: true },
    { name: 'review_pr', title: 'Review PR', description: 'Summarize a PR for review', readOnly: true, destructive: false },
    { name: 'merge_pull_request', title: 'Merge Pull Request', description: 'Merge a PR', readOnly: false, destructive: true },
  ],
  DEFAULT_ENABLED_TOOLSETS.includes('pulls')
);

// Search toolset
toolsetManager.registerToolset(
  'search',
  'Code search across repositories',
  true, // Read-only
  [
    { name: 'search_code', title: 'Search Code', description: 'Search for code patterns', readOnly: true, destructive: false },
    { name: 'search_repositories', title: 'Search Repositories', description: 'Search repositories', readOnly: true, destructive: false },
  ],
  DEFAULT_ENABLED_TOOLSETS.includes('search')
);

// Utility toolset
toolsetManager.registerToolset(
  'utility',
  'Utility tools: build validation, rate limits, caching',
  false, // clear_cache can modify state
  [
    { name: 'validate_build', title: 'Validate Build', description: 'Clone and build to verify compilation', readOnly: true, destructive: false },
    { name: 'check_github_rate_limit', title: 'Check Rate Limit', description: 'Check API rate limit status', readOnly: true, destructive: false },
    { name: 'clear_cache', title: 'Clear Cache', description: 'Clear content cache', readOnly: false, destructive: false },
  ],
  DEFAULT_ENABLED_TOOLSETS.includes('utility')
);

// Vercel toolset (conditional)
if (VERCEL_TOKEN) {
  toolsetManager.registerToolset(
    'vercel',
    'Vercel deployment monitoring',
    true, // Read-only
    [
      { name: 'list_vercel_deployments', title: 'List Deployments', description: 'List recent Vercel deployments', readOnly: true, destructive: false },
      { name: 'get_vercel_deployment', title: 'Get Deployment', description: 'Get deployment details', readOnly: true, destructive: false },
    ],
    DEFAULT_ENABLED_TOOLSETS.includes('vercel')
  );
}

// Toolsets meta-toolset (always enabled, cannot be disabled)
toolsetManager.registerToolset(
  'toolsets',
  'Toolset management: discover and toggle tool groups',
  false, // Has enable/disable which modify state
  [
    { name: 'list_toolsets', title: 'List Toolsets', description: 'List all available toolsets', readOnly: true, destructive: false },
    { name: 'get_toolset', title: 'Get Toolset', description: 'Get tools in a specific toolset', readOnly: true, destructive: false },
    { name: 'enable_toolset', title: 'Enable Toolset', description: 'Enable a toolset', readOnly: false, destructive: false },
    { name: 'disable_toolset', title: 'Disable Toolset', description: 'Disable a toolset', readOnly: false, destructive: false },
  ],
  true // Always enabled
);

// ============================================================================
// TOOLSET DISCOVERY TOOLS
// ============================================================================

// list_toolsets
registerTool(
  'list_toolsets',
  {
    title: 'List Toolsets',
    description: 'List all available toolsets and their status. Use this to discover what tool groups are available.',
    inputSchema: {
      include_tools: z.boolean().optional().describe('Include list of tools in each toolset'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ include_tools }) => {
    const toolsets = toolsetManager.getAllToolsets();
    const stats = toolsetManager.getStats();

    const result = {
      summary: {
        total_toolsets: stats.totalToolsets,
        enabled_toolsets: stats.enabledToolsets,
        total_tools: stats.totalTools,
        enabled_tools: stats.enabledTools,
      },
      toolsets: toolsets.map(ts => ({
        name: ts.name,
        description: ts.description,
        enabled: ts.enabled,
        readOnly: ts.readOnly,
        tool_count: ts.toolCount,
        ...(include_tools ? { tools: ts.tools } : {}),
      })),
    };

    return {
      content: [{ type: 'text' as const, text: JSON.stringify(result, null, 2) }],
    };
  }
);

// get_toolset
registerTool(
  'get_toolset',
  {
    title: 'Get Toolset',
    description: 'Get detailed information about a specific toolset and its tools.',
    inputSchema: {
      name: z.string().describe('Name of the toolset'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ name }) => {
    const tools = toolsetManager.getToolsetTools(name);

    if (!tools) {
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify({
            error: true,
            message: `Toolset "${name}" not found`,
            available_toolsets: toolsetManager.getAllToolsets().map(ts => ts.name),
          }, null, 2),
        }],
      };
    }

    const toolsets = toolsetManager.getAllToolsets();
    const toolset = toolsets.find(ts => ts.name === name)!;

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          name: toolset.name,
          description: toolset.description,
          enabled: toolset.enabled,
          readOnly: toolset.readOnly,
          tools: tools.map(t => ({
            name: t.name,
            title: t.title,
            description: t.description,
            readOnly: t.readOnly,
            destructive: t.destructive,
            available: toolset.enabled,
          })),
        }, null, 2),
      }],
    };
  }
);

// enable_toolset
registerTool(
  'enable_toolset',
  {
    title: 'Enable Toolset',
    description: 'Enable a toolset to make its tools available. Cannot enable the "toolsets" meta-toolset (always enabled).',
    inputSchema: {
      name: z.string().describe('Name of the toolset to enable'),
    },
  },
  async ({ name }) => {
    const success = toolsetManager.enableToolset(name);

    if (!success) {
      const exists = toolsetManager.getToolsetTools(name);
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify({
            success: false,
            message: exists ? `Cannot modify the "${name}" toolset` : `Toolset "${name}" not found`,
          }, null, 2),
        }],
      };
    }

    const stats = toolsetManager.getStats();
    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          success: true,
          message: `Toolset "${name}" enabled`,
          enabled_tools: stats.enabledTools,
          total_tools: stats.totalTools,
        }, null, 2),
      }],
    };
  }
);

// disable_toolset
registerTool(
  'disable_toolset',
  {
    title: 'Disable Toolset',
    description: 'Disable a toolset to hide its tools. Cannot disable the "toolsets" meta-toolset.',
    inputSchema: {
      name: z.string().describe('Name of the toolset to disable'),
    },
  },
  async ({ name }) => {
    const success = toolsetManager.disableToolset(name);

    if (!success) {
      const exists = toolsetManager.getToolsetTools(name);
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify({
            success: false,
            message: exists ? `Cannot disable the "${name}" toolset` : `Toolset "${name}" not found`,
          }, null, 2),
        }],
      };
    }

    const stats = toolsetManager.getStats();
    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          success: true,
          message: `Toolset "${name}" disabled`,
          enabled_tools: stats.enabledTools,
          total_tools: stats.totalTools,
        }, null, 2),
      }],
    };
  }
);

// ============================================================================
// GITHUB TOOLS - Direct API Implementation
// ============================================================================

// get_file_contents
registerTool(
  'get_file_contents',
  {
    title: 'Get File Contents',
    description: 'Read the contents of a file from a GitHub repository. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      path: z.string().describe('File path'),
      branch: z.string().optional().describe('Branch name (optional)'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, path: filePath, branch }) => {
    try {
      // Validate file path to prevent path traversal
      validateFilePath(filePath);

      const cacheKey = cache.fileKey(owner, repo, filePath, branch);
      const cached = cache.get(cacheKey);
      if (cached) return { content: [{ type: 'text' as const, text: cached }], _cached: true };

      let url = `/repos/${owner}/${repo}/contents/${filePath}`;
      if (branch) url += `?ref=${branch}`;

      const data = await withConcurrencyLimit(() => githubAPI(url));

      let content: string;
      if (data.content) {
        if (typeof data.size === 'number' && data.size > MAX_FILE_BYTES) {
          throw new Error(`File "${filePath}" exceeds max size (${data.size} bytes > ${MAX_FILE_BYTES} bytes)`);
        }
        const buffer = Buffer.from(data.content, 'base64');
        if (!ALLOW_BINARY_FILES && isBinaryBuffer(buffer)) {
          throw new Error(`File "${filePath}" appears to be binary. Set ALLOW_BINARY_FILES=true to allow.`);
        }
        content = buffer.toString('utf-8');
      } else if (Array.isArray(data)) {
        content = JSON.stringify(data.map((f: any) => ({ name: f.name, type: f.type, path: f.path })), null, 2);
      } else {
        content = JSON.stringify(data, null, 2);
      }

      cache.set(cacheKey, content, CACHE_TTL_FILE_MS);
      return { content: [{ type: 'text' as const, text: content }] };
    } catch (error) {
      return createSafeErrorResponse(error, 'get_file_contents');
    }
  }
);

// batch_read_files
registerTool(
  'batch_read_files',
  {
    title: 'Batch Read Files',
    description: 'Read multiple files from a GitHub repository in a single call. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      paths: z.array(z.string()).describe('Array of file paths (or JSON string of array)'),
      branch: z.string().optional().describe('Branch name (optional)'),
      timeout_ms: z.number().optional().describe('Per-file timeout in milliseconds (default: 30000)'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, paths: rawPaths, branch, timeout_ms }) => {
    try {
      // Pre-parse paths in case they're double-encoded JSON
      const paths = preParseValue<string[]>(rawPaths);
      if (!Array.isArray(paths)) {
        throw new Error('paths must be an array of file path strings');
      }

      // Validate all file paths before processing
      validateFilePaths(paths);

      // Initialize progress tracker
      const progress = new ProgressTracker(paths.length, 'batch_read_files');

      // Process files with concurrency limiting (queue manages parallel requests)
      const results = await Promise.all(paths.map(async (filePath) => {
        try {
          const cacheKey = cache.fileKey(owner, repo, filePath, branch);
          const cached = cache.get(cacheKey);
          if (cached) {
            progress.success(filePath);
            return { path: filePath, content: cached, cached: true };
          }

          let url = `/repos/${owner}/${repo}/contents/${filePath}`;
          if (branch) url += `?ref=${branch}`;

          // Use concurrency limiter with configurable timeout
          const data = await withConcurrencyLimit(() =>
            githubAPI(url, { timeoutMs: timeout_ms || DEFAULT_TIMEOUT_MS })
          );
          if (data.content) {
            if (typeof data.size === 'number' && data.size > MAX_FILE_BYTES) {
              throw new Error(`File "${filePath}" exceeds max size (${data.size} bytes > ${MAX_FILE_BYTES} bytes)`);
            }
            const buffer = Buffer.from(data.content, 'base64');
            if (!ALLOW_BINARY_FILES && isBinaryBuffer(buffer)) {
              throw new Error(`File "${filePath}" appears to be binary. Set ALLOW_BINARY_FILES=true to allow.`);
            }
            const content = buffer.toString('utf-8');
            cache.set(cacheKey, content, CACHE_TTL_BATCH_MS);
            progress.success(filePath);
            return { path: filePath, content, cached: false };
          }

          const content = JSON.stringify(data);
          cache.set(cacheKey, content, CACHE_TTL_BATCH_MS);
          progress.success(filePath);
          return { path: filePath, content, cached: false };
        } catch (error: any) {
          // Track failure with sanitized error
          const errorMsg = error instanceof Error ? error.message : String(error);
          progress.failure(filePath, errorMsg);
          return { path: filePath, content: null, error: sanitizeErrorMessage(errorMsg) };
        }
      }));

      // Get progress summary
      const summary = progress.getSummary();
      const filesRead = results.filter(r => r.content !== null).length;
      const cachedFiles = results.filter(r => r.cached === true).length;

      // Create dual output with human-readable summary and structured data
      const textSummary = `📁 Read ${filesRead}/${paths.length} files from ${owner}/${repo}${branch ? ` (${branch})` : ''}\n` +
        `   • Successful: ${summary.successful}\n` +
        `   • From cache: ${cachedFiles}\n` +
        `   • Failed: ${summary.failed}\n` +
        `   • Duration: ${summary.durationMs}ms (${summary.itemsPerSecond} files/sec)`;

      return createDualOutput(textSummary, {
        repository: `${owner}/${repo}`,
        branch: branch || 'default',
        files_requested: paths.length,
        files_read: filesRead,
        from_cache: cachedFiles,
        progress: summary,
        queue_stats: requestQueue.getStats(),
        files: results,
      });
    } catch (error) {
      return createSafeErrorResponse(error, 'batch_read_files');
    }
  }
);

// get_diff
registerTool(
  'get_diff',
  {
    title: 'Get Diff',
    description: 'Get a diff between two refs in a repository. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      base: z.string().describe('Base ref (branch or SHA)'),
      head: z.string().describe('Head ref (branch or SHA)'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, base, head }) => {
    const compare = await githubAPI(`/repos/${owner}/${repo}/compare/${base}...${head}`);
    const files = (compare.files || []).map((f: any) => ({
      filename: f.filename,
      status: f.status,
      additions: f.additions,
      deletions: f.deletions,
      changes: f.changes,
      patch: f.patch,
    }));

    return createDualOutput(`🧾 Diff ${base} → ${head} (${files.length} files)`, {
      repository: `${owner}/${repo}`,
      base,
      head,
      ahead_by: compare.ahead_by,
      behind_by: compare.behind_by,
      total_commits: compare.total_commits,
      files,
    });
  }
);

// suggest_changes
registerTool(
  'suggest_changes',
  {
    title: 'Suggest Changes',
    description: 'Summarize a diff and suggest review focus areas. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      base: z.string().optional().describe('Base ref'),
      head: z.string().optional().describe('Head ref'),
      diff: z.string().optional().describe('Unified diff text (optional)'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, base, head, diff }) => {
    let files: Array<{ filename: string; additions: number; deletions: number }> = [];
    let totalAdditions = 0;
    let totalDeletions = 0;

    if (diff) {
      const lines = diff.split('\n');
      let currentFile = '';
      for (const line of lines) {
        if (line.startsWith('+++ b/')) {
          currentFile = line.replace('+++ b/', '');
          if (!files.find(f => f.filename === currentFile)) {
            files.push({ filename: currentFile, additions: 0, deletions: 0 });
          }
        } else if (line.startsWith('+') && !line.startsWith('+++')) {
          totalAdditions += 1;
          const file = files.find(f => f.filename === currentFile);
          if (file) file.additions += 1;
        } else if (line.startsWith('-') && !line.startsWith('---')) {
          totalDeletions += 1;
          const file = files.find(f => f.filename === currentFile);
          if (file) file.deletions += 1;
        }
      }
    } else if (base && head) {
      const compare = await githubAPI(`/repos/${owner}/${repo}/compare/${base}...${head}`);
      files = (compare.files || []).map((f: any) => ({
        filename: f.filename,
        additions: f.additions,
        deletions: f.deletions,
      }));
      totalAdditions = compare.files.reduce((sum: number, f: any) => sum + f.additions, 0);
      totalDeletions = compare.files.reduce((sum: number, f: any) => sum + f.deletions, 0);
    } else {
      throw new Error('Provide either diff or base/head refs.');
    }

    const suggestions: string[] = [];
    if (totalAdditions + totalDeletions > 500) suggestions.push('Large change set; prioritize high-risk files.');
    if (files.some(f => f.filename.includes('migrations') || f.filename.includes('schema'))) suggestions.push('Review schema/migration changes carefully.');
    if (files.some(f => f.filename.endsWith('.env') || f.filename.includes('secret'))) suggestions.push('Check for secrets and configuration changes.');
    if (suggestions.length === 0) suggestions.push('Standard review recommended.');

    return createDualOutput('🧠 Suggested review focus areas', {
      repository: `${owner}/${repo}`,
      base,
      head,
      files,
      totals: { additions: totalAdditions, deletions: totalDeletions },
      suggestions,
    });
  }
);

// apply_patch
registerTool(
  'apply_patch',
  {
    title: 'Apply Patch',
    description: 'Apply a unified diff patch to a file in a repository.',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      path: z.string().describe('File path to patch'),
      patch: z.string().describe('Unified diff patch'),
      branch: z.string().optional().describe('Branch name (optional)'),
      message: z.string().optional().describe('Commit message'),
    },
    annotations: { destructiveHint: true },
  },
  async ({ owner, repo, path: filePath, patch, branch, message }) => {
    try {
      validateFilePath(filePath);
      let url = `/repos/${owner}/${repo}/contents/${filePath}`;
      if (branch) url += `?ref=${branch}`;

      const data = await githubAPI(url);
      if (!data.content || !data.sha) {
        throw new Error('File content not found for patching');
      }

      if (typeof data.size === 'number' && data.size > MAX_FILE_BYTES) {
        throw new Error(`File "${filePath}" exceeds max size (${data.size} bytes > ${MAX_FILE_BYTES} bytes)`);
      }

      const buffer = Buffer.from(data.content, 'base64');
      if (!ALLOW_BINARY_FILES && isBinaryBuffer(buffer)) {
        throw new Error(`File "${filePath}" appears to be binary. Set ALLOW_BINARY_FILES=true to allow.`);
      }

      const original = buffer.toString('utf-8');
      const updated = applyUnifiedDiff(original, patch);

      if (updated === original) {
        throw new Error('Patch produced no changes');
      }

      if (Buffer.byteLength(updated, 'utf8') > MAX_FILE_BYTES) {
        throw new Error(`Patched content exceeds max size (${MAX_FILE_BYTES} bytes)`);
      }

      const commitMessage = message || `Apply patch to ${filePath}`;
      const result = await githubAPI(`/repos/${owner}/${repo}/contents/${filePath}`, {
        method: 'PUT',
        body: JSON.stringify({
          message: commitMessage,
          content: Buffer.from(updated).toString('base64'),
          sha: data.sha,
          ...(branch ? { branch } : {}),
        }),
      });

      cache.invalidateRepo(owner, repo);

      return createSuccessOutput('Patch applied', {
        repository: `${owner}/${repo}`,
        branch: branch || 'default',
        file: filePath,
        commit: {
          sha: result.commit.sha,
          url: result.commit.html_url,
        },
      });
    } catch (error) {
      return createSafeErrorResponse(error, 'apply_patch');
    }
  }
);

// create_or_update_file
registerTool(
  'create_or_update_file',
  {
    title: 'Create or Update File',
    description: 'Create a new file or update an existing file in a GitHub repository.',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      path: z.string().describe('File path'),
      content: z.string().describe('File content'),
      message: z.string().describe('Commit message'),
      branch: z.string().optional().describe('Branch name (optional)'),
      sha: z.string().optional().describe('SHA of file being replaced (required for updates)'),
      timeout_ms: z.number().optional().describe('Request timeout in milliseconds (default: 30000)'),
    },
    annotations: { destructiveHint: true },
  },
  async ({ owner, repo, path: filePath, content, message, branch, sha, timeout_ms }) => {
    try {
      // Validate file path to prevent path traversal
      validateFilePath(filePath);

      const isUpdate = !!sha;
      let fileSha = sha;

      // If no SHA provided and file exists, get it (this is an update)
      if (!fileSha) {
        try {
          let url = `/repos/${owner}/${repo}/contents/${filePath}`;
          if (branch) url += `?ref=${branch}`;
          const existing = await withConcurrencyLimit(() =>
            githubAPI(url, { timeoutMs: timeout_ms, maxRetries: 1 })
          );
          fileSha = existing.sha;
        } catch (error) {
          // File doesn't exist - this will be a creation
          if (!(error instanceof GitHubNotFoundError)) {
            throw error; // Re-throw non-404 errors
          }
        }
      }

      const body: Record<string, unknown> = {
        message,
        content: Buffer.from(content).toString('base64'),
      };
      if (branch) body.branch = branch;
      if (fileSha) body.sha = fileSha;

      const result = await withConcurrencyLimit(() =>
        githubAPI(`/repos/${owner}/${repo}/contents/${filePath}`, {
          method: 'PUT',
          body: JSON.stringify(body),
          timeoutMs: timeout_ms,
        })
      );

      cache.invalidateRepo(owner, repo);

      const operation = fileSha ? 'Updated' : 'Created';
      const textSummary = `✅ ${operation} ${filePath} in ${owner}/${repo}${branch ? `@${branch}` : ''}\n` +
        `   • Commit: ${result.commit.sha.slice(0, 7)}\n` +
        `   • Message: ${message.split('\n')[0]}\n` +
        `   • Size: ${content.length} bytes`;

      return createDualOutput(textSummary, {
        success: true,
        operation: fileSha ? 'update' : 'create',
        commit: {
          sha: result.commit.sha,
          short_sha: result.commit.sha.slice(0, 7),
          message: result.commit.message,
          url: `https://github.com/${owner}/${repo}/commit/${result.commit.sha}`,
        },
        file: {
          path: result.content.path,
          sha: result.content.sha,
          size: content.length,
          url: result.content.html_url,
        },
        repository: `${owner}/${repo}`,
        branch: branch || 'default',
      });
    } catch (error) {
      return createSafeErrorResponse(error, 'create_or_update_file');
    }
  }
);

// push_files
registerTool(
  'push_files',
  {
    title: 'Push Files',
    description: 'Commit multiple files to a GitHub repository in a single commit.',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      branch: z.string().describe('Branch name'),
      message: z.string().describe('Commit message'),
      files: z.any().describe('Array of {path, content} objects (or JSON string of array)'),
      timeout_ms: z.number().optional().describe('Request timeout in milliseconds (default: 30000)'),
    },
    annotations: { destructiveHint: true },
  },
  async ({ owner, repo, branch, message, files: rawFiles, timeout_ms }) => {
    try {
      // Pre-parse files in case they're double-encoded JSON from Claude Desktop
      const files = parseFilesInput(rawFiles);

      // Validate all file paths before processing
      validateFilePaths(files.map(f => f.path));

      // Initialize progress tracker
      const progress = new ProgressTracker(files.length + 4, 'push_files'); // +4 for git operations
      const startTime = Date.now();

      // Get the reference for the branch
      const ref = await withConcurrencyLimit(() =>
        githubAPI(`/repos/${owner}/${repo}/git/refs/heads/${branch}`, { timeoutMs: timeout_ms })
      );
      const latestCommitSha = ref.object.sha;
      progress.success('get_ref');

      // Get the tree of the latest commit
      const latestCommit = await withConcurrencyLimit(() =>
        githubAPI(`/repos/${owner}/${repo}/git/commits/${latestCommitSha}`, { timeoutMs: timeout_ms })
      );
      const baseTreeSha = latestCommit.tree.sha;
      progress.success('get_tree');

      // Create blobs for each file (with concurrency limiting and progress tracking)
      const blobResults: Array<{ path: string; sha: string; success: boolean; error?: string }> = [];
      const treeItems = await Promise.all(files.map(async (file) => {
        try {
          const blob = await withConcurrencyLimit(() =>
            githubAPI(`/repos/${owner}/${repo}/git/blobs`, {
              method: 'POST',
              body: JSON.stringify({
                content: file.content,
                encoding: 'utf-8',
              }),
              timeoutMs: timeout_ms,
            })
          );
          progress.success(file.path);
          blobResults.push({ path: file.path, sha: blob.sha, success: true });
          return {
            path: file.path,
            mode: '100644' as const,
            type: 'blob' as const,
            sha: blob.sha,
          };
        } catch (error: any) {
          const errorMsg = error instanceof Error ? error.message : String(error);
          progress.failure(file.path, errorMsg);
          blobResults.push({ path: file.path, sha: '', success: false, error: sanitizeErrorMessage(errorMsg) });
          return null;
        }
      }));

      // Check if any blobs failed
      const failedBlobs = blobResults.filter(r => !r.success);
      if (failedBlobs.length > 0) {
        throw new Error(`Failed to create blobs for ${failedBlobs.length} files: ${failedBlobs.map(f => f.path).join(', ')}`);
      }

      // Filter out null entries (shouldn't happen if we throw above, but be safe)
      const validTreeItems = treeItems.filter((item): item is NonNullable<typeof item> => item !== null);

      // Create a new tree
      const newTree = await withConcurrencyLimit(() =>
        githubAPI(`/repos/${owner}/${repo}/git/trees`, {
          method: 'POST',
          body: JSON.stringify({
            base_tree: baseTreeSha,
            tree: validTreeItems,
          }),
          timeoutMs: timeout_ms,
        })
      );
      progress.success('create_tree');

      // Create a new commit
      const newCommit = await withConcurrencyLimit(() =>
        githubAPI(`/repos/${owner}/${repo}/git/commits`, {
          method: 'POST',
          body: JSON.stringify({
            message,
            tree: newTree.sha,
            parents: [latestCommitSha],
          }),
          timeoutMs: timeout_ms,
        })
      );
      progress.success('create_commit');

      // Update the reference
      await withConcurrencyLimit(() =>
        githubAPI(`/repos/${owner}/${repo}/git/refs/heads/${branch}`, {
          method: 'PATCH',
          body: JSON.stringify({ sha: newCommit.sha }),
          timeoutMs: timeout_ms,
        })
      );

      cache.invalidateRepo(owner, repo);

      const durationMs = Date.now() - startTime;
      const summary = progress.getSummary();

      // Create dual output with human-readable summary
      const textSummary = `✅ Committed ${files.length} files to ${owner}/${repo}@${branch}\n` +
        `   • Commit: ${newCommit.sha.slice(0, 7)}\n` +
        `   • Message: ${message.split('\n')[0]}\n` +
        `   • Duration: ${durationMs}ms\n` +
        `   • Files: ${files.map(f => f.path).join(', ')}`;

      return createDualOutput(textSummary, {
        success: true,
        commit: {
          sha: newCommit.sha,
          short_sha: newCommit.sha.slice(0, 7),
          message: newCommit.message,
          url: `https://github.com/${owner}/${repo}/commit/${newCommit.sha}`,
        },
        repository: `${owner}/${repo}`,
        branch,
        files_committed: files.length,
        files: blobResults.map(r => ({ path: r.path, sha: r.sha.slice(0, 7) })),
        progress: summary,
        duration_ms: durationMs,
      });
    } catch (error) {
      return createSafeErrorResponse(error, 'push_files');
    }
  }
);

// list_issues
registerTool(
  'list_issues',
  {
    title: 'List Issues',
    description: 'List issues in a GitHub repository. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      state: z.enum(['open', 'closed', 'all']).optional().describe('Filter by state'),
      per_page: z.number().optional().describe('Results per page (max 100)'),
      include_pull_requests: z.boolean().optional().describe('Include pull requests in results'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, state, per_page, include_pull_requests }) => {
    let url = `/repos/${owner}/${repo}/issues?`;
    if (state) url += `state=${state}&`;
    if (per_page) url += `per_page=${per_page}`;

    const issues = await githubAPI(url);
    const filtered = include_pull_requests ? issues : issues.filter((i: any) => !i.pull_request);
    const formatted = filtered.map((i: any) => ({
      number: i.number,
      title: i.title,
      state: i.state,
      user: i.user.login,
      created_at: i.created_at,
      updated_at: i.updated_at,
      labels: i.labels.map((l: any) => l.name),
      url: i.html_url,
    }));

    const textSummary = `📋 ${formatted.length} issues in ${owner}/${repo}${state ? ` (${state})` : ''}`;
    return createDualOutput(textSummary, {
      repository: `${owner}/${repo}`,
      state: state || 'open',
      count: formatted.length,
      issues: formatted,
    });
  }
);

// get_issue
registerTool(
  'get_issue',
  {
    title: 'Get Issue',
    description: 'Get details for a GitHub issue. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      issue_number: z.number().describe('Issue number'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, issue_number }) => {
    const issue = await githubAPI(`/repos/${owner}/${repo}/issues/${issue_number}`);
    const textSummary = `🧾 Issue #${issue.number}: ${issue.title}`;
    return createDualOutput(textSummary, {
      number: issue.number,
      title: issue.title,
      state: issue.state,
      user: issue.user.login,
      assignees: issue.assignees.map((a: any) => a.login),
      labels: issue.labels.map((l: any) => l.name),
      created_at: issue.created_at,
      updated_at: issue.updated_at,
      body: issue.body,
      url: issue.html_url,
    });
  }
);

// search_issues
registerTool(
  'search_issues',
  {
    title: 'Search Issues',
    description: 'Search issues and pull requests. [Read-Only]',
    inputSchema: {
      query: z.string().describe('Search query (GitHub search syntax)'),
      per_page: z.number().optional().describe('Results per page (max 100)'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ query, per_page }) => {
    const q = encodeURIComponent(query);
    const url = `/search/issues?q=${q}${per_page ? `&per_page=${per_page}` : ''}`;
    const results = await githubAPI(url);
    const formatted = results.items.map((i: any) => ({
      number: i.number,
      title: i.title,
      state: i.state,
      repository_url: i.repository_url,
      url: i.html_url,
      user: i.user?.login,
      created_at: i.created_at,
    }));
    return createDualOutput(`🔎 Found ${formatted.length} results`, {
      total_count: results.total_count,
      items: formatted,
    });
  }
);

// list_issue_comments
registerTool(
  'list_issue_comments',
  {
    title: 'List Issue Comments',
    description: 'List comments on an issue. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      issue_number: z.number().describe('Issue number'),
      per_page: z.number().optional().describe('Results per page (max 100)'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, issue_number, per_page }) => {
    const url = `/repos/${owner}/${repo}/issues/${issue_number}/comments${per_page ? `?per_page=${per_page}` : ''}`;
    const comments = await githubAPI(url);
    const formatted = comments.map((c: any) => ({
      id: c.id,
      user: c.user.login,
      created_at: c.created_at,
      body: c.body,
      url: c.html_url,
    }));
    return createDualOutput(`💬 ${formatted.length} comments on #${issue_number}`, {
      issue_number,
      comments: formatted,
    });
  }
);

// create_issue
registerTool(
  'create_issue',
  {
    title: 'Create Issue',
    description: 'Create a new issue.',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      title: z.string().describe('Issue title'),
      body: z.string().optional().describe('Issue body'),
      labels: z.array(z.string()).optional().describe('Labels'),
      assignees: z.array(z.string()).optional().describe('Assignees'),
    },
    annotations: { destructiveHint: true },
  },
  async ({ owner, repo, title, body, labels, assignees }) => {
    const issue = await githubAPI(`/repos/${owner}/${repo}/issues`, {
      method: 'POST',
      body: JSON.stringify({ title, body, labels, assignees }),
    });
    return createSuccessOutput('Issue created', {
      number: issue.number,
      title: issue.title,
      state: issue.state,
      url: issue.html_url,
    });
  }
);

// update_issue
registerTool(
  'update_issue',
  {
    title: 'Update Issue',
    description: 'Update an existing issue.',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      issue_number: z.number().describe('Issue number'),
      title: z.string().optional().describe('New title'),
      body: z.string().optional().describe('New body'),
      state: z.enum(['open', 'closed']).optional().describe('State'),
      labels: z.array(z.string()).optional().describe('Labels'),
      assignees: z.array(z.string()).optional().describe('Assignees'),
    },
    annotations: { destructiveHint: true },
  },
  async ({ owner, repo, issue_number, title, body, state, labels, assignees }) => {
    const issue = await githubAPI(`/repos/${owner}/${repo}/issues/${issue_number}`, {
      method: 'PATCH',
      body: JSON.stringify({ title, body, state, labels, assignees }),
    });
    return createSuccessOutput('Issue updated', {
      number: issue.number,
      title: issue.title,
      state: issue.state,
      url: issue.html_url,
    });
  }
);

// add_issue_comment
registerTool(
  'add_issue_comment',
  {
    title: 'Add Issue Comment',
    description: 'Add a comment to an issue.',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      issue_number: z.number().describe('Issue number'),
      body: z.string().describe('Comment body'),
    },
    annotations: { destructiveHint: true },
  },
  async ({ owner, repo, issue_number, body }) => {
    const comment = await githubAPI(`/repos/${owner}/${repo}/issues/${issue_number}/comments`, {
      method: 'POST',
      body: JSON.stringify({ body }),
    });
    return createSuccessOutput('Issue comment added', {
      id: comment.id,
      url: comment.html_url,
      created_at: comment.created_at,
    });
  }
);

// create_pull_request
registerTool(
  'create_pull_request',
  {
    title: 'Create Pull Request',
    description: 'Create a new pull request.',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      title: z.string().describe('PR title'),
      head: z.string().describe('Branch containing changes'),
      base: z.string().describe('Branch to merge into'),
      body: z.string().optional().describe('PR description'),
    },
    annotations: { destructiveHint: true },
  },
  async ({ owner, repo, title, head, base, body }) => {
    const pr = await githubAPI(`/repos/${owner}/${repo}/pulls`, {
      method: 'POST',
      body: JSON.stringify({ title, head, base, body }),
    });

    return createSuccessOutput('Pull request created', {
      number: pr.number,
      url: pr.html_url,
      title: pr.title,
      state: pr.state,
      head: pr.head.ref,
      base: pr.base.ref,
    });
  }
);

// update_pull_request
registerTool(
  'update_pull_request',
  {
    title: 'Update Pull Request',
    description: 'Update a pull request.',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      pull_number: z.number().describe('PR number'),
      title: z.string().optional().describe('New title'),
      body: z.string().optional().describe('New body'),
      state: z.enum(['open', 'closed']).optional().describe('State'),
    },
    annotations: { destructiveHint: true },
  },
  async ({ owner, repo, pull_number, title, body, state }) => {
    const pr = await githubAPI(`/repos/${owner}/${repo}/pulls/${pull_number}`, {
      method: 'PATCH',
      body: JSON.stringify({ title, body, state }),
    });
    return createSuccessOutput('Pull request updated', {
      number: pr.number,
      url: pr.html_url,
      title: pr.title,
      state: pr.state,
    });
  }
);

// list_pull_requests
registerTool(
  'list_pull_requests',
  {
    title: 'List Pull Requests',
    description: 'List pull requests in a repository. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      state: z.enum(['open', 'closed', 'all']).optional().describe('Filter by state'),
      per_page: z.number().optional().describe('Results per page (max 100)'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, state, per_page }) => {
    let url = `/repos/${owner}/${repo}/pulls?`;
    if (state) url += `state=${state}&`;
    if (per_page) url += `per_page=${per_page}`;
    const prs = await githubAPI(url);
    const formatted = prs.map((pr: any) => ({
      number: pr.number,
      title: pr.title,
      state: pr.state,
      user: pr.user.login,
      head: pr.head.ref,
      base: pr.base.ref,
      url: pr.html_url,
    }));
    return createDualOutput(`📌 ${formatted.length} pull requests in ${owner}/${repo}`, {
      repository: `${owner}/${repo}`,
      count: formatted.length,
      pull_requests: formatted,
    });
  }
);

// get_pull_request
registerTool(
  'get_pull_request',
  {
    title: 'Get Pull Request',
    description: 'Get details of a pull request. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      pull_number: z.number().describe('PR number'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, pull_number }) => {
    const pr = await githubAPI(`/repos/${owner}/${repo}/pulls/${pull_number}`);

    return createDualOutput(`🔍 PR #${pr.number}: ${pr.title}`, {
      number: pr.number,
      title: pr.title,
      state: pr.state,
      merged: pr.merged,
      mergeable: pr.mergeable,
      head: pr.head.ref,
      base: pr.base.ref,
      user: pr.user.login,
      url: pr.html_url,
    });
  }
);

// get_pull_request_files
registerTool(
  'get_pull_request_files',
  {
    title: 'Get Pull Request Files',
    description: 'List files in a pull request. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      pull_number: z.number().describe('PR number'),
      per_page: z.number().optional().describe('Results per page (max 100)'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, pull_number, per_page }) => {
    let url = `/repos/${owner}/${repo}/pulls/${pull_number}/files`;
    if (per_page) url += `?per_page=${per_page}`;
    const files = await githubAPI(url);
    const formatted = files.map((f: any) => ({
      filename: f.filename,
      status: f.status,
      additions: f.additions,
      deletions: f.deletions,
      changes: f.changes,
      patch: f.patch,
    }));
    return createDualOutput(`📂 ${formatted.length} files in PR #${pull_number}`, {
      pull_number,
      files: formatted,
    });
  }
);

// get_pull_request_comments
registerTool(
  'get_pull_request_comments',
  {
    title: 'Get Pull Request Comments',
    description: 'List issue comments on a pull request. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      pull_number: z.number().describe('PR number'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, pull_number }) => {
    const comments = await githubAPI(`/repos/${owner}/${repo}/issues/${pull_number}/comments`);
    const formatted = comments.map((c: any) => ({
      id: c.id,
      user: c.user.login,
      created_at: c.created_at,
      body: c.body,
      url: c.html_url,
    }));
    return createDualOutput(`💬 ${formatted.length} comments on PR #${pull_number}`, {
      pull_number,
      comments: formatted,
    });
  }
);

// get_pull_request_reviews
registerTool(
  'get_pull_request_reviews',
  {
    title: 'Get Pull Request Reviews',
    description: 'List reviews on a pull request. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      pull_number: z.number().describe('PR number'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, pull_number }) => {
    const reviews = await githubAPI(`/repos/${owner}/${repo}/pulls/${pull_number}/reviews`);
    const formatted = reviews.map((r: any) => ({
      id: r.id,
      user: r.user.login,
      state: r.state,
      submitted_at: r.submitted_at,
      body: r.body,
    }));
    return createDualOutput(`🧾 ${formatted.length} reviews on PR #${pull_number}`, {
      pull_number,
      reviews: formatted,
    });
  }
);

// create_pull_request_review
registerTool(
  'create_pull_request_review',
  {
    title: 'Create Pull Request Review',
    description: 'Create a review on a pull request.',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      pull_number: z.number().describe('PR number'),
      body: z.string().optional().describe('Review body'),
      event: z.enum(['APPROVE', 'REQUEST_CHANGES', 'COMMENT']).optional().describe('Review event'),
      comments: z.array(z.object({
        path: z.string().describe('File path'),
        position: z.number().optional().describe('Position in diff'),
        body: z.string().describe('Comment text'),
      })).optional().describe('Inline comments'),
    },
    annotations: { destructiveHint: true },
  },
  async ({ owner, repo, pull_number, body, event, comments }) => {
    const review = await githubAPI(`/repos/${owner}/${repo}/pulls/${pull_number}/reviews`, {
      method: 'POST',
      body: JSON.stringify({ body, event, comments }),
    });
    return createSuccessOutput('Review created', {
      id: review.id,
      state: review.state,
      url: review.html_url,
    });
  }
);

// review_pr
registerTool(
  'review_pr',
  {
    title: 'Review Pull Request',
    description: 'Summarize a pull request for review. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      pull_number: z.number().describe('PR number'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, pull_number }) => {
    const pr = await githubAPI(`/repos/${owner}/${repo}/pulls/${pull_number}`);
    const files = await githubAPI(`/repos/${owner}/${repo}/pulls/${pull_number}/files`);
    const reviews = await githubAPI(`/repos/${owner}/${repo}/pulls/${pull_number}/reviews`);
    const comments = await githubAPI(`/repos/${owner}/${repo}/issues/${pull_number}/comments`);

    const fileSummary = files.map((f: any) => ({
      filename: f.filename,
      additions: f.additions,
      deletions: f.deletions,
      changes: f.changes,
    }));

    return createDualOutput(`🧩 Review summary for PR #${pull_number}`, {
      pull_request: {
        number: pr.number,
        title: pr.title,
        state: pr.state,
        user: pr.user.login,
        base: pr.base.ref,
        head: pr.head.ref,
        url: pr.html_url,
      },
      files: fileSummary,
      reviews: reviews.map((r: any) => ({ id: r.id, user: r.user.login, state: r.state })),
      comments: comments.map((c: any) => ({ id: c.id, user: c.user.login, created_at: c.created_at })),
    });
  }
);

// merge_pull_request
registerTool(
  'merge_pull_request',
  {
    title: 'Merge Pull Request',
    description: 'Merge a pull request.',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      pull_number: z.number().describe('PR number'),
      merge_method: z.enum(['merge', 'squash', 'rebase']).optional().describe('Merge method'),
    },
    annotations: { destructiveHint: true },
  },
  async ({ owner, repo, pull_number, merge_method }) => {
    const result = await githubAPI(`/repos/${owner}/${repo}/pulls/${pull_number}/merge`, {
      method: 'PUT',
      body: JSON.stringify({ merge_method: merge_method || 'merge' }),
    });

    cache.invalidateRepo(owner, repo);

    return createSuccessOutput('Pull request merged', {
      merged: result.merged,
      message: result.message,
      sha: result.sha?.slice(0, 7),
    });
  }
);

// create_branch
registerTool(
  'create_branch',
  {
    title: 'Create Branch',
    description: 'Create a new branch from an existing ref.',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      branch: z.string().describe('New branch name'),
      from_branch: z.string().optional().describe('Source branch (default: main)'),
    },
  },
  async ({ owner, repo, branch, from_branch }) => {
    // Get SHA of source branch
    const sourceRef = await githubAPI(`/repos/${owner}/${repo}/git/refs/heads/${from_branch || 'main'}`);

    // Create new branch
    const newRef = await githubAPI(`/repos/${owner}/${repo}/git/refs`, {
      method: 'POST',
      body: JSON.stringify({
        ref: `refs/heads/${branch}`,
        sha: sourceRef.object.sha,
      }),
    });

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          branch,
          sha: newRef.object.sha.slice(0, 7),
          url: `https://github.com/${owner}/${repo}/tree/${branch}`,
        }, null, 2),
      }],
    };
  }
);

// search_code
registerTool(
  'search_code',
  {
    title: 'Search Code',
    description: 'Search for code in repositories. [Read-Only]',
    inputSchema: {
      q: z.string().describe('Search query (e.g., "function repo:owner/repo")'),
      per_page: z.number().optional().describe('Results per page (max 100)'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ q, per_page }) => {
    let url = `/search/code?q=${encodeURIComponent(q)}`;
    if (per_page) url += `&per_page=${per_page}`;

    const result = await githubAPI(url);

    return createDualOutput(`🔎 Found ${result.items.length} code matches`, {
      total_count: result.total_count,
      items: result.items.slice(0, 20).map((i: any) => ({
        name: i.name,
        path: i.path,
        repository: i.repository.full_name,
        url: i.html_url,
      })),
    });
  }
);

// search_repositories
registerTool(
  'search_repositories',
  {
    title: 'Search Repositories',
    description: 'Search repositories. [Read-Only]',
    inputSchema: {
      q: z.string().describe('Search query'),
      per_page: z.number().optional().describe('Results per page (max 100)'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ q, per_page }) => {
    let url = `/search/repositories?q=${encodeURIComponent(q)}`;
    if (per_page) url += `&per_page=${per_page}`;
    const result = await githubAPI(url);
    return createDualOutput(`📦 Found ${result.items.length} repositories`, {
      total_count: result.total_count,
      items: result.items.slice(0, 20).map((r: any) => ({
        full_name: r.full_name,
        description: r.description,
        stars: r.stargazers_count,
        url: r.html_url,
      })),
    });
  }
);

// list_commits
registerTool(
  'list_commits',
  {
    title: 'List Commits',
    description: 'List commits in a repository. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      sha: z.string().optional().describe('Branch or SHA to list from'),
      per_page: z.number().optional().describe('Results per page'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, sha, per_page }) => {
    let url = `/repos/${owner}/${repo}/commits?`;
    if (sha) url += `sha=${sha}&`;
    if (per_page) url += `per_page=${per_page}`;

    const commits = await githubAPI(url);

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify(commits.slice(0, 20).map((c: any) => ({
          sha: c.sha.slice(0, 7),
          message: c.commit.message.split('\n')[0],
          author: c.commit.author.name,
          date: c.commit.author.date,
        })), null, 2),
      }],
    };
  }
);

// ============================================================================
// CUSTOM TOOLS (validate_build, Vercel, rate limit, cache)
// ============================================================================

// validate_build
registerTool(
  'validate_build',
  {
    title: 'Validate Build',
    description: `Validate a repository's TypeScript compilation.

Modes:
- "quick" (default): Fast typecheck only (tsc --noEmit) - completes in ~20-30s
- "full": Full build with npm run build - may timeout on large repos

Quick mode is optimized for serverless timeouts and catches most TypeScript errors.`,
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      branch: z.string().optional().describe('Branch name (default: main)'),
      mode: z.enum(['quick', 'full']).optional().describe('Validation mode: "quick" (typecheck only, default) or "full" (complete build)'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, branch, mode }) => {
    const branchName = branch || 'main';
    const validationMode = mode || 'quick';
    const requestedAt = new Date().toISOString();

    if (VALIDATE_BUILD_MODE === 'disabled') {
      return createSafeErrorResponse(new Error('validate_build is disabled in this environment'), 'validate_build');
    }

    if (VALIDATE_BUILD_MODE === 'actions') {
      if (!VALIDATE_BUILD_WORKFLOW) {
        return createSafeErrorResponse(new Error('VALIDATE_BUILD_WORKFLOW is not configured'), 'validate_build');
      }

      const dispatchPayload = {
        ref: VALIDATE_BUILD_WORKFLOW_REF,
        inputs: {
          owner,
          repo,
          branch: branchName,
          mode: validationMode,
          requested_at: requestedAt,
        },
      };

      const response = await fetchWithTimeout(
        `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${VALIDATE_BUILD_WORKFLOW}/dispatches`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${GITHUB_TOKEN}`,
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(dispatchPayload),
        },
        DEFAULT_TIMEOUT_MS
      );

      if (!response.ok) {
        const responseText = await response.text();
        if (response.status === 404 || response.status === 422) {
          return createSuccessOutput(
            'Build workflow dispatch skipped',
            {
              repository: `${owner}/${repo}`,
              branch: branchName,
              mode: validationMode,
              workflow: VALIDATE_BUILD_WORKFLOW,
              ref: VALIDATE_BUILD_WORKFLOW_REF,
              requested_at: requestedAt,
              skipped: true,
              reason: responseText,
            },
            'Workflow dispatch skipped. Ensure the workflow exists and supports workflow_dispatch.'
          );
        }
        parseGitHubError(response, response.url, responseText);
      }

      return createSuccessOutput('Build queued via GitHub Actions', {
        repository: `${owner}/${repo}`,
        branch: branchName,
        mode: validationMode,
        workflow: VALIDATE_BUILD_WORKFLOW,
        ref: VALIDATE_BUILD_WORKFLOW_REF,
        requested_at: requestedAt,
      }, 'Workflow dispatch created. Check Actions in the target repo for status.');
    }

    if (!VALIDATE_BUILD_WORKER_URL) {
      return createSafeErrorResponse(new Error('VALIDATE_BUILD_WORKER_URL is not configured'), 'validate_build');
    }

    const response = await fetchWithTimeout(
      VALIDATE_BUILD_WORKER_URL,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          owner,
          repo,
          branch: branchName,
          mode: validationMode,
          requested_at: requestedAt,
        }),
      },
      DEFAULT_TIMEOUT_MS
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Worker enqueue failed: ${errorText || response.status}`);
    }

    let payload: any = {};
    try {
      payload = await response.json();
    } catch {
      payload = { message: 'Build request accepted' };
    }

    return createSuccessOutput('Build queued', {
      repository: `${owner}/${repo}`,
      branch: branchName,
      mode: validationMode,
      requested_at: requestedAt,
      worker: VALIDATE_BUILD_WORKER_URL,
      job: payload,
    }, payload.message || 'Worker accepted the build request.');
  }
);

// check_github_rate_limit
registerTool(
  'check_github_rate_limit',
  {
    title: 'Check GitHub Rate Limit',
    description: 'Check GitHub API rate limit status. [Read-Only]',
    inputSchema: {},
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async () => {
    const data = await githubAPI('/rate_limit');
    const core = data.resources.core;

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          limit: core.limit,
          remaining: core.remaining,
          used: core.used,
          reset_at: new Date(core.reset * 1000).toISOString(),
          status: core.remaining > 100 ? '✅ Healthy' : core.remaining > 10 ? '⚠️ Low' : '🛑 Critical',
        }, null, 2),
      }],
    };
  }
);

// clear_cache
registerTool(
  'clear_cache',
  {
    title: 'Clear Cache',
    description: 'Clear the file content cache.',
    inputSchema: {
      owner: z.string().optional().describe('Owner to clear (optional)'),
      repo: z.string().optional().describe('Repo to clear (requires owner)'),
    },
  },
  async ({ owner, repo }) => {
    const statsBefore = cache.getStats();
    let cleared = 0;

    if (owner && repo) {
      cache.invalidateRepo(owner, repo);
    } else {
      cleared = cache.clear();
    }

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({ cleared, statsBefore, statsAfter: cache.getStats() }, null, 2),
      }],
    };
  }
);

// Vercel API helper
async function vercelAPI(endpoint: string, options: RequestInit = {}): Promise<any> {
  if (!VERCEL_TOKEN) throw new Error('VERCEL_TOKEN not configured');
  const url = new URL(`https://api.vercel.com${endpoint}`);
  if (VERCEL_TEAM_ID && !url.searchParams.has('teamId')) {
    url.searchParams.set('teamId', VERCEL_TEAM_ID);
  }
  const response = await fetch(url, {
    ...options,
    headers: {
      'Authorization': `Bearer ${VERCEL_TOKEN}`,
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Vercel API error ${response.status}: ${error}`);
  }
  return response.json();
}

// Vercel tools
if (VERCEL_TOKEN) {
  registerTool(
    'list_vercel_deployments',
    {
      title: 'List Vercel Deployments',
      description: 'List recent Vercel deployments. [Read-Only]',
      inputSchema: {
        projectId: z.string().optional().describe('Project ID or name'),
        limit: z.number().optional().describe('Number of results'),
      },
      annotations: { readOnlyHint: true, idempotentHint: true },
    },
    async ({ projectId, limit }) => {
      const url = new URL('/v6/deployments', 'https://api.vercel.com');
      url.searchParams.set('limit', String(Math.min(limit || 10, 100)));
      if (projectId) url.searchParams.set('projectId', projectId);

      const response = await vercelAPI(`${url.pathname}${url.search}`);

      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify({
            count: response.deployments?.length || 0,
            deployments: (response.deployments || []).map((d: any) => ({
              id: d.uid,
              name: d.name,
              state: d.readyState || d.state,
              url: d.url ? `https://${d.url}` : null,
              branch: d.meta?.githubCommitRef || null,
            })),
          }, null, 2),
        }],
      };
    }
  );

  registerTool(
    'get_vercel_deployment',
    {
      title: 'Get Vercel Deployment',
      description: 'Get Vercel deployment details. [Read-Only]',
      inputSchema: { deploymentId: z.string().describe('Deployment ID') },
      annotations: { readOnlyHint: true, idempotentHint: true },
    },
    async ({ deploymentId }) => {
      const d = await vercelAPI(`/v13/deployments/${deploymentId}`);

      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify({
            id: d.uid || d.id,
            name: d.name,
            state: d.readyState,
            url: d.url ? `https://${d.url}` : null,
            errorCode: d.errorCode || null,
            errorMessage: d.errorMessage || null,
          }, null, 2),
        }],
      };
    }
  );
}

// ============================================================================
// EXPRESS APP
// ============================================================================

const app = express();
app.use(express.json({ limit: '10mb' }));

app.get('/health', (_req, res) => {
  const toolsetStats = toolsetManager.getStats();
  res.json({
    status: 'ok',
    version: '4.3.2',
    toolsets: toolsetStats,
    cache: cache.getStats(),
    requestQueue: requestQueue.getStats(),
    memory: process.memoryUsage(),
    vercelEnabled: !!VERCEL_TOKEN,
    vercelTeamId: VERCEL_TEAM_ID || null,
    buildCommand: BUILD_COMMAND,
    maxConcurrentRequests: MAX_CONCURRENT_REQUESTS,
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

    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
      enableJsonResponse: true,
    });

    await server.connect(transport);
    res.on('close', () => transport.close().catch(console.error));
    await transport.handleRequest(req, res, req.body);
  } catch (error) {
    console.error('MCP error:', error);
    // Sanitize error message before returning
    const errorMessage = error instanceof Error ? sanitizeErrorMessage(error.message) : 'Internal server error';
    res.status(500).json({
      jsonrpc: '2.0',
      error: { code: -32603, message: errorMessage },
      id: null,
    });
  }
});

app.options('/mcp', (_req, res) => res.status(204).send());

// ============================================================================
// STARTUP
// ============================================================================

const startupStats = toolsetManager.getStats();
console.log('========================================');
console.log('  Nova OS GitHub Connector v4.3.0');
console.log('  (MCP SDK Patterns + Enterprise)');
console.log('========================================');
console.log('');
console.log('Core Features:');
console.log('  ✓ Dynamic toolset management');
console.log('  ✓ Path traversal validation');
console.log('  ✓ Error message sanitization');
console.log(`  ✓ Concurrent request queue (max ${MAX_CONCURRENT_REQUESTS})`);
console.log('');
console.log('MCP SDK Patterns (v4.3.0):');
console.log('  ✓ Domain-specific exceptions (RateLimit, NotFound, Auth, etc.)');
console.log('  ✓ Retry with exponential backoff + jitter');
console.log('  ✓ Dual output format (text + structured data)');
console.log('  ✓ Progress tracking for bulk operations');
console.log('  ✓ Pre-parse JSON inputs (Claude Desktop compatibility)');
console.log(`  ✓ Per-request configurable timeouts (default: ${DEFAULT_TIMEOUT_MS}ms)`);
console.log('');
console.log('Toolsets:');
toolsetManager.getAllToolsets().forEach(ts => {
  const status = ts.enabled ? '✓' : '✗';
  console.log(`  [${status}] ${ts.name}: ${ts.toolCount} tools (${ts.readOnly ? 'read-only' : 'read/write'})`);
});
console.log('');
console.log(`Summary: ${startupStats.enabledToolsets}/${startupStats.totalToolsets} toolsets, ${startupStats.enabledTools}/${startupStats.totalTools} tools`);
console.log('');

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
  console.log('');
  console.log('URL: https://nova-os-connector.onrender.com/mcp');
  console.log('');
});
