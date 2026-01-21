import 'dotenv/config';
import express from 'express';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { z } from 'zod';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';

const execAsync = promisify(exec);

// ============================================================================
// CONFIGURATION
// ============================================================================

const PORT = parseInt(process.env.PORT || '8000', 10);
const GITHUB_TOKEN = process.env.GITHUB_PERSONAL_ACCESS_TOKEN;
const VERCEL_TOKEN = process.env.VERCEL_TOKEN;

// Configurable build command (defaults to 'npm run build')
const BUILD_COMMAND = process.env.BUILD_COMMAND || 'npm run build';

// Request queue configuration
const MAX_CONCURRENT_REQUESTS = parseInt(process.env.MAX_CONCURRENT_REQUESTS || '5', 10);

// Default enabled toolsets (can be overridden via env)
const DEFAULT_ENABLED_TOOLSETS = (process.env.ENABLED_TOOLSETS || 'repos,issues,pulls,search,utility,vercel,toolsets').split(',').map(s => s.trim());

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
function createSafeErrorResponse(error: unknown, toolName: string): { content: Array<{ type: 'text'; text: string }> } {
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
  }

  return {
    content: [{
      type: 'text' as const,
      text: JSON.stringify({
        error: true,
        code: errorCode,
        message: sanitizedMessage,
        tool: toolName,
        retryable,
        ...(retryAfter !== undefined ? { retry_after_seconds: retryAfter } : {}),
        suggestion: retryable
          ? `The operation failed but may succeed if retried${retryAfter ? ` after ${retryAfter} seconds` : ''}.`
          : 'The operation failed. You may need to try a different approach.',
      }, null, 2),
    }],
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

interface CacheEntry {
  data: any;
  timestamp: number;
  ttl: number;
  hits: number;
}

class ContentCache {
  private cache = new Map<string, CacheEntry>();
  private stats = { hits: 0, misses: 0, evictions: 0, invalidations: 0 };

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

  set(key: string, data: any, ttl: number = CACHE_TTL_MS): void {
    if (this.cache.size >= CACHE_MAX_ENTRIES) this.evictOldest();
    this.cache.set(key, { data, timestamp: Date.now(), ttl, hits: 0 });
  }

  invalidateRepo(owner: string, repo: string): void {
    const prefix = `file:${owner}/${repo}/`;
    let count = 0;
    for (const key of this.cache.keys()) {
      if (key.startsWith(prefix)) { this.cache.delete(key); count++; }
    }
    this.stats.invalidations += count;
  }

  clear(): number {
    const size = this.cache.size;
    this.cache.clear();
    return size;
  }

  private evictOldest(): void {
    let oldest: { key: string; timestamp: number } | null = null;
    for (const [key, entry] of this.cache.entries()) {
      if (!oldest || entry.timestamp < oldest.timestamp) oldest = { key, timestamp: entry.timestamp };
    }
    if (oldest) { this.cache.delete(oldest.key); this.stats.evictions++; }
  }

  getStats() {
    const hitRate = this.stats.hits + this.stats.misses > 0
      ? ((this.stats.hits / (this.stats.hits + this.stats.misses)) * 100).toFixed(1)
      : '0';
    return { entries: this.cache.size, hitRate: `${hitRate}%`, ...this.stats };
  }
}

const cache = new ContentCache();

// ============================================================================
// DUAL OUTPUT FORMAT (Text + Structured)
// ============================================================================

/**
 * Structured content that can be returned alongside text content.
 * This allows AI assistants to get both human-readable and machine-parseable output.
 */
interface DualOutput {
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

// ============================================================================
// MCP SERVER
// ============================================================================

const server = new McpServer({
  name: 'nova-os-github-connector',
  version: '4.3.2',
  description: 'Direct GitHub API connector for Nova OS with dynamic toolsets',
});

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
    { name: 'create_or_update_file', title: 'Create or Update File', description: 'Create or update a single file', readOnly: false, destructive: true },
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
  true, // Currently read-only (only has list)
  [
    { name: 'list_issues', title: 'List Issues', description: 'List issues in a repository', readOnly: true, destructive: false },
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
    { name: 'get_pull_request', title: 'Get Pull Request', description: 'Get PR details', readOnly: true, destructive: false },
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
server.registerTool(
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
server.registerTool(
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
server.registerTool(
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
server.registerTool(
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
server.registerTool(
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
        content = Buffer.from(data.content, 'base64').toString('utf-8');
      } else if (Array.isArray(data)) {
        content = JSON.stringify(data.map((f: any) => ({ name: f.name, type: f.type, path: f.path })), null, 2);
      } else {
        content = JSON.stringify(data, null, 2);
      }

      cache.set(cacheKey, content);
      return { content: [{ type: 'text' as const, text: content }] };
    } catch (error) {
      return createSafeErrorResponse(error, 'get_file_contents');
    }
  }
);

// batch_read_files
server.registerTool(
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
          const content = data.content ? Buffer.from(data.content, 'base64').toString('utf-8') : JSON.stringify(data);
          cache.set(cacheKey, content);
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

// create_or_update_file
server.registerTool(
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
server.registerTool(
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
server.registerTool(
  'list_issues',
  {
    title: 'List Issues',
    description: 'List issues in a GitHub repository. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      state: z.enum(['open', 'closed', 'all']).optional().describe('Filter by state'),
      per_page: z.number().optional().describe('Results per page (max 100)'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, state, per_page }) => {
    let url = `/repos/${owner}/${repo}/issues?`;
    if (state) url += `state=${state}&`;
    if (per_page) url += `per_page=${per_page}`;

    const issues = await githubAPI(url);

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify(issues.map((i: any) => ({
          number: i.number,
          title: i.title,
          state: i.state,
          user: i.user.login,
          created_at: i.created_at,
          labels: i.labels.map((l: any) => l.name),
        })), null, 2),
      }],
    };
  }
);

// create_pull_request
server.registerTool(
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

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          number: pr.number,
          url: pr.html_url,
          title: pr.title,
          state: pr.state,
          head: pr.head.ref,
          base: pr.base.ref,
        }, null, 2),
      }],
    };
  }
);

// get_pull_request
server.registerTool(
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

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          number: pr.number,
          title: pr.title,
          state: pr.state,
          merged: pr.merged,
          mergeable: pr.mergeable,
          head: pr.head.ref,
          base: pr.base.ref,
          user: pr.user.login,
          url: pr.html_url,
        }, null, 2),
      }],
    };
  }
);

// merge_pull_request
server.registerTool(
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

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          merged: result.merged,
          message: result.message,
          sha: result.sha?.slice(0, 7),
        }, null, 2),
      }],
    };
  }
);

// create_branch
server.registerTool(
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
server.registerTool(
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

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          total_count: result.total_count,
          items: result.items.slice(0, 20).map((i: any) => ({
            name: i.name,
            path: i.path,
            repository: i.repository.full_name,
            url: i.html_url,
          })),
        }, null, 2),
      }],
    };
  }
);

// list_commits
server.registerTool(
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
server.registerTool(
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
    const startTime = Date.now();

    // Use authenticated HTTPS URL for private repos
    const repoUrl = GITHUB_TOKEN
      ? `https://x-access-token:${GITHUB_TOKEN}@github.com/${owner}/${repo}.git`
      : `https://github.com/${owner}/${repo}.git`;
    const tempDir = path.join(os.tmpdir(), `validate-build-${Date.now()}`);

    let success = false;
    let errorSummary = '';
    let output = '';
    const steps: Array<{ step: string; status: string; durationMs?: number }> = [];

    try {
      await fs.mkdir(tempDir, { recursive: true });

      // Step 1: Clone (shallow, single branch for speed)
      const cloneStart = Date.now();
      try {
        await execAsync(
          `git clone --depth 1 --single-branch --branch ${branchName} ${repoUrl} .`,
          { cwd: tempDir, timeout: 30000 }
        );
        steps.push({ step: 'clone', status: 'success', durationMs: Date.now() - cloneStart });
      } catch (error: any) {
        steps.push({ step: 'clone', status: 'failed', durationMs: Date.now() - cloneStart });
        throw new Error(`Clone failed: ${error.message}`);
      }

      // Step 2: Install dependencies (fast mode with ignore-scripts)
      const installStart = Date.now();
      try {
        // Use npm ci with flags for speed: ignore scripts, prefer offline cache
        // 120s timeout for monorepos with many packages
        await execAsync(
          'npm ci --ignore-scripts --prefer-offline --no-audit --no-fund 2>&1 || npm install --ignore-scripts --prefer-offline --no-audit --no-fund 2>&1',
          { cwd: tempDir, timeout: 120000 }
        );
        steps.push({ step: 'install', status: 'success', durationMs: Date.now() - installStart });
      } catch (error: any) {
        steps.push({ step: 'install', status: 'failed', durationMs: Date.now() - installStart });
        throw new Error(`Install failed: ${error.message}`);
      }

      // Step 3: Validate (typecheck or full build based on mode)
      const validateStart = Date.now();
      const validateCmd = validationMode === 'quick'
        ? 'npx tsc --noEmit 2>&1 || npm run typecheck 2>&1'
        : `${BUILD_COMMAND} 2>&1`;

      try {
        const result = await execAsync(validateCmd, {
          cwd: tempDir,
          timeout: validationMode === 'quick' ? 45000 : 120000,
          maxBuffer: 5 * 1024 * 1024,
        });
        output = (result.stdout || '') + (result.stderr || '');
        success = true;
        steps.push({ step: 'validate', status: 'success', durationMs: Date.now() - validateStart });
      } catch (error: any) {
        output = (error.stdout || '') + (error.stderr || '');
        steps.push({ step: 'validate', status: 'failed', durationMs: Date.now() - validateStart });

        // Extract TypeScript errors
        const errorLines = output.split('\n').filter((line: string) =>
          line.includes('error TS') ||
          line.includes('Error:') ||
          line.includes('Module not found') ||
          line.includes('Cannot find')
        );
        errorSummary = errorLines.slice(0, 15).join('\n');
      }
    } catch (error: any) {
      errorSummary = sanitizeErrorMessage(error.message);
    } finally {
      // Cleanup temp directory
      await fs.rm(tempDir, { recursive: true, force: true }).catch(() => {});
    }

    const totalDurationMs = Date.now() - startTime;
    const emoji = success ? '✅' : '❌';
    const modeLabel = validationMode === 'quick' ? 'Typecheck' : 'Full Build';

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          repository: `${owner}/${repo}`,
          branch: branchName,
          mode: validationMode,
          success,
          summary: `${emoji} ${modeLabel} ${success ? 'passed' : 'failed'}`,
          duration_ms: totalDurationMs,
          steps,
          error_summary: errorSummary ? sanitizeErrorMessage(errorSummary) : null,
          output_tail: sanitizeErrorMessage(output.slice(-2000)),
          tip: validationMode === 'quick' && !success
            ? 'Run with mode="full" for complete build validation'
            : null,
        }, null, 2),
      }],
    };
  }
);

// check_github_rate_limit
server.registerTool(
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
server.registerTool(
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
  const url = `https://api.vercel.com${endpoint}`;
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
  server.registerTool(
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
      let url = `/v6/deployments?limit=${Math.min(limit || 10, 100)}`;
      if (projectId) url += `&projectId=${projectId}`;

      const response = await vercelAPI(url);

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

  server.registerTool(
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
    vercelEnabled: !!VERCEL_TOKEN,
    buildCommand: BUILD_COMMAND,
    maxConcurrentRequests: MAX_CONCURRENT_REQUESTS,
  });
});

app.post('/mcp', async (req, res) => {
  try {
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
