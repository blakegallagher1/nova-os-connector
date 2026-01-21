/**
 * Tests for MCP SDK Patterns
 * - Domain-specific exceptions
 * - Retry with exponential backoff
 * - Dual output format
 * - Progress tracking
 * - Pre-parse JSON inputs
 */

// ============================================================================
// DOMAIN-SPECIFIC EXCEPTIONS (Copy for testing)
// ============================================================================

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

class GitHubConflictError extends GitHubAPIError {
  constructor(message: string, endpoint: string, rawResponse?: string) {
    super(message, 409, endpoint, rawResponse);
    this.name = 'GitHubConflictError';
  }
}

class GitHubTimeoutError extends GitHubAPIError {
  constructor(endpoint: string, public readonly timeoutMs: number) {
    super(`Request to ${endpoint} timed out after ${timeoutMs}ms`, 408, endpoint);
    this.name = 'GitHubTimeoutError';
  }
}

// ============================================================================
// RETRY UTILITIES (Copy for testing)
// ============================================================================

function isRetryableError(error: unknown): boolean {
  if (error instanceof GitHubRateLimitError) return true;
  if (error instanceof GitHubTimeoutError) return true;
  if (error instanceof GitHubConflictError) return true;
  if (error instanceof GitHubAPIError) {
    return error.statusCode >= 500 && error.statusCode < 600;
  }
  if (error instanceof TypeError && error.message.includes('fetch')) return true;
  return false;
}

function calculateBackoffDelay(attempt: number, baseDelayMs: number, error?: unknown): number {
  if (error instanceof GitHubRateLimitError && error.retryAfter > 0) {
    return error.retryAfter * 1000;
  }
  const exponentialDelay = baseDelayMs * Math.pow(2, attempt);
  const jitter = 0.5 + Math.random();
  return Math.min(exponentialDelay * jitter, 60000);
}

// ============================================================================
// DUAL OUTPUT FORMAT (Copy for testing)
// ============================================================================

interface DualOutput {
  content: Array<{ type: 'text'; text: string }>;
  structuredContent?: Record<string, unknown>;
  _cached?: boolean;
  _progress?: { completed: number; total: number; message?: string };
}

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

// ============================================================================
// PROGRESS TRACKER (Copy for testing)
// ============================================================================

class ProgressTracker {
  private completed = 0;
  private errors: Array<{ item: string; error: string }> = [];
  private startTime = Date.now();

  constructor(private total: number, private operationName: string) {}

  success(item?: string): void {
    this.completed++;
  }

  failure(item: string, error: string): void {
    this.completed++;
    this.errors.push({ item, error });
  }

  getPercentage(): number {
    return this.total > 0 ? Math.round((this.completed / this.total) * 100) : 0;
  }

  getProgressMessage(): string {
    return `[${this.operationName}] ${this.completed}/${this.total} (${this.getPercentage()}%)`;
  }

  getSummary() {
    const durationMs = Date.now() - this.startTime;
    const successful = this.completed - this.errors.length;
    return {
      operation: this.operationName,
      total: this.total,
      successful,
      failed: this.errors.length,
      errors: this.errors.slice(0, 10),
      durationMs,
      itemsPerSecond: durationMs > 0 ? Math.round((this.completed / durationMs) * 1000 * 10) / 10 : 0,
    };
  }
}

// ============================================================================
// PRE-PARSE JSON (Copy for testing)
// ============================================================================

function preParseInputs<T extends Record<string, unknown>>(
  inputs: T,
  jsonFields?: string[]
): T {
  const result = { ...inputs } as Record<string, unknown>;

  for (const [key, value] of Object.entries(result)) {
    if (typeof value !== 'string') continue;
    if (jsonFields && !jsonFields.includes(key)) continue;

    const trimmed = value.trim();
    if (
      (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
      (trimmed.startsWith('[') && trimmed.endsWith(']'))
    ) {
      try {
        result[key] = JSON.parse(trimmed);
      } catch {
        // Not valid JSON
      }
    }
  }

  return result as T;
}

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
      // Not valid JSON
    }
  }

  return value as T;
}

function parseFilesInput(files: unknown): Array<{ path: string; content: string }> {
  const parsed = preParseValue<Array<{ path: string; content: string }>>(files);

  if (!Array.isArray(parsed)) {
    throw new Error('Files must be an array of {path, content} objects');
  }

  for (const file of parsed) {
    if (typeof file.path !== 'string' || typeof file.content !== 'string') {
      throw new Error('Each file must have "path" (string) and "content" (string) properties');
    }
  }

  return parsed;
}

// ============================================================================
// TESTS
// ============================================================================

describe('Domain-Specific Exceptions', () => {
  describe('GitHubRateLimitError', () => {
    it('should contain rate limit information', () => {
      const error = new GitHubRateLimitError(5000, 0, 1700000000, 60, '/repos/test');

      expect(error.limit).toBe(5000);
      expect(error.remaining).toBe(0);
      expect(error.resetTimestamp).toBe(1700000000);
      expect(error.retryAfter).toBe(60);
      expect(error.statusCode).toBe(429);
      expect(error.name).toBe('GitHubRateLimitError');
    });

    it('should format message with reset time', () => {
      const error = new GitHubRateLimitError(5000, 0, 1700000000, 60, '/repos/test');
      expect(error.message).toContain('rate limit exceeded');
      expect(error.message).toContain('5000');
    });
  });

  describe('GitHubNotFoundError', () => {
    it('should contain resource information', () => {
      const error = new GitHubNotFoundError('contents', '/src/file.ts', '/repos/owner/repo/contents/src/file.ts');

      expect(error.resourceType).toBe('contents');
      expect(error.resourcePath).toBe('/src/file.ts');
      expect(error.statusCode).toBe(404);
      expect(error.name).toBe('GitHubNotFoundError');
    });
  });

  describe('GitHubAuthError', () => {
    it('should handle 401 errors', () => {
      const error = new GitHubAuthError('Bad credentials', 401, '/user');

      expect(error.statusCode).toBe(401);
      expect(error.message).toBe('Bad credentials');
    });

    it('should handle 403 errors with required scopes', () => {
      const error = new GitHubAuthError('Requires admin scope', 403, '/repos/owner/repo/settings', ['admin:repo']);

      expect(error.statusCode).toBe(403);
      expect(error.requiredScopes).toEqual(['admin:repo']);
    });
  });

  describe('GitHubValidationError', () => {
    it('should contain validation errors', () => {
      const error = new GitHubValidationError('Validation Failed', '/repos/owner/repo/pulls', [
        { resource: 'PullRequest', field: 'head', code: 'missing_field' }
      ]);

      expect(error.statusCode).toBe(422);
      expect(error.errors).toHaveLength(1);
      expect(error.errors![0].field).toBe('head');
    });
  });

  describe('GitHubTimeoutError', () => {
    it('should contain timeout information', () => {
      const error = new GitHubTimeoutError('/repos/owner/repo/contents', 30000);

      expect(error.timeoutMs).toBe(30000);
      expect(error.statusCode).toBe(408);
      expect(error.message).toContain('timed out');
      expect(error.message).toContain('30000ms');
    });
  });
});

describe('Retry Utilities', () => {
  describe('isRetryableError', () => {
    it('should return true for rate limit errors', () => {
      const error = new GitHubRateLimitError(5000, 0, 1700000000, 60, '/test');
      expect(isRetryableError(error)).toBe(true);
    });

    it('should return true for timeout errors', () => {
      const error = new GitHubTimeoutError('/test', 30000);
      expect(isRetryableError(error)).toBe(true);
    });

    it('should return true for conflict errors', () => {
      const error = new GitHubConflictError('Merge conflict', '/test');
      expect(isRetryableError(error)).toBe(true);
    });

    it('should return true for 5xx server errors', () => {
      const error = new GitHubAPIError('Internal error', 500, '/test');
      expect(isRetryableError(error)).toBe(true);

      const error502 = new GitHubAPIError('Bad gateway', 502, '/test');
      expect(isRetryableError(error502)).toBe(true);
    });

    it('should return false for 4xx client errors', () => {
      const error = new GitHubNotFoundError('File', '/test', '/test');
      expect(isRetryableError(error)).toBe(false);

      const error400 = new GitHubAPIError('Bad request', 400, '/test');
      expect(isRetryableError(error400)).toBe(false);
    });

    it('should return false for auth errors', () => {
      const error = new GitHubAuthError('Unauthorized', 401, '/test');
      expect(isRetryableError(error)).toBe(false);
    });
  });

  describe('calculateBackoffDelay', () => {
    it('should use retry-after for rate limit errors', () => {
      const error = new GitHubRateLimitError(5000, 0, 1700000000, 60, '/test');
      const delay = calculateBackoffDelay(0, 1000, error);
      expect(delay).toBe(60000); // 60 seconds * 1000ms
    });

    it('should use exponential backoff for other errors', () => {
      // With jitter, the delay will be between 0.5x and 1.5x the exponential value
      const delay0 = calculateBackoffDelay(0, 1000); // 1000 * 2^0 * (0.5 to 1.5) = 500-1500
      const delay1 = calculateBackoffDelay(1, 1000); // 1000 * 2^1 * (0.5 to 1.5) = 1000-3000
      const delay2 = calculateBackoffDelay(2, 1000); // 1000 * 2^2 * (0.5 to 1.5) = 2000-6000

      expect(delay0).toBeGreaterThanOrEqual(500);
      expect(delay0).toBeLessThanOrEqual(1500);
      expect(delay1).toBeGreaterThanOrEqual(1000);
      expect(delay1).toBeLessThanOrEqual(3000);
      expect(delay2).toBeGreaterThanOrEqual(2000);
      expect(delay2).toBeLessThanOrEqual(6000);
    });

    it('should cap delay at 60 seconds', () => {
      const delay = calculateBackoffDelay(10, 1000); // Would be huge without cap
      expect(delay).toBeLessThanOrEqual(60000);
    });
  });
});

describe('Dual Output Format', () => {
  it('should create output with both text and structured content', () => {
    const output = createDualOutput('File created successfully', {
      success: true,
      path: '/src/file.ts',
      commit: 'abc1234',
    });

    expect(output.content).toHaveLength(1);
    expect(output.content[0].type).toBe('text');
    expect(output.content[0].text).toContain('File created successfully');
    expect(output.content[0].text).toContain('Structured Data');
    expect(output.structuredContent).toEqual({
      success: true,
      path: '/src/file.ts',
      commit: 'abc1234',
    });
  });

  it('should include cached flag when specified', () => {
    const output = createDualOutput('Data fetched', { data: 'test' }, { cached: true });
    expect(output._cached).toBe(true);
  });

  it('should include progress when specified', () => {
    const output = createDualOutput('Processing', { status: 'in_progress' }, {
      progress: { completed: 5, total: 10, message: 'Processing item 5' }
    });
    expect(output._progress).toEqual({
      completed: 5,
      total: 10,
      message: 'Processing item 5',
    });
  });
});

describe('Progress Tracker', () => {
  it('should track successful completions', () => {
    const tracker = new ProgressTracker(5, 'test_operation');

    tracker.success('item1');
    tracker.success('item2');

    expect(tracker.getPercentage()).toBe(40);
    expect(tracker.getProgressMessage()).toBe('[test_operation] 2/5 (40%)');
  });

  it('should track failures with error messages', () => {
    const tracker = new ProgressTracker(3, 'test_operation');

    tracker.success('item1');
    tracker.failure('item2', 'File not found');
    tracker.success('item3');

    const summary = tracker.getSummary();
    expect(summary.total).toBe(3);
    expect(summary.successful).toBe(2);
    expect(summary.failed).toBe(1);
    expect(summary.errors).toHaveLength(1);
    expect(summary.errors[0]).toEqual({ item: 'item2', error: 'File not found' });
  });

  it('should calculate items per second', async () => {
    const tracker = new ProgressTracker(10, 'test');

    // Add some items
    for (let i = 0; i < 10; i++) {
      tracker.success(`item${i}`);
    }

    const summary = tracker.getSummary();
    // Duration might be 0 if operations complete very fast
    expect(summary.durationMs).toBeGreaterThanOrEqual(0);
    // Items per second will be 0 if duration is 0, which is valid
    expect(summary.itemsPerSecond).toBeGreaterThanOrEqual(0);
    // Verify the formula is correct when duration > 0
    if (summary.durationMs > 0) {
      const expectedRate = Math.round((10 / summary.durationMs) * 1000 * 10) / 10;
      expect(summary.itemsPerSecond).toBe(expectedRate);
    }
  });

  it('should limit errors to 10 in summary', () => {
    const tracker = new ProgressTracker(20, 'test');

    for (let i = 0; i < 15; i++) {
      tracker.failure(`item${i}`, `Error ${i}`);
    }

    const summary = tracker.getSummary();
    expect(summary.failed).toBe(15);
    expect(summary.errors).toHaveLength(10); // Limited to 10
  });

  it('should handle zero total gracefully', () => {
    const tracker = new ProgressTracker(0, 'empty');
    expect(tracker.getPercentage()).toBe(0);
  });
});

describe('Pre-Parse JSON Inputs', () => {
  describe('preParseInputs', () => {
    it('should parse JSON string values', () => {
      const input = {
        name: 'test',
        data: '{"key": "value"}',
        array: '["a", "b", "c"]',
      };

      const result = preParseInputs(input);

      expect(result.name).toBe('test');
      expect(result.data).toEqual({ key: 'value' });
      expect(result.array).toEqual(['a', 'b', 'c']);
    });

    it('should not modify non-JSON strings', () => {
      const input = {
        text: 'just a string',
        number: 42,
        bool: true,
      };

      const result = preParseInputs(input);

      expect(result.text).toBe('just a string');
      expect(result.number).toBe(42);
      expect(result.bool).toBe(true);
    });

    it('should handle invalid JSON gracefully', () => {
      const input = {
        badJson: '{"incomplete": ',
        valid: '{"good": true}',
      };

      const result = preParseInputs(input);

      expect(result.badJson).toBe('{"incomplete": '); // Unchanged
      expect(result.valid).toEqual({ good: true });
    });

    it('should only parse specified fields when jsonFields provided', () => {
      const input = {
        shouldParse: '{"a": 1}',
        shouldNotParse: '{"b": 2}',
      };

      const result = preParseInputs(input, ['shouldParse']);

      expect(result.shouldParse).toEqual({ a: 1 });
      expect(result.shouldNotParse).toBe('{"b": 2}');
    });

    it('should handle whitespace around JSON', () => {
      const input = {
        data: '  {"key": "value"}  ',
      };

      const result = preParseInputs(input);
      expect(result.data).toEqual({ key: 'value' });
    });
  });

  describe('preParseValue', () => {
    it('should parse JSON objects', () => {
      expect(preParseValue('{"a": 1}')).toEqual({ a: 1 });
    });

    it('should parse JSON arrays', () => {
      expect(preParseValue('[1, 2, 3]')).toEqual([1, 2, 3]);
    });

    it('should return non-strings unchanged', () => {
      expect(preParseValue(42)).toBe(42);
      expect(preParseValue({ a: 1 })).toEqual({ a: 1 });
      expect(preParseValue(['a'])).toEqual(['a']);
    });
  });

  describe('parseFilesInput', () => {
    it('should parse valid files array', () => {
      const files = [
        { path: 'src/file1.ts', content: 'content1' },
        { path: 'src/file2.ts', content: 'content2' },
      ];

      const result = parseFilesInput(files);
      expect(result).toEqual(files);
    });

    it('should parse stringified files array', () => {
      const filesJson = JSON.stringify([
        { path: 'src/file.ts', content: 'test' },
      ]);

      const result = parseFilesInput(filesJson);
      expect(result).toHaveLength(1);
      expect(result[0].path).toBe('src/file.ts');
    });

    it('should throw for non-array input', () => {
      expect(() => parseFilesInput('not an array')).toThrow('must be an array');
      expect(() => parseFilesInput({ path: 'a', content: 'b' })).toThrow('must be an array');
    });

    it('should throw for invalid file objects', () => {
      expect(() => parseFilesInput([{ path: 123, content: 'test' }])).toThrow('path');
      expect(() => parseFilesInput([{ path: 'test' }])).toThrow('content');
      expect(() => parseFilesInput([{ content: 'test' }])).toThrow('path');
    });
  });
});
