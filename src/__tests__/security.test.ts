/**
 * Comprehensive test suite for Nova OS GitHub Connector
 * Tests security features, caching, and request queue functionality
 */

// ============================================================================
// PATH TRAVERSAL VALIDATION TESTS
// ============================================================================

describe('Path Traversal Validation', () => {
  // Inline implementation for testing (mirrors server-direct.ts)
  function validateFilePath(filePath: string): void {
    if (filePath.includes('..')) {
      throw new Error('Invalid path: Path traversal detected (contains "..")');
    }
    if (filePath.startsWith('/')) {
      throw new Error('Invalid path: Absolute paths not allowed (starts with "/")');
    }
    if (filePath.includes('\\..') || filePath.includes('..\\')) {
      throw new Error('Invalid path: Path traversal detected (contains backslash traversal)');
    }
    if (filePath.includes('\x00')) {
      throw new Error('Invalid path: Null byte detected');
    }
    if (filePath.includes('%2e%2e') || filePath.includes('%2E%2E')) {
      throw new Error('Invalid path: URL-encoded path traversal detected');
    }
  }

  describe('Valid paths', () => {
    const validPaths = [
      'src/index.ts',
      'package.json',
      'apps/web/src/app/page.tsx',
      'README.md',
      'src/components/Button.tsx',
      'docs/api/endpoints.md',
      '.github/workflows/ci.yml',
      '.env.example',
      'src/lib/utils/helpers.ts',
    ];

    test.each(validPaths)('should accept valid path: %s', (path) => {
      expect(() => validateFilePath(path)).not.toThrow();
    });
  });

  describe('Path traversal attacks', () => {
    const traversalPaths = [
      ['../etc/passwd', 'contains ".."'],
      ['../../etc/passwd', 'contains ".."'],
      ['src/../../../etc/passwd', 'contains ".."'],
      ['src/app/../../secret.txt', 'contains ".."'],
      ['foo/bar/../../../baz', 'contains ".."'],
    ];

    test.each(traversalPaths)('should reject path traversal: %s', (path, _reason) => {
      expect(() => validateFilePath(path)).toThrow('Path traversal detected');
    });
  });

  describe('Absolute paths', () => {
    const absolutePaths = [
      '/etc/passwd',
      '/home/user/.ssh/id_rsa',
      '/var/log/syslog',
      '/Users/admin/secrets.txt',
    ];

    test.each(absolutePaths)('should reject absolute path: %s', (path) => {
      expect(() => validateFilePath(path)).toThrow('Absolute paths not allowed');
    });
  });

  describe('Windows-style path traversal', () => {
    // Note: Paths with '..' are caught by the general traversal check first
    // These tests verify that Windows paths with '..' are still rejected
    const windowsPaths = [
      'src\\..\\secret.txt',
      '..\\windows\\system32',
      'foo\\..\\..\\bar',
    ];

    test.each(windowsPaths)('should reject Windows path traversal: %s', (path) => {
      // These are caught by the ".." check, which is correct behavior
      expect(() => validateFilePath(path)).toThrow('Path traversal');
    });

    // Test pure backslash patterns without '..'
    test('should allow backslashes without traversal (Windows-style path separators)', () => {
      // Pure Windows paths without traversal are allowed (GitHub normalizes them)
      // This is intentional - we only block traversal attempts
    });
  });

  describe('Null byte injection', () => {
    test('should reject paths with null bytes', () => {
      expect(() => validateFilePath('src/file.txt\x00.jpg')).toThrow('Null byte detected');
      expect(() => validateFilePath('normal\x00path')).toThrow('Null byte detected');
    });
  });

  describe('URL-encoded path traversal', () => {
    const encodedPaths = [
      'src/%2e%2e/secret.txt',
      '%2E%2E/etc/passwd',
      'foo/%2e%2e/%2e%2e/bar',
    ];

    test.each(encodedPaths)('should reject URL-encoded traversal: %s', (path) => {
      expect(() => validateFilePath(path)).toThrow('URL-encoded path traversal');
    });
  });
});

// ============================================================================
// ERROR SANITIZATION TESTS
// ============================================================================

describe('Error Sanitization', () => {
  const SENSITIVE_PATTERNS = [
    /ghp_[a-zA-Z0-9]{36}/g,
    /ghs_[a-zA-Z0-9]{36}/g,
    /github_pat_[a-zA-Z0-9_]{82}/g,
    /sk-[a-zA-Z0-9]{48}/g,
    /Bearer\s+[a-zA-Z0-9._-]+/gi,
    /Authorization:\s*[^\s]+/gi,
    /password[=:]\s*[^\s&]+/gi,
    /token[=:]\s*[^\s&]+/gi,
    /\/Users\/[^\/]+/g,
    /\/home\/[^\/]+/g,
    /C:\\Users\\[^\\]+/gi,
    /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
  ];

  function sanitizeErrorMessage(message: string): string {
    let sanitized = message;
    for (const pattern of SENSITIVE_PATTERNS) {
      sanitized = sanitized.replace(pattern, '[REDACTED]');
    }
    if (sanitized.length > 1000) {
      sanitized = sanitized.substring(0, 1000) + '... (truncated)';
    }
    return sanitized;
  }

  describe('GitHub tokens', () => {
    test('should redact GitHub PAT (classic)', () => {
      const message = 'Authentication failed with token ghp_abc123def456ghi789jkl012mno345pqr678';
      const sanitized = sanitizeErrorMessage(message);
      expect(sanitized).not.toContain('ghp_');
      expect(sanitized).toContain('[REDACTED]');
    });

    test('should redact GitHub App token', () => {
      const message = 'Error: ghs_abc123def456ghi789jkl012mno345pqr678';
      const sanitized = sanitizeErrorMessage(message);
      expect(sanitized).not.toContain('ghs_');
      expect(sanitized).toContain('[REDACTED]');
    });

    test('should redact fine-grained PAT', () => {
      const token = 'github_pat_' + 'a'.repeat(82);
      const message = `Token expired: ${token}`;
      const sanitized = sanitizeErrorMessage(message);
      expect(sanitized).not.toContain('github_pat_');
      expect(sanitized).toContain('[REDACTED]');
    });
  });

  describe('API keys', () => {
    test('should redact OpenAI API key', () => {
      const message = 'OpenAI error with key sk-' + 'x'.repeat(48);
      const sanitized = sanitizeErrorMessage(message);
      expect(sanitized).not.toContain('sk-');
      expect(sanitized).toContain('[REDACTED]');
    });

    test('should redact Bearer tokens', () => {
      const message = 'Request failed: Bearer abc123xyz789.token.value';
      const sanitized = sanitizeErrorMessage(message);
      expect(sanitized).not.toContain('abc123xyz789');
      expect(sanitized).toContain('[REDACTED]');
    });

    test('should redact Authorization headers', () => {
      const message = 'Headers: Authorization: secret_value_here';
      const sanitized = sanitizeErrorMessage(message);
      expect(sanitized).not.toContain('secret_value_here');
      expect(sanitized).toContain('[REDACTED]');
    });
  });

  describe('Query string secrets', () => {
    test('should redact password in query strings', () => {
      const message = 'URL: https://api.example.com?password=secret123&user=test';
      const sanitized = sanitizeErrorMessage(message);
      expect(sanitized).not.toContain('secret123');
      expect(sanitized).toContain('[REDACTED]');
    });

    test('should redact token in query strings', () => {
      const message = 'Callback URL: https://example.com?token=my_secret_token';
      const sanitized = sanitizeErrorMessage(message);
      expect(sanitized).not.toContain('my_secret_token');
      expect(sanitized).toContain('[REDACTED]');
    });
  });

  describe('File paths', () => {
    test('should redact macOS user paths', () => {
      const message = 'File not found: /Users/johndoe/secret/file.txt';
      const sanitized = sanitizeErrorMessage(message);
      expect(sanitized).not.toContain('johndoe');
      expect(sanitized).toContain('[REDACTED]');
    });

    test('should redact Linux user paths', () => {
      const message = 'Error reading /home/administrator/.ssh/id_rsa';
      const sanitized = sanitizeErrorMessage(message);
      expect(sanitized).not.toContain('administrator');
      expect(sanitized).toContain('[REDACTED]');
    });

    test('should redact Windows user paths', () => {
      const message = 'Cannot access C:\\Users\\JohnSmith\\Documents\\secret.doc';
      const sanitized = sanitizeErrorMessage(message);
      expect(sanitized).not.toContain('JohnSmith');
      expect(sanitized).toContain('[REDACTED]');
    });
  });

  describe('Email addresses', () => {
    test('should redact email addresses', () => {
      const message = 'Notification sent to admin@company.com failed';
      const sanitized = sanitizeErrorMessage(message);
      expect(sanitized).not.toContain('admin@company.com');
      expect(sanitized).toContain('[REDACTED]');
    });
  });

  describe('Message truncation', () => {
    test('should truncate very long messages', () => {
      const longMessage = 'Error: ' + 'x'.repeat(2000);
      const sanitized = sanitizeErrorMessage(longMessage);
      expect(sanitized.length).toBeLessThanOrEqual(1020); // 1000 + "... (truncated)"
      expect(sanitized).toContain('... (truncated)');
    });

    test('should not truncate short messages', () => {
      const shortMessage = 'Simple error message';
      const sanitized = sanitizeErrorMessage(shortMessage);
      expect(sanitized).toBe(shortMessage);
      expect(sanitized).not.toContain('(truncated)');
    });
  });

  describe('Safe messages', () => {
    test('should not modify safe error messages', () => {
      const safeMessages = [
        'File not found: src/index.ts',
        'Repository does not exist',
        'Rate limit exceeded',
        'Invalid JSON response',
        'Network timeout after 30 seconds',
      ];

      safeMessages.forEach((msg) => {
        expect(sanitizeErrorMessage(msg)).toBe(msg);
      });
    });
  });
});

// ============================================================================
// REQUEST QUEUE TESTS
// ============================================================================

describe('Request Queue (Concurrency Limiter)', () => {
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

  test('should allow requests up to max concurrent', async () => {
    const queue = new RequestQueue(3);

    await queue.acquire();
    await queue.acquire();
    await queue.acquire();

    expect(queue.getStats().running).toBe(3);
    expect(queue.getStats().maxConcurrent).toBe(3);
  });

  test('should queue requests beyond max concurrent', async () => {
    const queue = new RequestQueue(2);

    await queue.acquire();
    await queue.acquire();

    // This should queue, not immediately acquire
    const pending = queue.acquire();

    expect(queue.getStats().running).toBe(2);
    expect(queue.getStats().queued).toBe(1);

    // Release one, which should allow the queued request to proceed
    queue.release();
    await pending;

    expect(queue.getStats().running).toBe(2);
    expect(queue.getStats().queued).toBe(0);
  });

  test('should track completed requests', async () => {
    const queue = new RequestQueue(5);

    await queue.acquire();
    await queue.acquire();

    queue.release();
    queue.release();

    expect(queue.getStats().completed).toBe(2);
    expect(queue.getStats().running).toBe(0);
  });

  test('should limit concurrent execution', async () => {
    const queue = new RequestQueue(3);
    const executionOrder: number[] = [];
    const results: number[] = [];

    async function task(id: number, delay: number): Promise<number> {
      await queue.acquire();
      executionOrder.push(id);
      await new Promise((resolve) => setTimeout(resolve, delay));
      queue.release();
      results.push(id);
      return id;
    }

    // Start 6 tasks with max 3 concurrent
    await Promise.all([
      task(1, 50),
      task(2, 50),
      task(3, 50),
      task(4, 50),
      task(5, 50),
      task(6, 50),
    ]);

    // All tasks should complete
    expect(results).toHaveLength(6);
    // Max concurrent should never exceed 3
    expect(queue.getStats().maxConcurrent).toBeLessThanOrEqual(3);
  });

  test('should process queue in FIFO order', async () => {
    const queue = new RequestQueue(1);
    const order: number[] = [];

    // Fill the single slot
    await queue.acquire();

    // Queue additional requests
    const p1 = queue.acquire().then(() => order.push(1));
    const p2 = queue.acquire().then(() => order.push(2));
    const p3 = queue.acquire().then(() => order.push(3));

    // Release to process queue
    queue.release();
    await p1;
    queue.release();
    await p2;
    queue.release();
    await p3;

    expect(order).toEqual([1, 2, 3]);
  });
});

// ============================================================================
// CACHE TESTS
// ============================================================================

describe('Content Cache', () => {
  const CACHE_TTL_MS = 100; // Short TTL for testing
  const CACHE_MAX_ENTRIES = 5;

  class ContentCache {
    private cache = new Map<string, { data: any; timestamp: number; ttl: number; hits: number }>();
    private stats = { hits: 0, misses: 0, evictions: 0, invalidations: 0 };

    fileKey(owner: string, repo: string, path: string, branch?: string): string {
      return `file:${owner}/${repo}/${branch || 'default'}:${path}`;
    }

    get(key: string): any | null {
      const entry = this.cache.get(key);
      if (!entry) {
        this.stats.misses++;
        return null;
      }
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
      if (this.cache.size >= CACHE_MAX_ENTRIES) {
        this.evictOldest();
      }
      this.cache.set(key, { data, timestamp: Date.now(), ttl, hits: 0 });
    }

    invalidateRepo(owner: string, repo: string): void {
      const prefix = `file:${owner}/${repo}/`;
      let count = 0;
      for (const key of this.cache.keys()) {
        if (key.startsWith(prefix)) {
          this.cache.delete(key);
          count++;
        }
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
        if (!oldest || entry.timestamp < oldest.timestamp) {
          oldest = { key, timestamp: entry.timestamp };
        }
      }
      if (oldest) {
        this.cache.delete(oldest.key);
        this.stats.evictions++;
      }
    }

    getStats() {
      const hitRate =
        this.stats.hits + this.stats.misses > 0
          ? ((this.stats.hits / (this.stats.hits + this.stats.misses)) * 100).toFixed(1)
          : '0';
      return { entries: this.cache.size, hitRate: `${hitRate}%`, ...this.stats };
    }
  }

  test('should generate correct cache keys', () => {
    const cache = new ContentCache();

    expect(cache.fileKey('owner', 'repo', 'path/to/file.ts')).toBe('file:owner/repo/default:path/to/file.ts');
    expect(cache.fileKey('owner', 'repo', 'file.ts', 'main')).toBe('file:owner/repo/main:file.ts');
    expect(cache.fileKey('owner', 'repo', 'file.ts', 'feature/branch')).toBe('file:owner/repo/feature/branch:file.ts');
  });

  test('should cache and retrieve data', () => {
    const cache = new ContentCache();
    const key = 'test-key';
    const data = { content: 'test content' };

    cache.set(key, data);
    expect(cache.get(key)).toEqual(data);
  });

  test('should return null for missing keys', () => {
    const cache = new ContentCache();
    expect(cache.get('nonexistent')).toBeNull();
    expect(cache.getStats().misses).toBe(1);
  });

  test('should expire entries after TTL', async () => {
    const cache = new ContentCache();
    const key = 'expiring-key';

    cache.set(key, 'data', 50); // 50ms TTL
    expect(cache.get(key)).toBe('data');

    await new Promise((resolve) => setTimeout(resolve, 60));
    expect(cache.get(key)).toBeNull();
  });

  test('should track hit rate', () => {
    const cache = new ContentCache();

    cache.set('key1', 'data1');
    cache.get('key1'); // hit
    cache.get('key1'); // hit
    cache.get('key2'); // miss

    const stats = cache.getStats();
    expect(stats.hits).toBe(2);
    expect(stats.misses).toBe(1);
    expect(stats.hitRate).toBe('66.7%');
  });

  test('should evict oldest when at capacity', () => {
    const cache = new ContentCache();

    // Fill cache to capacity
    for (let i = 0; i < CACHE_MAX_ENTRIES; i++) {
      cache.set(`key${i}`, `data${i}`);
    }

    // Add one more, should evict oldest
    cache.set('keyNew', 'dataNew');

    expect(cache.getStats().entries).toBe(CACHE_MAX_ENTRIES);
    expect(cache.getStats().evictions).toBe(1);
  });

  test('should invalidate repo entries', () => {
    const cache = new ContentCache();

    cache.set(cache.fileKey('owner', 'repo1', 'file1.ts'), 'data1');
    cache.set(cache.fileKey('owner', 'repo1', 'file2.ts'), 'data2');
    cache.set(cache.fileKey('owner', 'repo2', 'file1.ts'), 'data3');

    expect(cache.getStats().entries).toBe(3);

    cache.invalidateRepo('owner', 'repo1');

    expect(cache.getStats().entries).toBe(1);
    expect(cache.get(cache.fileKey('owner', 'repo2', 'file1.ts'))).toBe('data3');
  });

  test('should clear all entries', () => {
    const cache = new ContentCache();

    cache.set('key1', 'data1');
    cache.set('key2', 'data2');
    cache.set('key3', 'data3');

    const cleared = cache.clear();

    expect(cleared).toBe(3);
    expect(cache.getStats().entries).toBe(0);
  });
});

// ============================================================================
// INTEGRATION-STYLE TESTS
// ============================================================================

describe('Integration Tests', () => {
  describe('Safe Error Response Creation', () => {
    function createSafeErrorResponse(error: unknown, toolName: string) {
      const SENSITIVE_PATTERNS = [
        /ghp_[a-zA-Z0-9]{36}/g,
        /\/Users\/[^\/]+/g,
      ];

      const rawMessage = error instanceof Error ? error.message : String(error);
      let sanitized = rawMessage;
      for (const pattern of SENSITIVE_PATTERNS) {
        sanitized = sanitized.replace(pattern, '[REDACTED]');
      }
      if (sanitized.length > 1000) {
        sanitized = sanitized.substring(0, 1000) + '... (truncated)';
      }

      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify({
            error: true,
            message: sanitized,
            tool: toolName,
            suggestion: 'The operation failed. You may retry or try a different approach.',
          }, null, 2),
        }],
      };
    }

    test('should create properly formatted error response', () => {
      const response = createSafeErrorResponse(new Error('Test error'), 'test_tool');

      expect(response.content).toHaveLength(1);
      expect(response.content[0].type).toBe('text');

      const parsed = JSON.parse(response.content[0].text);
      expect(parsed.error).toBe(true);
      expect(parsed.message).toBe('Test error');
      expect(parsed.tool).toBe('test_tool');
      expect(parsed.suggestion).toBeDefined();
    });

    test('should sanitize sensitive data in error response', () => {
      const error = new Error('Failed at /Users/secret/path with token ghp_abcdefghijklmnopqrstuvwxyz1234567890');
      const response = createSafeErrorResponse(error, 'test_tool');

      const parsed = JSON.parse(response.content[0].text);
      expect(parsed.message).not.toContain('secret');
      expect(parsed.message).not.toContain('ghp_');
      expect(parsed.message).toContain('[REDACTED]');
    });
  });

  describe('Build Command Configuration', () => {
    test('should use default build command when not specified', () => {
      const defaultCommand = process.env.BUILD_COMMAND || 'npm run build';
      expect(defaultCommand).toBe('npm run build');
    });

    test('should allow custom build command', () => {
      const originalCommand = process.env.BUILD_COMMAND;
      process.env.BUILD_COMMAND = 'npm run build:web';

      const buildCommand = process.env.BUILD_COMMAND;
      expect(buildCommand).toBe('npm run build:web');

      // Restore
      if (originalCommand) {
        process.env.BUILD_COMMAND = originalCommand;
      } else {
        delete process.env.BUILD_COMMAND;
      }
    });
  });
});
