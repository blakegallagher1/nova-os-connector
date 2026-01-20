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

if (!GITHUB_TOKEN) {
  console.error('ERROR: GITHUB_PERSONAL_ACCESS_TOKEN is required');
  process.exit(1);
}

// ============================================================================
// GITHUB API HELPER
// ============================================================================

async function githubAPI(endpoint: string, options: RequestInit = {}): Promise<any> {
  const url = endpoint.startsWith('http') ? endpoint : `https://api.github.com${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      'Authorization': `Bearer ${GITHUB_TOKEN}`,
      'Accept': 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`GitHub API error ${response.status}: ${error}`);
  }

  return response.json();
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
// MCP SERVER
// ============================================================================

const server = new McpServer({
  name: 'nova-os-github-connector',
  version: '4.0.0',
  description: 'Direct GitHub API connector for Nova OS (no child processes)',
});

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
    const cacheKey = cache.fileKey(owner, repo, filePath, branch);
    const cached = cache.get(cacheKey);
    if (cached) return { content: [{ type: 'text' as const, text: cached }], _cached: true };

    let url = `/repos/${owner}/${repo}/contents/${filePath}`;
    if (branch) url += `?ref=${branch}`;

    const data = await githubAPI(url);

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
      paths: z.array(z.string()).describe('Array of file paths'),
      branch: z.string().optional().describe('Branch name (optional)'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, paths, branch }) => {
    const results = await Promise.all(paths.map(async (filePath) => {
      try {
        const cacheKey = cache.fileKey(owner, repo, filePath, branch);
        const cached = cache.get(cacheKey);
        if (cached) return { path: filePath, content: cached, cached: true };

        let url = `/repos/${owner}/${repo}/contents/${filePath}`;
        if (branch) url += `?ref=${branch}`;

        const data = await githubAPI(url);
        const content = data.content ? Buffer.from(data.content, 'base64').toString('utf-8') : JSON.stringify(data);
        cache.set(cacheKey, content);
        return { path: filePath, content, cached: false };
      } catch (error: any) {
        return { path: filePath, content: null, error: error.message };
      }
    }));

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          repository: `${owner}/${repo}`,
          branch: branch || 'default',
          files_requested: paths.length,
          files_read: results.filter(r => r.content !== null).length,
          files: results,
        }, null, 2),
      }],
    };
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
    },
    annotations: { destructiveHint: true },
  },
  async ({ owner, repo, path: filePath, content, message, branch, sha }) => {
    // If no SHA provided and file exists, get it
    let fileSha = sha;
    if (!fileSha) {
      try {
        let url = `/repos/${owner}/${repo}/contents/${filePath}`;
        if (branch) url += `?ref=${branch}`;
        const existing = await githubAPI(url);
        fileSha = existing.sha;
      } catch {
        // File doesn't exist, that's fine for creation
      }
    }

    const body: any = {
      message,
      content: Buffer.from(content).toString('base64'),
    };
    if (branch) body.branch = branch;
    if (fileSha) body.sha = fileSha;

    const result = await githubAPI(`/repos/${owner}/${repo}/contents/${filePath}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    });

    cache.invalidateRepo(owner, repo);

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          success: true,
          commit: result.commit.sha.slice(0, 7),
          message: result.commit.message,
          path: result.content.path,
        }, null, 2),
      }],
    };
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
      files: z.array(z.object({
        path: z.string().describe('File path'),
        content: z.string().describe('File content'),
      })).describe('Array of files to commit'),
    },
    annotations: { destructiveHint: true },
  },
  async ({ owner, repo, branch, message, files }) => {
    // Get the reference for the branch
    const ref = await githubAPI(`/repos/${owner}/${repo}/git/refs/heads/${branch}`);
    const latestCommitSha = ref.object.sha;

    // Get the tree of the latest commit
    const latestCommit = await githubAPI(`/repos/${owner}/${repo}/git/commits/${latestCommitSha}`);
    const baseTreeSha = latestCommit.tree.sha;

    // Create blobs for each file
    const treeItems = await Promise.all(files.map(async (file) => {
      const blob = await githubAPI(`/repos/${owner}/${repo}/git/blobs`, {
        method: 'POST',
        body: JSON.stringify({
          content: file.content,
          encoding: 'utf-8',
        }),
      });
      return {
        path: file.path,
        mode: '100644' as const,
        type: 'blob' as const,
        sha: blob.sha,
      };
    }));

    // Create a new tree
    const newTree = await githubAPI(`/repos/${owner}/${repo}/git/trees`, {
      method: 'POST',
      body: JSON.stringify({
        base_tree: baseTreeSha,
        tree: treeItems,
      }),
    });

    // Create a new commit
    const newCommit = await githubAPI(`/repos/${owner}/${repo}/git/commits`, {
      method: 'POST',
      body: JSON.stringify({
        message,
        tree: newTree.sha,
        parents: [latestCommitSha],
      }),
    });

    // Update the reference
    await githubAPI(`/repos/${owner}/${repo}/git/refs/heads/${branch}`, {
      method: 'PATCH',
      body: JSON.stringify({ sha: newCommit.sha }),
    });

    cache.invalidateRepo(owner, repo);

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          success: true,
          commit: newCommit.sha.slice(0, 7),
          message: newCommit.message,
          files_committed: files.length,
          branch,
        }, null, 2),
      }],
    };
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
    description: 'Clone a repository and run the build to verify it compiles. [Read-Only]',
    inputSchema: {
      owner: z.string().describe('Repository owner'),
      repo: z.string().describe('Repository name'),
      branch: z.string().optional().describe('Branch name'),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ owner, repo, branch }) => {
    const branchName = branch || 'main';
    const repoUrl = `https://github.com/${owner}/${repo}.git`;
    const tempDir = path.join(os.tmpdir(), `validate-build-${Date.now()}`);

    let success = false;
    let errorSummary = '';
    let buildOutput = '';

    try {
      await fs.mkdir(tempDir, { recursive: true });

      // Clone
      await execAsync(`git clone --depth 1 --branch ${branchName} ${repoUrl} .`, { cwd: tempDir, timeout: 120000 });

      // Install
      await execAsync('npm ci --prefer-offline 2>&1 || npm install 2>&1', { cwd: tempDir, timeout: 300000 });

      // Build
      try {
        const result = await execAsync('npm run build:web 2>&1', { cwd: tempDir, timeout: 300000, maxBuffer: 10 * 1024 * 1024 });
        buildOutput = result.stdout + result.stderr;
        success = true;
      } catch (error: any) {
        buildOutput = error.stdout + error.stderr;
        const errorLines = buildOutput.split('\n').filter((line: string) =>
          line.includes('Error:') || line.includes('error TS') || line.includes('Module not found')
        );
        errorSummary = errorLines.slice(0, 20).join('\n');
      }
    } catch (error: any) {
      errorSummary = error.message;
    } finally {
      await fs.rm(tempDir, { recursive: true, force: true }).catch(() => {});
    }

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          repository: `${owner}/${repo}`,
          branch: branchName,
          success,
          summary: success ? '✅ Build successful!' : '❌ Build failed',
          error_summary: errorSummary || null,
          build_output: buildOutput.slice(-4000),
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
  res.json({
    status: 'ok',
    tools: 13 + (VERCEL_TOKEN ? 2 : 0),
    cache: cache.getStats(),
    vercelEnabled: !!VERCEL_TOKEN,
    version: '4.0.0',
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
    res.status(500).json({
      jsonrpc: '2.0',
      error: { code: -32603, message: error instanceof Error ? error.message : 'Error' },
      id: null,
    });
  }
});

app.options('/mcp', (_req, res) => res.status(204).send());

// ============================================================================
// STARTUP
// ============================================================================

console.log('========================================');
console.log('  Nova OS GitHub Connector v4');
console.log('  (Direct API - No Child Processes)');
console.log('========================================');
console.log('');
console.log(`Tools: 13 GitHub + ${VERCEL_TOKEN ? '2 Vercel + ' : ''}3 utility = ${13 + (VERCEL_TOKEN ? 2 : 0) + 3} total`);
console.log('');

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
  console.log('');
  console.log('URL: https://nova-os-connector.onrender.com/mcp');
  console.log('');
});
