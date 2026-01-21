# Nova OS GitHub Connector

Full "Cursor-like" management of the `nova-os` repository directly within ChatGPT Developer Mode.

## Quick Start

### 1. Create GitHub Personal Access Token

1. Go to [GitHub Settings > Tokens (classic)](https://github.com/settings/tokens)
2. Click **Generate new token** > **Generate new token (classic)**
3. Note: "Nova OS Connector"
4. Select scopes: `repo`, `user`
5. Click **Generate token** and copy it (starts with `ghp_...`)

### 2. Configure Environment

```bash
cd nova-os-connector
cp .env.example .env
# Edit .env and paste your token
```

Or export directly:
```bash
export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_token_here
```

### 3. Start the Server

**Recommended (v3 Wrapper with optimizations):**
```bash
./start-v3.sh
```

**Legacy (v2 mcp-proxy):**
```bash
./start-v2.sh
```

### 4. Connect to ChatGPT

1. Open [ChatGPT](https://chatgpt.com)
2. Go to **Settings** > **Apps/Data** > Enable **Developer Mode**
3. Go to **Connectors** tab > **Create New**
4. Fill in:
   - **Name**: `Nova OS Builder`
   - **Description**: `Full access to manage the nova-os repository`
   - **URL**: `https://nova-connector.gallagherpropco.com/mcp`
   - **Authentication**: None
5. Click **Create**

### 5. Test It

Start a new chat and try:
- "Read README.md, CLAUDE.md, and AGENTS.md from blakegallagher1/nova-os using batch_read_files"
- "List the files in the root of blakegallagher1/nova-os"
- "Create a file docs/mcp-test.md with the text 'Connection successful'"

## Version Comparison

| Feature | v2 (mcp-proxy) | v3 (Wrapper) |
|---------|----------------|--------------|
| Tools | 26 | 27 |
| Tool Annotations | No | Yes |
| batch_read_files | No | Yes |
| Read-only hints | No | 15 tools |
| Confirmation prompts | All tools | Write tools only |

## v3 Wrapper Features

### Tool Annotations
Tools are marked with `readOnlyHint: true` for read operations, allowing ChatGPT to execute them without confirmation prompts:
- `get_file_contents`, `search_code`, `search_issues`, `search_repositories`, `search_users`
- `get_issue`, `get_pull_request`, `get_pull_request_*`, `list_*`

### batch_read_files
Read multiple files in a single call (much faster than individual `get_file_contents` calls):
```
batch_read_files({
  owner: "blakegallagher1",
  repo: "nova-os",
  paths: ["README.md", "CLAUDE.md", "AGENTS.md", "PLAN.md"]
})
```

Returns all file contents in one response, reducing round-trips from 4 to 1.

## Available Tools (27 total)

### Read-Only (no confirmation)
| Tool | Description |
|------|-------------|
| `get_file_contents` | Read file contents |
| `batch_read_files` | Read multiple files at once |
| `search_code` | Search code in repos |
| `search_issues` | Search issues |
| `search_repositories` | Search repos |
| `search_users` | Search users |
| `get_issue` | Get issue details |
| `get_pull_request` | Get PR details |
| `get_pull_request_files` | Get PR file list |
| `get_pull_request_status` | Get PR status |
| `get_pull_request_comments` | Get PR comments |
| `get_pull_request_reviews` | Get PR reviews |
| `list_commits` | List commits |
| `list_issues` | List issues |
| `list_pull_requests` | List PRs |

### Write Operations (requires confirmation)
| Tool | Description |
|------|-------------|
| `create_or_update_file` | Create/edit files |
| `push_files` | Commit multiple files |
| `create_branch` | Create branch |
| `create_issue` | Create issue |
| `create_pull_request` | Create PR |
| `create_pull_request_review` | Review PR |
| `create_repository` | Create repo |
| `fork_repository` | Fork repo |
| `merge_pull_request` | Merge PR |
| `update_issue` | Update issue |
| `update_pull_request_branch` | Update PR branch |
| `add_issue_comment` | Comment on issue |

## Architecture

```
v3 (Wrapper):
ChatGPT → Cloudflare Tunnel → localhost:8000 → Wrapper Server → GitHub MCP (stdio)
                                    ↓
                              Adds annotations
                              + batch_read_files

v2 (mcp-proxy):
ChatGPT → Cloudflare Tunnel → localhost:8000 → mcp-proxy → GitHub MCP (stdio)
```

## Troubleshooting

- **"Error connecting to server"**: Ensure server is running and Cloudflare tunnel is active
- **"Authentication failed"**: Check `GITHUB_PERSONAL_ACCESS_TOKEN` is set
- **"Tool not found"**: Refresh the connector in ChatGPT settings
- **Read operations slow**: Switch to v3 wrapper with tool annotations

## Validate Build via GitHub Actions (Recommended)

The `validate_build` tool now supports offloading to GitHub Actions to avoid memory spikes on Render.

1) Copy this workflow into each target repo you want to validate:
- `nova-os-connector/.github/workflows/nova-validate-build.yml`

2) Set these env vars on the connector:
```bash
VALIDATE_BUILD_MODE=actions
VALIDATE_BUILD_WORKFLOW=nova-validate-build.yml
VALIDATE_BUILD_WORKFLOW_REF=main
```

3) Ensure the GitHub token used by the connector has `workflow` permission (or `actions:write` for fine‑grained PATs).

## Repository Allowlist (Required)

This connector is locked to a specific repo by default:

```bash
ALLOWED_REPOS=blakegallagher1/nova-os
```

Provide a comma‑separated list to expand access.

## Performance Tuning Defaults (Aggressive)

These defaults are tuned for heavy usage. Override via env vars as needed:

```bash
MAX_FILE_BYTES=4194304            # 4MB
CACHE_MAX_ENTRY_BYTES=1048576     # 1MB
CACHE_MAX_TOTAL_BYTES=268435456   # 256MB
CACHE_TTL_FILE_MS=300000          # 5 min
CACHE_TTL_BATCH_MS=300000         # 5 min
```

Tool rate limits (override via `TOOL_RATE_LIMITS` JSON):
```bash
TOOL_RATE_LIMITS='{
  "batch_read_files": {"limit": 30, "windowMs": 60000},
  "push_files": {"limit": 20, "windowMs": 60000},
  "apply_patch": {"limit": 50, "windowMs": 60000},
  "validate_build": {"limit": 10, "windowMs": 600000}
}'
```
