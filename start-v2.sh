#!/bin/bash
# Nova OS GitHub Connector v2 - Using mcp-proxy for better stability
# This script starts the GitHub MCP server with HTTP/SSE transport

set -e

cd "$(dirname "$0")"

# Use Node 22 for mcp-proxy compatibility
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 22 > /dev/null 2>&1 || echo "Note: Using default Node version"

# Check for GitHub token
if [ -z "$GITHUB_PERSONAL_ACCESS_TOKEN" ]; then
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi
fi

if [ -z "$GITHUB_PERSONAL_ACCESS_TOKEN" ]; then
    echo "ERROR: GITHUB_PERSONAL_ACCESS_TOKEN is not set."
    echo ""
    echo "Set it by either:"
    echo "  1. export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_token_here"
    echo "  2. Create a .env file with: GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_token_here"
    echo ""
    echo "Generate a token at: https://github.com/settings/tokens"
    echo "Required scopes: repo, user"
    exit 1
fi

PORT=${PORT:-8000}

echo "========================================"
echo "  Nova OS GitHub Connector v2"
echo "========================================"
echo ""
echo "Starting on port $PORT with mcp-proxy..."
echo ""
echo "Endpoints available:"
echo "  - SSE:    http://localhost:$PORT/sse"
echo "  - Stream: http://localhost:$PORT/mcp"
echo ""
echo "Cloudflare Tunnel URL:"
echo "  https://nova-connector.gallagherpropco.com/sse"
echo ""
echo "Timeouts configured:"
echo "  - Connection: 120 seconds"
echo "  - Request:    5 minutes"
echo ""

# Use mcp-proxy for better stability
# --connectionTimeout: 120s for initial connection
# --requestTimeout: 300s (5 min) for tool operations
# --debug: Enable debug logging
npx -y mcp-proxy \
    --port $PORT \
    --connectionTimeout 120000 \
    --requestTimeout 300000 \
    --debug \
    -- npx -y @modelcontextprotocol/server-github
