#!/bin/bash
# Nova OS GitHub Connector - Startup Script
# This script starts the GitHub MCP server with HTTP/SSE transport

set -e

# Check for GitHub token
if [ -z "$GITHUB_PERSONAL_ACCESS_TOKEN" ]; then
    if [ -f .env ]; then
        export $(cat .env | xargs)
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

echo "Starting Nova OS GitHub Connector on port 8000..."
echo "MCP endpoint will be available at: http://localhost:8000/sse"
echo ""
echo "Next steps:"
echo "  1. In a new terminal, run: ngrok http 8000"
echo "  2. Copy the https://...ngrok-free.app URL"
echo "  3. Add to ChatGPT Connectors with URL: https://...ngrok-free.app/sse"
echo ""

# Run the server using supergateway to bridge stdio to HTTP/SSE
npx -y supergateway --stdio "npx -y @modelcontextprotocol/server-github" --port 8000
