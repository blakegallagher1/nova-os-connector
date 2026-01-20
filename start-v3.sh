#!/bin/bash
# Nova OS GitHub Connector v3 - Wrapper with tool annotations and batch operations
# This script starts the optimized GitHub MCP wrapper server

set -e

cd "$(dirname "$0")"

# Use Node 22 for ESM and modern features
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
export PORT

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo "========================================"
echo "  Nova OS GitHub Connector v3"
echo "  (Wrapper with Annotations)"
echo "========================================"
echo ""
echo "Features:"
echo "  - Tool annotations for faster read operations"
echo "  - batch_read_files for multi-file reads"
echo "  - 27 total tools (26 GitHub + 1 batch)"
echo ""
echo "Starting on port $PORT..."
echo ""

# Start the wrapper server
npm start
