#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-https://nova-os-connector.onrender.com}"
PROJECT_ID="${2:-nova-os}"

echo "Health check: ${BASE_URL}/health"
health="$(curl -sS "${BASE_URL}/health")"
if ! echo "$health" | grep -q '"status":"ok"'; then
  echo "Health check failed"
  echo "$health"
  exit 1
fi

echo "MCP Vercel check: list_vercel_deployments (${PROJECT_ID})"
resp="$(curl -sS -X POST "${BASE_URL}/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_vercel_deployments","arguments":{"projectId":"'"${PROJECT_ID}"'","limit":1}}}')"

if echo "$resp" | grep -q '"invalidToken":true'; then
  echo "Vercel token invalid"
  echo "$resp"
  exit 1
fi
if ! echo "$resp" | grep -q '"success":true'; then
  echo "Vercel MCP check failed"
  echo "$resp"
  exit 1
fi

echo "OK: health + Vercel MCP check passed"
