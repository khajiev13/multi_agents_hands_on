#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

exec "$PROJECT_ROOT/.venv/bin/mcp-neo4j-cypher" "$@" 2>>"$LOG_DIR/mcp-neo4j-cypher.log"
