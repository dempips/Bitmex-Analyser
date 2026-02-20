#!/usr/bin/env bash
# Start frontend only (run in Terminal for full network access).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/frontend"
HOST=127.0.0.1 npm start
