#!/usr/bin/env bash
# Start backend only (run in Terminal for full network access).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
./venv/bin/python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
