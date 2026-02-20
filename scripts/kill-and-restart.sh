#!/usr/bin/env bash
# Kill anything on backend/frontend ports, then start backend and frontend.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Killing processes on ports 8000, 8001, 8002, 3000..."
for port in 8000 8001 8002 3000; do
  # macOS-friendly: list TCP listeners on this port, print only PIDs.
  pids=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null) || true
  if [ -n "$pids" ]; then
    kill -9 $pids 2>/dev/null || true
    echo "  Killed $pids on $port"
  fi
done
sleep 1
echo "Ports cleared."

echo ""
echo "Starting backend on http://localhost:8000 ..."
cd "$ROOT/backend"
./venv/bin/python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
sleep 2

echo "Starting frontend on http://localhost:3000 ..."
cd "$ROOT/frontend"
HOST=127.0.0.1 npm start &
FRONTEND_PID=$!

cleanup() {
  # Best-effort cleanup on Ctrl+C / script exit.
  kill "$FRONTEND_PID" 2>/dev/null || true
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "Backend PID: $BACKEND_PID (http://localhost:8000)"
echo "Frontend PID: $FRONTEND_PID (http://localhost:3000)"
echo "Press Ctrl+C to stop both."
wait
