#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$ROOT_DIR/app.log"
PID_FILE="$ROOT_DIR/app.pid"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Python was not found in PATH." >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Policy GPT is already running with PID $EXISTING_PID"
    echo "Log: $LOG_FILE"
    exit 0
  fi
fi

cd "$ROOT_DIR"
touch "$LOG_FILE"
nohup "$PYTHON_BIN" app.py >>"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"

echo "Policy GPT started in background."
echo "PID: $PID"
echo "Log: $LOG_FILE"
