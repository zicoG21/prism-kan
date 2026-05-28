#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="results/formula_fidelity_minisuite/overnight_tmlr_${STAMP}"
mkdir -p "$RUN_ROOT/logs"

PID_FILE="$RUN_ROOT/overnight.pid"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Existing overnight job is still running: PID $(cat "$PID_FILE")"
  echo "Run root: $RUN_ROOT"
  exit 1
fi

nohup timeout 10h bash scripts/run_tmlr_minisuite_overnight.sh "$STAMP" \
  > "$RUN_ROOT/logs/launcher.log" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "Started TMLR overnight mini-suite"
echo "PID: $PID"
echo "Run root: $RUN_ROOT"
echo "Launcher log: $RUN_ROOT/logs/launcher.log"
echo "Main log: $RUN_ROOT/logs/overnight.log"
echo "Report target: $RUN_ROOT/overnight_report.md"
