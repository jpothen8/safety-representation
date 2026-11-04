#!/usr/bin/env bash
# Invoke from the native server shell. This owns a detached tmux job.
set -euo pipefail
cd "$(dirname "$0")/.."
session=${1:-cubes-full}
output=${2:-artifacts/full_experiment}
if [[ ! "$session" =~ ^[a-zA-Z0-9_-]+$ ]]; then echo 'Invalid tmux session name' >&2; exit 2; fi
if [[ -f "$output/completed.json" ]]; then echo "Experiment already complete: $output/summary.md"; exit 0; fi
if tmux has-session -t "$session" 2>/dev/null; then
    echo "Session already exists: $session. Inspect it before launching another experiment."
    exit 0
fi
tmux new-session -d -s "$session" -c "$PWD" bash scripts/run_full.sh run "$output"
echo "Started tmux session: $session"
echo "Attach with: tmux attach -t $session"
echo "Logs: $PWD/$output/logs/"
echo 'Preflight verifies native GPU and online WandB before model training.'
