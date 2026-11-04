#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${TMUX:-}" ]]; then echo 'Launch from native tmux so the GPU job survives remote disconnection.' >&2; exit 2; fi
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
unset DISPLAY
python_bin=${SAFETYCUBES_PYTHON:-.venv/bin/python}
if [[ ! -x "$python_bin" ]]; then python_bin="$HOME/Documents/LeRobot-SafetyCubes/.venv/bin/python"; fi
stage=${1:-run}
output=${2:-artifacts/full_experiment}
mkdir -p "$output/logs"
exec 9> "$output/.full-experiment.lock"
flock -n 9 || { echo 'A full experiment process already owns this output directory.' >&2; exit 3; }
"$python_bin" -u -m recovery_experiment.full "$stage" --spec configs/full_experiment.json --output "$output" 2>&1 | tee "$output/logs/${stage}_$(date -u +%Y%m%dT%H%M%SZ).log"
