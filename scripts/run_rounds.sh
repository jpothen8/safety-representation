#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${TMUX:-}" ]]; then echo 'Start this from tmux so remote disconnects do not own the job.' >&2; exit 2; fi
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
unset DISPLAY
stage=${1:-run}
config=${2:-configs/round1_cpu.json}
output=${3:-artifacts/round1}
python_bin=${SAFETYCUBES_PYTHON:-.venv/bin/python}
if [[ ! -x "$python_bin" ]]; then
    python_bin=/home/john/Documents/LeRobot-SafetyCubes/.venv/bin/python
fi
mkdir -p "$output/logs"
exec 9> "$output/.rounds-${stage}.lock"
flock -n 9 || { echo 'This stage is already running for the output directory.' >&2; exit 3; }
"$python_bin" -m recovery_experiment.rounds "$stage" --config "$config" --output "$output" 2>&1 | tee "$output/logs/${stage}_$(date -u +%Y%m%dT%H%M%SZ).log"
