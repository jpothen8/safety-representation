#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -d .publication/git ]]; then
    source_git=(git --git-dir="$PWD/.publication/git" --work-tree="$PWD")
else
    source_git=(git)
fi
"${source_git[@]}" add .gitignore LICENSE NOTICE README.md LESSONS_REVIEW.md RESULTS.md FULL_EXPERIMENT.md requirements.txt docs configs recovery_experiment scripts sim tests
"${source_git[@]}" diff --cached --check
if ! "${source_git[@]}" diff --cached --quiet; then
    "${source_git[@]}" commit -m 'Audit task port and prepare full three-seed recovery experiment'
fi
"${source_git[@]}" -c 'credential.helper=!gh auth git-credential' push origin main
