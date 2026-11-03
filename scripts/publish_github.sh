#!/usr/bin/env bash
# Run from the native authenticated terminal; creates a NEW repository only.
set -euo pipefail
cd "$(dirname "$0")/.."
owner=jpothen8
name=${1:-safety-representation}
if [[ ! "$name" =~ ^[a-zA-Z0-9._-]+$ ]]; then echo 'Invalid repository name' >&2; exit 2; fi
login=$(gh api user --jq .login)
identifier=$(gh api user --jq .id)
if [[ "$login" != "$owner" ]]; then echo "Expected authenticated account $owner, got $login" >&2; exit 2; fi
# Do not overwrite an existing repository. gh repo create also fails if taken.
if gh repo view "$owner/$name" >/dev/null 2>&1; then echo "Repository already exists: $owner/$name; choose a new name" >&2; exit 2; fi
root=$PWD
metadata="$root/.publication/git"
mkdir -p "$root/.publication"
git --git-dir="$metadata" --work-tree="$root" init -b main
git --git-dir="$metadata" config user.name "${GIT_AUTHOR_NAME:-jpothen8}"
git --git-dir="$metadata" config user.email "${GIT_AUTHOR_EMAIL:-${identifier}+${owner}@users.noreply.github.com}"
# Explicit pathspec: no data, checkpoints, credentials, logs or videos.
git --git-dir="$metadata" --work-tree="$root" add .gitignore LICENSE NOTICE README.md LESSONS_REVIEW.md RESULTS.md requirements.txt docs configs recovery_experiment scripts sim tests
if ! git --git-dir="$metadata" --work-tree="$root" diff --cached --quiet; then
    git --git-dir="$metadata" --work-tree="$root" commit -m 'Add local cube task and imitation-only diffusion recovery rounds'
fi
gh repo create "$owner/$name" --public --description 'Imitation-only diffusion policy: full-task expert training versus targeted near-collision expert recovery rounds'
git --git-dir="$metadata" remote add origin "https://github.com/$owner/$name.git"
git -c 'credential.helper=!gh auth git-credential' --git-dir="$metadata" --work-tree="$root" push -u origin main
echo "https://github.com/$owner/$name"
