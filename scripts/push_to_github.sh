#!/usr/bin/env bash
# First upload of this folder to a new, empty GitHub repository.
# Usage: scripts/push_to_github.sh https://github.com/<user>/<repo>.git
set -euo pipefail
url=${1:?"give the repository URL, e.g. https://github.com/<user>/mars-pinn-gapfill.git"}
cd "$(dirname "$0")/.."
git init -b main
git add .
git commit -m "Martian Profile Gap Lab: pipeline, simulator, paper"
git remote add origin "$url"
git push -u origin main
echo "Now enable Pages: Settings -> Pages -> Source: GitHub Actions"
