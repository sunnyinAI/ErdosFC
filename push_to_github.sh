#!/usr/bin/env bash
#
# Push Erdos Federated Computing to GitHub as the repo "ErdosFC".
#
# Usage:
#   cd ~/Desktop/ErdosComputing
#   bash push_to_github.sh
#
# Requires either the GitHub CLI (`gh`) authenticated, OR that you create an
# empty repo named ErdosFC on github.com first (see the fallback message).

set -euo pipefail

REPO_NAME="ErdosFC"
GH_USER="sunnyinAI"
VISIBILITY="public"   # change to "private" if you prefer
COMMIT_MSG="Erdos Federated Computing 0.1.0"

# Always operate in the folder this script lives in.
cd "$(dirname "$0")"

# 1) Initialise the repository (idempotent).
if [ ! -d .git ]; then
  git init
fi

git add .
git commit -m "$COMMIT_MSG" || echo "(nothing new to commit)"
git branch -M main

# 2) Create the GitHub repo and push.
if command -v gh >/dev/null 2>&1; then
  echo ">> Using GitHub CLI to create $GH_USER/$REPO_NAME and push..."
  gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push
  echo ">> Done: https://github.com/$GH_USER/$REPO_NAME"
else
  cat <<EOF

GitHub CLI (gh) was not found, so finish the last step manually:

  1. Create an EMPTY repo named "$REPO_NAME" at https://github.com/new
     (do NOT add a README, .gitignore, or license — this repo already has them).

  2. Then run:

     git remote add origin https://github.com/$GH_USER/$REPO_NAME.git
     git push -u origin main

EOF
fi
