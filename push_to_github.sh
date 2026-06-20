#!/usr/bin/env bash
#
# Publish this repository (Erdos FC + Erdos Platform) to GitHub as
# "ErdosComputing", push main, and point you at the GitHub Pages setup so the
# Erdos Platform marketing site goes live.
#
# Usage:
#   cd ~/Desktop/ErdosComputing
#   bash push_to_github.sh
#
# Requires either the GitHub CLI (`gh`) authenticated, OR that you create an
# empty repo named ErdosComputing on github.com first (see the fallback below).

set -euo pipefail

REPO_NAME="ErdosComputing"
GH_USER="sunnyinAI"
VISIBILITY="public"   # change to "private" if you prefer (Pages then needs a paid plan)

# Always operate in the folder this script lives in.
cd "$(dirname "$0")"

# 1) Make sure we have a repo and are on main (commits already exist).
[ -d .git ] || git init
git branch -M main

# 2) Create the GitHub repo and push main.
if command -v gh >/dev/null 2>&1; then
  echo ">> Using GitHub CLI to create $GH_USER/$REPO_NAME and push main..."
  gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push
  echo ">> Pushed: https://github.com/$GH_USER/$REPO_NAME"
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

cat <<EOF

>> Enable the Erdos Platform website (GitHub Pages):
   Repo -> Settings -> Pages -> Build and deployment -> Source: "GitHub Actions".
   The bundled workflow (.github/workflows/pages.yml) then deploys platform/web
   on every push to main. Your site will be served at:

     https://$GH_USER.github.io/$REPO_NAME/

EOF
