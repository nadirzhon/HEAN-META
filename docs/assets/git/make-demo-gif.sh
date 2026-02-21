#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ASSET_DIR="$ROOT_DIR/docs/assets/git"
CAST_FILE="$ASSET_DIR/workflow-demo.cast"
GIF_FILE="$ASSET_DIR/workflow-demo.gif"

if ! command -v asciinema >/dev/null 2>&1; then
  echo "Error: asciinema is not installed."
  echo "Install: brew install asciinema"
  exit 1
fi

if ! command -v agg >/dev/null 2>&1; then
  echo "Error: agg is not installed."
  echo "Install: brew install charmbracelet/tap/agg"
  exit 1
fi

echo "Recording terminal demo to: $CAST_FILE"
echo "Tip: run a short flow (git sync, git sw -c ..., git add -p, git commit, git push)"
echo "Press Ctrl+D when finished."
asciinema rec "$CAST_FILE"

echo "Rendering GIF to: $GIF_FILE"
agg "$CAST_FILE" "$GIF_FILE"
echo "Done."

