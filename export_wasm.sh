#!/usr/bin/env bash
# Build a static WASM bundle of the app for previewing locally.
#
# The canonical deployment is handled by GitHub Actions. This script writes to
# build/, which is intentionally ignored by git.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export UV_LINK_MODE=copy

OUT_DIR="${1:-build}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "Exporting WASM bundle to $OUT_DIR/index.html ..."
uv run python -m marimo export html-wasm app.py -o "$OUT_DIR" --mode run -f

echo
echo "Done. To preview locally:"
echo "  cd $OUT_DIR && python -m http.server 8000"
echo "  then open http://localhost:8000/"
