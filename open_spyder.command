#!/bin/zsh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

open -na "Spyder" --args \
    --workdir "$PROJECT_DIR" \
    "$PROJECT_DIR/run_fast_mm.py"
