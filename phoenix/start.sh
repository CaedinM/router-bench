#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PHOENIX_WORKING_DIR="$ROOT/phoenix/storage"
export PHOENIX_HOST=127.0.0.1
mkdir -p "$PHOENIX_WORKING_DIR"
exec "$ROOT/.venv/bin/phoenix" serve
