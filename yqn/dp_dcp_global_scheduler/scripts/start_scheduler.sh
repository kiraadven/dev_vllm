#!/usr/bin/env bash
# Start the standalone GlobalScheduler process.
# Front-ends and engines connect to the printed addresses.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PROJ_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing Python executable: $PYTHON_BIN"
    exit 1
fi

# 1 prefill rank (engine 0) + 3 decode ranks (engines 1..3) on a single
# NUMA half (GPU 0..3 are NV4+PIX; GPU 4..7 likewise on the other NUMA).
ENGINES="${ENGINES:-4}"
ROLES="${ROLES:-prefill,decode,decode,decode}"

cd "$PROJ_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
exec "$PYTHON_BIN" -m src.global_scheduler \
    --engines "$ENGINES" \
    --roles "$ROLES" \
    --front-addr "tcp://*:5570" \
    --back-pull-addr "tcp://*:5571" \
    --back-pub-addr "tcp://*:5572"
