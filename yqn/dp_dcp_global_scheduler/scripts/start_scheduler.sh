#!/bin/bash
# Start the standalone GlobalScheduler process.
# Front-ends and engines connect to the printed addresses.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PY=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/qining/dev_vllm/.venv/bin/python

# 1 prefill rank (engine 0) + 4 decode ranks (engines 1..4) on a single
# NUMA half (GPU 0..3 are NV4+PIX; GPU 4..7 likewise on the other NUMA).
ENGINES=5
ROLES="prefill,decode,decode,decode,decode"

cd "$PROJ_DIR"
exec $PY -m src.global_scheduler \
    --engines "$ENGINES" \
    --roles "$ROLES" \
    --front-addr "tcp://*:5570" \
    --back-pull-addr "tcp://*:5571" \
    --back-pub-addr "tcp://*:5572"
