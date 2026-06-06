#!/usr/bin/env bash
# Launch a prefill EngineCore (engine_index=0) pointing at the scheduler.
#
# Hardware mapping (matches your nvidia-smi topo -m output):
#   Prefill on GPU 0 (NUMA 0), decode on GPU 1..3 (NUMA 0 NV4+PIX).
#   Keep DP=4 strictly within one NUMA half to avoid SYS cross-NUMA hops.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PROJ_DIR/../.." && pwd)"
MODEL="${MODEL:-/data/yqn/Qwen1.5-MoE-A2.7B}"
VLLM_BIN="${VLLM_BIN:-$ROOT_DIR/.venv/bin/vllm}"
PREFILL_GPU="${PREFILL_GPU:-0}"
PREFILL_PORT="${PREFILL_PORT:-8100}"

if [[ ! -x "$VLLM_BIN" ]]; then
    echo "Missing vLLM executable: $VLLM_BIN"
    exit 1
fi

export PYTHONHASHSEED=${VLLM_PYTHON_HASH_SEED:-123}
export LMCACHE_CONFIG_FILE="$PROJ_DIR/configs/lmcache_prefill.yaml"
export LMCACHE_USE_EXPERIMENTAL=True
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export UCX_TLS=cuda_ipc,cuda_copy,tcp
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Addresses where the GlobalScheduler is listening (matches
# scripts/start_scheduler.sh defaults).
export GLOBAL_SCHED_PULL_ADDR="${GLOBAL_SCHED_PULL_ADDR:-tcp://127.0.0.1:5571}"
export GLOBAL_SCHED_PUB_ADDR="${GLOBAL_SCHED_PUB_ADDR:-tcp://127.0.0.1:5572}"
export ENGINE_INDEX=0
export ENGINE_ROLE=prefill
# Opt-in to the cross-DP flash attention patch. Set to the number of DP
# ranks in this DP-DCP unit (4 = prefill + 3 decoders). If you only want
# Phase 1 (placement + handoff routing, no patched attention), unset this.
export DP_ATTN_WORLD_SIZE="${DP_ATTN_WORLD_SIZE:-4}"

CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
    "$VLLM_BIN" \
    serve "$MODEL" \
    --port "$PREFILL_PORT" \
    --host 0.0.0.0 \
    --enforce-eager \
    --data-parallel-size 1 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"producer1"}}'
