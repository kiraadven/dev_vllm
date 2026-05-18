#!/bin/bash
# Launch 4 decode EngineCores (engine_index=1..4) on GPU 1..3 on the same
# NUMA node as the prefill rank.  Each starts with its own LMCache rpc
# port and connects to the scheduler over ZMQ.
#
# Usage:
#   ./start_decoders.sh           # foreground, joins all 4
#   ./start_decoders.sh --bg      # background, writes pid file

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen2.5-MoE-A14B-Instruct}"

export PYTHONHASHSEED=${VLLM_PYTHON_HASH_SEED:-123}
export LMCACHE_CONFIG_FILE="$PROJ_DIR/configs/lmcache_decode.yaml"
export LMCACHE_USE_EXPERIMENTAL=True
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export UCX_TLS=cuda_ipc,cuda_copy,tcp
export GLOBAL_SCHED_PULL_ADDR="${GLOBAL_SCHED_PULL_ADDR:-tcp://127.0.0.1:5571}"
export GLOBAL_SCHED_PUB_ADDR="${GLOBAL_SCHED_PUB_ADDR:-tcp://127.0.0.1:5572}"
export DP_ATTN_WORLD_SIZE="${DP_ATTN_WORLD_SIZE:-4}"

VLLM=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/qining/dev_vllm/.venv/bin/vllm

start_one() {
    local engine_index=$1
    local gpu=$2
    local port=$3
    local rpc=$4

    ENGINE_INDEX=$engine_index ENGINE_ROLE=decode \
    CUDA_VISIBLE_DEVICES=$gpu \
        $VLLM serve "$MODEL" \
        --port "$port" \
        --enforce-eager \
        --data-parallel-size 1 \
        --kv-transfer-config \
        "{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"discard_partial_chunks\":false,\"lmcache_rpc_port\":\"$rpc\"}}"
}

if [[ "$1" == "--bg" ]]; then
    for spec in "1 1 8201 consumer1" "2 2 8202 consumer2" "3 3 8203 consumer3"; do
        # shellcheck disable=SC2086
        start_one $spec &
    done
    wait
else
    # Foreground: only run engine 1 in this shell; user runs the others in
    # separate shells. Keeps stdout readable for debugging.
    start_one 1 1 8201 consumer1
fi
