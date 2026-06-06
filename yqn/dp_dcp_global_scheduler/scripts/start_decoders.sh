#!/usr/bin/env bash
# Launch 4 decode EngineCores (engine_index=1..4) on GPU 1..3 on the same
# NUMA node as the prefill rank.  Each starts with its own LMCache rpc
# port and connects to the scheduler over ZMQ.
#
# Usage:
#   ./start_decoders.sh           # foreground, joins all 4
#   ./start_decoders.sh --bg      # background, writes pid file

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PROJ_DIR/../.." && pwd)"
MODEL="${MODEL:-/data/yqn/Qwen1.5-MoE-A2.7B}"
VLLM_BIN="${VLLM_BIN:-$ROOT_DIR/.venv/bin/vllm}"
DECODE_GPUS="${DECODE_GPUS:-1,2,3}"
DECODE_PORTS="${DECODE_PORTS:-8201,8202,8203}"
DECODE_RPC_NAMES="${DECODE_RPC_NAMES:-consumer1,consumer2,consumer3}"

if [[ ! -x "$VLLM_BIN" ]]; then
    echo "Missing vLLM executable: $VLLM_BIN"
    exit 1
fi

export PYTHONHASHSEED=${VLLM_PYTHON_HASH_SEED:-123}
export LMCACHE_CONFIG_FILE="$PROJ_DIR/configs/lmcache_decode.yaml"
export LMCACHE_USE_EXPERIMENTAL=True
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export UCX_TLS=cuda_ipc,cuda_copy,tcp
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export GLOBAL_SCHED_PULL_ADDR="${GLOBAL_SCHED_PULL_ADDR:-tcp://127.0.0.1:5571}"
export GLOBAL_SCHED_PUB_ADDR="${GLOBAL_SCHED_PUB_ADDR:-tcp://127.0.0.1:5572}"
export DP_ATTN_WORLD_SIZE="${DP_ATTN_WORLD_SIZE:-4}"

declare -a DECODE_GPU_ARRAY=()
declare -a DECODE_PORT_ARRAY=()
declare -a DECODE_RPC_ARRAY=()

parse_csv_array() {
    local -n target_array="$1"
    local raw="$2"
    IFS=',' read -r -a target_array <<< "$raw"
}

validate_arrays() {
    if [[ "${#DECODE_GPU_ARRAY[@]}" -ne 3 ]]; then
        echo "DECODE_GPUS must contain exactly 3 entries."
        exit 1
    fi
    if [[ "${#DECODE_PORT_ARRAY[@]}" -ne 3 ]]; then
        echo "DECODE_PORTS must contain exactly 3 entries."
        exit 1
    fi
    if [[ "${#DECODE_RPC_ARRAY[@]}" -ne 3 ]]; then
        echo "DECODE_RPC_NAMES must contain exactly 3 entries."
        exit 1
    fi
}

start_one() {
    local engine_index=$1
    local gpu=$2
    local port=$3
    local rpc=$4

    ENGINE_INDEX=$engine_index ENGINE_ROLE=decode \
    CUDA_VISIBLE_DEVICES=$gpu \
        "$VLLM_BIN" serve "$MODEL" \
        --port "$port" \
        --host 0.0.0.0 \
        --enforce-eager \
        --data-parallel-size 1 \
        --trust-remote-code \
        --kv-transfer-config \
        "{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"discard_partial_chunks\":false,\"lmcache_rpc_port\":\"$rpc\"}}"
}

parse_csv_array DECODE_GPU_ARRAY "$DECODE_GPUS"
parse_csv_array DECODE_PORT_ARRAY "$DECODE_PORTS"
parse_csv_array DECODE_RPC_ARRAY "$DECODE_RPC_NAMES"
validate_arrays

for idx in "${!DECODE_GPU_ARRAY[@]}"; do
    start_one \
        "$((idx + 1))" \
        "${DECODE_GPU_ARRAY[$idx]}" \
        "${DECODE_PORT_ARRAY[$idx]}" \
        "${DECODE_RPC_ARRAY[$idx]}" &
done
wait
