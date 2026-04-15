#!/usr/bin/env bash
# Online disaggregated P/D benchmark on a single node with 4 GPUs:
# - 2 prefill instances + 2 decode instances (2 P/D pairs)
# - Round-robin dispatch by trace CSV arrive_time
# - 5 full measurement rounds (startup -> replay -> shutdown)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YQN_DIR="${ROOT_DIR}/yqn"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

MODEL_NAME="${HF_MODEL_NAME:-/data/yqn/Qwen1.5-MoE-A2.7B}"
TRACE_FILE="${TRACE_FILE:-${YQN_DIR}/traces/sharegpt_x/sharegpt_x_rate1p0.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${YQN_DIR}/stats/online_round_robin}"

RUNS="${RUNS:-5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-7200}"
VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
PROFILER_CONFIG_JSON="${PROFILER_CONFIG_JSON:-{\"profiler\":\"cuda\",\"delay_iterations\":3,\"max_iterations\":10}}"

PREFILL_GPU_LIST="${PREFILL_GPU_LIST:-0,2}"
DECODE_GPU_LIST="${DECODE_GPU_LIST:-1,3}"

IFS=',' read -r -a PREFILL_GPUS <<< "${PREFILL_GPU_LIST}"
IFS=',' read -r -a DECODE_GPUS <<< "${DECODE_GPU_LIST}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing python interpreter: ${PYTHON_BIN}"
    echo "Please create .venv first (uv venv --python 3.12) and install deps."
    exit 1
fi

if [[ ! -f "${TRACE_FILE}" ]]; then
    echo "Trace file not found: ${TRACE_FILE}"
    echo "Set TRACE_FILE=/abs/path/to/trace.csv"
    exit 1
fi

if [[ ${#PREFILL_GPUS[@]} -ne 2 || ${#DECODE_GPUS[@]} -ne 2 ]]; then
    echo "Need exactly 2 prefill GPUs and 2 decode GPUs."
    echo "Example: PREFILL_GPU_LIST=0,1 DECODE_GPU_LIST=2,3"
    exit 1
fi

if ! "${PYTHON_BIN}" -c "import aiohttp, quart" >/dev/null 2>&1; then
    echo "Missing aiohttp/quart in .venv."
    echo "Install with: /root/.local/bin/uv pip install aiohttp quart"
    exit 1
fi

# install quart first -- required for disagg prefill proxy serve
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi

PREFILL_PORTS=(8100 8101)
DECODE_PORTS=(8200 8201)
PROXY_PORTS=(8000 8001)
KV_PROXY_PORTS=(30001 30002)
PREFILL_KV_PORTS=(14579 14581)
DECODE_KV_PORTS=(14580 14582)

PIDS=()

record_pid() {
    PIDS+=("$1")
}

wait_for_server() {
    local port="$1"
    local retries=1200
    local i=0
    while [[ $i -lt $retries ]]; do
        if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    echo "Timeout waiting for server on port ${port}"
    return 1
}

stop_all_processes() {
    if [[ ${#PIDS[@]} -eq 0 ]]; then
        return
    fi

    local pid
    for ((idx=${#PIDS[@]}-1; idx>=0; idx--)); do
        pid="${PIDS[$idx]}"
        kill "${pid}" >/dev/null 2>&1 || true
    done
    sleep 2
    for ((idx=${#PIDS[@]}-1; idx>=0; idx--)); do
        pid="${PIDS[$idx]}"
        kill -9 "${pid}" >/dev/null 2>&1 || true
    done
    PIDS=()
}

cleanup() {
    stop_all_processes
}
trap cleanup EXIT INT TERM

start_pd_pair() {
    local idx="$1"
    local prefill_gpu="$2"
    local decode_gpu="$3"
    local prefill_port="${PREFILL_PORTS[$idx]}"
    local decode_port="${DECODE_PORTS[$idx]}"
    local kv_proxy_port="${KV_PROXY_PORTS[$idx]}"
    local prefill_kv_port="${PREFILL_KV_PORTS[$idx]}"
    local decode_kv_port="${DECODE_KV_PORTS[$idx]}"
    local proxy_port="${PROXY_PORTS[$idx]}"

    local prefill_kv_cfg
    prefill_kv_cfg=$(cat <<JSON
{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":"1e9","kv_port":"${prefill_kv_port}","kv_connector_extra_config":{"proxy_ip":"${VLLM_HOST_IP}","proxy_port":"${kv_proxy_port}","http_ip":"${VLLM_HOST_IP}","http_port":"${prefill_port}","send_type":"PUT_ASYNC"}}
JSON
)

    local decode_kv_cfg
    decode_kv_cfg=$(cat <<JSON
{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":"1e10","kv_port":"${decode_kv_port}","kv_connector_extra_config":{"proxy_ip":"${VLLM_HOST_IP}","proxy_port":"${kv_proxy_port}","http_ip":"${VLLM_HOST_IP}","http_port":"${decode_port}","send_type":"PUT_ASYNC"}}
JSON
)

    CUDA_VISIBLE_DEVICES="${prefill_gpu}" vllm serve "${MODEL_NAME}" \
        --host 0.0.0.0 \
        --port "${prefill_port}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --profiler-config "${PROFILER_CONFIG_JSON}" \
        --enable-layerwise-nvtx-tracing \
        --enable-logging-iteration-details \
        --kv-transfer-config "${prefill_kv_cfg}" &
    record_pid "$!"

    CUDA_VISIBLE_DEVICES="${decode_gpu}" vllm serve "${MODEL_NAME}" \
        --host 0.0.0.0 \
        --port "${decode_port}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --profiler-config "${PROFILER_CONFIG_JSON}" \
        --enable-layerwise-nvtx-tracing \
        --enable-logging-iteration-details \
        --kv-transfer-config "${decode_kv_cfg}" &
    record_pid "$!"

    wait_for_server "${prefill_port}"
    wait_for_server "${decode_port}"

    "${PYTHON_BIN}" "${ROOT_DIR}/benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py" \
        --port "${proxy_port}" \
        --prefill-url "http://${VLLM_HOST_IP}:${prefill_port}" \
        --decode-url "http://${VLLM_HOST_IP}:${decode_port}" \
        --kv-host "${VLLM_HOST_IP}" \
        --prefill-kv-port "${prefill_kv_port}" \
        --decode-kv-port "${decode_kv_port}" &
    record_pid "$!"
}

run_one_round() {
    local round_idx="$1"
    local round_dir="${OUTPUT_ROOT}/run_${round_idx}"
    mkdir -p "${round_dir}"

    start_pd_pair 0 "${PREFILL_GPUS[0]}" "${DECODE_GPUS[0]}"
    start_pd_pair 1 "${PREFILL_GPUS[1]}" "${DECODE_GPUS[1]}"

    # Give proxy servers a short stabilization window.
    sleep 2

    "${PYTHON_BIN}" "${YQN_DIR}/online_round_robin_dispatch.py" \
        --trace-file "${TRACE_FILE}" \
        --model "${MODEL_NAME}" \
        --proxy-urls "http://${VLLM_HOST_IP}:${PROXY_PORTS[0]},http://${VLLM_HOST_IP}:${PROXY_PORTS[1]}" \
        --timeout "${REQUEST_TIMEOUT_S}" \
        --output-dir "${round_dir}"
}

mkdir -p "${OUTPUT_ROOT}"

echo "Model: ${MODEL_NAME}"
echo "Trace: ${TRACE_FILE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Host: ${VLLM_HOST_IP}"
echo "Prefill GPUs: ${PREFILL_GPU_LIST}"
echo "Decode GPUs: ${DECODE_GPU_LIST}"
echo "Rounds: ${RUNS}"

for round_idx in $(seq 1 "${RUNS}"); do
    echo "===== Measurement round ${round_idx}/${RUNS} ====="
    if ! run_one_round "${round_idx}"; then
        echo "Round ${round_idx} failed."
        stop_all_processes
        exit 1
    fi
    stop_all_processes
    sleep 2
done

echo "All ${RUNS} rounds finished successfully."
