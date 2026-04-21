#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
VLLM_BIN="${ROOT_DIR}/.venv/bin/vllm"
PROXY_SCRIPT="${SCRIPT_DIR}/disagg_proxy_p2p_nccl_2p2d.py"

MODEL="${MODEL:-/data/yqn/Qwen1.5-MoE-A2.7B}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"

PROXY_DISCOVERY_PORT="30001"
PROXY_HTTP_PORT="10001"

PREFILL_GPUS="${PREFILL_GPUS:-0,2}"
DECODE_GPUS="${DECODE_GPUS:-1,3}"
PREFILL_PORTS="${PREFILL_PORTS:-20003,20013}"
DECODE_PORTS="${DECODE_PORTS:-20005,20015}"
PREFILL_KV_PORTS="${PREFILL_KV_PORTS:-21001,21011}"
DECODE_KV_PORTS="${DECODE_KV_PORTS:-22001,22011}"

RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
BENCH_BACKEND="${BENCH_BACKEND:-openai-chat}"
BENCH_ENDPOINT="${BENCH_ENDPOINT:-/v1/chat/completions}"
SHAREGPT_DATASET_PATH="${SHAREGPT_DATASET_PATH:-/data/yqn/ShareGPT-X/ChatGPT-Simple.jsonl}"
SHAREGPT_OUTPUT_LEN="${SHAREGPT_OUTPUT_LEN:-256}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-1000}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-4}"
BENCH_BURSTINESS="${BENCH_BURSTINESS:-1.0}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-6}"
BENCH_SEED="${BENCH_SEED:-$(date +%s)}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs_2p2d}"

export NO_PROXY="127.0.0.1,localhost,0.0.0.0,${NO_PROXY:-}"
export no_proxy="${NO_PROXY}"
export DISAGG_MODEL="${MODEL}"

declare -a PIDS=()
declare -a PREFILL_GPU_ARRAY=()
declare -a DECODE_GPU_ARRAY=()
declare -a PREFILL_PORT_ARRAY=()
declare -a DECODE_PORT_ARRAY=()
declare -a PREFILL_KV_PORT_ARRAY=()
declare -a DECODE_KV_PORT_ARRAY=()

parse_csv_array() {
    local -n target_array="$1"
    local raw="$2"
    IFS=',' read -r -a target_array <<< "$raw"
}

launch_in_new_session() {
    local log_file="$1"
    shift
    setsid "$@" >"${log_file}" 2>&1 &
    PIDS+=("$!")
}

check_required_files() {
    local missing=0
    for path in "${PYTHON_BIN}" "${VLLM_BIN}" "${PROXY_SCRIPT}"; do
        if [[ ! -e "${path}" ]]; then
            echo "Missing required path: ${path}"
            missing=1
        fi
    done
    if [[ "${missing}" -ne 0 ]]; then
        exit 1
    fi
}

ensure_python_module_installed() {
    local module_name="$1"
    if ! "${PYTHON_BIN}" -c "import ${module_name}" >/dev/null 2>&1; then
        echo "Python module '${module_name}' is not available in ${PYTHON_BIN}."
        exit 1
    fi
}

check_num_gpus() {
    local num_gpus
    num_gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
    if [[ "${num_gpus}" -lt 4 ]]; then
        echo "This 2P2D script requires at least 4 GPUs, found ${num_gpus}."
        exit 1
    fi
}

validate_array_sizes() {
    if [[ "${#PREFILL_GPU_ARRAY[@]}" -ne 2 || "${#DECODE_GPU_ARRAY[@]}" -ne 2 ]]; then
        echo "Need exactly 2 prefill GPUs and 2 decode GPUs."
        exit 1
    fi
    if [[ "${#PREFILL_PORT_ARRAY[@]}" -ne 2 || "${#DECODE_PORT_ARRAY[@]}" -ne 2 ]]; then
        echo "Need exactly 2 prefill API ports and 2 decode API ports."
        exit 1
    fi
    if [[ "${#PREFILL_KV_PORT_ARRAY[@]}" -ne 2 || "${#DECODE_KV_PORT_ARRAY[@]}" -ne 2 ]]; then
        echo "Need exactly 2 prefill KV ports and 2 decode KV ports."
        exit 1
    fi
}

validate_ports() {
    declare -A seen_ports=()
    local port=""

    for port in \
        "${PROXY_DISCOVERY_PORT}" \
        "${PROXY_HTTP_PORT}" \
        "${PREFILL_PORT_ARRAY[@]}" \
        "${DECODE_PORT_ARRAY[@]}" \
        "${PREFILL_KV_PORT_ARRAY[@]}" \
        "${DECODE_KV_PORT_ARRAY[@]}"; do
        if [[ ! "${port}" =~ ^[0-9]+$ ]]; then
            echo "Invalid port: ${port}"
            exit 1
        fi
        if (( port < 1 || port > 65535 )); then
            echo "Port out of range: ${port}"
            exit 1
        fi
        if [[ -n "${seen_ports[${port}]:-}" ]]; then
            echo "Duplicate port detected: ${port}"
            exit 1
        fi
        seen_ports["${port}"]=1
    done
}

validate_benchmark_args() {
    if [[ "${RUN_BENCHMARK}" == "1" && -z "${SHAREGPT_DATASET_PATH}" ]]; then
        echo "SHAREGPT_DATASET_PATH must be set when RUN_BENCHMARK=1."
        exit 1
    fi
    if [[ -n "${SHAREGPT_DATASET_PATH}" && ! -f "${SHAREGPT_DATASET_PATH}" ]]; then
        echo "ShareGPT dataset file does not exist: ${SHAREGPT_DATASET_PATH}"
        exit 1
    fi
}

wait_for_http() {
    local url="$1"
    local label="$2"
    local start_time

    start_time="$(date +%s)"
    echo "Waiting for ${label}: ${url}"
    while true; do
        if curl --noproxy '*' --silent --output /dev/null --fail "${url}"; then
            echo "${label} is ready."
            return 0
        fi
        if (( "$(date +%s)" - start_time >= TIMEOUT_SECONDS )); then
            echo "Timed out waiting for ${label}: ${url}"
            return 1
        fi
        sleep 1
    done
}

cleanup() {
    local pid=""
    local deadline=""
    trap - EXIT INT TERM
    echo "Cleaning up processes..."
    for pid in "${PIDS[@]:-}"; do
        kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
        kill -TERM "${pid}" >/dev/null 2>&1 || true
    done
    deadline=$((SECONDS + 10))
    while (( SECONDS < deadline )); do
        local any_alive=0
        for pid in "${PIDS[@]:-}"; do
            if kill -0 "${pid}" >/dev/null 2>&1; then
                any_alive=1
                break
            fi
        done
        if (( any_alive == 0 )); then
            break
        fi
        sleep 1
    done
    for pid in "${PIDS[@]:-}"; do
        kill -KILL -- "-${pid}" >/dev/null 2>&1 || true
        kill -KILL "${pid}" >/dev/null 2>&1 || true
        wait "${pid}" >/dev/null 2>&1 || true
    done
}

print_config() {
    cat <<EOF
2P2D Disaggregated Serving Configuration
  Model: ${MODEL}
  Proxy discovery port: ${PROXY_DISCOVERY_PORT}
  Proxy HTTP port: ${PROXY_HTTP_PORT}
  Prefill GPUs: ${PREFILL_GPUS}
  Decode GPUs: ${DECODE_GPUS}
  Prefill API ports: ${PREFILL_PORTS}
  Decode API ports: ${DECODE_PORTS}
  Prefill KV ports: ${PREFILL_KV_PORTS}
  Decode KV ports: ${DECODE_KV_PORTS}
  Benchmark enabled: ${RUN_BENCHMARK}
  ShareGPT dataset: ${SHAREGPT_DATASET_PATH}
  Logs: ${LOG_DIR}
EOF
}

launch_proxy() {
    echo "Starting proxy: ${PROXY_SCRIPT}"
    launch_in_new_session "${LOG_DIR}/proxy.log" \
        "${PYTHON_BIN}" "${PROXY_SCRIPT}" \
        --host 0.0.0.0 \
        --port "${PROXY_HTTP_PORT}" \
        --discovery-port "${PROXY_DISCOVERY_PORT}"
    wait_for_http "http://127.0.0.1:${PROXY_HTTP_PORT}/health" "proxy"
}

launch_prefill_servers() {
    local i="" gpu_id="" api_port="" kv_port="" kv_config=""

    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        gpu_id="${PREFILL_GPU_ARRAY[$i]}"
        api_port="${PREFILL_PORT_ARRAY[$i]}"
        kv_port="${PREFILL_KV_PORT_ARRAY[$i]}"
        kv_config="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"${kv_port}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"${PROXY_DISCOVERY_PORT}\",\"http_port\":\"${api_port}\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}"

        echo "Starting prefill $((i + 1)): GPU=${gpu_id}, api_port=${api_port}, kv_port=${kv_port}"
        launch_in_new_session "${LOG_DIR}/prefill$((i + 1)).log" \
            env \
            VLLM_ENGINE_READY_TIMEOUT_S="${TIMEOUT_SECONDS}" \
            CUDA_VISIBLE_DEVICES="${gpu_id}" \
            "${VLLM_BIN}" serve "${MODEL}" \
            --enforce-eager \
            --host 0.0.0.0 \
            --port "${api_port}" \
            --tensor-parallel-size 1 \
            --seed 1024 \
            --dtype float16 \
            --max-model-len 8192 \
            --max-num-batched-tokens 10000 \
            --max-num-seqs 256 \
            --trust-remote-code \
            --gpu-memory-utilization 0.9 \
            --kv-transfer-config "${kv_config}"
    done

    for api_port in "${PREFILL_PORT_ARRAY[@]}"; do
        wait_for_http "http://127.0.0.1:${api_port}/health" "prefill:${api_port}"
    done
}

launch_decode_servers() {
    local i="" gpu_id="" api_port="" kv_port="" kv_config=""

    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        gpu_id="${DECODE_GPU_ARRAY[$i]}"
        api_port="${DECODE_PORT_ARRAY[$i]}"
        kv_port="${DECODE_KV_PORT_ARRAY[$i]}"
        kv_config="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"${kv_port}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"${PROXY_DISCOVERY_PORT}\",\"http_port\":\"${api_port}\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}"

        echo "Starting decode $((i + 1)): GPU=${gpu_id}, api_port=${api_port}, kv_port=${kv_port}"
        launch_in_new_session "${LOG_DIR}/decode$((i + 1)).log" \
            env \
            VLLM_ENGINE_READY_TIMEOUT_S="${TIMEOUT_SECONDS}" \
            CUDA_VISIBLE_DEVICES="${gpu_id}" \
            "${VLLM_BIN}" serve "${MODEL}" \
            --enforce-eager \
            --host 0.0.0.0 \
            --port "${api_port}" \
            --tensor-parallel-size 1 \
            --seed 1024 \
            --dtype float16 \
            --max-model-len 8192 \
            --max-num-batched-tokens 10000 \
            --max-num-seqs 256 \
            --trust-remote-code \
            --gpu-memory-utilization 0.7 \
            --kv-transfer-config "${kv_config}"
    done

    for api_port in "${DECODE_PORT_ARRAY[@]}"; do
        wait_for_http "http://127.0.0.1:${api_port}/health" "decode:${api_port}"
    done
}

run_sharegpt_benchmark() {
    echo "Starting ShareGPT benchmark through proxy..."
    "${VLLM_BIN}" bench serve \
        --backend "${BENCH_BACKEND}" \
        --base-url "http://127.0.0.1:${PROXY_HTTP_PORT}" \
        --endpoint "${BENCH_ENDPOINT}" \
        --model "${MODEL}" \
        --dataset-name sharegpt \
        --dataset-path "${SHAREGPT_DATASET_PATH}" \
        --sharegpt-output-len "${SHAREGPT_OUTPUT_LEN}" \
        --num-prompts "${BENCH_NUM_PROMPTS}" \
        --request-rate "${BENCH_REQUEST_RATE}" \
        --burstiness "${BENCH_BURSTINESS}" \
        --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
        --seed "${BENCH_SEED}" \
        2>&1 | tee "${LOG_DIR}/benchmark.log"
}

main() {
    trap cleanup EXIT INT TERM
    mkdir -p "${LOG_DIR}"

    check_required_files
    ensure_python_module_installed msgpack
    ensure_python_module_installed pandas
    ensure_python_module_installed datasets

    parse_csv_array PREFILL_GPU_ARRAY "${PREFILL_GPUS}"
    parse_csv_array DECODE_GPU_ARRAY "${DECODE_GPUS}"
    parse_csv_array PREFILL_PORT_ARRAY "${PREFILL_PORTS}"
    parse_csv_array DECODE_PORT_ARRAY "${DECODE_PORTS}"
    parse_csv_array PREFILL_KV_PORT_ARRAY "${PREFILL_KV_PORTS}"
    parse_csv_array DECODE_KV_PORT_ARRAY "${DECODE_KV_PORTS}"

    validate_array_sizes
    validate_ports
    validate_benchmark_args
    check_num_gpus
    print_config

    launch_proxy
    launch_prefill_servers
    launch_decode_servers

    echo "All proxy/prefill/decode servers are ready."
    echo "Proxy completions endpoint: http://127.0.0.1:${PROXY_HTTP_PORT}/v1/completions"
    echo "Proxy chat endpoint: http://127.0.0.1:${PROXY_HTTP_PORT}/v1/chat/completions"

    if [[ "${RUN_BENCHMARK}" == "1" ]]; then
        run_sharegpt_benchmark
    else
        echo "RUN_BENCHMARK=0, leaving servers running until interrupted."
        while true; do
            sleep 60
        done
    fi
}

main "$@"
