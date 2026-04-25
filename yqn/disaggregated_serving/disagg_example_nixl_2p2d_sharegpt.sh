#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
VLLM_BIN="${ROOT_DIR}/.venv/bin/vllm"
PROXY_SCRIPT="${SCRIPT_DIR}/disagg_proxy_nixl_2p2d.py"
NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"

MODEL="${MODEL:-/data/yqn/Qwen1.5-MoE-A2.7B}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"

PROXY_HTTP_PORT="${PROXY_HTTP_PORT:-10001}"
UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
UCX_TLS="${UCX_TLS:-tcp,cuda,self}"
UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
NIXL_KV_BUFFER_DEVICE="${NIXL_KV_BUFFER_DEVICE:-cuda}"
NIXL_KV_LOAD_FAILURE_POLICY="${NIXL_KV_LOAD_FAILURE_POLICY:-fail}"

PREFILL_GPUS="${PREFILL_GPUS:-4,6}"
DECODE_GPUS="${DECODE_GPUS:-5,7}"
PREFILL_PORTS="${PREFILL_PORTS:-20003,20013}"
DECODE_PORTS="${DECODE_PORTS:-20005,20015}"
PREFILL_NIXL_SIDE_CHANNEL_PORTS="${PREFILL_NIXL_SIDE_CHANNEL_PORTS:-21001,21011}"
DECODE_NIXL_SIDE_CHANNEL_PORTS="${DECODE_NIXL_SIDE_CHANNEL_PORTS:-22001,22011}"

RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
BENCH_BACKEND="${BENCH_BACKEND:-openai-chat}"
BENCH_ENDPOINT="${BENCH_ENDPOINT:-/v1/chat/completions}"
SHAREGPT_DATASET_PATH="${SHAREGPT_DATASET_PATH:-/data/yqn/datasets/ShareGPT-X/ChatGPT-Simple.jsonl}"
SHAREGPT_OUTPUT_LEN="${SHAREGPT_OUTPUT_LEN:-}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-1000}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-64}"
BENCH_BURSTINESS="${BENCH_BURSTINESS:-100}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"
BENCH_SEED="${BENCH_SEED:-$(date +%s)}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs_2p2d_nixl}"
DECODE_ATTN_NSYS="${DECODE_ATTN_NSYS:-1}"
DECODE_ATTN_VERIFY="${DECODE_ATTN_VERIFY:-0}"
DECODE_ATTN_VERIFY_MAX_LOGS="${DECODE_ATTN_VERIFY_MAX_LOGS:-50}"
DECODE_ATTN_NSYS_DIR="${DECODE_ATTN_NSYS_DIR:-${LOG_DIR}/decode_attn_nsys}"
DECODE_ATTN_NSYS_TRACE="${DECODE_ATTN_NSYS_TRACE:-cuda,nvtx,osrt}"
DECODE_ATTN_NSYS_SAMPLE="${DECODE_ATTN_NSYS_SAMPLE:-none}"
DECODE_ATTN_NSYS_TRACE_FORK_BEFORE_EXEC="${DECODE_ATTN_NSYS_TRACE_FORK_BEFORE_EXEC:-true}"
DECODE_ATTN_NSYS_WAIT="${DECODE_ATTN_NSYS_WAIT:-all}"

export NO_PROXY="127.0.0.1,localhost,0.0.0.0,${NO_PROXY:-}"
export no_proxy="${NO_PROXY}"
export DISAGG_MODEL="${MODEL}"

declare -a PIDS=()
declare -a PREFILL_GPU_ARRAY=()
declare -a DECODE_GPU_ARRAY=()
declare -a PREFILL_PORT_ARRAY=()
declare -a DECODE_PORT_ARRAY=()
declare -a PREFILL_SIDE_CHANNEL_PORT_ARRAY=()
declare -a DECODE_SIDE_CHANNEL_PORT_ARRAY=()

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
    if [[ "${DECODE_ATTN_NSYS}" == "1" && -z "${NSYS_BIN}" ]]; then
        echo "DECODE_ATTN_NSYS=1 but 'nsys' was not found in PATH."
        missing=1
    fi
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
    if [[ "${#PREFILL_SIDE_CHANNEL_PORT_ARRAY[@]}" -ne 2 || "${#DECODE_SIDE_CHANNEL_PORT_ARRAY[@]}" -ne 2 ]]; then
        echo "Need exactly 2 prefill side-channel ports and 2 decode side-channel ports."
        exit 1
    fi
}

validate_ports() {
    declare -A seen_ports=()
    local port=""

    for port in \
        "${PROXY_HTTP_PORT}" \
        "${PREFILL_PORT_ARRAY[@]}" \
        "${DECODE_PORT_ARRAY[@]}" \
        "${PREFILL_SIDE_CHANNEL_PORT_ARRAY[@]}" \
        "${DECODE_SIDE_CHANNEL_PORT_ARRAY[@]}"; do
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

build_nixl_kv_config() {
    if [[ "${NIXL_KV_BUFFER_DEVICE}" == "cuda" ]]; then
        printf '%s' \
            "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"${NIXL_KV_LOAD_FAILURE_POLICY}\"}"
        return 0
    fi

    printf '%s' \
        "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"${NIXL_KV_BUFFER_DEVICE}\",\"kv_load_failure_policy\":\"${NIXL_KV_LOAD_FAILURE_POLICY}\"}"
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
    export_decode_nsys_sqlite
}

export_decode_nsys_sqlite() {
    local i=""
    local rep_path=""
    local sqlite_path=""

    if [[ "${DECODE_ATTN_NSYS}" != "1" ]]; then
        return
    fi

    echo "Exporting decode nsys reports to sqlite..."
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        rep_path="${DECODE_ATTN_NSYS_DIR}/decode$((i + 1)).nsys-rep"
        sqlite_path="${DECODE_ATTN_NSYS_DIR}/decode$((i + 1)).sqlite"
        if [[ ! -f "${rep_path}" ]]; then
            echo "Skipping missing report: ${rep_path}"
            continue
        fi
        echo "  ${rep_path} -> ${sqlite_path}"
        "${NSYS_BIN}" export \
            --type sqlite \
            --force-overwrite true \
            --quiet false \
            --output "${sqlite_path}" \
            "${rep_path}"
    done
}

print_config() {
    cat <<EOF
2P2D Disaggregated Serving Configuration
  Connector: NixlConnector
  Model: ${MODEL}
  Proxy HTTP port: ${PROXY_HTTP_PORT}
  Proxy script: ${PROXY_SCRIPT}
  UCX_NET_DEVICES: ${UCX_NET_DEVICES}
  UCX_TLS: ${UCX_TLS}
  UCX_MEMTYPE_CACHE: ${UCX_MEMTYPE_CACHE}
  NIXL kv_buffer_device: ${NIXL_KV_BUFFER_DEVICE}
  NIXL kv_load_failure_policy: ${NIXL_KV_LOAD_FAILURE_POLICY}
  Prefill GPUs: ${PREFILL_GPUS}
  Decode GPUs: ${DECODE_GPUS}
  Prefill API ports: ${PREFILL_PORTS}
  Decode API ports: ${DECODE_PORTS}
  Prefill NIXL side-channel ports: ${PREFILL_NIXL_SIDE_CHANNEL_PORTS}
  Decode NIXL side-channel ports: ${DECODE_NIXL_SIDE_CHANNEL_PORTS}
  Benchmark enabled: ${RUN_BENCHMARK}
  ShareGPT dataset: ${SHAREGPT_DATASET_PATH}
  Logs: ${LOG_DIR}
  Decode attention nsys: ${DECODE_ATTN_NSYS}
  Decode attention verify: ${DECODE_ATTN_VERIFY}
  Decode attention verify max logs: ${DECODE_ATTN_VERIFY_MAX_LOGS}
  Decode attention nsys dir: ${DECODE_ATTN_NSYS_DIR}
  Decode attention nsys trace-fork-before-exec: ${DECODE_ATTN_NSYS_TRACE_FORK_BEFORE_EXEC}
  Decode attention nsys wait: ${DECODE_ATTN_NSYS_WAIT}
EOF
}

launch_proxy() {
    echo "Starting proxy: ${PROXY_SCRIPT}"
    launch_in_new_session "${LOG_DIR}/proxy.log" \
        "${PYTHON_BIN}" "${PROXY_SCRIPT}" \
        --host 0.0.0.0 \
        --port "${PROXY_HTTP_PORT}" \
        --prefiller-hosts 127.0.0.1 127.0.0.1 \
        --prefiller-ports "${PREFILL_PORT_ARRAY[@]}" \
        --decoder-hosts 127.0.0.1 127.0.0.1 \
        --decoder-ports "${DECODE_PORT_ARRAY[@]}"
    wait_for_http "http://127.0.0.1:${PROXY_HTTP_PORT}/healthcheck" "proxy"
}

launch_prefill_servers() {
    local i="" gpu_id="" api_port="" side_channel_port="" kv_config=""

    kv_config="$(build_nixl_kv_config)"

    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        gpu_id="${PREFILL_GPU_ARRAY[$i]}"
        api_port="${PREFILL_PORT_ARRAY[$i]}"
        side_channel_port="${PREFILL_SIDE_CHANNEL_PORT_ARRAY[$i]}"

        echo "Starting prefill $((i + 1)): GPU=${gpu_id}, api_port=${api_port}, side_channel_port=${side_channel_port}"
        launch_in_new_session "${LOG_DIR}/prefill$((i + 1)).log" \
            env \
            VLLM_ENGINE_READY_TIMEOUT_S="${TIMEOUT_SECONDS}" \
            UCX_NET_DEVICES="${UCX_NET_DEVICES}" \
            UCX_TLS="${UCX_TLS}" \
            UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE}" \
            VLLM_NIXL_SIDE_CHANNEL_PORT="${side_channel_port}" \
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
            --gpu-memory-utilization 0.8 \
            --kv-transfer-config "${kv_config}"
    done

    for api_port in "${PREFILL_PORT_ARRAY[@]}"; do
        wait_for_http "http://127.0.0.1:${api_port}/health" "prefill:${api_port}"
    done
}

launch_decode_servers() {
    local i="" gpu_id="" api_port="" side_channel_port="" kv_config=""

    kv_config="$(build_nixl_kv_config)"
    if [[ "${DECODE_ATTN_NSYS}" == "1" ]]; then
        mkdir -p "${DECODE_ATTN_NSYS_DIR}"
    fi

    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        gpu_id="${DECODE_GPU_ARRAY[$i]}"
        api_port="${DECODE_PORT_ARRAY[$i]}"
        side_channel_port="${DECODE_SIDE_CHANNEL_PORT_ARRAY[$i]}"

        echo "Starting decode $((i + 1)): GPU=${gpu_id}, api_port=${api_port}, side_channel_port=${side_channel_port}"
        if [[ "${DECODE_ATTN_NSYS}" == "1" ]]; then
            launch_in_new_session "${LOG_DIR}/decode$((i + 1)).log" \
                env \
                VLLM_ENGINE_READY_TIMEOUT_S="${TIMEOUT_SECONDS}" \
                UCX_NET_DEVICES="${UCX_NET_DEVICES}" \
                UCX_TLS="${UCX_TLS}" \
                UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE}" \
                VLLM_NIXL_SIDE_CHANNEL_PORT="${side_channel_port}" \
                VLLM_QWEN2MOE_DECODE_ATTN_NVTX="1" \
                VLLM_QWEN2MOE_DECODE_ATTN_VERIFY="${DECODE_ATTN_VERIFY}" \
                VLLM_QWEN2MOE_DECODE_ATTN_VERIFY_MAX_LOGS="${DECODE_ATTN_VERIFY_MAX_LOGS}" \
                CUDA_VISIBLE_DEVICES="${gpu_id}" \
                "${NSYS_BIN}" profile \
                --trace "${DECODE_ATTN_NSYS_TRACE}" \
                --sample "${DECODE_ATTN_NSYS_SAMPLE}" \
                --force-overwrite true \
                --output "${DECODE_ATTN_NSYS_DIR}/decode$((i + 1))" \
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
                --gpu-memory-utilization 0.8 \
                --kv-transfer-config "${kv_config}"
        else
            launch_in_new_session "${LOG_DIR}/decode$((i + 1)).log" \
                env \
                VLLM_ENGINE_READY_TIMEOUT_S="${TIMEOUT_SECONDS}" \
                UCX_NET_DEVICES="${UCX_NET_DEVICES}" \
                UCX_TLS="${UCX_TLS}" \
                UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE}" \
                VLLM_NIXL_SIDE_CHANNEL_PORT="${side_channel_port}" \
                VLLM_QWEN2MOE_DECODE_ATTN_NVTX="0" \
                VLLM_QWEN2MOE_DECODE_ATTN_VERIFY="${DECODE_ATTN_VERIFY}" \
                VLLM_QWEN2MOE_DECODE_ATTN_VERIFY_MAX_LOGS="${DECODE_ATTN_VERIFY_MAX_LOGS}" \
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
        fi
    done

    for api_port in "${DECODE_PORT_ARRAY[@]}"; do
        wait_for_http "http://127.0.0.1:${api_port}/health" "decode:${api_port}"
    done
}

run_sharegpt_benchmark() {
    local -a bench_args=(
        --backend "${BENCH_BACKEND}"
        --base-url "http://127.0.0.1:${PROXY_HTTP_PORT}"
        --endpoint "${BENCH_ENDPOINT}"
        --model "${MODEL}"
        --dataset-name sharegpt
        --dataset-path "${SHAREGPT_DATASET_PATH}"
        --num-prompts "${BENCH_NUM_PROMPTS}"
        --request-rate "${BENCH_REQUEST_RATE}"
        --burstiness "${BENCH_BURSTINESS}"
        --max-concurrency "${BENCH_MAX_CONCURRENCY}"
        --seed "${BENCH_SEED}"
    )

    if [[ -n "${SHAREGPT_OUTPUT_LEN}" ]]; then
        bench_args+=(--sharegpt-output-len "${SHAREGPT_OUTPUT_LEN}")
    fi

    echo "Starting ShareGPT benchmark through proxy..."
    "${VLLM_BIN}" bench serve "${bench_args[@]}" 2>&1 | tee "${LOG_DIR}/benchmark.log"
}

main() {
    trap cleanup EXIT INT TERM
    mkdir -p "${LOG_DIR}"

    check_required_files
    ensure_python_module_installed pandas
    ensure_python_module_installed datasets
    ensure_python_module_installed msgpack
    ensure_python_module_installed nixl

    parse_csv_array PREFILL_GPU_ARRAY "${PREFILL_GPUS}"
    parse_csv_array DECODE_GPU_ARRAY "${DECODE_GPUS}"
    parse_csv_array PREFILL_PORT_ARRAY "${PREFILL_PORTS}"
    parse_csv_array DECODE_PORT_ARRAY "${DECODE_PORTS}"
    parse_csv_array PREFILL_SIDE_CHANNEL_PORT_ARRAY "${PREFILL_NIXL_SIDE_CHANNEL_PORTS}"
    parse_csv_array DECODE_SIDE_CHANNEL_PORT_ARRAY "${DECODE_NIXL_SIDE_CHANNEL_PORTS}"

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
