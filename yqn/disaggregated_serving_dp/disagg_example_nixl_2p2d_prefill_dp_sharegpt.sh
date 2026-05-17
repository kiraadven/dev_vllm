#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
VLLM_BIN="${ROOT_DIR}/.venv/bin/vllm"
PROXY_SCRIPT="${ROOT_DIR}/yqn/disaggregated_serving_dp/disagg_proxy_nixl_2p2d.py"
NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"

MODEL="${MODEL:-/data/yqn/Qwen1.5-MoE-A2.7B}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"

PROXY_HTTP_PORT="${PROXY_HTTP_PORT:-10101}"
UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
UCX_TLS="${UCX_TLS:-tcp,cuda,self}"
UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
NIXL_KV_BUFFER_DEVICE="${NIXL_KV_BUFFER_DEVICE:-cuda}"
NIXL_KV_LOAD_FAILURE_POLICY="${NIXL_KV_LOAD_FAILURE_POLICY:-fail}"

PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-2}"
PREFILL_DP_SIZE_LOCAL="${PREFILL_DP_SIZE_LOCAL:-2}"
PREFILL_DP_MASTER_IP="${PREFILL_DP_MASTER_IP:-127.0.0.1}"
PREFILL_DP_RPC_PORT="${PREFILL_DP_RPC_PORT:-13445}"
PREFILL_DP_GPUS="${PREFILL_DP_GPUS:-4,6}"
PREFILL_DP_PORTS="${PREFILL_DP_PORTS:-20103}"
PREFILL_DP_NIXL_SIDE_CHANNEL_BASE_PORT="${PREFILL_DP_NIXL_SIDE_CHANNEL_BASE_PORT:-21101}"

DECODE_GPUS="${DECODE_GPUS:-5,7}"
DECODE_PORTS="${DECODE_PORTS:-20105,20115}"
DECODE_NIXL_SIDE_CHANNEL_PORTS="${DECODE_NIXL_SIDE_CHANNEL_PORTS:-22101,22111}"

RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
BENCH_BACKEND="${BENCH_BACKEND:-openai-chat}"
BENCH_ENDPOINT="${BENCH_ENDPOINT:-/v1/chat/completions}"
SHAREGPT_DATASET_PATH="${SHAREGPT_DATASET_PATH:-/data/yqn/datasets/ShareGPT-X/ChatGPT-Simple.jsonl}"
SHAREGPT_OUTPUT_LEN="${SHAREGPT_OUTPUT_LEN:-}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-2000}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-512}"
BENCH_BURSTINESS="${BENCH_BURSTINESS:-0.5}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-1024}"
BENCH_SEED="${BENCH_SEED:-$(date +%s)}"
DP_COORDINATOR_TRACE="${DP_COORDINATOR_TRACE:-1}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs_dp_2p2d_nixl}"
DECODE_ATTN_NSYS="${DECODE_ATTN_NSYS:-1}"
DECODE_ATTN_VERIFY="${DECODE_ATTN_VERIFY:-0}"
DECODE_ATTN_VERIFY_MAX_LOGS="${DECODE_ATTN_VERIFY_MAX_LOGS:-1000000}"
DECODE_ATTN_NSYS_DIR="${DECODE_ATTN_NSYS_DIR:-${LOG_DIR}/decode_attn_nsys}"
DECODE_ATTN_NSYS_TRACE="${DECODE_ATTN_NSYS_TRACE:-cuda,nvtx,osrt}"
DECODE_ATTN_NSYS_SAMPLE="${DECODE_ATTN_NSYS_SAMPLE:-none}"
DECODE_ATTN_NSYS_TRACE_FORK_BEFORE_EXEC="${DECODE_ATTN_NSYS_TRACE_FORK_BEFORE_EXEC:-true}"
DECODE_ATTN_NSYS_WAIT="${DECODE_ATTN_NSYS_WAIT:-all}"
CLEANUP_GRACE_SECONDS="${CLEANUP_GRACE_SECONDS:-10}"
NSYS_CLEANUP_GRACE_SECONDS="${NSYS_CLEANUP_GRACE_SECONDS:-120}"
NSYS_REPORT_STABLE_POLLS="${NSYS_REPORT_STABLE_POLLS:-3}"
NSYS_REPORT_STABLE_INTERVAL_SECONDS="${NSYS_REPORT_STABLE_INTERVAL_SECONDS:-2}"

export NO_PROXY="127.0.0.1,localhost,0.0.0.0,${NO_PROXY:-}"
export no_proxy="${NO_PROXY}"
export DISAGG_MODEL="${MODEL}"

declare -a PIDS=()
declare -a PREFILL_DP_GPU_ARRAY=()
declare -a PREFILL_DP_PORT_ARRAY=()
declare -a PREFILL_DP_SIDE_CHANNEL_PORT_ARRAY=()
declare -a DECODE_GPU_ARRAY=()
declare -a DECODE_PORT_ARRAY=()
declare -a DECODE_SIDE_CHANNEL_PORT_ARRAY=()

parse_csv_array() {
    local -n target_array="$1"
    local raw="$2"
    IFS=',' read -r -a target_array <<< "$raw"
}

build_prefill_side_channel_ports() {
    local base_port="$1"
    local rank=""

    PREFILL_DP_SIDE_CHANNEL_PORT_ARRAY=()
    for (( rank = 0; rank < PREFILL_DP_SIZE_LOCAL; rank += 1 )); do
        PREFILL_DP_SIDE_CHANNEL_PORT_ARRAY+=("$((base_port + rank))")
    done
}

launch_in_new_session() {
    local log_file="$1"
    shift
    # Run in background. Bash automatically puts background jobs in their
    # own process group (PGID == PID), so `kill -- -$PID` in cleanup()
    # reaches the entire process tree (nsys → vllm → workers).
    "$@" >"${log_file}" 2>&1 &
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
        echo "This prefill-DP 2P2D script requires at least 4 GPUs, found ${num_gpus}."
        exit 1
    fi
}

validate_array_sizes() {
    if [[ "${PREFILL_DP_SIZE}" -ne 2 ]]; then
        echo "This script currently supports exactly PREFILL_DP_SIZE=2."
        exit 1
    fi
    if [[ "${PREFILL_DP_SIZE_LOCAL}" -ne "${PREFILL_DP_SIZE}" ]]; then
        echo "Internal-LB prefill DP requires PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE}."
        exit 1
    fi
    if [[ "${#PREFILL_DP_GPU_ARRAY[@]}" -ne 2 ]]; then
        echo "Need exactly 2 prefill DP GPUs."
        exit 1
    fi
    if [[ "${#PREFILL_DP_PORT_ARRAY[@]}" -ne 1 ]]; then
        echo "Need exactly 1 prefill DP API port for internal LB."
        exit 1
    fi
    if [[ "${#DECODE_GPU_ARRAY[@]}" -ne 2 ]]; then
        echo "Need exactly 2 decode GPUs."
        exit 1
    fi
    if [[ "${#DECODE_PORT_ARRAY[@]}" -ne 2 ]]; then
        echo "Need exactly 2 decode API ports."
        exit 1
    fi
    if [[ "${#DECODE_SIDE_CHANNEL_PORT_ARRAY[@]}" -ne 2 ]]; then
        echo "Need exactly 2 decode NIXL side-channel ports."
        exit 1
    fi
}

validate_ports() {
    declare -A seen_ports=()
    local port=""

    for port in \
        "${PROXY_HTTP_PORT}" \
        "${PREFILL_DP_RPC_PORT}" \
        "${PREFILL_DP_PORT_ARRAY[@]}" \
        "${DECODE_PORT_ARRAY[@]}" \
        "${PREFILL_DP_SIDE_CHANNEL_PORT_ARRAY[@]}" \
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
    trap - EXIT INT TERM
    echo "Cleaning up processes..."

    local grace="${CLEANUP_GRACE_SECONDS}"
    if [[ "${DECODE_ATTN_NSYS}" == "1" ]]; then
        grace="${NSYS_CLEANUP_GRACE_SECONDS}"
    fi

    # SIGTERM for graceful shutdown.
    # - vLLM's SIGTERM handler dumps step-profiler CSV then re-raises.
    # - nsys finalizes .nsys-rep on SIGTERM of its child.
    for pid in "${PIDS[@]:-}"; do
        kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
        kill -TERM "${pid}" >/dev/null 2>&1 || true
    done

    # Wait up to $grace seconds for processes to exit.
    local deadline=$((SECONDS + grace))
    while (( SECONDS < deadline )); do
        local any_alive=0
        for pid in "${PIDS[@]:-}"; do
            if kill -0 "${pid}" >/dev/null 2>&1; then
                any_alive=1
                break
            fi
        done
        (( any_alive == 0 )) && break
        sleep 1
    done

    # Force-kill stragglers.
    for pid in "${PIDS[@]:-}"; do
        kill -KILL -- "-${pid}" >/dev/null 2>&1 || true
        kill -KILL "${pid}" >/dev/null 2>&1 || true
        wait "${pid}" >/dev/null 2>&1 || true
    done

    export_decode_nsys_sqlite
}

wait_for_stable_file() {
    local path="$1"
    local stable_polls="${2:-3}"
    local interval_seconds="${3:-2}"
    local stable_count=0
    local prev_size="-1"
    local current_size=""

    if [[ ! -f "${path}" ]]; then
        return 1
    fi

    while (( stable_count < stable_polls )); do
        current_size="$(stat -c '%s' "${path}" 2>/dev/null || echo -1)"
        if [[ "${current_size}" == "${prev_size}" && "${current_size}" != "-1" ]]; then
            stable_count=$((stable_count + 1))
        else
            stable_count=0
            prev_size="${current_size}"
        fi
        sleep "${interval_seconds}"
    done
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
        echo "  Waiting for stable report file: ${rep_path}"
        wait_for_stable_file \
            "${rep_path}" \
            "${NSYS_REPORT_STABLE_POLLS}" \
            "${NSYS_REPORT_STABLE_INTERVAL_SECONDS}"
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
Prefill-DP 2P2D Disaggregated Serving Configuration
  Connector: NixlConnector
  Model: ${MODEL}
  Proxy HTTP port: ${PROXY_HTTP_PORT}
  Proxy script: ${PROXY_SCRIPT}
  UCX_NET_DEVICES: ${UCX_NET_DEVICES}
  UCX_TLS: ${UCX_TLS}
  UCX_MEMTYPE_CACHE: ${UCX_MEMTYPE_CACHE}
  NIXL kv_buffer_device: ${NIXL_KV_BUFFER_DEVICE}
  NIXL kv_load_failure_policy: ${NIXL_KV_LOAD_FAILURE_POLICY}
  Prefill DP size: ${PREFILL_DP_SIZE}
  Prefill DP size local: ${PREFILL_DP_SIZE_LOCAL}
  Prefill LB mode: internal
  Prefill DP master IP: ${PREFILL_DP_MASTER_IP}
  Prefill DP RPC port: ${PREFILL_DP_RPC_PORT}
  Prefill DP GPUs: ${PREFILL_DP_GPUS}
  Prefill DP API port: ${PREFILL_DP_PORTS}
  Prefill DP NIXL side-channel base port: ${PREFILL_DP_NIXL_SIDE_CHANNEL_BASE_PORT}
  Prefill DP NIXL side-channel ports: ${PREFILL_DP_SIDE_CHANNEL_PORT_ARRAY[*]}
  Decode GPUs: ${DECODE_GPUS}
  Decode API ports: ${DECODE_PORTS}
  Decode NIXL side-channel ports: ${DECODE_NIXL_SIDE_CHANNEL_PORTS}
  Benchmark enabled: ${RUN_BENCHMARK}
  ShareGPT dataset: ${SHAREGPT_DATASET_PATH}
  Logs: ${LOG_DIR}
  Coordinator log: ${LOG_DIR}/dp_coordinator.log
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
        --prefiller-hosts 127.0.0.1 \
        --prefiller-ports "${PREFILL_DP_PORT_ARRAY[0]}" \
        --decoder-hosts 127.0.0.1 127.0.0.1 \
        --decoder-ports "${DECODE_PORT_ARRAY[@]}"
    wait_for_http "http://127.0.0.1:${PROXY_HTTP_PORT}/healthcheck" "proxy"
}

launch_prefill_dp_servers() {
    local api_port="" kv_config=""

    api_port="${PREFILL_DP_PORT_ARRAY[0]}"
    kv_config="$(build_nixl_kv_config)"

    echo "Starting prefill internal-LB server: GPUs=${PREFILL_DP_GPUS}, api_port=${api_port}, side_channel_base_port=${PREFILL_DP_NIXL_SIDE_CHANNEL_BASE_PORT}"
    launch_in_new_session "${LOG_DIR}/prefill_internal_lb.log" \
        env \
        VLLM_ENGINE_READY_TIMEOUT_S="${TIMEOUT_SECONDS}" \
        VLLM_DP_COORDINATOR_TRACE="${DP_COORDINATOR_TRACE}" \
        VLLM_DP_COORDINATOR_LOG_PATH="${LOG_DIR}/dp_coordinator.log" \
        UCX_NET_DEVICES="${UCX_NET_DEVICES}" \
        UCX_TLS="${UCX_TLS}" \
        UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE}" \
        VLLM_NIXL_SIDE_CHANNEL_PORT="${PREFILL_DP_NIXL_SIDE_CHANNEL_BASE_PORT}" \
        CUDA_VISIBLE_DEVICES="${PREFILL_DP_GPUS}" \
        "${VLLM_BIN}" serve "${MODEL}" \
        --enforce-eager \
        --host 0.0.0.0 \
        --port "${api_port}" \
        --api-server-count 1 \
        --tensor-parallel-size 1 \
        --data-parallel-size "${PREFILL_DP_SIZE}" \
        --data-parallel-size-local "${PREFILL_DP_SIZE_LOCAL}" \
        --data-parallel-address "${PREFILL_DP_MASTER_IP}" \
        --data-parallel-rpc-port "${PREFILL_DP_RPC_PORT}" \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 8192 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config "${kv_config}"

    wait_for_http "http://127.0.0.1:${api_port}/health" "prefill-dp:${api_port}"
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
                VLLM_INSTANCE_ID="${i}" \
                VLLM_INSTANCE_LOCAL_ID="${gpu_id}" \
                CUDA_VISIBLE_DEVICES="${gpu_id}" \
                "${NSYS_BIN}" profile \
                --trace "${DECODE_ATTN_NSYS_TRACE}" \
                --sample "${DECODE_ATTN_NSYS_SAMPLE}" \
                --trace-fork-before-exec "${DECODE_ATTN_NSYS_TRACE_FORK_BEFORE_EXEC}" \
                --wait "${DECODE_ATTN_NSYS_WAIT}" \
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
                --gpu-memory-utilization 0.9 \
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
                VLLM_INSTANCE_ID="${i}" \
                VLLM_INSTANCE_LOCAL_ID="${gpu_id}" \
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
    ensure_python_module_installed msgpack
    ensure_python_module_installed pandas
    ensure_python_module_installed datasets
    ensure_python_module_installed nixl

    parse_csv_array PREFILL_DP_GPU_ARRAY "${PREFILL_DP_GPUS}"
    parse_csv_array PREFILL_DP_PORT_ARRAY "${PREFILL_DP_PORTS}"
    build_prefill_side_channel_ports "${PREFILL_DP_NIXL_SIDE_CHANNEL_BASE_PORT}"
    parse_csv_array DECODE_GPU_ARRAY "${DECODE_GPUS}"
    parse_csv_array DECODE_PORT_ARRAY "${DECODE_PORTS}"
    parse_csv_array DECODE_SIDE_CHANNEL_PORT_ARRAY "${DECODE_NIXL_SIDE_CHANNEL_PORTS}"

    validate_array_sizes
    validate_ports
    validate_benchmark_args
    check_num_gpus
    print_config

    launch_proxy
    launch_prefill_dp_servers
    launch_decode_servers

    echo "All proxy/prefill-dp/decode servers are ready."
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
