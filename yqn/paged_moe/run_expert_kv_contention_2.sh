#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv/bin/vllm}"
MODEL="${MODEL:-/data/yqn/Qwen1.5-MoE-A2.7B}"
SHAREGPT_DATASET_PATH="${SHAREGPT_DATASET_PATH:-/data/yqn/datasets/ShareGPT-X/ChatGPT-Simple.vllm.json}"

GPU_ID="${GPU_ID:-6}"
if [[ -z "${NUMA_NODE:-}" ]]; then
    if (( GPU_ID <= 3 )); then
        NUMA_NODE=0
    else
        NUMA_NODE=1
    fi
fi
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
SERVER_STOP_TIMEOUT="${SERVER_STOP_TIMEOUT:-30}"

# RTX A6000 has ~48GiB HBM; Qwen1.5-MoE-A2.7B weights are ~27GiB.
# This second-run default uses NUMA node 1 and a lower memory budget to amplify
# KV/expert HBM contention while avoiding the primary experiment's port/GPU.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

PREFETCH_STEP="${PREFETCH_STEP:-2}"
OFFLOAD_PARAMS="${OFFLOAD_PARAMS:-w13_weight w2_weight}"
R_CONFIGS="${R_CONFIGS:-R0 R25 R50 R75 R100}"
REQUEST_RATES="${REQUEST_RATES:-16 64 128 256 512}"

BENCH_BACKEND="${BENCH_BACKEND:-openai}"
BENCH_ENDPOINT="${BENCH_ENDPOINT:-/v1/completions}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-256}"
BENCH_BURSTINESS="${BENCH_BURSTINESS:-1.0}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-256}"
BENCH_SEED="${BENCH_SEED:-1024}"
SHAREGPT_OUTPUT_LEN="${SHAREGPT_OUTPUT_LEN:-1024}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/results_expert_kv_contention}"
RUN_DIR="${RESULT_ROOT}/${RUN_ID}"
SKIP_DONE="${SKIP_DONE:-1}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"

SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"

CUDA_RUNTIME_LIB_DIR="${CUDA_RUNTIME_LIB_DIR:-${ROOT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cu13/lib}"
VLLM_STABLE_EXT_PATH="${VLLM_STABLE_EXT_PATH:-}"
if [[ -d "${CUDA_RUNTIME_LIB_DIR}" ]]; then
    VLLM_LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
    VLLM_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
fi
VLLM_PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

SERVER_PID=""
CURRENT_STATUS_FILE=""

declare -a COMMON_SERVER_ARGS=()
declare -a SERVER_OFFLOAD_ARGS=()

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

split_words() {
    local -n target_array="$1"
    local raw="$2"
    target_array=()
    if [[ -n "${raw}" ]]; then
        read -r -a target_array <<< "${raw}"
    fi
}

mark_status() {
    local status="$1"
    if [[ -n "${CURRENT_STATUS_FILE}" ]]; then
        printf '%s\n' "${status}" > "${CURRENT_STATUS_FILE}"
    fi
}

is_server_ready() {
    curl --noproxy '*' --silent --output /dev/null --fail --max-time 2 \
        "${BASE_URL}/v1/models"
}

ensure_no_unmanaged_server() {
    if is_server_ready; then
        log "ERROR: ${BASE_URL} is already responding before this script starts a server."
        log "Stop that server or choose another PORT; refusing to benchmark an unmanaged server."
        exit 1
    fi
}

cleanup_current_server() {
    local pid="${SERVER_PID}"
    local deadline=""

    if [[ -z "${pid}" ]]; then
        return 0
    fi

    if kill -0 "${pid}" >/dev/null 2>&1; then
        log "Stopping vLLM server process group ${pid}"
        kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
        kill -TERM "${pid}" >/dev/null 2>&1 || true

        deadline=$((SECONDS + SERVER_STOP_TIMEOUT))
        while kill -0 "${pid}" >/dev/null 2>&1; do
            if (( SECONDS >= deadline )); then
                log "Server did not stop in ${SERVER_STOP_TIMEOUT}s; sending SIGKILL to process group ${pid}"
                kill -KILL -- "-${pid}" >/dev/null 2>&1 || true
                kill -KILL "${pid}" >/dev/null 2>&1 || true
                break
            fi
            sleep 1
        done
    fi

    wait "${pid}" >/dev/null 2>&1 || true
    SERVER_PID=""
}

cleanup_and_exit() {
    local exit_code="$1"
    trap - EXIT INT TERM
    if [[ "${exit_code}" -eq 130 ]]; then
        log "Interrupted. Cleaning up current server so GPU resources are released."
        mark_status "interrupted"
    fi
    cleanup_current_server
    exit "${exit_code}"
}

trap 'cleanup_and_exit 130' INT TERM
trap 'cleanup_and_exit $?' EXIT

check_required_paths() {
    local missing=0
    for path in "${VLLM_BIN}" "${SHAREGPT_DATASET_PATH}"; do
        if [[ ! -e "${path}" ]]; then
            log "Missing required path: ${path}"
            missing=1
        fi
    done
    if ! command -v curl >/dev/null 2>&1; then
        log "Missing required command: curl"
        missing=1
    fi
    if ! command -v stdbuf >/dev/null 2>&1; then
        log "Missing required command: stdbuf"
        missing=1
    fi
    if ! command -v numactl >/dev/null 2>&1; then
        log "Missing required command: numactl"
        missing=1
    fi
    if [[ "${missing}" -ne 0 ]]; then
        exit 1
    fi
}

build_common_server_args() {
    COMMON_SERVER_ARGS=(
        serve "${MODEL}"
        --host "${HOST}"
        --port "${PORT}"
        --seed 1024
        --dtype "${DTYPE}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
        --max-model-len "${MAX_MODEL_LEN}"
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
        --max-num-seqs "${MAX_NUM_SEQS}"
    )

    if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
        COMMON_SERVER_ARGS+=(--trust-remote-code)
    fi
    if [[ "${ENFORCE_EAGER}" == "1" ]]; then
        COMMON_SERVER_ARGS+=(--enforce-eager)
    fi
}

set_offload_args() {
    local r_label="$1"
    SERVER_OFFLOAD_ARGS=()

    local group_size=""
    local num_in_group=""
    case "${r_label}" in
        R0)
            return 0
            ;;
        R25)
            group_size=4
            num_in_group=1
            ;;
        R50)
            group_size=4
            num_in_group=2
            ;;
        R75)
            group_size=4
            num_in_group=3
            ;;
        R100)
            group_size=1
            num_in_group=1
            ;;
        *)
            log "ERROR: unknown R config '${r_label}'. Supported: R0 R25 R50 R75 R100"
            return 1
            ;;
    esac

    SERVER_OFFLOAD_ARGS=(
        --offload-backend prefetch
        --offload-group-size "${group_size}"
        --offload-num-in-group "${num_in_group}"
        --offload-prefetch-step "${PREFETCH_STEP}"
    )

    local offload_param_args=()
    split_words offload_param_args "${OFFLOAD_PARAMS}"
    if (( ${#offload_param_args[@]} > 0 )); then
        SERVER_OFFLOAD_ARGS+=(--offload-params "${offload_param_args[@]}")
    fi
}

write_run_config() {
    mkdir -p "${RUN_DIR}"
    cat > "${RUN_DIR}/run_config.env" <<EOF
VLLM_BIN=${VLLM_BIN}
MODEL=${MODEL}
SHAREGPT_DATASET_PATH=${SHAREGPT_DATASET_PATH}
GPU_ID=${GPU_ID}
NUMA_NODE=${NUMA_NODE}
HOST=${HOST}
PORT=${PORT}
BASE_URL=${BASE_URL}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS}
SERVER_STOP_TIMEOUT=${SERVER_STOP_TIMEOUT}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
DTYPE=${DTYPE}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}
ENFORCE_EAGER=${ENFORCE_EAGER}
PREFETCH_STEP=${PREFETCH_STEP}
OFFLOAD_PARAMS=${OFFLOAD_PARAMS}
R_CONFIGS=${R_CONFIGS}
REQUEST_RATES=${REQUEST_RATES}
BENCH_BACKEND=${BENCH_BACKEND}
BENCH_ENDPOINT=${BENCH_ENDPOINT}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS}
BENCH_BURSTINESS=${BENCH_BURSTINESS}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY}
BENCH_SEED=${BENCH_SEED}
SHAREGPT_OUTPUT_LEN=${SHAREGPT_OUTPUT_LEN}
RUN_ID=${RUN_ID}
RESULT_ROOT=${RESULT_ROOT}
RUN_DIR=${RUN_DIR}
SKIP_DONE=${SKIP_DONE}
STOP_ON_FAILURE=${STOP_ON_FAILURE}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS}
BENCH_EXTRA_ARGS=${BENCH_EXTRA_ARGS}
CUDA_RUNTIME_LIB_DIR=${CUDA_RUNTIME_LIB_DIR}
VLLM_STABLE_EXT_PATH=${VLLM_STABLE_EXT_PATH}
VLLM_PYTHONPATH=${VLLM_PYTHONPATH}
EOF
}

print_config() {
    cat <<EOF
Expert/KV Contention Matrix Configuration
  Model: ${MODEL}
  ShareGPT dataset: ${SHAREGPT_DATASET_PATH}
  GPU: ${GPU_ID}
  NUMA node: ${NUMA_NODE}
  Server: ${BASE_URL}
  R configs: ${R_CONFIGS}
  Request rates: ${REQUEST_RATES}
  gpu_memory_utilization: ${GPU_MEMORY_UTILIZATION}
  dtype: ${DTYPE}
  enforce eager: ${ENFORCE_EAGER}
  prefetch_step: ${PREFETCH_STEP}
  offload_params: ${OFFLOAD_PARAMS}
  num_prompts: ${BENCH_NUM_PROMPTS}
  burstiness: ${BENCH_BURSTINESS}
  max_concurrency: ${BENCH_MAX_CONCURRENCY}
  run dir: ${RUN_DIR}

HBM note:
  gpu_memory_utilization fixes the vLLM memory budget, not the KV/expert split.
  The budget contains model weights, resident expert weights, prefetch buffers,
  KV cache blocks, CUDA graph/runtime workspace, activations, and allocator overhead.
  Changing R changes resident expert weights and prefetch buffers; the KV cache
  size is whatever remains inside the fixed budget.
EOF
}

wait_for_server_ready() {
    local server_log="$1"
    local start_time="$(date +%s)"

    log "Waiting for server readiness: ${BASE_URL}/v1/models"
    while true; do
        if is_server_ready; then
            log "Server is ready."
            return 0
        fi

        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
            log "ERROR: server exited before becoming ready. See ${server_log}"
            return 1
        fi

        if (( "$(date +%s)" - start_time >= TIMEOUT_SECONDS )); then
            log "ERROR: timed out waiting for server. See ${server_log}"
            return 1
        fi

        sleep 2
    done
}

start_server() {
    local r_label="$1"
    local out_dir="$2"
    local trace_path="$3"
    local server_log="${out_dir}/server.log"

    build_common_server_args
    set_offload_args "${r_label}"

    local server_extra_args=()
    split_words server_extra_args "${SERVER_EXTRA_ARGS}"

    local server_args=(
        "${COMMON_SERVER_ARGS[@]}"
        "${SERVER_OFFLOAD_ARGS[@]}"
        "${server_extra_args[@]}"
    )

    printf '%q ' "${VLLM_BIN}" "${server_args[@]}" > "${out_dir}/server_cmd.txt"
    printf '\n' >> "${out_dir}/server_cmd.txt"

    local server_env=(
        CUDA_VISIBLE_DEVICES="${GPU_ID}"
        LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}"
        PYTHONPATH="${VLLM_PYTHONPATH}"
        PYTHONUNBUFFERED=1
        VLLM_EXPERT_KV_CONTENTION_TRACE="${trace_path}"
    )
    if [[ -n "${VLLM_STABLE_EXT_PATH}" ]]; then
        server_env+=(VLLM_STABLE_EXT_PATH="${VLLM_STABLE_EXT_PATH}")
    fi

    log "Starting server for ${r_label}; log=${server_log}"
    setsid env "${server_env[@]}" \
        numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
        stdbuf -oL -eL "${VLLM_BIN}" "${server_args[@]}" \
        > "${server_log}" 2>&1 &
    SERVER_PID="$!"
    printf '%s\n' "${SERVER_PID}" > "${out_dir}/server.pid"

    wait_for_server_ready "${server_log}"
}

run_sharegpt_benchmark() {
    local r_label="$1"
    local request_rate="$2"
    local out_dir="$3"
    local trace_path="$4"
    local bench_log="${out_dir}/bench.log"

    local bench_extra_args=()
    split_words bench_extra_args "${BENCH_EXTRA_ARGS}"

    local bench_args=(
        bench serve
        --backend "${BENCH_BACKEND}"
        --base-url "${BASE_URL}"
        --endpoint "${BENCH_ENDPOINT}"
        --model "${MODEL}"
        --dataset-name sharegpt
        --dataset-path "${SHAREGPT_DATASET_PATH}"
        --num-prompts "${BENCH_NUM_PROMPTS}"
        --request-rate "${request_rate}"
        --burstiness "${BENCH_BURSTINESS}"
        --seed "${BENCH_SEED}"
        --save-result
        --result-dir "${out_dir}"
        --result-filename bench.json
        --metadata
        "run_id=${RUN_ID}"
        "r_label=${r_label}"
        "request_rate=${request_rate}"
        "trace_path=${trace_path}"
    )

    if [[ -n "${BENCH_MAX_CONCURRENCY}" ]]; then
        bench_args+=(--max-concurrency "${BENCH_MAX_CONCURRENCY}")
    fi
    if [[ -n "${SHAREGPT_OUTPUT_LEN}" ]]; then
        bench_args+=(--sharegpt-output-len "${SHAREGPT_OUTPUT_LEN}")
    fi

    bench_args+=("${bench_extra_args[@]}")

    printf '%q ' "${VLLM_BIN}" "${bench_args[@]}" > "${out_dir}/bench_cmd.txt"
    printf '\n' >> "${out_dir}/bench_cmd.txt"

    local bench_env=(
        NO_PROXY="127.0.0.1,localhost,0.0.0.0,${NO_PROXY:-}"
        no_proxy="127.0.0.1,localhost,0.0.0.0,${no_proxy:-}"
        LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}"
        PYTHONPATH="${VLLM_PYTHONPATH}"
        PYTHONUNBUFFERED=1
        VLLM_EXPERT_KV_CONTENTION_TRACE="${trace_path}"
        VLLM_EXPERT_KV_CONTENTION_SUMMARY=1
    )
    if [[ -n "${VLLM_STABLE_EXT_PATH}" ]]; then
        bench_env+=(VLLM_STABLE_EXT_PATH="${VLLM_STABLE_EXT_PATH}")
    fi

    log "Running benchmark ${r_label}, request_rate=${request_rate}; log=${bench_log}"
    env "${bench_env[@]}" \
        stdbuf -oL -eL "${VLLM_BIN}" "${bench_args[@]}" \
        > "${bench_log}" 2>&1
}

run_point() {
    local r_label="$1"
    local request_rate="$2"
    local out_dir="${RUN_DIR}/${r_label}_N${request_rate}"
    local trace_path="${out_dir}/expert_kv_trace.jsonl"
    local result_json="${out_dir}/bench.json"

    CURRENT_STATUS_FILE="${out_dir}/status.txt"

    if [[ "${SKIP_DONE}" == "1" && -s "${result_json}" ]]; then
        log "Skipping completed point ${r_label}, N=${request_rate}: ${result_json} exists"
        return 0
    fi

    mkdir -p "${out_dir}"
    mark_status "running"
    : > "${trace_path}"

    ensure_no_unmanaged_server

    if ! start_server "${r_label}" "${out_dir}" "${trace_path}"; then
        mark_status "server_failed"
        cleanup_current_server
        return 1
    fi

    if ! run_sharegpt_benchmark "${r_label}" "${request_rate}" "${out_dir}" "${trace_path}"; then
        mark_status "bench_failed"
        cleanup_current_server
        return 1
    fi

    cleanup_current_server
    mark_status "done"
    log "Finished ${r_label}, N=${request_rate}. Data: ${out_dir}"
}

main() {
    cd "${ROOT_DIR}"
    check_required_paths
    write_run_config
    print_config

    local r_labels=()
    local request_rates=()
    split_words r_labels "${R_CONFIGS}"
    split_words request_rates "${REQUEST_RATES}"

    for r_label in "${r_labels[@]}"; do
        for request_rate in "${request_rates[@]}"; do
            if ! run_point "${r_label}" "${request_rate}"; then
                log "Point failed: ${r_label}, N=${request_rate}"
                if [[ "${STOP_ON_FAILURE}" == "1" ]]; then
                    log "STOP_ON_FAILURE=1, stopping matrix. Current server has been cleaned up."
                    exit 1
                fi
            fi
        done
    done

    log "All requested matrix points finished. Results are under ${RUN_DIR}"
}

main "$@"
