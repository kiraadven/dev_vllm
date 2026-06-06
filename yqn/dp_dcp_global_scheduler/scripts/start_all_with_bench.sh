#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PROJ_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-$ROOT_DIR/.venv/bin/vllm}"
MODEL="${MODEL:-/data/yqn/Qwen1.5-MoE-A2.7B}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"

SCHEDULER_PORT_FRONT="${SCHEDULER_PORT_FRONT:-5570}"
SCHEDULER_PORT_PULL="${SCHEDULER_PORT_PULL:-5571}"
SCHEDULER_PORT_PUB="${SCHEDULER_PORT_PUB:-5572}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORTS="${DECODE_PORTS:-8201,8202,8203}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPUS="${DECODE_GPUS:-1,2,3}"

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

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs_global_scheduler}"
export NO_PROXY="127.0.0.1,localhost,0.0.0.0,${NO_PROXY:-}"
export no_proxy="${NO_PROXY}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

declare -a PIDS=()
declare -a DECODE_GPU_ARRAY=()
declare -a DECODE_PORT_ARRAY=()

parse_csv_array() {
    local -n target_array="$1"
    local raw="$2"
    IFS=',' read -r -a target_array <<< "$raw"
}

check_required_files() {
    local missing=0
    for path in \
        "$PYTHON_BIN" \
        "$VLLM_BIN" \
        "$SCRIPT_DIR/start_scheduler.sh" \
        "$SCRIPT_DIR/start_prefill.sh" \
        "$SCRIPT_DIR/start_decoders.sh"; do
        if [[ ! -e "$path" ]]; then
            echo "Missing required path: $path"
            missing=1
        fi
    done
    if [[ "$missing" -ne 0 ]]; then
        exit 1
    fi
}

validate_array_sizes() {
    if [[ "${#DECODE_GPU_ARRAY[@]}" -ne 3 ]]; then
        echo "DECODE_GPUS must contain exactly 3 GPUs."
        exit 1
    fi
    if [[ "${#DECODE_PORT_ARRAY[@]}" -ne 3 ]]; then
        echo "DECODE_PORTS must contain exactly 3 ports."
        exit 1
    fi
}

validate_benchmark_args() {
    if [[ "$RUN_BENCHMARK" == "1" && ! -f "$SHAREGPT_DATASET_PATH" ]]; then
        echo "ShareGPT dataset file does not exist: $SHAREGPT_DATASET_PATH"
        exit 1
    fi
}

launch_in_new_session() {
    local log_file="$1"
    shift
    "$@" >"$log_file" 2>&1 &
    PIDS+=("$!")
}

wait_for_http() {
    local url="$1"
    local label="$2"
    local start_time

    start_time="$(date +%s)"
    echo "Waiting for $label: $url"
    while true; do
        if curl --noproxy '*' --silent --output /dev/null --fail "$url"; then
            echo "$label is ready."
            return 0
        fi
        if (( "$(date +%s)" - start_time >= TIMEOUT_SECONDS )); then
            echo "Timed out waiting for $label: $url"
            return 1
        fi
        sleep 1
    done
}

cleanup() {
    trap - EXIT INT TERM
    echo "Cleaning up processes..."
    for pid in "${PIDS[@]:-}"; do
        kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
        kill -TERM "${pid}" >/dev/null 2>&1 || true
    done
    sleep 3
    for pid in "${PIDS[@]:-}"; do
        kill -KILL -- "-${pid}" >/dev/null 2>&1 || true
        kill -KILL "${pid}" >/dev/null 2>&1 || true
        wait "${pid}" >/dev/null 2>&1 || true
    done
}

print_config() {
    cat <<EOF
DP-DCP Global Scheduler Launcher
  Model: ${MODEL}
  Scheduler front addr: tcp://127.0.0.1:${SCHEDULER_PORT_FRONT}
  Scheduler back pull addr: tcp://127.0.0.1:${SCHEDULER_PORT_PULL}
  Scheduler back pub addr: tcp://127.0.0.1:${SCHEDULER_PORT_PUB}
  Prefill GPU: ${PREFILL_GPU}
  Prefill API port: ${PREFILL_PORT}
  Decode GPUs: ${DECODE_GPUS}
  Decode API ports: ${DECODE_PORTS}
  Benchmark enabled: ${RUN_BENCHMARK}
  ShareGPT dataset: ${SHAREGPT_DATASET_PATH}
  Logs: ${LOG_DIR}
EOF
}

launch_scheduler() {
    echo "Starting global scheduler..."
    launch_in_new_session "${LOG_DIR}/scheduler.log" \
        env \
        PYTHON_BIN="${PYTHON_BIN}" \
        ENGINES=4 \
        ROLES="prefill,decode,decode,decode" \
        "${SCRIPT_DIR}/start_scheduler.sh"
    sleep 3
}

launch_prefill() {
    echo "Starting prefill server..."
    launch_in_new_session "${LOG_DIR}/prefill.log" \
        env \
        PYTHONPATH="${PYTHONPATH}" \
        VLLM_BIN="${VLLM_BIN}" \
        MODEL="${MODEL}" \
        PREFILL_GPU="${PREFILL_GPU}" \
        PREFILL_PORT="${PREFILL_PORT}" \
        GLOBAL_SCHED_PULL_ADDR="tcp://127.0.0.1:${SCHEDULER_PORT_PULL}" \
        GLOBAL_SCHED_PUB_ADDR="tcp://127.0.0.1:${SCHEDULER_PORT_PUB}" \
        DP_ATTN_WORLD_SIZE=4 \
        "${SCRIPT_DIR}/start_prefill.sh"
    wait_for_http "http://127.0.0.1:${PREFILL_PORT}/health" "prefill:${PREFILL_PORT}"
}

launch_decodes() {
    local port=""

    echo "Starting decode servers..."
    launch_in_new_session "${LOG_DIR}/decoders.log" \
        env \
        PYTHONPATH="${PYTHONPATH}" \
        VLLM_BIN="${VLLM_BIN}" \
        MODEL="${MODEL}" \
        GLOBAL_SCHED_PULL_ADDR="tcp://127.0.0.1:${SCHEDULER_PORT_PULL}" \
        GLOBAL_SCHED_PUB_ADDR="tcp://127.0.0.1:${SCHEDULER_PORT_PUB}" \
        DP_ATTN_WORLD_SIZE=4 \
        "${SCRIPT_DIR}/start_decoders.sh"

    for port in "${DECODE_PORT_ARRAY[@]}"; do
        wait_for_http "http://127.0.0.1:${port}/health" "decode:${port}"
    done
}

run_sharegpt_benchmark() {
    local -a bench_args=(
        serve
        --backend "${BENCH_BACKEND}"
        --base-url "http://127.0.0.1:${PREFILL_PORT}"
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

    echo "Starting ShareGPT benchmark against prefill entrypoint..."
    "${VLLM_BIN}" bench "${bench_args[@]}" 2>&1 | tee "${LOG_DIR}/benchmark.log"
}

main() {
    trap cleanup EXIT INT TERM
    mkdir -p "$LOG_DIR"

    check_required_files
    parse_csv_array DECODE_GPU_ARRAY "${DECODE_GPUS}"
    parse_csv_array DECODE_PORT_ARRAY "${DECODE_PORTS}"
    validate_array_sizes
    validate_benchmark_args
    print_config

    launch_scheduler
    launch_prefill
    launch_decodes

    echo "All scheduler/prefill/decode servers are ready."
    echo "Prefill chat endpoint: http://127.0.0.1:${PREFILL_PORT}/v1/chat/completions"
    echo "Prefill completions endpoint: http://127.0.0.1:${PREFILL_PORT}/v1/completions"

    if [[ "${RUN_BENCHMARK}" == "1" ]]; then
        run_sharegpt_benchmark
    else
        while true; do
            sleep 60
        done
    fi
}

main "$@"
