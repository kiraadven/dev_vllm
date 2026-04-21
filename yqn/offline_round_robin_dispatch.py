# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Input:
- Trace CSV(s): arrive_time,input_tokens_length,output_tokens_length (offline mode ignores arrive_time)
- Model path (--model) with config.json; max_model_len is auto-read from config.json
- Device/PD args (--data-parallel-size, --pd-ratio, ...)

Output:
- Per-trace assignment csv: <trace-output-root>/<trace_name>/dispatch_requests.csv
- Optional one nsys report per trace: <nsys-output>_<trace_name>.nsys-rep (+ sqlite if enabled)

Usage:
- Run all traces: /root/vllm/.venv/bin/python offline_round_robin_dispatch.py --data-parallel-size 4 --pd-ratio 1:1
- Run one trace:  /root/vllm/.venv/bin/python offline_round_robin_dispatch.py --trace-file traces/sharegpt_x/sharegpt_x_rate1p0.csv
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

from vllm import EngineArgs, LLM, SamplingParams
from vllm.config import KVTransferConfig, ProfilerConfig
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_ip, get_open_port

logger = init_logger(__name__)


@dataclass(frozen=True)
class TraceRequest:
    """One request row from trace csv."""

    trace_name: str
    request_id: int
    input_tokens: int
    output_tokens: int
    in_str: str
    out_str: str


@dataclass(frozen=True)
class CommonRunConfig:
    """Shared runtime config passed to worker processes."""

    timeout: int
    model: str
    max_model_len: int
    gpu_memory_utilization: float
    enable_expert_parallel: bool
    enforce_eager: bool


def create_parser() -> FlexibleArgumentParser:
    """Create CLI parser for offline trace execution and optional nsys mode."""
    parser = FlexibleArgumentParser(description="Offline disaggregated prefill/decode")
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        model="/data/yqn/Qwen1.5-MoE-A2.7B",
        enable_expert_parallel=True,
        enforce_eager=True,
        data_parallel_size=2,
        tensor_parallel_size=1,
    )

    for name, t, default in (
        ("--timeout", int, 600),
        ("--pd-ratio", str, "1:1"),
        ("--trace-dir", str, "/root/vllm/yqn/traces/sharegpt_x"),
        ("--trace-glob", str, "sharegpt_x_rate*.csv"),
        ("--trace-file", str, ""),
        ("--trace-output-root", str, "stats/offline_round_robin/by_trace"),
        ("--nsys-profile-output", str, ""),
    ):
        parser.add_argument(name, type=t, default=default)

    parser.add_argument("--nsys-export-sqlite", action="store_true")
    return parser


def build_pd_rank_plan(dp_size: int, pd_ratio: str) -> tuple[list[int], list[int]]:
    """Convert pd_ratio to prefill/decode GPU index lists."""
    m = re.fullmatch(r"\s*(\d+)\s*[:/xX]\s*(\d+)\s*", pd_ratio)
    if not m:
        raise ValueError(f"Invalid --pd-ratio={pd_ratio!r}")
    pw, dw = int(m.group(1)), int(m.group(2))
    if pw <= 0 or dw <= 0 or dp_size <= 0:
        raise ValueError("pd_ratio and dp_size must be positive")

    pattern = ["P"] * pw + ["D"] * dw
    prefill, decode = [], []
    for rank in range(dp_size):
        (prefill if pattern[rank % len(pattern)] == "P" else decode).append(rank)
    return prefill, decode


def build_1p1d_pairs(
    prefill_gpus: list[int], decode_gpus: list[int]
) -> list[tuple[int, int]]:
    """Build one-to-one prefill/decode GPU pairs."""
    if len(prefill_gpus) != len(decode_gpus):
        raise ValueError(
            "This script runs independent 1P1D pairs, so prefill/decode counts "
            f"must match. Got prefill={len(prefill_gpus)} decode={len(decode_gpus)}. "
            "Use --pd-ratio 1:1 with an even --data-parallel-size."
        )
    return list(zip(prefill_gpus, decode_gpus, strict=True))


def allocate_pair_ports(pair_count: int) -> list[tuple[int, int]]:
    """Allocate dedicated (prefill_port, decode_port) for each pair."""
    if pair_count <= 0:
        return []

    used: set[int] = set()
    ports: list[tuple[int, int]] = []
    for _ in range(pair_count):
        prefill_port = get_open_port()
        while prefill_port in used:
            prefill_port = get_open_port()
        used.add(prefill_port)

        decode_port = get_open_port()
        while decode_port in used:
            decode_port = get_open_port()
        used.add(decode_port)

        ports.append((prefill_port, decode_port))
    return ports


def resolve_max_model_len_from_config(model: str) -> int:
    """Read max_model_len from <model>/config.json."""
    cfg_path = Path(model).expanduser() / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Cannot find config.json: {cfg_path}")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    candidate_keys = (
            "max_model_len",
            "model_max_length",
            "max_position_embeddings",
            "seq_length",
            "n_positions",
            "n_ctx",
    )
    for key in candidate_keys:
        val = cfg.get(key)
        if isinstance(val, int) and 0 < val < 10_000_000:
            return val
        if isinstance(val, float) and val.is_integer() and 0 < val < 10_000_000:
            return int(val)
    raise ValueError(f"No max length key in {cfg_path}")


def load_trace_requests(trace_path: Path) -> list[TraceRequest]:
    """Load one trace csv in original row order (no sorting)."""
    requests: list[TraceRequest] = []
    with open(trace_path, newline="", encoding="utf-8") as f:
        for request_id, row in enumerate(csv.DictReader(f)):
            in_tok = int(row["input_tokens_length"])
            out_tok = int(row["output_tokens_length"])
            requests.append(
                TraceRequest(
                    trace_name=trace_path.stem,
                    request_id=request_id,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    in_str="hi" * max(1, in_tok),
                    out_str="hi" * max(1, out_tok),
                )
            )
    if not requests:
        raise ValueError(f"Trace is empty: {trace_path}")
    return requests


def dispatch_round_robin(
    requests: list[TraceRequest], prefill_gpus: list[int]
) -> dict[int, list[TraceRequest]]:
    """Round-robin assign requests to prefill GPUs."""
    if not prefill_gpus:
        raise ValueError("prefill_gpus is empty")
    assigned = {gpu: [] for gpu in prefill_gpus}
    for i, req in enumerate(requests):
        assigned[prefill_gpus[i % len(prefill_gpus)]].append(req)
    return assigned


def write_dispatch_csv(
    trace_output_root: Path,
    trace_name: str,
    assignments: dict[int, list[TraceRequest]],
) -> None:
    """Persist per-trace request assignment result to csv."""
    out_dir = trace_output_root / trace_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "dispatch_requests.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trace_name",
                "prefill_rank",
                "request_id",
                "input_tokens",
                "output_tokens",
            ],
        )
        writer.writeheader()
        for prefill_gpu, reqs in assignments.items():
            for req in reqs:
                writer.writerow(
                    {
                        "trace_name": req.trace_name,
                        "prefill_rank": prefill_gpu,
                        "request_id": req.request_id,
                        "input_tokens": req.input_tokens,
                        "output_tokens": req.output_tokens,
                    }
                )


def set_worker_env(visible_device: int) -> None:
    """Apply per-worker process environment."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)


def generate_with_request_ids(
    llm: LLM,
    prompts: list[str],
    request_ids: list[str],
    sampling_params: SamplingParams,
) -> None:
    """Submit prompts with explicit request ids and drive engine until done."""
    if len(prompts) != len(request_ids):
        raise ValueError("prompts and request_ids must have the same length")

    for prompt, request_id in zip(prompts, request_ids, strict=True):
        llm.llm_engine.add_request(request_id, prompt, sampling_params)

    while llm.llm_engine.has_unfinished_requests():
        llm.llm_engine.step()


def run_prefill_worker(
    prefill_ready,
    shutdown_event,
    prompts: list[str],
    request_ids: list[str],
    prefill_gpu: int,
    kv_port: int,
    pair_idx: int,
    cfg: CommonRunConfig,
) -> None:
    """Prefill process: run all prompts once, signal ready, wait shutdown event."""
    try:
        set_worker_env(prefill_gpu)
        llm = LLM(
            model=cfg.model,
            kv_transfer_config=KVTransferConfig(
                kv_connector="P2pNcclConnector",
                kv_role="kv_producer",
                kv_rank=0,
                kv_parallel_size=2,
                kv_port=kv_port,
            ),
            profiler_config=ProfilerConfig(
                profiler="cuda",
                delay_iterations=3,
                max_iterations=10,
            ),
            enable_layerwise_nvtx_tracing=True,
            enable_logging_iteration_details=True,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enable_expert_parallel=cfg.enable_expert_parallel,
            enforce_eager=cfg.enforce_eager,
            data_parallel_size=1,
            tensor_parallel_size=1,
        )
        generate_with_request_ids(
            llm,
            prompts,
            request_ids,
            SamplingParams(temperature=0, top_p=0.95, max_tokens=1),
        )
        prefill_ready.set()
        shutdown_event.wait()
    except Exception:
        logger.error(
            "prefill pair=%d gpu=%d failed\n%s",
            pair_idx,
            prefill_gpu,
            traceback.format_exc(),
        )
        prefill_ready.set()


def run_decode_worker(
    prefill_ready,
    prompts: list[str],
    request_ids: list[str],
    decode_gpu: int,
    kv_port: int,
    pair_idx: int,
    cfg: CommonRunConfig,
) -> None:
    """Decode process: initialize consumer, wait prefill, then decode."""
    set_worker_env(decode_gpu)
    llm = LLM(
        model=cfg.model,
        kv_transfer_config=KVTransferConfig(
            kv_connector="P2pNcclConnector",
            kv_role="kv_consumer",
            kv_rank=1,
            kv_parallel_size=2,
            kv_port=kv_port,
        ),
        profiler_config=ProfilerConfig(
            profiler="cuda",
            delay_iterations=3,
            max_iterations=10,
        ),
        enable_layerwise_nvtx_tracing=True,
        enable_logging_iteration_details=True,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enable_expert_parallel=cfg.enable_expert_parallel,
        enforce_eager=cfg.enforce_eager,
        data_parallel_size=1,
        tensor_parallel_size=1,
    )
    if not prefill_ready.wait(timeout=cfg.timeout):
        raise TimeoutError(
            f"decode pair {pair_idx} (gpu {decode_gpu}) timed out waiting prefill"
        )
    generate_with_request_ids(
        llm,
        prompts,
        request_ids,
        SamplingParams(temperature=0, top_p=0.95),
    )


def run_single_trace(
    requests: list[TraceRequest],
    pair_gpus: list[tuple[int, int]],
    pair_ports: list[tuple[int, int]],
    cfg: CommonRunConfig,
    trace_output_root: Path,
) -> int:
    """Run one trace end-to-end; after decode ends, shut down all prefill instances."""
    prefill_gpus = [prefill_gpu for prefill_gpu, _ in pair_gpus]
    assignments = dispatch_round_robin(requests, prefill_gpus)
    write_dispatch_csv(trace_output_root, requests[0].trace_name, assignments)

    mp_ctx = get_context("spawn")
    pairs: list[dict[str, object]] = []
    local_ip = get_ip()

    for pair_idx, ((prefill_gpu, decode_gpu), (prefill_port, decode_port)) in enumerate(
        zip(pair_gpus, pair_ports, strict=True)
    ):
        pair_reqs = assignments[prefill_gpu]
        in_str = [r.in_str for r in pair_reqs] or ["Placeholder"]
        out_str = [r.out_str for r in pair_reqs] or ["Placeholder"]
        request_ids = [
            (
                f"___prefill_addr_{local_ip}:{prefill_port}"
                f"___decode_addr_{local_ip}:{decode_port}_{r.request_id}"
            )
            for r in pair_reqs
        ] or [
            (
                f"___prefill_addr_{local_ip}:{prefill_port}"
                f"___decode_addr_{local_ip}:{decode_port}_placeholder"
            )
        ]

        prefill_ready = mp_ctx.Event()
        shutdown_event = mp_ctx.Event()
        prefill_proc = mp_ctx.Process(
            target=run_prefill_worker,
            args=(
                prefill_ready,
                shutdown_event,
                in_str,
                request_ids,
                prefill_gpu,
                prefill_port,
                pair_idx,
                cfg,
            ),
        )
        decode_proc = mp_ctx.Process(
            target=run_decode_worker,
            args=(
                prefill_ready,
                out_str,
                request_ids,
                decode_gpu,
                decode_port,
                pair_idx,
                cfg,
            ),
        )

        prefill_proc.start()
        decode_proc.start()
        pairs.append(
            {
                "pair_idx": pair_idx,
                "prefill_gpu": prefill_gpu,
                "decode_gpu": decode_gpu,
                "prefill_proc": prefill_proc,
                "decode_proc": decode_proc,
                "shutdown_event": shutdown_event,
            }
        )

    exit_code = 0
    for pair in pairs:
        decode_proc = pair["decode_proc"]
        decode_proc.join(timeout=cfg.timeout)  # type: ignore[attr-defined]
        if decode_proc.exitcode is None:  # type: ignore[attr-defined]
            logger.error(
                "decode timeout pair=%d p_gpu=%d->d_gpu=%d",
                pair["pair_idx"],
                pair["prefill_gpu"],
                pair["decode_gpu"],
            )
            decode_proc.kill()  # type: ignore[attr-defined]
            exit_code = 1
        elif decode_proc.exitcode != 0:  # type: ignore[attr-defined]
            logger.error(
                "decode failed pair=%d p_gpu=%d->d_gpu=%d code=%s",
                pair["pair_idx"],
                pair["prefill_gpu"],
                pair["decode_gpu"],
                decode_proc.exitcode,
            )
            exit_code = int(decode_proc.exitcode)  # type: ignore[arg-type]

        pair["shutdown_event"].set()  # type: ignore[attr-defined]
        prefill_proc = pair["prefill_proc"]
        prefill_proc.join(timeout=5)  # type: ignore[attr-defined]
        if prefill_proc.is_alive():  # type: ignore[attr-defined]
            prefill_proc.terminate()  # type: ignore[attr-defined]
            prefill_proc.join(timeout=5)  # type: ignore[attr-defined]

    return exit_code


def launch_nsys_for_trace(trace_path: Path, output_base: Path, export_sqlite: bool) -> int:
    """Launch one child run under nsys for a single trace and optionally export sqlite."""
    per_trace_base = output_base.parent / f"{output_base.name}_{trace_path.stem}"
    per_trace_base.parent.mkdir(parents=True, exist_ok=True)

    child_argv: list[str] = []
    skip_next = False
    for token in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in {"--nsys-profile-output", "--trace-file"}:
            skip_next = True
            continue
        if token == "--nsys-export-sqlite":
            continue
        child_argv.append(token)
    child_argv.extend(["--trace-file", str(trace_path), "--nsys-profile-output", ""])

    cmd = [
        "nsys",
        "profile",
        "-t",
        "cuda,osrt",
        "--force-overwrite",
        "true",
        "-o",
        str(per_trace_base),
        sys.executable,
        sys.argv[0],
        *child_argv,
    ]
    env = os.environ.copy()
    env["VLLM_RR_NSYS_CHILD"] = "1"
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        return rc

    if export_sqlite:
        rep_path = f"{per_trace_base}.nsys-rep"
        export_cmd = [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite",
            "true",
            "-o",
            str(per_trace_base),
            rep_path,
        ]
        rc = subprocess.run(export_cmd, env=env).returncode
    return rc


def main() -> int:
    """Entry point: parse args, optional nsys mode, then run traces sequentially."""
    parser = create_parser()
    args = vars(parser.parse_args())

    device_count = int(args.pop("data_parallel_size"))
    timeout = int(args.pop("timeout"))
    pd_ratio = str(args.pop("pd_ratio"))

    trace_dir = Path(str(args.pop("trace_dir")))
    trace_glob = str(args.pop("trace_glob"))
    trace_file = str(args.pop("trace_file"))
    trace_output_root = Path(str(args.pop("trace_output_root")))
    nsys_profile_output = str(args.pop("nsys_profile_output"))
    nsys_export_sqlite = bool(args.pop("nsys_export_sqlite"))

    prefill_gpus, decode_gpus = build_pd_rank_plan(device_count, pd_ratio)
    pair_gpus = build_1p1d_pairs(prefill_gpus, decode_gpus)

    trace_paths = [Path(trace_file)] if trace_file else list(trace_dir.glob(trace_glob))
    if not trace_paths:
        raise FileNotFoundError(
            "No trace files found from "
            f"trace_dir={trace_dir} trace_glob={trace_glob} trace_file={trace_file}"
        )

    if nsys_profile_output and os.environ.get("VLLM_RR_NSYS_CHILD") != "1":
        out_base = Path(nsys_profile_output)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        rc = 0
        for trace_path in trace_paths:
            rc = launch_nsys_for_trace(trace_path, out_base, nsys_export_sqlite)
            if rc != 0:
                break
        return rc

    model = str(args.get("model", "/data/yqn/Qwen1.5-MoE-A2.7B"))
    cfg = CommonRunConfig(
        timeout=timeout,
        model=model,
        max_model_len=resolve_max_model_len_from_config(model),
        gpu_memory_utilization=float(args.get("gpu_memory_utilization", 0.8) or 0.8),
        enable_expert_parallel=bool(args.get("enable_expert_parallel", True)),
        enforce_eager=bool(args.get("enforce_eager", True)),
    )

    logger.info(
        "model=%s max_model_len=%d pd_ratio=%s pair_gpus=%s",
        cfg.model,
        cfg.max_model_len,
        pd_ratio,
        pair_gpus,
    )

    final_exit_code = 0
    for trace_path in trace_paths:
        pair_ports = allocate_pair_ports(len(pair_gpus))
        requests = load_trace_requests(trace_path)
        logger.info(
            "start trace=%s reqs=%d pair_ports=%s",
            trace_path.stem,
            len(requests),
            pair_ports,
        )
        rc = run_single_trace(requests, pair_gpus, pair_ports, cfg, trace_output_root)
        if rc != 0:
            final_exit_code = rc
            break
        logger.info("done trace=%s; all instances shutdown", trace_path.stem)

    return final_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
