"""
Entry point: analyze token-length distributions and generate traces
for ShareGPT-GPT4, ShareGPT-X, LMSYS-Chat-1M, and LEval.

Usage
─────
    # Run all datasets
    python trace_generated.py

    # Run a single dataset
    python trace_generated.py --dataset sharegpt_gpt4
    python trace_generated.py --dataset sharegpt_x
    python trace_generated.py --dataset lmsys_chat_1m
    python trace_generated.py --dataset leval

    # Skip the analysis step (only generate traces)
    python trace_generated.py --no-analyze

    # Skip the trace generation step (only analyze)
    python trace_generated.py --no-traces

Output layout (all relative to experiments/):
    figures/<dataset>/length_distribution.{pdf,png}
    figures/<dataset>/arrival_rate_*.{pdf,png}        ← req/s + tok/s plots
    stats/<dataset>/distribution_stats.json
    traces/<dataset>/<dataset>_rate<X>p<Y>.csv        ← sharegpt_gpt4, lmsys, leval
    traces/sharegpt_x/sharegpt_x_real.csv             ← real-timestamp trace
"""

from __future__ import annotations

import argparse
from pathlib import Path

# from leval_dataset import LEvalDataset
# from lmsys_chat_dataset import LMSYSChatDataset
# from sharegpt_dataset import ShareGPTDataset
from sharegpt_x_dataset import ShareGPTXDataset

# ── Output root ───────────────────────────────────────────────────────────────
OUTPUT_ROOT = Path(__file__).parent

# ── Poisson trace settings (ShareGPT-GPT4 and LMSYS-Chat-1M) ─────────────────
N_REQUESTS    = 1000
RANDOM_SEED   = 42
POISSON_RATES = [0.5, 1.0, 1.5, 2.0, 2.5]   # req/s — one CSV per rate

# ── ShareGPT-X token controls ────────────────────────────────────────────────
MAX_INPUT_TOKENS  = 32_768
MAX_OUTPUT_TOKENS = 32_768
IO_DIVISOR        = 10      # divide input/output token lengths by this factor

# ── Dataset registry ──────────────────────────────────────────────────────────
#   key            → (class, extra kwargs for generate_traces)
_REGISTRY = {
    # "sharegpt_gpt4": (
    #     ShareGPTDataset,
    #     {},                                     # uses Poisson traces
    # ),
    "sharegpt_x": (
        ShareGPTXDataset,
        {},                                     # uses real arrival times
    ),
    # "lmsys_chat_1m": (
    #     LMSYSChatDataset,
    #     {},                                     # uses Poisson traces
    # ),
    # "leval": (
    #     LEvalDataset,
    #     {},                                     # uses Poisson traces
    # ),
}


def _build_dataset(
    name: str,
    max_input_tokens: int,
    max_output_tokens: int,
    io_divisor: int,
):
    cls, _ = _REGISTRY[name]
    return cls(
        output_root=OUTPUT_ROOT,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        length_divisor=io_divisor,
    )


def _run(
    name: str,
    do_analyze: bool,
    do_traces: bool,
    max_input_tokens: int,
    max_output_tokens: int,
    io_divisor: int,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Dataset : {name}")
    print(f"{'=' * 60}")

    ds = _build_dataset(
        name,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        io_divisor=io_divisor,
    )

    if do_analyze:
        ds.analyze()

    if do_traces:
        ds.generate_traces(
            n_requests=N_REQUESTS,
            seed=RANDOM_SEED,
            rates=POISSON_RATES,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze datasets and generate LLM serving traces."
    )
    parser.add_argument(
        "--dataset",
        choices=list(_REGISTRY.keys()),
        default=None,
        help="Run a single dataset (default: run all).",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip the distribution analysis + figures step.",
    )
    parser.add_argument(
        "--no-traces",
        action="store_true",
        help="Skip the trace generation step.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=MAX_INPUT_TOKENS,
        help="Truncate input text to this many tokens before counting.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help="Truncate output text to this many tokens before counting.",
    )
    parser.add_argument(
        "--io-divisor",
        type=int,
        default=IO_DIVISOR,
        help="Divide both input/output token lengths by this value (ceil).",
    )
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset else list(_REGISTRY.keys())
    do_analyze = not args.no_analyze
    do_traces  = not args.no_traces

    for name in targets:
        _run(
            name,
            do_analyze,
            do_traces,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            io_divisor=args.io_divisor,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()