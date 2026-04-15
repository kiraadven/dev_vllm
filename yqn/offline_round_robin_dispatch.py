# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import re
from time import sleep
import time, csv
from multiprocessing import Event, Process

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_open_port
from vllm.logger import init_logger
from vllm.config import KVTransferConfig

logger = init_logger(__name__)

def create_parser():
    parser = FlexibleArgumentParser(
        description="Data Parallel Inference with Disaggregated Prefill"
    )

    # Add all engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        model="/data/yqn/Qwen1.5-MoE-A2.7B",
        enable_expert_parallel=True,
        enforce_eager=True,
        data_parallel_size=2,
        tensor_parallel_size=1,
    )

    # Add DP-specific args (separate from engine args to avoid conflicts)
    parser.add_argument(
        "--dp-num-nodes",
        type=int,
        default=1,
        help="Total number of nodes for data parallel.",
    )
    parser.add_argument(
        "--dp-node-rank",
        type=int,
        default=0,
        help="Rank of the current node for data parallel.",
    )
    parser.add_argument(
        "--dp-master-addr",
        type=str,
        default="",
        help="Master node IP address for DP coordination.",
    )
    parser.add_argument(
        "--dp-master-port",
        type=int,
        default=0,
        help="Master node port for DP coordination.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Number of seconds before unresponsive process is killed.",
    )

    # PD placement args.
    parser.add_argument(
        "--pd-ratio",
        type=str,
        default="1:1",
        help=(
            "Prefill:Decode ratio used to assign DP ranks automatically, "
            "e.g. 1:1, 1:2, 2/3."
        ),
    )
    # parser.add_argument(
    #     "--prefill-select-strategy",
    #     type=str,
    #     choices=["round_robin"],
    #     default="round_robin",
    #     help=(
    #         "How a decode rank selects its prefill rank. "
    #         "Currently only round_robin is supported."
    #     ),
    # )
    return parser


def parse_pd_ratio(pd_ratio: str) -> tuple[int, int]:
    """Parse a ratio string like '1:2' into (1, 2)."""
    match = re.fullmatch(r"\s*(\d+)\s*[:/xX]\s*(\d+)\s*", pd_ratio)
    if match is None:
        raise ValueError(
            f"Invalid --pd-ratio={pd_ratio!r}. Use formats like 1:1, 1:2, 2/3."
        )

    prefill_weight = int(match.group(1))
    decode_weight = int(match.group(2))
    if prefill_weight <= 0 or decode_weight <= 0:
        raise ValueError(
            f"Invalid --pd-ratio={pd_ratio!r}. Both sides must be > 0."
        )

    return prefill_weight, decode_weight


def build_pd_rank_plan(dp_size: int, pd_ratio: str) -> tuple[list[int], list[int]]:
    """
    Build rank roles from pd ratio.

    Example:
        dp_size=4, pd_ratio=1:1 -> prefill=[0,2], decode=[1,3]
        dp_size=6, pd_ratio=1:2 -> prefill=[0,3], decode=[1,2,4,5]
    """
    if dp_size <= 0:
        raise ValueError(f"dp_size must be > 0, got {dp_size}.")

    prefill_weight, decode_weight = parse_pd_ratio(pd_ratio)
    pattern = ["P"] * prefill_weight + ["D"] * decode_weight

    prefill_ranks: list[int] = []
    decode_ranks: list[int] = []

    for global_rank in range(dp_size):
        if pattern[global_rank % len(pattern)] == "P":
            prefill_ranks.append(global_rank)
        else:
            decode_ranks.append(global_rank)

    prefill_ranks.sort()
    decode_ranks.sort()
    return prefill_ranks, decode_ranks


def select_decode_rank_for_prefill(
    prefill_rank: int,
    decode_ranks: list[int],
    strategy: str = "round_robin",
) -> int:
    if not decode_ranks:
        raise ValueError("decode_ranks is empty; cannot select decode rank.")

    if strategy == "round_robin":
        return decode_ranks[prefill_rank % len(decode_ranks)]

    raise ValueError(f"Unsupported --prefill-select-strategy={strategy!r}")

def run_prefill(prefill_done, prefill_rank, prompts):
    # We use GPU 0 for prefill node.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(prefill_rank)

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Using P2pNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the prefill node (kv_producer, rank 0).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for P2pNcclConnector.
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_producer",
        kv_rank=prefill_rank,
        kv_parallel_size=2,
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.8,
    )

    llm.generate(prompts, sampling_params)
    logger.info("Prefill node is finished for rank %d.", prefill_rank)
    prefill_done.set()

    # To keep the prefill node running in case the decode node is not done;
    # otherwise, the script might exit prematurely, causing incomplete decoding.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script stopped by user.")


def run_decode(prefill_done, decode_rank):
    # We use the specified GPU for the decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(decode_rank)
    sampling_params = SamplingParams(temperature=0, top_p=0.95)

    # Using P2pNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the decode node (kv_consumer, rank 1).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for P2pNcclConnector.
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_consumer",
        kv_rank=decode_rank,
        kv_parallel_size=2,
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.8,
    )

    # Wait for the producer to start the pipe
    logger.info("Waiting for prefill node to finish...,rank %d", decode_rank)
    prefill_done.wait()

    # At this point when the prefill_done is set, the kv-cache should have been
    # transferred to this decode node, so we can start decoding.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def main(
    prompts,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    engine_args,
    p_rank,
    d_rank,
):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # With DP, each rank should process different prompts.
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def start(rank: int) -> int:
        return rank * floor + min(rank, remainder)

    prompts = prompts[start(global_dp_rank):start(global_dp_rank + 1)]
    if len(prompts) == 0:
        prompts = ["Placeholder"]

    logger.info(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

    prefill_done = Event()
    prefill_process = Process(target=run_prefill, args=(prefill_done, p_rank, prompts))
    decode_process = Process(target=run_decode, args=(prefill_done, d_rank))

    # Start prefill node
    prefill_process.start()

    # Start decode node
    decode_process.start()

    # Terminate the prefill node when decode is finished
    decode_process.join()
    prefill_process.terminate()

    # Give engines time to pause their processing loops before exiting.
    sleep(1)


if __name__ == "__main__":
    parser = create_parser()
    args = vars(parser.parse_args())

    # Extract DP-specific args (pop to remove from engine_args)
    dp_size = args.pop("data_parallel_size")
    dp_num_nodes = args.pop("dp_num_nodes")
    dp_node_rank = args.pop("dp_node_rank")
    dp_master_addr = args.pop("dp_master_addr")
    dp_master_port = args.pop("dp_master_port")
    timeout = args.pop("timeout")
    pd_ratio = args.pop("pd_ratio")

    # Remaining args are engine args
    engine_args = args

    if dp_num_nodes == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port_val = get_open_port()
    else:
        dp_master_ip = dp_master_addr
        dp_master_port_val = dp_master_port

    assert dp_size % dp_num_nodes == 0, "dp_size should be divisible by dp_num_nodes"
    dp_per_node = dp_size // dp_num_nodes

    prefill_ranks, decode_ranks = build_pd_rank_plan(dp_size, pd_ratio)
    prefill_to_decode = {
        p_rank: select_decode_rank_for_prefill(
            p_rank,
            decode_ranks,
        )
        for p_rank in prefill_ranks
    }

    logger.info(
        f"PD ratio {pd_ratio} with dp_size={dp_size} => "
        f"prefill_ranks={prefill_ranks}, decode_ranks={decode_ranks}"
    )
    logger.info(f"Prefill->Decode mapping: {prefill_to_decode}")

    def load_trace(csv_path):
      rows = []
      with open(csv_path) as f:
          for r in csv.DictReader(f):
              rows.append((float(r["arrive_time"]), int(r["input_tokens_length"])))
      rows.sort(key=lambda x: x[0])
      t0 = rows[0][0]
      return [(t - t0, n) for t, n in rows]  # 相对时间

    def synth_prompt(n_tokens):
        return "hello " * max(1, n_tokens)

    trace = load_trace("vllm/yqn/traces/sharegpt_x/sharegpt_x_rate0p5.csv")

    from multiprocessing import get_context

    # This launcher spawns DP processes that each create vLLM worker
    # subprocesses. Using "spawn" avoids fork-related CUDA init hangs.
    mp_ctx = get_context("spawn")

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(dp_node_rank * dp_per_node, (dp_node_rank + 1) * dp_per_node)
    ):
        p_rank = prefill_ranks.index(global_dp_rank) if global_dp_rank in prefill_ranks else None
        # FIXME: This selection logic is only for 1:1 ratio. We need a more general way to assign decode ranks to prefill ranks
        d_rank = select_decode_rank_for_prefill(p_rank, decode_ranks) # type: ignore
        proc = mp_ctx.Process(
            target=main,
            args=(
                prompts,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port_val,
                engine_args,
                p_rank,
                d_rank,
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=timeout)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
