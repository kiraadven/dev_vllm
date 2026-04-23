# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import os
import socket
from typing import Iterable

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use torch.distributed + NCCL to verify that rank0 and rank1 "
            "can exchange CUDA tensors between two GPUs."
        )
    )
    parser.add_argument(
        "--tensor-size",
        type=int,
        default=16,
        help="Number of elements in the CUDA tensor used for communication.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "int64"),
        default="float32",
        help="Tensor dtype used for send/recv and collective checks.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Process group timeout in seconds.",
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def log(message: str) -> None:
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(
        f"{now()} | rank={rank} | local_rank={local_rank} | {message}",
        flush=True,
    )


def tensor_preview(tensor: torch.Tensor, limit: int = 8) -> str:
    flat = tensor.detach().flatten()
    preview = flat[:limit].cpu().tolist()
    return (
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, "
        f"numel={tensor.numel()}, sum={flat.sum().item()}, preview={preview}"
    )


def build_test_tensor(
    *,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    tensor_size: int,
) -> torch.Tensor:
    base = torch.arange(tensor_size, device=device, dtype=dtype)
    if dtype.is_floating_point:
        return base + float(rank * 1000)
    return base + rank * 1000


def require_env(names: Iterable[str]) -> None:
    missing = [name for name in names if name not in os.environ]
    if missing:
        raise RuntimeError(
            "Missing torchrun environment variables: "
            + ", ".join(sorted(missing))
        )


def verify_environment() -> tuple[int, int, int, torch.device]:
    require_env(("RANK", "LOCAL_RANK", "WORLD_SIZE"))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False. "
            "This script only validates GPU-to-GPU NCCL communication."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size != 2:
        raise RuntimeError(
            f"This script expects exactly 2 ranks, but WORLD_SIZE={world_size}."
        )

    visible_count = torch.cuda.device_count()
    if visible_count < 2:
        raise RuntimeError(
            f"Need at least 2 visible CUDA devices, but found {visible_count}."
        )

    if local_rank >= visible_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is out of range for {visible_count} visible "
            "CUDA devices."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    log(
        "environment ready: "
        f"hostname={socket.gethostname()}, pid={os.getpid()}, "
        f"world_size={world_size}, visible_cuda_devices={visible_count}, "
        f"current_device={torch.cuda.current_device()}, "
        f"device_name={torch.cuda.get_device_name(device)}"
    )
    log(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    return rank, local_rank, world_size, device


def verify_send_recv(
    *,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    tensor_size: int,
) -> None:
    send_src = 0
    send_dst = 1

    if rank not in (send_src, send_dst):
        raise RuntimeError(f"Unexpected rank {rank}; expected only 0 or 1.")

    source_tensor = build_test_tensor(
        rank=send_src,
        device=device,
        dtype=dtype,
        tensor_size=tensor_size,
    )

    if rank == send_src:
        tensor_to_send = source_tensor.clone()
        log(
            "rank0 prepared outbound CUDA tensor for rank1: "
            + tensor_preview(tensor_to_send)
        )
        log("rank0 launching NCCL isend to rank1")
        req = dist.isend(tensor_to_send, dst=send_dst)
        log("rank0 waiting for outbound isend completion")
        req.wait()
        torch.cuda.synchronize(device)
        log("rank0 outbound isend completed")
    else:
        tensor_to_recv = torch.empty(
            tensor_size,
            device=device,
            dtype=dtype,
        )
        log(f"rank1 prepared empty receive buffer on {device} for rank0 tensor")
        log("rank1 launching NCCL irecv from rank0")
        req = dist.irecv(tensor_to_recv, src=send_src)
        log("rank1 waiting for inbound irecv completion")
        req.wait()
        torch.cuda.synchronize(device)
        log("rank1 received tensor from rank0: " + tensor_preview(tensor_to_recv))
        expected = source_tensor
        torch.testing.assert_close(tensor_to_recv, expected)
        log("rank1 receive content matches rank0 source tensor")

    dist.barrier()
    log("one-way send/recv phase passed on all ranks")


def main() -> int:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    if not dtype.is_floating_point and dtype not in (torch.int64,):
        raise RuntimeError(f"Unsupported dtype: {dtype}")

    rank, local_rank, world_size, device = verify_environment()

    log(
        "initializing process group with backend=nccl, "
        f"timeout={args.timeout_seconds}s"
    )
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=args.timeout_seconds),
    )

    try:
        log(
            "process group initialized successfully: "
            f"rank={rank}, local_rank={local_rank}, world_size={world_size}, "
            f"device={device}, dtype={dtype}, tensor_size={args.tensor_size}"
        )
        verify_send_recv(
            rank=rank,
            device=device,
            dtype=dtype,
            tensor_size=args.tensor_size,
        )

        if rank == 0:
            log("SUCCESS: rank1 received the CUDA tensor from rank0")
        return 0
    finally:
        if dist.is_initialized():
            log("destroying process group")
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise
