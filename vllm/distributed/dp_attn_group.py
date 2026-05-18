# Cross-DP attention group.
#
# A NCCL communicator that spans the per-DP-rank GPUs participating in the
# DP-DCP scheme (one GPU per DP rank in this fork). It is conceptually the
# same shape as `_DCP` (decode context parallel) but built across DP ranks
# instead of inside a TP shard. Used by the cross-DP flash attention path
# (yqn/dp_dcp_global_scheduler/src/flash_attn_dp_dcp.py) to all_gather Q
# and to combine partial attention + LSE.
#
# This file is intentionally separate from parallel_state.py so the fork
# does not have to touch upstream init plumbing. Bring up the group from
# the engine init hook (see engine_integration.py), after vLLM's own
# distributed init has run.

from __future__ import annotations

import os

import torch.distributed as dist

from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_world_group,
    init_model_parallel_group,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_DP_ATTN: GroupCoordinator | None = None
_DP_ATTN_LOCAL_RANK: int | None = None
_DP_ATTN_WORLD_SIZE: int | None = None


def init_dp_attn_group(
    dp_world_size: int,
    backend: str | None = None,
) -> GroupCoordinator:
    """Build a NCCL group across the *dp_world_size* GPUs of this DP-DCP unit.

    Layout assumption: each DP rank is a single-GPU EngineCore (no TP).
    The N participating EngineCores form one contiguous block of global
    ranks [0, dp_world_size). For multi-unit deployments (e.g. one DP-DCP
    unit per NUMA half), call this once per unit with its own torch.dist
    process group already initialized.

    Idempotent: returns the existing group if already initialized.
    """
    global _DP_ATTN, _DP_ATTN_LOCAL_RANK, _DP_ATTN_WORLD_SIZE

    if _DP_ATTN is not None:
        return _DP_ATTN

    if not dist.is_initialized():
        raise RuntimeError(
            "init_dp_attn_group requires torch.distributed to be initialized"
        )

    backend = backend or dist.get_backend()
    world = get_world_group()
    if world.world_size < dp_world_size:
        raise ValueError(
            f"dp_world_size={dp_world_size} exceeds world_size={world.world_size}"
        )

    # Single ring spanning ranks [0, dp_world_size). Override via env if a
    # different mapping is needed (e.g. NUMA-disjoint pools).
    ranks_str = os.environ.get("DP_ATTN_RANKS")
    if ranks_str:
        ranks = [int(x) for x in ranks_str.split(",") if x]
    else:
        ranks = list(range(dp_world_size))

    if world.rank not in ranks:
        # Engines that are not part of any DP-DCP unit don't need the group.
        return None  # type: ignore[return-value]

    _DP_ATTN = init_model_parallel_group(
        group_ranks=[ranks],
        local_rank=world.local_rank,
        backend=backend,
        group_name="dp_attn",
    )
    _DP_ATTN_LOCAL_RANK = ranks.index(world.rank)
    _DP_ATTN_WORLD_SIZE = len(ranks)
    logger.info(
        "DP attn group up: ranks=%s, local_rank=%s/%s",
        ranks,
        _DP_ATTN_LOCAL_RANK,
        _DP_ATTN_WORLD_SIZE,
    )
    return _DP_ATTN


def get_dp_attn_group() -> GroupCoordinator:
    assert _DP_ATTN is not None, (
        "dp_attn group is not initialized; call init_dp_attn_group() first"
    )
    return _DP_ATTN


def get_dp_attn_world_size() -> int:
    assert _DP_ATTN_WORLD_SIZE is not None
    return _DP_ATTN_WORLD_SIZE


def get_dp_attn_rank() -> int:
    assert _DP_ATTN_LOCAL_RANK is not None
    return _DP_ATTN_LOCAL_RANK


def destroy_dp_attn_group() -> None:
    global _DP_ATTN, _DP_ATTN_LOCAL_RANK, _DP_ATTN_WORLD_SIZE
    if _DP_ATTN is not None:
        _DP_ATTN.destroy()
    _DP_ATTN = None
    _DP_ATTN_LOCAL_RANK = None
    _DP_ATTN_WORLD_SIZE = None
