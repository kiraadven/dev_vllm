# Phase 3 placeholder: shadow KVCacheManager.
#
# Future-work scaffold for moving block id allocation out of each
# EngineCore and into the central scheduler. Today (Phase 1) every
# EngineCore still owns its own vllm.v1.core.kv_cache_manager and the
# global scheduler only decides per-request placement + per-block owner
# rank. That works because round-robin block assignment is purely
# determined by (req_id, num_blocks, dp_world_size) — no shared state.
#
# When we promote to Phase 3 we will:
#   1. centralize the BlockPool here, keyed by (engine_index, block_id);
#   2. push allocate() / free() RPCs from engines to the scheduler;
#   3. compute global prefix-cache hits across ranks so the routing can
#      bias a request's first block to the rank that already holds a
#      hit, while keeping round-robin for the remaining blocks.
#
# Until then this module raises NotImplementedError on use; the file
# exists so the import path is stable and so the interface surface for
# Phase 3 is reviewable up front.

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _RankBlockPool:
    free: int = 0
    total: int = 0
    held_by_req: dict[str, int] = field(default_factory=dict)


@dataclass
class ShadowKVCacheManagerConfig:
    num_engines: int
    block_size: int
    blocks_per_engine: int


class ShadowKVCacheManager:
    """Phase 3 shadow allocator. Not used in Phase 1.

    Engine-side BlockPools remain authoritative for actual GPU block
    bookkeeping. This shadow only mirrors the same counts so the
    scheduler can answer "do we have capacity for an N-block request"
    without an RPC. Phase 3 will flip the ownership and make the engine
    a passive applier of scheduler-issued (allocate, free) commands.
    """

    def __init__(self, config: ShadowKVCacheManagerConfig) -> None:
        self.config = config
        self._pools: dict[int, _RankBlockPool] = {
            i: _RankBlockPool(free=config.blocks_per_engine,
                              total=config.blocks_per_engine)
            for i in range(config.num_engines)
        }

    def absorb_report(
        self,
        engine_index: int,
        free_gpu_blocks: int,
        total_gpu_blocks: int,
        held_blocks_per_req: dict[str, int],
    ) -> None:
        pool = self._pools.get(engine_index)
        if pool is None:
            return
        pool.free = free_gpu_blocks
        pool.total = total_gpu_blocks
        pool.held_by_req = dict(held_blocks_per_req)

    def have_capacity(self, total_blocks_needed: int) -> bool:
        return sum(p.free for p in self._pools.values()) >= total_blocks_needed

    # ---- the Phase 3 surface (not implemented yet) ----------------------

    def allocate(self, req_id: str, num_blocks: int) -> list[tuple[int, int]]:
        """Return [(engine_index, block_id), ...] for the new request."""
        raise NotImplementedError("Phase 3: centralized allocation")

    def free(self, req_id: str) -> None:
        raise NotImplementedError("Phase 3: centralized free")
