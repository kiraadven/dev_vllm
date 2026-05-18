# Placement algorithms for the global scheduler.
#
# Three decisions live here:
#   1) prefill_owner       : which prefill rank handles a new request
#   2) decode_handoff      : after prefill, how to shard KV blocks across
#                            decode ranks and pick the decode owner
#   3) finished cleanup    : reconcile global view after a request ends
#
# Design choices (locked in earlier discussion):
#   - per-request KV blocks are split EVENLY across all decode ranks via
#     round-robin: block_k -> rank (k mod N_dec). This guarantees per-request
#     self-balance and natural cleanup on request exit.
#   - decode_owner is chosen INDEPENDENTLY of the KV holders, by least-loaded
#     "owned_reqs" count. Owner runs MLP/MoE/sample for the request.
#   - prefill_owner picks the prefill rank with the fewest waiting+running
#     requests.
#
# No hash routing, no water-fill, no rebalancing.

from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import BlockPlacement, PlacementDecision, Phase


@dataclass
class RankView:
    """Scheduler's view of one DP rank. Refreshed every stats report."""

    engine_index: int
    role: str  # "prefill" | "decode" | "hybrid"

    free_gpu_blocks: int = 0
    total_gpu_blocks: int = 0
    num_waiting_reqs: int = 0
    num_running_reqs: int = 0
    owned_req_ids: set[str] = field(default_factory=set)
    held_blocks_per_req: dict[str, int] = field(default_factory=dict)

    lmcache_dram_used: int = 0
    lmcache_dram_capacity: int = 0

    last_step_id: int = -1
    last_wave: int = -1

    @property
    def total_held_blocks(self) -> int:
        return sum(self.held_blocks_per_req.values())


@dataclass
class RequestView:
    """Scheduler's view of one in-flight request."""

    req_id: str
    phase: Phase = Phase.NEW

    prompt_len: int = 0
    expected_max_decode_len: int = 0
    prompt_hash: str = ""

    prefill_owner: int | None = None
    decode_owner: int | None = None
    decode_kv_holders: list[int] = field(default_factory=list)
    num_blocks: int = 0
    block_size: int = 0
    lmcache_key_prefix: str = ""


# ---------------------------------------------------------------------------
# (1) prefill_owner selection
# ---------------------------------------------------------------------------


def pick_prefill_owner(ranks: dict[int, RankView]) -> int:
    """Choose the prefill rank with the lightest queue.

    Tiebreak: more free GPU blocks first, then lowest engine_index for
    determinism (helpful for reproducible tests).
    """
    candidates = [r for r in ranks.values() if r.role in ("prefill", "hybrid")]
    if not candidates:
        raise RuntimeError("no prefill/hybrid ranks registered")
    candidates.sort(
        key=lambda r: (
            r.num_waiting_reqs + r.num_running_reqs,
            -r.free_gpu_blocks,
            r.engine_index,
        )
    )
    return candidates[0].engine_index


# ---------------------------------------------------------------------------
# (2) decode handoff: even round-robin KV split + independent owner pick
# ---------------------------------------------------------------------------


def _round_robin_block_owners(num_blocks: int, decode_ranks: list[int]) -> list[int]:
    """block_k -> decode_ranks[k mod N]. Pure even split, no load awareness."""
    n = len(decode_ranks)
    return [decode_ranks[k % n] for k in range(num_blocks)]


def pick_decode_owner(
    ranks: dict[int, RankView], candidate_decode_ranks: list[int]
) -> int:
    """Pick the decode rank with the fewest currently owned requests.

    Owner runs the request's query through QKV proj, MLP, MoE router/expert,
    o_proj, sample. This load is per-token-not-per-context, so even
    distribution of owner counts is the relevant metric.

    Tiebreak: fewest active query tokens (already-rolling load), then index.
    """
    candidates = [ranks[i] for i in candidate_decode_ranks]
    candidates.sort(
        key=lambda r: (
            len(r.owned_req_ids),
            r.num_running_reqs,
            r.engine_index,
        )
    )
    return candidates[0].engine_index


def make_decode_handoff(
    req: RequestView,
    ranks: dict[int, RankView],
) -> PlacementDecision:
    """Build the DECODE_HANDOFF placement: KV split + owner.

    Pre-condition: req.num_blocks, req.block_size, req.lmcache_key_prefix
    are already set from the PrefillDone report. The caller should also
    have checked that there is enough aggregate free capacity across the
    decode ranks.
    """
    decode_rank_ids = sorted(
        i for i, r in ranks.items() if r.role in ("decode", "hybrid")
    )
    if not decode_rank_ids:
        raise RuntimeError("no decode/hybrid ranks registered")

    owners = _round_robin_block_owners(req.num_blocks, decode_rank_ids)
    placements = [
        BlockPlacement(
            block_idx=idx,
            owner_rank=owners[idx],
            lmcache_key=f"{req.lmcache_key_prefix}:{idx}",
        )
        for idx in range(req.num_blocks)
    ]

    decode_owner = pick_decode_owner(ranks, decode_rank_ids)

    # Mutate the request view to reflect the new placement.
    req.phase = Phase.DECODE_HANDOFF
    req.decode_kv_holders = decode_rank_ids
    req.decode_owner = decode_owner

    return PlacementDecision(
        req_id=req.req_id,
        phase=Phase.DECODE_HANDOFF,
        decode_owner=decode_owner,
        block_placements=placements,
        decode_world_size=len(decode_rank_ids),
    )


def make_prefill_placement(
    req: RequestView, ranks: dict[int, RankView]
) -> PlacementDecision:
    """Build the PREFILL placement: only sets prefill_owner."""
    owner = pick_prefill_owner(ranks)
    req.phase = Phase.PREFILL
    req.prefill_owner = owner
    return PlacementDecision(
        req_id=req.req_id,
        phase=Phase.PREFILL,
        prefill_owner=owner,
    )


# ---------------------------------------------------------------------------
# Capacity check helpers
# ---------------------------------------------------------------------------


def have_capacity_for(num_blocks: int, ranks: dict[int, RankView]) -> bool:
    """Aggregate free-block check across decode ranks.

    With round-robin split, each decode rank takes ceil(num_blocks / N) at
    most. Be conservative: require every decode rank to have that many free.
    """
    decode_ranks = [r for r in ranks.values() if r.role in ("decode", "hybrid")]
    if not decode_ranks:
        return False
    n = len(decode_ranks)
    per_rank_share = (num_blocks + n - 1) // n
    return all(r.free_gpu_blocks >= per_rank_share for r in decode_ranks)


def cleanup_finished(req_id: str, ranks: dict[int, RankView]) -> None:
    """Remove a finished request from all rank views in one pass."""
    for r in ranks.values():
        r.owned_req_ids.discard(req_id)
        r.held_blocks_per_req.pop(req_id, None)
