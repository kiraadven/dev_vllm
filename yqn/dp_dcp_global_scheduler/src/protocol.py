# Protocol structs exchanged between the global scheduler and per-DP-rank
# EngineCore processes.
#
# Wire format is msgpack via msgspec, matching what the existing
# DPCoordinator pipeline already uses (vllm/v1/engine/coordinator.py).
# We reuse the same back_output (PULL) and back_publish (XPUB) sockets;
# the only thing we change is the payload schema.
#
# Direction conventions:
#   ENG -> SCHED : engine reports state on the existing PULL socket
#   SCHED -> ENG : scheduler broadcasts decisions on the existing XPUB socket
#   FE   -> SCHED: front-end asks for placement before sending a request
#   SCHED -> FE  : scheduler answers placement on the front-publish XPUB

from __future__ import annotations

import enum
from typing import Optional

import msgspec


# ---------------------------------------------------------------------------
# Common atoms
# ---------------------------------------------------------------------------


class Phase(str, enum.Enum):
    """Lifecycle phase a request is currently in."""

    NEW = "new"
    PREFILL = "prefill"
    DECODE_HANDOFF = "decode_handoff"
    DECODE = "decode"
    FINISHED = "finished"


# ---------------------------------------------------------------------------
# ENG -> SCHED: per-step stats
# ---------------------------------------------------------------------------


class RankBlockStats(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """KV pool occupancy for one DP rank.

    Reported every scheduler step end so the global scheduler can decide
    placement for new requests and detect imbalance.
    """

    # Per-rank GPU paged-buffer state
    free_gpu_blocks: int = 0
    total_gpu_blocks: int = 0

    # Per-request block holdings on THIS rank (slice of the request's KV
    # that physically lives here). Sum across all ranks recovers the
    # request's total block count.
    #   {req_id: block_count_on_this_rank}
    held_blocks_per_req: dict[str, int] = msgspec.field(default_factory=dict)

    # LMCache pool occupancy (DRAM tier on this node)
    lmcache_dram_used_bytes: int = 0
    lmcache_dram_capacity_bytes: int = 0


class RankRoleStats(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Compute / ownership load for one DP rank."""

    # Requests currently rooted on this rank as decode owner: this rank
    # holds their query, runs MLP/MoE/sample for them.
    owned_req_ids: list[str] = msgspec.field(default_factory=list)

    # Number of query tokens processed on this rank this step.
    # For pure decode == len(owned_req_ids); for prefill it's the chunk size.
    active_query_tokens: int = 0

    # Local scheduler queue lengths (replaces the old [waiting, running] int
    # pair carried in EngineState).
    num_waiting_reqs: int = 0
    num_running_reqs: int = 0


class EngineStatsReport(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Whole-step report from one EngineCore process to the scheduler."""

    engine_index: int
    step_id: int
    wave: int
    role: str  # "prefill" | "decode" | "hybrid"

    blocks: RankBlockStats
    load: RankRoleStats

    # Requests whose prefill just finished on this rank, ready to be handed
    # off to decode ranks. Each entry must be matched by a subsequent
    # decode_handoff PlacementDecision from the scheduler.
    prefill_done: list[PrefillDone] = msgspec.field(default_factory=list)

    # Requests that finished on this rank (decode generated EOS or was
    # aborted). Scheduler removes them from its global request table and
    # any per-rank held_blocks map.
    finished_req_ids: list[str] = msgspec.field(default_factory=list)


class PrefillDone(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Prefill-rank announces a request has its full KV ready."""

    req_id: str
    prompt_len: int
    num_blocks: int  # total KV blocks for the whole prompt
    block_size: int

    # LMCache key prefix the prefill rank used when staging KV into the
    # DRAM pool (per-block keys are formed as f"{lmcache_key_prefix}:{idx}").
    # The decode ranks read from the same keys.
    lmcache_key_prefix: str


# ---------------------------------------------------------------------------
# SCHED -> ENG: placement decisions
# ---------------------------------------------------------------------------


class BlockPlacement(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Where one logical KV block of a request must live."""

    block_idx: int  # 0..num_blocks-1 within the request
    owner_rank: int  # which DP rank physically holds this block
    lmcache_key: str  # key to fetch the KV from (or write to)


class PlacementDecision(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Scheduler tells engines how to place / migrate a request's KV.

    Broadcast on the back_publish XPUB. Each engine filters on whether it
    is mentioned (in prefill_owner, decode_owner, or owns at least one
    block in block_placements).
    """

    req_id: str
    phase: Phase

    # phase=PREFILL: which rank runs prefill
    prefill_owner: Optional[int] = None

    # phase=DECODE_HANDOFF: how the KV must be sharded onto decode ranks
    decode_owner: Optional[int] = None
    block_placements: list[BlockPlacement] = msgspec.field(default_factory=list)

    # Number of decode ranks the KV is sharded across; used by the
    # cross-DP DCP attention path to size its NCCL group.
    decode_world_size: Optional[int] = None


# ---------------------------------------------------------------------------
# FE <-> SCHED
# ---------------------------------------------------------------------------


class NewRequestHint(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Front-end asks the scheduler to place a request."""

    req_id: str
    prompt_token_count: int
    expected_max_decode_len: int = 0
    # Optional prompt hash for LMCache lookup. Empty if hashing disabled.
    prompt_hash: str = ""


class PlacementAnswer(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Scheduler answers front-end which rank should receive a request."""

    req_id: str
    target_engine_index: int
    # If non-empty, the scheduler already knows decode placement and the
    # engine can skip the prefill stage entirely (LMCache full-prompt hit).
    skip_prefill: bool = False


# ---------------------------------------------------------------------------
# Message envelopes (sent over the wire)
# ---------------------------------------------------------------------------


class MessageKind(str, enum.Enum):
    # ENG -> SCHED
    ENGINE_STATS = "engine_stats"
    # SCHED -> ENG
    PLACEMENT_DECISION = "placement_decision"
    EXECUTE_HINT = "execute_hint"
    # FE -> SCHED
    NEW_REQUEST_HINT = "new_request_hint"
    # SCHED -> FE
    PLACEMENT_ANSWER = "placement_answer"
    LOAD_REPORT = "load_report"  # mirrors old (counts, wave, running) tuple


class Envelope(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Wire envelope so a single socket can carry multiple message kinds."""

    kind: MessageKind
    payload: bytes  # msgpack-encoded body keyed by kind


# Helpers
_ENVELOPE_ENC = msgspec.msgpack.Encoder()
_ENVELOPE_DEC = msgspec.msgpack.Decoder(Envelope)


def pack(kind: MessageKind, body) -> bytes:
    """Encode body and wrap in an envelope (raw bytes ready for zmq.send)."""
    return _ENVELOPE_ENC.encode(
        Envelope(kind=kind, payload=msgspec.msgpack.encode(body))
    )


def unpack(buf: bytes) -> tuple[MessageKind, bytes]:
    """Return (kind, raw_body_bytes). Caller decodes body with the right type."""
    env = _ENVELOPE_DEC.decode(buf)
    return env.kind, env.payload
