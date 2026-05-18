# EngineCore-side adapter that bridges the local vLLM Scheduler with the
# external GlobalScheduler process.
#
# Two duties:
#   (A) Push per-step EngineStatsReport to the scheduler over ZMQ.
#   (B) Pull PlacementDecision broadcasts and translate them into local
#       actions: populate the LMCache routing registry, mark requests as
#       pending decode-handoff, etc.
#
# This file is a sidecar: it spawns a single I/O thread per EngineCore
# process and exposes a small API to be called from the engine's main
# scheduling loop. It does not modify any vllm source files; integration
# is a 5-line patch to vllm/v1/engine/core.py (documented in README).

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue

import msgspec
import zmq

from .lmcache_handoff import (
    BlockRoute,
    RequestRouting,
    get_routing_registry,
)
from .protocol import (
    EngineStatsReport,
    MessageKind,
    Phase,
    PlacementDecision,
    PrefillDone,
    RankBlockStats,
    RankRoleStats,
    pack,
    unpack,
)

logger = logging.getLogger("engine_adapter")


@dataclass
class EngineAdapterConfig:
    engine_index: int
    role: str  # "prefill" | "decode" | "hybrid"
    back_output_address: str  # PUSH to scheduler (PULL on other side)
    back_publish_address: str  # SUB from scheduler (XPUB on other side)
    # Optional topic filter; default subscribes to everything.
    subscribe_topic: bytes = b""


@dataclass
class _AdapterShared:
    """Cross-thread state. Main thread writes stats, IO thread reads."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    pending_prefill_done: deque[PrefillDone] = field(default_factory=deque)
    pending_finished: list[str] = field(default_factory=list)
    last_pushed_step: int = -1
    # Decisions delivered to engine for action (mostly for prefill_owner
    # routing; KV routing is consumed directly by the connector registry).
    decision_outbox: Queue[PlacementDecision] = field(default_factory=Queue)
    stop: threading.Event = field(default_factory=threading.Event)


class EngineAdapter:
    """Sidecar IO thread connecting one EngineCore to the GlobalScheduler.

    Usage in engine main loop (pseudo-code):

        adapter = EngineAdapter(config)
        adapter.start()

        # in scheduler step body, before returning outputs:
        adapter.report_step(
            step_id=..., wave=...,
            free_gpu_blocks=..., total_gpu_blocks=...,
            held_blocks_per_req=...,
            owned_req_ids=..., active_query_tokens=...,
            num_waiting=..., num_running=...,
            finished_req_ids=...,
        )

        # when a prefill finishes:
        adapter.notify_prefill_done(req_id, prompt_len, num_blocks, block_size,
                                    lmcache_key_prefix)

        # to consume placement decisions:
        for decision in adapter.drain_decisions():
            ...
    """

    def __init__(self, config: EngineAdapterConfig) -> None:
        self.config = config
        self.shared = _AdapterShared()
        self._thread: threading.Thread | None = None
        self._ctx = zmq.Context.instance()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._io_loop, name="EngineAdapterIO", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self.shared.stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # -- main-thread API --------------------------------------------

    def notify_prefill_done(
        self,
        req_id: str,
        prompt_len: int,
        num_blocks: int,
        block_size: int,
        lmcache_key_prefix: str,
    ) -> None:
        done = PrefillDone(
            req_id=req_id,
            prompt_len=prompt_len,
            num_blocks=num_blocks,
            block_size=block_size,
            lmcache_key_prefix=lmcache_key_prefix,
        )
        with self.shared.lock:
            self.shared.pending_prefill_done.append(done)

    def notify_finished(self, req_ids: list[str]) -> None:
        if not req_ids:
            return
        with self.shared.lock:
            self.shared.pending_finished.extend(req_ids)

    def report_step(
        self,
        *,
        step_id: int,
        wave: int,
        free_gpu_blocks: int,
        total_gpu_blocks: int,
        held_blocks_per_req: dict[str, int],
        owned_req_ids: list[str],
        active_query_tokens: int,
        num_waiting: int,
        num_running: int,
        lmcache_dram_used_bytes: int = 0,
        lmcache_dram_capacity_bytes: int = 0,
    ) -> None:
        with self.shared.lock:
            if step_id <= self.shared.last_pushed_step:
                # Idempotency guard against retries.
                return
            self.shared.last_pushed_step = step_id
            prefill_done = list(self.shared.pending_prefill_done)
            self.shared.pending_prefill_done.clear()
            finished = list(self.shared.pending_finished)
            self.shared.pending_finished.clear()

        report = EngineStatsReport(
            engine_index=self.config.engine_index,
            step_id=step_id,
            wave=wave,
            role=self.config.role,
            blocks=RankBlockStats(
                free_gpu_blocks=free_gpu_blocks,
                total_gpu_blocks=total_gpu_blocks,
                held_blocks_per_req=dict(held_blocks_per_req),
                lmcache_dram_used_bytes=lmcache_dram_used_bytes,
                lmcache_dram_capacity_bytes=lmcache_dram_capacity_bytes,
            ),
            load=RankRoleStats(
                owned_req_ids=list(owned_req_ids),
                active_query_tokens=active_query_tokens,
                num_waiting_reqs=num_waiting,
                num_running_reqs=num_running,
            ),
            prefill_done=prefill_done,
            finished_req_ids=finished,
        )
        # Stash for the IO thread to send. We use a separate queue so the
        # main thread never blocks on the socket.
        self._stats_to_send.put(report)

    def drain_decisions(self) -> list[PlacementDecision]:
        out: list[PlacementDecision] = []
        try:
            while True:
                out.append(self.shared.decision_outbox.get_nowait())
        except Empty:
            pass
        return out

    # -- IO thread ---------------------------------------------------

    _stats_to_send: "Queue[EngineStatsReport]" = None  # type: ignore[assignment]

    def _io_loop(self) -> None:
        self._stats_to_send = Queue()

        push = self._ctx.socket(zmq.PUSH)
        push.connect(self.config.back_output_address)
        sub = self._ctx.socket(zmq.SUB)
        sub.connect(self.config.back_publish_address)
        sub.setsockopt(zmq.SUBSCRIBE, self.config.subscribe_topic)
        # Subscribe sentinel so the XPUB on the other side sees us.
        # zmq.SUB <-> zmq.XPUB does this automatically via SUBSCRIBE.

        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)

        decision_decoder = msgspec.msgpack.Decoder(PlacementDecision)

        while not self.shared.stop.is_set():
            # Drain stats queue first (non-blocking)
            self._drain_stats(push)

            # Poll for incoming decisions
            events = dict(poller.poll(timeout=20))
            if sub in events:
                buf = sub.recv()
                self._handle_inbound(buf, decision_decoder)

        push.close(0)
        sub.close(0)

    def _drain_stats(self, push) -> None:
        # Pull as many as we have without blocking the IO loop.
        sent = 0
        while sent < 64:
            try:
                report = self._stats_to_send.get_nowait()
            except Empty:
                return
            try:
                push.send(pack(MessageKind.ENGINE_STATS, report), zmq.NOBLOCK)
                sent += 1
            except zmq.Again:
                # Re-queue and bail until next loop iteration.
                self._stats_to_send.put(report)
                return

    def _handle_inbound(self, buf: bytes, decoder) -> None:
        try:
            kind, body = unpack(buf)
        except Exception:
            logger.exception("failed to unpack scheduler msg")
            return
        if kind != MessageKind.PLACEMENT_DECISION:
            # LOAD_REPORTs etc go to front-ends, not us
            return
        try:
            decision: PlacementDecision = decoder.decode(body)
        except Exception:
            logger.exception("failed to decode PlacementDecision")
            return
        self._apply_decision(decision)

    # -- decision application ---------------------------------------

    def _apply_decision(self, decision: PlacementDecision) -> None:
        # Always surface to the main thread for any extra actions.
        self.shared.decision_outbox.put(decision)

        # KV routing belongs in the LMCache registry so the connector
        # can pick it up when building per-step metadata.
        if decision.phase == Phase.DECODE_HANDOFF and decision.block_placements:
            routing = RequestRouting(
                req_id=decision.req_id,
                block_size=0,  # block_size known to engine, not needed here
                routes=[
                    BlockRoute(
                        block_idx=bp.block_idx,
                        owner_rank=bp.owner_rank,
                        lmcache_key=bp.lmcache_key,
                    )
                    for bp in decision.block_placements
                ],
            )
            get_routing_registry().set(routing)
        elif decision.phase == Phase.FINISHED:
            get_routing_registry().discard(decision.req_id)
