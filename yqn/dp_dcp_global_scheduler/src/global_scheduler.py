# Centralized scheduler process that replaces vllm/v1/engine/coordinator.py
# DPCoordinator.
#
# Reused from DPCoordinator (the socket plumbing is unchanged):
#   - bind 3 ZMQ sockets: publish_front (XPUB), output_back (PULL),
#     publish_back (XPUB)
#   - mp.Process spawn + zmq_addr_pipe handshake
#   - wave-coordination state (engines_running, current_wave, START_DP_WAVE)
#   - 100ms publish interval to front-ends
#
# Added on top:
#   - per-rank view richer than [waiting, running]
#   - per-request lifecycle table
#   - NewRequestHint -> PlacementAnswer round trip (front-end asks before
#     dispatching new requests)
#   - PrefillDone -> DECODE_HANDOFF PlacementDecision flow
#   - Cleanup on finished_req_ids
#
# This file is self-contained: it does NOT modify any vLLM source. To wire
# it in, run it as a side-car process and point the front-end / engine
# clients at its ZMQ addresses instead of DPCoordinator's.

from __future__ import annotations

import argparse
import copy
import logging
import multiprocessing
import multiprocessing.connection
import time
import weakref
from collections import deque
from dataclasses import dataclass

import msgspec
import zmq

from .placement import (
    RankView,
    RequestView,
    cleanup_finished,
    have_capacity_for,
    make_decode_handoff,
    make_prefill_placement,
)
from .protocol import (
    EngineStatsReport,
    MessageKind,
    NewRequestHint,
    Phase,
    PlacementAnswer,
    PlacementDecision,
    pack,
    unpack,
)

logger = logging.getLogger("global_scheduler")


# ---------------------------------------------------------------------------
# Public handle (constructed in the main process)
# ---------------------------------------------------------------------------


@dataclass
class GlobalSchedulerConfig:
    engine_count: int
    # Per-engine role assignment, indexed by engine_index. Length must match
    # engine_count. Each entry is "prefill" | "decode" | "hybrid".
    roles: list[str]
    front_publish_address: str  # XPUB bind, front-ends subscribe
    back_publish_address: str  # XPUB bind, engines subscribe
    back_output_address: str  # PULL bind, engines push to
    enable_wave_coordination: bool = True
    publish_interval_ms: int = 100


class GlobalScheduler:
    """Spawns the scheduler subprocess and exposes its bind addresses.

    Drop-in replacement for vllm.v1.engine.coordinator.DPCoordinator from
    the front-end's perspective.
    """

    def __init__(self, config: GlobalSchedulerConfig) -> None:
        assert len(config.roles) == config.engine_count
        ctx = multiprocessing.get_context("spawn")
        parent_pipe, child_pipe = ctx.Pipe(duplex=False)
        self.proc = ctx.Process(
            target=_GlobalSchedulerProc.run,
            name="VLLM_GlobalScheduler",
            kwargs={"config": config, "addr_pipe": child_pipe},
            daemon=True,
        )
        self.proc.start()
        child_pipe.close()

        ready = multiprocessing.connection.wait(
            [parent_pipe, self.proc.sentinel], timeout=30
        )
        if not ready or parent_pipe not in ready:
            raise RuntimeError("GlobalScheduler failed to come up")
        addrs = parent_pipe.recv()
        parent_pipe.close()
        (
            self.front_publish_address,
            self.back_output_address,
            self.back_publish_address,
        ) = addrs

        self._finalizer = weakref.finalize(self, _terminate_proc, self.proc)

    def shutdown(self, timeout: float | None = 5.0) -> None:
        if self._finalizer.detach() is not None:
            _terminate_proc(self.proc, timeout)


def _terminate_proc(proc: multiprocessing.Process, timeout: float | None = 5.0) -> None:
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()


# ---------------------------------------------------------------------------
# Subprocess body
# ---------------------------------------------------------------------------


class _GlobalSchedulerProc:
    def __init__(self, config: GlobalSchedulerConfig) -> None:
        self.config = config
        self.ctx = zmq.Context()

        # Per-rank live view
        self.ranks: dict[int, RankView] = {
            i: RankView(engine_index=i, role=config.roles[i])
            for i in range(config.engine_count)
        }
        # Per-request lifecycle table
        self.requests: dict[str, RequestView] = {}
        # Outbox: decisions to broadcast on next loop iteration
        self.pending_engine_decisions: deque[PlacementDecision] = deque()
        self.pending_fe_answers: deque[PlacementAnswer] = deque()

        # Wave coord state (mirrors DPCoordinatorProc)
        self.current_wave = 0
        self.engines_running = False
        self.last_step_counts: list[list[int]] | None = None
        self.stats_changed = False
        self.last_stats_step = -1
        self.last_stats_wave = -1
        self.last_publish_ms = 0

        self._stats_decoder = msgspec.msgpack.Decoder(EngineStatsReport)
        self._new_req_decoder = msgspec.msgpack.Decoder(NewRequestHint)

    @staticmethod
    def run(config: GlobalSchedulerConfig, addr_pipe) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="[GlobalScheduler] %(asctime)s %(levelname)s %(message)s",
        )
        proc = _GlobalSchedulerProc(config)
        try:
            proc._serve(addr_pipe)
        except KeyboardInterrupt:
            logger.info("GlobalScheduler interrupted")
        finally:
            try:
                addr_pipe.close()
            except Exception:
                pass

    # -- main loop -------------------------------------------------------

    def _serve(self, addr_pipe) -> None:
        cfg = self.config
        with (
            self._bind(cfg.front_publish_address, zmq.XPUB) as publish_front,
            self._bind(cfg.back_output_address, zmq.PULL) as output_back,
            self._bind(cfg.back_publish_address, zmq.XPUB) as publish_back,
        ):
            addrs = (
                publish_front.getsockopt(zmq.LAST_ENDPOINT).decode(),
                output_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                publish_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
            )
            try:
                addr_pipe.send(addrs)
            finally:
                addr_pipe.close()

            # Wait for all engine subscriptions
            engines_seen = 0
            while engines_seen < cfg.engine_count:
                msg = publish_back.recv()
                if msg == b"\x01":
                    engines_seen += 1
                else:
                    logger.warning("unexpected back_publish msg during init: %r", msg)
            publish_back.send(b"READY")
            logger.info("All %d engines subscribed", cfg.engine_count)

            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)
            poller.register(publish_back, zmq.POLLIN)
            poller.register(output_back, zmq.POLLIN)

            while True:
                self._flush_outbox(publish_front, publish_back)
                self._maybe_publish_loads(publish_front)

                now_ms = int(time.time() * 1000)
                elapsed = now_ms - self.last_publish_ms
                timeout = (
                    cfg.publish_interval_ms if self.stats_changed else 5000
                ) - elapsed
                events = dict(poller.poll(timeout=max(0, timeout)))

                if output_back in events:
                    self._handle_engine_push(output_back.recv())

                if publish_front in events:
                    self._handle_front_msg(publish_front.recv(), publish_back)

                if publish_back in events:
                    msg = publish_back.recv()
                    if msg == b"\x01":
                        publish_back.send(b"READY")
                    elif msg != b"\x00":
                        logger.warning("unexpected back_publish msg: %r", msg)

    def _bind(self, address: str, kind: int):
        sock = self.ctx.socket(kind)
        sock.bind(address)
        return sock

    # -- engine push --------------------------------------------------

    def _handle_engine_push(self, buf: bytes) -> None:
        try:
            kind, body = unpack(buf)
        except Exception:
            logger.exception("failed to unpack engine push")
            return

        if kind != MessageKind.ENGINE_STATS:
            logger.warning("ignoring unexpected engine push kind=%s", kind)
            return

        try:
            report: EngineStatsReport = self._stats_decoder.decode(body)
        except Exception:
            logger.exception("failed to decode EngineStatsReport")
            return

        rank = self.ranks.get(report.engine_index)
        if rank is None:
            logger.warning("stats from unknown engine %d", report.engine_index)
            return

        # Ignore stale reports (e.g. duplicates on retry).
        if (report.wave, report.step_id) <= (rank.last_wave, rank.last_step_id):
            return
        rank.last_wave = report.wave
        rank.last_step_id = report.step_id

        rank.role = report.role
        rank.free_gpu_blocks = report.blocks.free_gpu_blocks
        rank.total_gpu_blocks = report.blocks.total_gpu_blocks
        rank.held_blocks_per_req = dict(report.blocks.held_blocks_per_req)
        rank.lmcache_dram_used = report.blocks.lmcache_dram_used_bytes
        rank.lmcache_dram_capacity = report.blocks.lmcache_dram_capacity_bytes
        rank.owned_req_ids = set(report.load.owned_req_ids)
        rank.num_waiting_reqs = report.load.num_waiting_reqs
        rank.num_running_reqs = report.load.num_running_reqs
        self.stats_changed = True

        # Handle finished requests first so freed capacity is visible to
        # the prefill_done handoff below.
        for rid in report.finished_req_ids:
            self.requests.pop(rid, None)
            cleanup_finished(rid, self.ranks)

        for done in report.prefill_done:
            self._on_prefill_done(done)

    def _on_prefill_done(self, done) -> None:
        req = self.requests.get(done.req_id)
        if req is None:
            # The request was finished/aborted before handoff. Drop.
            logger.warning(
                "prefill_done for unknown req %s; ignoring", done.req_id
            )
            return
        req.num_blocks = done.num_blocks
        req.block_size = done.block_size
        req.lmcache_key_prefix = done.lmcache_key_prefix

        if not have_capacity_for(req.num_blocks, self.ranks):
            # Soft drop: re-queue the decision. In a real system the
            # scheduler would trigger LMCache evictions or wait. For this
            # prototype we just retry on the next stats tick.
            logger.warning(
                "no capacity for handoff of %s (%d blocks); will retry",
                req.req_id, req.num_blocks,
            )
            return

        decision = make_decode_handoff(req, self.ranks)
        self.pending_engine_decisions.append(decision)

    # -- front-end msg -----------------------------------------------

    def _handle_front_msg(self, buf: bytes, publish_back) -> None:
        if buf in (b"\x01", b"\x00"):
            # XPUB subscribe/unsubscribe notifications
            return
        try:
            kind, body = unpack(buf)
        except Exception:
            # Older clients may speak the legacy DPCoordinator protocol
            # ((engine_to_exclude, wave) tuple). Fall back to that.
            try:
                decoded = msgspec.msgpack.decode(buf)
                if isinstance(decoded, (list, tuple)) and len(decoded) == 2:
                    self._handle_legacy_wave_msg(decoded, publish_back)
                    return
            except Exception:
                logger.exception("failed to decode front msg")
            return

        if kind == MessageKind.NEW_REQUEST_HINT:
            hint: NewRequestHint = self._new_req_decoder.decode(body)
            self._on_new_request(hint)
        else:
            logger.warning("ignoring unexpected front msg kind=%s", kind)

    def _on_new_request(self, hint: NewRequestHint) -> None:
        if hint.req_id in self.requests:
            logger.warning("duplicate NewRequestHint for %s", hint.req_id)
            return
        req = RequestView(
            req_id=hint.req_id,
            prompt_len=hint.prompt_token_count,
            expected_max_decode_len=hint.expected_max_decode_len,
            prompt_hash=hint.prompt_hash,
        )
        self.requests[hint.req_id] = req
        decision = make_prefill_placement(req, self.ranks)
        # Send to engine (so the prefill rank knows to expect this request)
        # AND answer the front-end so it can dispatch.
        self.pending_engine_decisions.append(decision)
        self.pending_fe_answers.append(
            PlacementAnswer(
                req_id=req.req_id,
                target_engine_index=req.prefill_owner,
            )
        )

    def _handle_legacy_wave_msg(self, decoded, publish_back) -> None:
        engine_to_exclude, wave = decoded
        if not self.engines_running:
            if wave < self.current_wave:
                engine_to_exclude = None
            self.engines_running = True
            wave_encoded = msgspec.msgpack.encode(
                (self.current_wave, engine_to_exclude)
            )
            # Wire-compatible with vllm.v1.engine.EngineCoreRequestType.START_DP_WAVE
            publish_back.send_multipart((b"\x02", wave_encoded))

    # -- outbox flush + load publish ---------------------------------

    def _flush_outbox(self, publish_front, publish_back) -> None:
        while self.pending_engine_decisions:
            decision = self.pending_engine_decisions.popleft()
            publish_back.send(pack(MessageKind.PLACEMENT_DECISION, decision))
        while self.pending_fe_answers:
            answer = self.pending_fe_answers.popleft()
            publish_front.send(pack(MessageKind.PLACEMENT_ANSWER, answer))

    def _maybe_publish_loads(self, publish_front) -> None:
        if not self.stats_changed:
            return
        now_ms = int(time.time() * 1000)
        if now_ms - self.last_publish_ms < self.config.publish_interval_ms:
            return
        # Pack as a list parallel to engine_index. Front-end clients use
        # this for load-balancing decisions (legacy interface).
        counts = [
            [r.num_waiting_reqs, r.num_running_reqs]
            for r in (self.ranks[i] for i in sorted(self.ranks))
        ]
        body = (counts, self.current_wave, self.engines_running)
        publish_front.send(pack(MessageKind.LOAD_REPORT, body))
        self.last_publish_ms = now_ms
        self.stats_changed = False
        self.last_step_counts = copy.deepcopy(counts)


# ---------------------------------------------------------------------------
# CLI entry (useful for debugging without spawning from vLLM)
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", type=int, required=True)
    parser.add_argument(
        "--roles",
        type=str,
        required=True,
        help="comma-separated roles, length = --engines (e.g. prefill,decode,decode,decode)",
    )
    parser.add_argument("--front-addr", default="tcp://*:5570")
    parser.add_argument("--back-pull-addr", default="tcp://*:5571")
    parser.add_argument("--back-pub-addr", default="tcp://*:5572")
    args = parser.parse_args()

    roles = args.roles.split(",")
    if len(roles) != args.engines:
        raise SystemExit("--roles length must equal --engines")

    cfg = GlobalSchedulerConfig(
        engine_count=args.engines,
        roles=roles,
        front_publish_address=args.front_addr,
        back_output_address=args.back_pull_addr,
        back_publish_address=args.back_pub_addr,
    )
    sched = GlobalScheduler(cfg)
    print(f"GlobalScheduler up:")
    print(f"  front_publish = {sched.front_publish_address}")
    print(f"  back_output   = {sched.back_output_address}")
    print(f"  back_publish  = {sched.back_publish_address}")
    try:
        sched.proc.join()
    except KeyboardInterrupt:
        sched.shutdown()


if __name__ == "__main__":
    _cli()
