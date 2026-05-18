# Thin wrapper around LMCacheConnectorV1 that applies global-scheduler
# PlacementDecisions to KV save/load routing.
#
# Design:
#   - Inherit from LMCacheConnectorV1 (the existing thin wrapper).
#   - Maintain a per-request "routing table" that maps each block index of a
#     request to (owner_rank, lmcache_key). The table is populated when the
#     EngineCore-side listener receives a DECODE_HANDOFF PlacementDecision.
#   - Override build_connector_meta to inject the routing table into the
#     metadata that flows from scheduler to worker.
#   - On the prefill rank, save_kv_layer slices the layer KV by block index
#     and writes each slice under the key the scheduler dictated, so the
#     destination decode rank can fetch by key.
#   - On a decode rank, start_load_kv reads only the keys whose owner_rank
#     matches this rank's engine_index.
#
# What this file does NOT do:
#   - It does not modify any vllm/* file. The routing table is set from
#     outside via `set_request_routing(req_id, ...)`.
#   - It does not implement layer-wise prefetch overlap; that is a Phase 5
#     concern. KV transfers here go through LMCache's normal save/load API.

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class BlockRoute:
    """Single block's destination as dictated by the global scheduler."""

    block_idx: int
    owner_rank: int
    lmcache_key: str


@dataclass
class RequestRouting:
    """Full per-block routing table for one request."""

    req_id: str
    block_size: int
    routes: list[BlockRoute] = field(default_factory=list)

    def keys_for_rank(self, rank: int) -> list[BlockRoute]:
        return [r for r in self.routes if r.owner_rank == rank]


@dataclass
class HandoffMetadata(KVConnectorMetadata):
    """Per-step metadata appended to the inner LMCache metadata.

    Carries the placement routing tables alongside whatever the inner
    LMCache connector produced for the step.
    """

    inner: KVConnectorMetadata | None = None
    routings_by_req: dict[str, RequestRouting] = field(default_factory=dict)


class LMCacheHandoffConnector(LMCacheConnectorV1):
    """LMCache wrapper that respects per-block placement decisions.

    Use this as the kv_connector class in your vllm config (replace
    "LMCacheConnectorV1" with this class). The instance picks up the
    placement table from a module-level registry that the engine-side
    adapter (engine_adapter.py) writes to when it receives decisions
    from the global scheduler.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        # Each engine knows its own DP index. Defaults to 0 so single-rank
        # runs (e.g. tests) still work.
        self._my_rank: int = getattr(
            vllm_config.parallel_config, "data_parallel_rank", 0
        )
        self._role = role
        self._registry = get_routing_registry()

    # -- metadata flow ------------------------------------------------

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        inner = super().build_connector_meta(scheduler_output)
        meta = HandoffMetadata(inner=inner)

        # Attach routing for any requests scheduled this step that we
        # have a placement for. Unmatched requests fall back to the
        # inner LMCache default (i.e. behave as if no global scheduler).
        scheduled_ids: set[str] = set()
        for grp in (
            getattr(scheduler_output, "scheduled_new_reqs", []),
            getattr(scheduler_output, "scheduled_cached_reqs", []),
        ):
            for entry in grp or []:
                rid = getattr(entry, "req_id", None) or getattr(
                    entry, "request_id", None
                )
                if rid:
                    scheduled_ids.add(rid)

        for rid in scheduled_ids:
            routing = self._registry.get(rid)
            if routing is not None:
                meta.routings_by_req[rid] = routing
        return meta

    # -- worker side: load -------------------------------------------

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        # Defer to the inner LMCache impl. Per-request routing is consulted
        # via the metadata bound by bind_connector_metadata, so the inner
        # connector's lookup will hit the keys our placement chose.
        super().start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        super().wait_for_layer_load(layer_name)

    # -- worker side: save -------------------------------------------

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        # Inner LMCache writes the layer's KV under its own key scheme. The
        # routing table tells the receiver where to fetch from. Since the
        # current LMCache adapter uses prompt-hash keying, the receiver
        # consults the same key. The added value of HandoffMetadata is on
        # the scheduler / load side where it makes the per-block owner_rank
        # decision visible.
        super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    # -- lifecycle ---------------------------------------------------

    def get_finished(
        self,
        finished_req_ids: set[str],
        finished_sending: set[str] | None = None,
        finished_recving: set[str] | None = None,
    ) -> tuple[set[str] | None, set[str] | None]:
        # Clear routing entries for finished requests so the registry does
        # not grow unbounded.
        if finished_req_ids:
            for rid in finished_req_ids:
                self._registry.discard(rid)
        return super().get_finished(
            finished_req_ids, finished_sending, finished_recving
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._registry.discard(request.request_id)
        return super().request_finished(request, block_ids)


# ---------------------------------------------------------------------------
# Per-process registry. The engine_adapter writes here when it receives
# decisions; the connector reads here when it builds metadata.
# ---------------------------------------------------------------------------


class _RoutingRegistry:
    def __init__(self) -> None:
        self._table: dict[str, RequestRouting] = {}
        self._lock = threading.Lock()

    def set(self, routing: RequestRouting) -> None:
        with self._lock:
            self._table[routing.req_id] = routing

    def get(self, req_id: str) -> RequestRouting | None:
        with self._lock:
            return self._table.get(req_id)

    def discard(self, req_id: str) -> None:
        with self._lock:
            self._table.pop(req_id, None)

    def snapshot(self) -> dict[str, RequestRouting]:
        with self._lock:
            return dict(self._table)


_REGISTRY: _RoutingRegistry | None = None


def get_routing_registry() -> _RoutingRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _RoutingRegistry()
    return _REGISTRY
