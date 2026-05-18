# DP-DCP Global Scheduler (Phase 1 scaffold)

Externalizes the per-DP-rank vLLM scheduler decisions into one central
process, while keeping each EngineCore's local Scheduler + KVCacheManager
intact. Reuses the existing DPCoordinator ZMQ plumbing; replaces only the
payload schema and the decision logic.

## What this codebase contains

```
src/
  protocol.py               # msgpack message types (scheduler <-> engine, FE <-> sched)
  placement.py              # round-robin KV split + owner picking algorithms
  global_scheduler.py       # central process, drop-in for DPCoordinator
  engine_adapter.py         # sidecar IO thread spawned inside each EngineCore
  engine_integration.py     # hooks wired into vllm/v1/engine/core.py
  lmcache_handoff.py        # LMCacheConnectorV1 subclass that respects placement
  flash_attn_dp_dcp.py      # cross-DP partial attention + LSE combine
  shadow_kv_cache_manager.py  # Phase 3 scaffold (NotImplementedError today)
configs/
  lmcache_prefill.yaml      # sender role, KV stays in CUDA buffer
  lmcache_decode.yaml       # receiver role, KV lands in host DRAM
scripts/
  start_scheduler.sh        # spawn the global scheduler standalone
  start_prefill.sh          # spawn 1 prefill EngineCore on GPU 0
  start_decoders.sh         # spawn decode EngineCores on GPU 1..3
vllm/distributed/
  dp_attn_group.py          # NCCL communicator across DP ranks (new file)
vllm/v1/engine/core.py      # 2 call sites added (guarded by env)
```

## Design summary (locked-in decisions)

1. **Per-request KV layout**: round-robin block split across decode ranks.
   `block_k -> decode_ranks[k mod N]`. No hash routing, no water-fill, no
   rebalance. Request exit frees all its blocks across all ranks atomically.
2. **Decode owner**: chosen independently of KV holders, by least-loaded
   `owned_reqs` count. Owner runs QKV proj / MLP / MoE / sample.
3. **LMCache role**: prefill writes KV slices over NIXL/RDMA to decode-node
   DRAM. Decode ranks pull from DRAM into their paged buffer when handoff
   arrives. DRAM also caches evicted decode requests (tier 1).
4. **Global scheduler scope**: placement only. Local Scheduler still
   composes batches and allocates GPU blocks. KVCacheManager unchanged.
5. **GPU prefix cache**: accepted as effectively disabled in cross-rank
   case. Prefix hits go through LMCache CPU pool. GPU layer pays a
   DRAM->GPU copy on hit (still much faster than re-prefill).

## Message flow

```
FE  --NewRequestHint-->        Scheduler
FE  <--PlacementAnswer--       Scheduler        (target_engine_index)
ENG --EngineStatsReport-->     Scheduler        (every step end)
ENG --EngineStatsReport.prefill_done-->  Scheduler
ENG <--PlacementDecision--     Scheduler        (PREFILL or DECODE_HANDOFF)
FE  <--LoadReport--            Scheduler        (legacy 100ms tick)
```

`EngineStatsReport` carries per-rank free blocks, held blocks per request,
owned req IDs, queue lengths, plus the prefill_done batch and finished IDs
for this step. `PlacementDecision` for DECODE_HANDOFF includes per-block
`(owner_rank, lmcache_key)` tuples.

## Engine-side integration

Already applied to `vllm/v1/engine/core.py` at the tail of
`EngineCoreProc.__init__` and at the end of `EngineCore.step()`. Both
sites delegate to `yqn/dp_dcp_global_scheduler/src/engine_integration.py`
and are guarded by env vars — when `GLOBAL_SCHED_PULL_ADDR` is unset the
hooks are no-ops, so upstream behavior is unchanged.

Env vars that turn the integration on:

```
GLOBAL_SCHED_PULL_ADDR=tcp://127.0.0.1:5571   # scheduler PULL (engines PUSH stats)
GLOBAL_SCHED_PUB_ADDR=tcp://127.0.0.1:5572    # scheduler XPUB (engines SUB decisions)
ENGINE_INDEX=<0..N-1>
ENGINE_ROLE=prefill|decode|hybrid
DP_ATTN_WORLD_SIZE=<N>                        # opt-in: patch FlashAttn._forward_with_dcp
DP_ATTN_RANKS=0,1,2,3                         # optional: explicit rank list
```

What the hooks do, mechanically:

- `maybe_install_engine_adapter(self)` — spawns the sidecar IO thread,
  attaches `EngineAdapter` to the engine instance.
- `maybe_install_attention_patch(vllm_config)` — calls
  `init_dp_attn_group()` and monkey-patches `FlashAttentionImpl
  ._forward_with_dcp` with `flash_attn_dp_dcp.forward_with_dp_attn`,
  switching attention to the cross-DP NCCL group.
- `maybe_report_step(self, scheduler_output)` — at every step end,
  pushes `EngineStatsReport`, detects locally-finished prefills and
  fires `notify_prefill_done`, and drains any inbound decisions.

## Replacing DPCoordinator

`scripts/start_scheduler.sh` spawns the global scheduler. Front-end clients
that currently connect to `DPCoordinator` addresses should be pointed at the
addresses printed at startup:

```
front_publish = tcp://0.0.0.0:5570
back_output   = tcp://0.0.0.0:5571
back_publish  = tcp://0.0.0.0:5572
```

The `LoadReport` message preserves the legacy `(counts, wave, running)`
tuple under a typed envelope; older front-end code that decodes raw tuples
needs a small adjustment to first unwrap the envelope.

## What still needs to be built

Phase 1+2 here cover placement + KV handoff routing + cross-DP attention
backend + EngineCore patch. Not yet implemented (deferred by design):

- **Layer-wise prefetch** (`prefetch_scheduler.py`): cudaMemcpyAsync
  DRAM->GPU overlapped with forward, using a dedicated copy stream.
  Deferred — first land DP-DCP correctness end-to-end, then optimize.
- **GlobalScheduler shadow KVCacheManager** (Phase 3): central allocation
  of block ids. Scaffolded in `src/shadow_kv_cache_manager.py` with the
  Phase 3 surface stubbed `NotImplementedError`. Today the per-engine
  KVCacheManager remains authoritative.

## Testing without GPUs

`global_scheduler.py` exposes a CLI; you can spin it up and use a tiny
PUSH client to inject `EngineStatsReport` and verify decisions show up on
the PUB socket. This is the recommended way to iterate on placement logic
without burning GPUs.

## NUMA pinning

Your 8-GPU box has two NUMA halves with NV4 + PIX inside each half and
SYS between halves. **Do not span a single DP-DCP group across NUMA**:
cross-NUMA collectives traverse a serial QPI/UPI hop and will dominate
latency. The recommended layout is one full 4-way DP-DCP unit per NUMA
half (GPU 0..3 and GPU 4..7), with the global scheduler treating the
two halves as two independent decode pools.
