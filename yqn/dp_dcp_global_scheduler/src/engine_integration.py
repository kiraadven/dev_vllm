# Engine-side integration shim.
#
# All EngineCore <-> GlobalScheduler wiring lives here so the touch on
# upstream vllm/v1/engine/core.py stays to four call sites, each guarded
# by `GLOBAL_SCHED_PULL_ADDR` so this code is a no-op when the global
# scheduler is not in use.
#
# Public surface:
#   maybe_install_engine_adapter(engine)   -- called at the end of
#       EngineCoreProc.__init__. Reads env, builds EngineAdapter, starts
#       it, stashes on engine.
#   maybe_report_step(engine, scheduler_output)  -- called at the end of
#       each EngineCore.step(), after update_from_output. Pushes a stats
#       report + drains decisions for any prefill-handoff actions.
#   maybe_install_attention_patch(vllm_config)   -- swap
#       FlashAttentionImpl._forward_with_dcp for the DP-DCP variant when
#       dp_attn group is available.

from __future__ import annotations

import logging
import os

logger = logging.getLogger("dp_dcp.engine_integration")

_ADAPTER_ATTR = "_gs_adapter"
_PATCHED_FLAG = "_dp_attn_patched"


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


def maybe_install_engine_adapter(engine) -> None:
    """Hook for EngineCoreProc.__init__ tail.

    No-op if `GLOBAL_SCHED_PULL_ADDR` is not set.
    """
    pull_addr = os.environ.get("GLOBAL_SCHED_PULL_ADDR")
    pub_addr = os.environ.get("GLOBAL_SCHED_PUB_ADDR")
    if not pull_addr or not pub_addr:
        return
    if getattr(engine, _ADAPTER_ATTR, None) is not None:
        return

    from .engine_adapter import EngineAdapter, EngineAdapterConfig

    engine_index = int(os.environ.get("ENGINE_INDEX", str(engine.engine_index)))
    role = os.environ.get("ENGINE_ROLE", "decode")
    adapter = EngineAdapter(
        EngineAdapterConfig(
            engine_index=engine_index,
            role=role,
            back_output_address=pull_addr,
            back_publish_address=pub_addr,
        )
    )
    adapter.start()
    setattr(engine, _ADAPTER_ATTR, adapter)
    logger.info(
        "[gs] EngineAdapter started (engine_index=%s role=%s pull=%s pub=%s)",
        engine_index,
        role,
        pull_addr,
        pub_addr,
    )


# ---------------------------------------------------------------------------
# Per-step reporting
# ---------------------------------------------------------------------------


def maybe_report_step(engine, scheduler_output) -> None:
    """Hook for the tail of EngineCore.step()."""
    adapter = getattr(engine, _ADAPTER_ATTR, None)
    if adapter is None:
        return

    scheduler = engine.scheduler
    kv_manager = getattr(scheduler, "kv_cache_manager", None)

    free_blocks = 0
    total_blocks = 0
    block_size = 0
    held: dict[str, int] = {}
    if kv_manager is not None:
        try:
            free_blocks = kv_manager.block_pool.get_num_free_blocks()
            total_blocks = kv_manager.block_pool.num_gpu_blocks
            block_size = kv_manager.block_size
        except AttributeError:
            pass
        for req in scheduler.running:
            try:
                held[req.request_id] = len(
                    kv_manager.get_blocks(req.request_id)
                )
            except Exception:
                held[req.request_id] = 0

    owned = [r.request_id for r in scheduler.running]
    active_tokens = sum(
        getattr(r, "num_computed_tokens", 0) for r in scheduler.running
    )

    # Step id: cheap monotonic counter derived from scheduler if exposed,
    # otherwise local.
    step_id = getattr(scheduler, "step_counter", None)
    if step_id is None:
        step_id = getattr(engine, "_gs_step_counter", 0)
        engine._gs_step_counter = step_id + 1
    wave = getattr(scheduler, "current_wave", 0)

    # Surface freshly-finished requests to the scheduler so it can prune
    # routing state.
    finished_ids: list[str] = []
    if scheduler_output is not None:
        finished_ids = list(getattr(scheduler_output, "finished_req_ids", []) or [])
    if finished_ids:
        adapter.notify_finished(finished_ids)

    adapter.report_step(
        step_id=int(step_id),
        wave=int(wave),
        free_gpu_blocks=int(free_blocks),
        total_gpu_blocks=int(total_blocks),
        held_blocks_per_req=held,
        owned_req_ids=owned,
        active_query_tokens=int(active_tokens),
        num_waiting=len(scheduler.waiting),
        num_running=len(scheduler.running),
    )

    # Prefill-finished detection: any request that just transitioned to
    # finished prefill stage on a prefill-role engine triggers a handoff
    # announce. Conservative: only fire when role == "prefill".
    if adapter.config.role == "prefill" and kv_manager is not None:
        prev_done = getattr(engine, "_gs_prefill_done", set())
        new_done = set()
        for req in scheduler.running:
            num_computed = getattr(req, "num_computed_tokens", 0)
            num_prompt = getattr(req, "num_prompt_tokens", 0)
            if num_prompt > 0 and num_computed >= num_prompt:
                new_done.add(req.request_id)
                if req.request_id not in prev_done:
                    try:
                        nb = len(kv_manager.get_blocks(req.request_id))
                    except Exception:
                        nb = 0
                    adapter.notify_prefill_done(
                        req_id=req.request_id,
                        prompt_len=num_prompt,
                        num_blocks=nb,
                        block_size=block_size or 0,
                        lmcache_key_prefix=f"req:{req.request_id}",
                    )
        engine._gs_prefill_done = new_done

    # Drain decisions so anything that needs main-thread action (logging,
    # local routing tweak) runs here. The LMCache registry has already
    # been updated from the IO thread inside the adapter.
    for _decision in adapter.drain_decisions():
        pass


# ---------------------------------------------------------------------------
# Attention backend patch
# ---------------------------------------------------------------------------


def maybe_install_attention_patch(vllm_config=None) -> None:
    """Replace FlashAttentionImpl._forward_with_dcp with the DP-DCP variant.

    Triggered by `DP_ATTN_WORLD_SIZE` env. Idempotent.
    """
    dp_world_env = os.environ.get("DP_ATTN_WORLD_SIZE")
    if not dp_world_env:
        return
    try:
        dp_world = int(dp_world_env)
    except ValueError:
        return
    if dp_world <= 1:
        return

    from vllm.distributed.dp_attn_group import init_dp_attn_group
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    if getattr(FlashAttentionImpl, _PATCHED_FLAG, False):
        return

    try:
        init_dp_attn_group(dp_world_size=dp_world)
    except Exception:
        logger.exception("[gs] failed to init dp_attn group; not patching")
        return

    from .flash_attn_dp_dcp import forward_with_dp_attn

    def _patched(self, query, key, value, key_cache, value_cache, output,
                 attn_metadata, q_descale=None, k_descale=None, v_descale=None):
        return forward_with_dp_attn(
            self, query, key, value, key_cache, value_cache, output,
            attn_metadata,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        )

    FlashAttentionImpl._forward_with_dcp = _patched
    FlashAttentionImpl._dp_attn_patched = True  # type: ignore[attr-defined]
    logger.info("[gs] FlashAttentionImpl._forward_with_dcp patched (dp_world=%d)",
                dp_world)
