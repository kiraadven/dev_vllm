# Cross-DP flash attention backend.
#
# Mirrors `_forward_with_dcp` in vllm/v1/attention/backends/flash_attn.py
# but pulls Q across DP ranks via `get_dp_attn_group()` instead of
# `get_dcp_group()`. The KV per request has been split round-robin across
# DP ranks by the global scheduler, so each rank holds 1/N of every
# decode request's KV in its paged buffer. This file does NOT modify the
# upstream backend — it is a drop-in that is used when:
#
#   parallel_config.decode_context_parallel_size == 1   AND
#   dp_attn_group is initialized                        AND
#   request is in DECODE phase
#
# The wrapper falls back to the stock single-rank forward when those
# conditions are not met (e.g. prefill, encoder attention).
#
# Activation: monkey-patch FlashAttentionImpl._forward_with_dcp at engine
# startup; see engine_integration.install_attention_patch().

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.attention.utils.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

if TYPE_CHECKING:
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
    )


def forward_with_dp_attn(
    impl: "FlashAttentionImpl",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    attn_metadata: "FlashAttentionMetadata",
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> torch.Tensor:
    """DP-DCP partial attention + LSE combine.

    Layout: each DP rank owns block_k for k where k % dp_world == this_rank.
    The QKV projection has already run on `decode_owner`; that rank is the
    one that calls forward(). To compute attention, we need:
      1. each rank to have Q (so we all_gather Q across DP ranks),
      2. each rank to compute partial attention against its slice of K/V,
      3. combine partials via LSE-aware reduce-scatter.

    `decode_owner` then runs the post-attention path (output proj / MLP /
    sample) using the combined output.
    """
    from vllm.distributed.dp_attn_group import (
        get_dp_attn_group,
        get_dp_attn_world_size,
    )

    assert impl.vllm_flash_attn_version is not None
    dp_world = get_dp_attn_world_size()
    dp_group = get_dp_attn_group()

    cu_seqlens_q = attn_metadata.query_start_loc
    max_seqlen_q = attn_metadata.max_query_len
    block_table = attn_metadata.block_table

    query = query.contiguous()
    # All ranks need Q. Gather along the head dim so the per-rank partials
    # produce head-major outputs that can be combined.
    query_across_dp = dp_group.all_gather(query, dim=1)

    sliding_window_size = (
        list(impl.sliding_window) if impl.sliding_window is not None else None
    )
    n = query_across_dp.shape[0]
    dp_dtype = impl._dcp_dtype if impl._dcp_dtype is not None else query.dtype
    dp_context_out = torch.empty(
        (n, impl.num_heads * dp_world, impl.head_size),
        dtype=dp_dtype,
        device=query.device,
    )

    # Partial attention against the local KV slice. seqused_k must reflect
    # the per-request token count physically resident on THIS rank — same
    # field DCP uses, populated by the build() path.
    context_attn_out, context_lse = flash_attn_varlen_func(
        q=query_across_dp,
        k=key_cache,
        v=value_cache,
        out=dp_context_out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=attn_metadata.dcp_context_kv_lens,
        max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
        softmax_scale=impl.scale,
        causal=False,
        alibi_slopes=impl.alibi_slopes,
        window_size=sliding_window_size,
        block_table=block_table,
        softcap=impl.logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=attn_metadata.scheduler_metadata,
        fa_version=impl.vllm_flash_attn_version,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        num_splits=attn_metadata.max_num_splits,
    )
    # all-gather output + reduce-scatter LSE.
    context_attn_out_cor, context_lse_cor = cp_lse_ag_out_rs(
        context_attn_out,
        context_lse.transpose(0, 1),
        dp_group,
        return_lse=True,
    )
    context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

    # The current step's Q/K/V (this token's contribution) is held only on
    # decode_owner. Merge it with the cross-DP context.
    dp_query_out = torch.empty(
        (query.shape[0], impl.num_heads, impl.head_size),
        dtype=dp_dtype,
        device=query.device,
    )
    query_attn_out, query_lse = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        out=dp_query_out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_k=cu_seqlens_q,
        max_seqlen_k=max_seqlen_q,
        softmax_scale=impl.scale,
        causal=attn_metadata.causal,
        alibi_slopes=impl.alibi_slopes,
        window_size=sliding_window_size,
        softcap=impl.logits_soft_cap,
        return_softmax_lse=True,
        fa_version=impl.vllm_flash_attn_version,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        num_splits=attn_metadata.max_num_splits,
    )
    assert context_attn_out_cor.shape == query_attn_out.shape
    assert context_lse_cor.shape == query_lse.shape
    merge_attn_states(
        output,
        context_attn_out_cor,
        context_lse_cor,
        query_attn_out,
        query_lse,
    )
    return output
