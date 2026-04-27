# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2MoE model compatible with HuggingFace weights."""

import os
from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
from itertools import islice
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2MoeConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

_DECODE_ATTN_NVTX_ENABLE_ENV = "VLLM_QWEN2MOE_DECODE_ATTN_NVTX"
_DECODE_ATTN_VERIFY_ENV = "VLLM_QWEN2MOE_DECODE_ATTN_VERIFY"
_DECODE_ATTN_VERIFY_MAX_LOGS_ENV = (
    "VLLM_QWEN2MOE_DECODE_ATTN_VERIFY_MAX_LOGS"
)
_DECODE_ATTN_NVTX_BATCH_INFO_KEY = "_qwen2_moe_decode_attn_nvtx_batch_info"
_DECODE_ATTN_NVTX_STEP_KEY = "_qwen2_moe_decode_attn_nvtx_step"
_DECODE_ATTN_NVTX_REQUEST_ID_KEYS = (
    "req_ids",
    "request_ids",
    "request_ids_output_copy",
    "scheduled_req_ids",
)
_DECODE_ATTN_NVTX_STEP_COUNTER = 0
_DECODE_ATTN_VERIFY_LOG_COUNTER = 0


def _infer_decode_counts(
    attn_metadata: Any,
) -> tuple[int, int, int]:
    """Return (num_decodes, num_decode_tokens, num_prefill_tokens).

    Works across all attention-metadata flavours:
    * FlashInfer / ROCm Aiter: have ``num_decodes``, ``num_decode_tokens``,
      ``num_prefill_tokens`` attributes directly.
    * FlashAttention (V1, NVIDIA): these attributes are absent.  We fall back
      to ``query_start_loc`` (cumulative query lengths per request) and count
      requests whose query length is exactly 1 as decode requests.
    """
    # Fast path: metadata already carries the fields (FlashInfer, ROCm Aiter).
    nd = getattr(attn_metadata, "num_decodes", None)
    ndt = getattr(attn_metadata, "num_decode_tokens", None)
    npt = getattr(attn_metadata, "num_prefill_tokens", None)
    if nd is not None and ndt is not None and npt is not None:
        return int(nd), int(ndt), int(npt)

    # Slow path: derive from query_start_loc (FlashAttention V1).
    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    if query_start_loc is not None and query_start_loc.numel() > 1:
        qsl = query_start_loc
        if qsl.device.type != "cpu":
            qsl = qsl.cpu()
        query_lens = (qsl[1:] - qsl[:-1]).tolist()
        num_decodes = 0
        num_decode_tokens = 0
        num_prefill_tokens = 0
        for qlen in query_lens:
            if qlen == 1:
                num_decodes += 1
                num_decode_tokens += 1
            else:
                num_prefill_tokens += qlen
        return num_decodes, num_decode_tokens, num_prefill_tokens

    return 0, 0, 0


def _decode_attn_nvtx_enabled() -> bool:
    return bool(int(os.getenv(_DECODE_ATTN_NVTX_ENABLE_ENV, "1")))


def _decode_attn_verify_enabled() -> bool:
    return bool(int(os.getenv(_DECODE_ATTN_VERIFY_ENV, "0")))


def _decode_attn_verify_max_logs() -> int:
    try:
        return max(0, int(os.getenv(_DECODE_ATTN_VERIFY_MAX_LOGS_ENV, "50")))
    except ValueError:
        return 50


def _slice_int_list(values: Any, limit: int) -> list[int]:
    if values is None:
        return []
    if torch.is_tensor(values):
        tensor = values[:limit]
        if tensor.device.type != "cpu":
            tensor = tensor.to("cpu")
        return [int(value) for value in tensor.tolist()]
    if hasattr(values, "tolist"):
        return [int(value) for value in values.tolist()[:limit]]
    return [int(value) for value in islice(values, limit)]


def _extract_decode_query_lens(
    attn_metadata: Any,
    num_decodes: int,
    num_decode_tokens: int,
) -> list[int]:
    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    if query_start_loc_cpu is not None:
        query_start_locs = _slice_int_list(query_start_loc_cpu, num_decodes + 1)
        if len(query_start_locs) == num_decodes + 1:
            query_lens = [
                query_start_locs[idx + 1] - query_start_locs[idx]
                for idx in range(num_decodes)
            ]
            if sum(query_lens) == num_decode_tokens:
                return query_lens

    if num_decodes <= 0:
        return []
    base, remainder = divmod(num_decode_tokens, num_decodes)
    return [base + (1 if idx < remainder else 0) for idx in range(num_decodes)]


def _extract_decode_seq_lens(attn_metadata: Any, num_decodes: int) -> list[int]:
    seq_lens = _slice_int_list(getattr(attn_metadata, "seq_lens_cpu", None), num_decodes)
    if len(seq_lens) == num_decodes:
        return seq_lens
    return _slice_int_list(getattr(attn_metadata, "seq_lens", None), num_decodes)


def _extract_decode_request_keys(
    additional_kwargs: dict[str, Any], num_decodes: int
) -> tuple[list[str], str]:
    if num_decodes <= 0:
        return [], "none"

    for key in _DECODE_ATTN_NVTX_REQUEST_ID_KEYS:
        value = additional_kwargs.get(key)
        if isinstance(value, dict):
            request_keys = [str(req_id) for req_id in value.keys()]
        elif isinstance(value, (list, tuple)):
            request_keys = [str(req_id) for req_id in value]
        else:
            continue
        if len(request_keys) >= num_decodes:
            return request_keys[:num_decodes], f"forward_context.{key}"

    return [f"batch_pos_{idx}" for idx in range(num_decodes)], "batch_position_fallback"


def _get_worker_rank(additional_kwargs: dict[str, Any]) -> str:
    rank = additional_kwargs.get("worker_rank")
    if rank is not None:
        return str(rank)
    return os.getenv("RANK", "na")


def _get_worker_local_rank(additional_kwargs: dict[str, Any]) -> str:
    local_rank = additional_kwargs.get("worker_local_rank")
    if local_rank is not None:
        return str(local_rank)
    return os.getenv("LOCAL_RANK", "na")


def _format_nvtx_list(values: list[Any], *, limit: int = 16) -> str:
    shown = [str(value) for value in values[:limit]]
    if len(values) > limit:
        shown.append("...")
    return "[" + ",".join(shown) + "]"


def _next_decode_attn_step(forward_context: Any) -> int:
    global _DECODE_ATTN_NVTX_STEP_COUNTER
    step_idx = forward_context.additional_kwargs.get(_DECODE_ATTN_NVTX_STEP_KEY)
    if isinstance(step_idx, int):
        return step_idx
    _DECODE_ATTN_NVTX_STEP_COUNTER += 1
    step_idx = _DECODE_ATTN_NVTX_STEP_COUNTER
    forward_context.additional_kwargs[_DECODE_ATTN_NVTX_STEP_KEY] = step_idx
    return step_idx


def _get_decode_attn_batch_info(
    forward_context: Any,
    attn_metadata: Any,
    num_decodes: int,
    num_decode_tokens: int,
) -> dict[str, Any]:
    batch_info = forward_context.additional_kwargs.get(_DECODE_ATTN_NVTX_BATCH_INFO_KEY)
    if (
        isinstance(batch_info, dict)
        and batch_info.get("num_decodes") == num_decodes
        and batch_info.get("num_decode_tokens") == num_decode_tokens
    ):
        return batch_info

    batch_info = {
        "num_decodes": num_decodes,
        "num_decode_tokens": num_decode_tokens,
        "request_keys": _extract_decode_request_keys(
            forward_context.additional_kwargs, num_decodes
        ),
        "query_lens": _extract_decode_query_lens(
            attn_metadata, num_decodes, num_decode_tokens
        ),
        "seq_lens": _extract_decode_seq_lens(attn_metadata, num_decodes),
    }
    forward_context.additional_kwargs[_DECODE_ATTN_NVTX_BATCH_INFO_KEY] = batch_info
    return batch_info


def _maybe_log_decode_attn_condition(
    *,
    forward_context: Any,
    attn_metadata: Any,
    layer_idx: int,
    layer_name: str,
    num_prefill_tokens: int,
    num_decode_tokens: int,
    num_decodes: int,
) -> None:
    global _DECODE_ATTN_VERIFY_LOG_COUNTER

    if not _decode_attn_verify_enabled():
        return

    condition_met = (
        num_prefill_tokens == 0 and num_decode_tokens > 0 and num_decodes > 0
    )
    if condition_met and layer_idx != 0:
        return

    if _DECODE_ATTN_VERIFY_LOG_COUNTER >= _decode_attn_verify_max_logs():
        return
    _DECODE_ATTN_VERIFY_LOG_COUNTER += 1

    request_keys = []
    request_key_kind = "na"
    query_lens = []
    seq_lens = []
    if num_decodes > 0:
        batch_info = _get_decode_attn_batch_info(
            forward_context, attn_metadata, num_decodes, num_decode_tokens
        )
        request_keys, request_key_kind = batch_info["request_keys"]
        query_lens = batch_info["query_lens"]
        seq_lens = batch_info["seq_lens"]

    log_fn = logger.info if condition_met else logger.warning
    log_fn(
        "qwen2_moe decode-attn verify layer=%s layer_name=%s rank=%s "
        "local_rank=%s condition_met=%s num_prefill_tokens=%s "
        "num_decode_tokens=%s num_decodes=%s request_key_kind=%s "
        "request_keys=%s query_lens=%s seq_lens=%s",
        layer_idx,
        layer_name,
        _get_worker_rank(forward_context.additional_kwargs),
        _get_worker_local_rank(forward_context.additional_kwargs),
        condition_met,
        num_prefill_tokens,
        num_decode_tokens,
        num_decodes,
        request_key_kind,
        request_keys,
        query_lens,
        seq_lens,
    )


def _build_decode_attn_nvtx_label(
    *,
    forward_context: Any,
    attn_metadata: Any,
    layer_idx: int,
    layer_name: str,
    num_decodes: int,
    num_decode_tokens: int,
) -> str:
    batch_info = _get_decode_attn_batch_info(
        forward_context, attn_metadata, num_decodes, num_decode_tokens
    )
    request_keys, request_key_kind = batch_info["request_keys"]
    query_lens = batch_info["query_lens"]
    seq_lens = batch_info["seq_lens"]
    step_idx = _next_decode_attn_step(forward_context)
    rank = _get_worker_rank(forward_context.additional_kwargs)
    local_rank = _get_worker_local_rank(forward_context.additional_kwargs)
    return (
        f"decode_attn"
        f" step={step_idx}"
        f" layer={layer_idx}"
        f" rank={rank}"
        f" local_rank={local_rank}"
        f" num_reqs={num_decodes}"
        f" num_tokens={num_decode_tokens}"
        f" req_key_kind={request_key_kind}"
        f" layer_name={layer_name}"
        f" req_keys={_format_nvtx_list(request_keys)}"
        f" qlens={_format_nvtx_list(query_lens)}"
        f" slens={_format_nvtx_list(seq_lens)}"
    )


@contextmanager
def _decode_attn_nvtx_range(label: str):
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "nvtx"):
        yield
        return
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class Qwen2MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.expert_gate = expert_gate

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)

        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out

        return out


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None

        self.experts = FusedMoE(
            shared_experts=self.shared_expert,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        return final_hidden_states.view(orig_shape)


class Qwen2MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_context = nullcontext()
        if ((_decode_attn_nvtx_enabled() or _decode_attn_verify_enabled())
                and is_forward_context_available()):
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata.get(self.attn.layer_name)
            num_decodes, num_decode_tokens, num_prefill_tokens = (
                _infer_decode_counts(attn_metadata)
            )
            _maybe_log_decode_attn_condition(
                forward_context=forward_context,
                attn_metadata=attn_metadata,
                layer_idx=self.layer_idx,
                layer_name=self.attn.layer_name,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                num_decodes=num_decodes,
            )
            if (_decode_attn_nvtx_enabled() and num_prefill_tokens == 0
                    and num_decode_tokens > 0 and num_decodes > 0):
                attn_context = _decode_attn_nvtx_range(
                    _build_decode_attn_nvtx_label(
                        forward_context=forward_context,
                        attn_metadata=attn_metadata,
                        layer_idx=self.layer_idx,
                        layer_name=self.attn.layer_name,
                        num_decodes=num_decodes,
                        num_decode_tokens=num_decode_tokens,
                    )
                )
        with attn_context:
            attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = Qwen2MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # Note: Qwen/Qwen2-57B-A14B-Instruct does not have
        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen2MoeSparseMoeBlock(
                config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Qwen2MoeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen2MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale"
                        )
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv_scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv_scale is not loaded.",  #  noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    # GGUF: make sure that shared_expert_gate is a 2D tensor.
                    if (
                        "mlp.shared_expert_gate" in name
                        and len(loaded_weight.shape) == 1
                    ):
                        loaded_weight = loaded_weight[None, :]
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen2MoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    fall_back_to_pt_during_load = False
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # Only perform the following mapping when Qwen2MoeMLP exists
        if (
            getattr(config, "mlp_only_layers", [])
            or config.shared_expert_intermediate_size > 0
        ):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]

        self.model = Qwen2MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
