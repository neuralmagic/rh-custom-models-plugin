# SPDX-License-Identifier: Apache-2.0
"""vLLM's FlexAttention backend with DeBERTa's disentangled attention.

DeBERTa adds two terms to every attention score: content-to-position,
``q_i . PK[r(i, j)]``, and position-to-content, ``k_j . PQ[r(i, j)]``, where
``r`` buckets the relative position ``i - j``. No fused backend can add these,
but FlexAttention's ``score_mod`` can. Each layer attaches its tables to its
attention layer before calling it, and the backend installs a ``score_mod``
that reads them, then runs vLLM's encoder-only flex path unchanged.
"""

import torch
from vllm.config import CacheConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.encoder_only_attention import (
    create_encoder_only_attention_backend,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionBackend,
    FlexAttentionImpl,
)
from vllm.v1.kv_cache_interface import KVCacheSpec


def _relative_score_mod(
    c2p: torch.Tensor,  # [tokens, heads, 2 * att_span], already scaled
    p2c: torch.Tensor,  # [tokens, heads, 2 * att_span], already scaled
    positions: torch.Tensor,  # [tokens]; restarts at 0 for each sequence
    rel_index: torch.Tensor,  # relative position + offset -> table index
    offset: int,
):
    def score_mod(score, b, h, q_idx, kv_idx):
        r = rel_index[positions[q_idx] - positions[kv_idx] + offset]
        return score + c2p[q_idx, h, r] + p2c[kv_idx, h, r]

    return score_mod


class DebertaFlexAttentionImpl(FlexAttentionImpl):
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, *args, **kw):
        if attn_metadata is not None:
            attn_metadata.transformed_score_mod = _relative_score_mod(
                *layer.relative_terms
            )
        return super().forward(
            layer, query, key, value, kv_cache, attn_metadata, *args, **kw
        )


class DebertaFlexAttentionBackend(FlexAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[FlexAttentionImpl]:
        return DebertaFlexAttentionImpl


class DebertaAttention(Attention):
    """Encoder-only attention on ``DebertaFlexAttentionBackend``.

    Set ``relative_terms`` before each call:
    ``(c2p, p2c, positions, rel_index, offset)``.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        cache_config: CacheConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            cache_config=cache_config,
            attn_backend=create_encoder_only_attention_backend(
                DebertaFlexAttentionBackend
            ),
            attn_type=AttentionType.ENCODER_ONLY,
            **kwargs,
        )
        self.relative_terms = None

    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec | None:
        return None
