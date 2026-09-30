# SPDX-License-Identifier: Apache-2.0
"""DeBERTa-v2/v3 encoder, the backbone of the DeBERTa GLiNER2 checkpoints.

Adapted from vllm-project/vllm#42094 (JLiu4Coding), not yet merged upstream.
The disentangled attention runs on vLLM's FlexAttention backend, with the
relative-position terms added by a ``score_mod`` (see ``deberta_flex.py``).
"""

import math

import torch
from torch import nn
from transformers import DebertaV2Config
from transformers.activations import ACT2FN
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.sequence import IntermediateTensors

from rh_custom_models_plugin.gliner2.deberta_flex import DebertaAttention


class DebertaV2Embeddings(nn.Module):
    """Word (+ optional token_type + position) embeddings for DeBERTa-v2/v3.

    DeBERTa-v3 sets ``position_biased_input=False`` and ``type_vocab_size=0``,
    so only word embeddings + LayerNorm are used in that variant.
    """

    def __init__(self, config: DebertaV2Config) -> None:
        super().__init__()
        self.config = config
        self.position_biased_input = getattr(config, "position_biased_input", True)

        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        if self.position_biased_input:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )

        if getattr(config, "type_vocab_size", 0) > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size
            )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)

        if self.position_biased_input:
            embeddings = embeddings + self.position_embeddings(positions)

        if token_type_ids is not None and hasattr(self, "token_type_embeddings"):
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        return self.LayerNorm(embeddings)


# ---------------------------------------------------------------------------
# Disentangled (relative-position) self-attention
# ---------------------------------------------------------------------------


class DebertaV2DisentangledSelfAttention(nn.Module):
    """DeBERTa disentangled attention: c2c + c2p + p2c."""

    def __init__(
        self,
        config: DebertaV2Config,
        max_relative_positions: int,
        max_len: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()

        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_heads})"
        )

        self.num_heads = num_heads // tp_size  # local heads per TP rank
        self.head_dim = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.head_dim  # local hidden

        self.max_relative_positions = max_relative_positions
        # For models with position_buckets (DeBERTa-v3), log-scale bucketing maps
        # linear relative distances to bucket indices; max_position_embeddings is
        # used as the "max_position" argument in the bucketing formula.
        self.position_buckets = getattr(config, "position_buckets", -1)
        self.max_position_embeddings = config.max_position_embeddings

        pos_att_type_raw = getattr(config, "pos_att_type", "p2c|c2p")
        # HF configs may store pos_att_type as a list or a "|"-separated string
        if isinstance(pos_att_type_raw, (list, tuple)):
            self.pos_att_type = [p.strip().lower() for p in pos_att_type_raw]
        else:
            self.pos_att_type = [p.strip() for p in pos_att_type_raw.lower().split("|")]
        if sorted(self.pos_att_type) != ["c2p", "p2c"]:
            raise NotImplementedError(
                f"pos_att_type {self.pos_att_type}: only c2p|p2c is supported"
            )
        self.share_att_key = getattr(config, "share_att_key", True)

        # scale = sqrt(head_dim * scale_factor) where scale_factor counts
        # the number of attention terms (1 c2c + n position terms)
        self.scale_factor = 1 + len(self.pos_att_type)
        self.scale = math.sqrt(self.head_dim * self.scale_factor)

        self.query_proj = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            prefix=f"{prefix}.query_proj",
        )
        self.key_proj = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            prefix=f"{prefix}.value_proj",
        )

        if not self.share_att_key:
            if "c2p" in self.pos_att_type:
                self.pos_key_proj = ColumnParallelLinear(
                    hidden_size,
                    hidden_size,
                    bias=True,
                    prefix=f"{prefix}.pos_key_proj",
                )
            if "p2c" in self.pos_att_type:
                self.pos_query_proj = ColumnParallelLinear(
                    hidden_size,
                    hidden_size,
                    bias=True,
                    prefix=f"{prefix}.pos_query_proj",
                )

        self.attn = DebertaAttention(
            self.num_heads,
            self.head_dim,
            1 / self.scale,
            prefix=f"{prefix}.attn",
        )
        # Table index for each relative position q - k in (-max_len, max_len).
        rel_pos = torch.arange(-max_len + 1, max_len)
        if self.position_buckets > 0:
            rel_pos = self._log_bucket_positions(rel_pos)
        att_span = self.max_relative_positions
        self.rel_offset = max_len - 1
        self.register_buffer(
            "rel_index",
            (rel_pos + att_span).clamp(0, 2 * att_span - 1),
            persistent=False,
        )

    def _log_bucket_positions(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """Apply DeBERTa-v2 log-scale position bucketing.

        Mirrors HF's ``make_log_bucket_position``:
          - positions within ±mid are kept linear
          - larger distances are log-compressed into [±mid, ±(bucket_size-1)]

        Args:
            rel_pos: [L, L] LongTensor of raw (linear) relative positions q-k.

        Returns:
            [L, L] LongTensor of bucketed positions in
            [-(bucket_size-1), bucket_size-1].
        """
        mid = self.position_buckets // 2  # half the bucket count (e.g. 128)
        max_pos = self.max_position_embeddings  # e.g. 512

        sign = torch.sign(rel_pos).float()
        abs_pos = torch.where(
            (rel_pos < mid) & (rel_pos > -mid),
            torch.full_like(rel_pos, mid - 1, dtype=torch.float),
            rel_pos.abs().float(),
        )
        log_ratio = math.log((max_pos - 1) / mid)
        log_pos = (
            torch.ceil(torch.log(abs_pos / mid) / log_ratio * (mid - 1)).float() + mid
        )
        bucket_pos = torch.where(
            rel_pos.abs() < mid,
            rel_pos.float(),
            log_pos * sign,
        )
        return bucket_pos.long()

    def _project_rel_emb(
        self, rel_emb: torch.Tensor, proj: ColumnParallelLinear
    ) -> torch.Tensor:
        """Project relative embeddings and reshape to [num_heads, head_dim, N]."""
        out, _ = proj(rel_emb)  # [N, all_head_size]
        # [N, num_heads, head_dim] → [num_heads, head_dim, N]
        return out.view(-1, self.num_heads, self.head_dim).permute(1, 2, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [tokens, hidden_size]
        positions: torch.Tensor,  # [tokens]; restarts at 0 for each sequence
        rel_embeddings: torch.Tensor,  # [2 * att_span, hidden_size]
    ) -> torch.Tensor:  # [tokens, all_head_size]
        q = self.query_proj(hidden_states)[0]
        k = self.key_proj(hidden_states)[0]
        v = self.value_proj(hidden_states)[0]

        def heads(x: torch.Tensor) -> torch.Tensor:  # -> [tokens, heads, head_dim]
            return x.view(-1, self.num_heads, self.head_dim)

        # [tokens, heads, 2 * att_span], scaled like the c2c scores
        proj = self.key_proj if self.share_att_key else self.pos_key_proj
        pos_key = self._project_rel_emb(rel_embeddings, proj)
        c2p = torch.einsum("thd,hds->ths", heads(q), pos_key) / self.scale
        proj = self.query_proj if self.share_att_key else self.pos_query_proj
        pos_query = self._project_rel_emb(rel_embeddings, proj)
        p2c = torch.einsum("thd,hds->ths", heads(k), pos_query) / self.scale

        self.attn.relative_terms = (
            c2p,
            p2c,
            positions,
            self.rel_index,
            self.rel_offset,
        )
        return self.attn(q, k, v)


# ---------------------------------------------------------------------------
# Attention output, intermediate, layer-output sub-modules
# ---------------------------------------------------------------------------


class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config: DebertaV2Config, prefix: str = "") -> None:
        super().__init__()
        self.dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        return self.LayerNorm(hidden_states + residual)


class DebertaV2Attention(nn.Module):
    def __init__(
        self,
        config: DebertaV2Config,
        max_relative_positions: int,
        max_len: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self = DebertaV2DisentangledSelfAttention(
            config, max_relative_positions, max_len, prefix=f"{prefix}.self"
        )
        self.output = DebertaV2SelfOutput(config, prefix=f"{prefix}.output")

    def forward(
        self,
        hidden_states: torch.Tensor,
        rel_embeddings: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        self_out = self.self(hidden_states, positions, rel_embeddings)
        return self.output(self_out, hidden_states)


class DebertaV2Intermediate(nn.Module):
    def __init__(self, config: DebertaV2Config, prefix: str = "") -> None:
        super().__init__()
        self.dense = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            prefix=f"{prefix}.dense",
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        return self.act_fn(hidden_states)


class DebertaV2LayerOutput(nn.Module):
    def __init__(self, config: DebertaV2Config, prefix: str = "") -> None:
        super().__init__()
        self.dense = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        return self.LayerNorm(hidden_states + residual)


# ---------------------------------------------------------------------------
# Transformer layer and encoder
# ---------------------------------------------------------------------------


class DebertaV2Layer(nn.Module):
    def __init__(
        self,
        config: DebertaV2Config,
        max_relative_positions: int,
        max_len: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attention = DebertaV2Attention(
            config, max_relative_positions, max_len, prefix=f"{prefix}.attention"
        )
        self.intermediate = DebertaV2Intermediate(
            config, prefix=f"{prefix}.intermediate"
        )
        self.output = DebertaV2LayerOutput(config, prefix=f"{prefix}.output")

    def forward(
        self,
        hidden_states: torch.Tensor,
        rel_embeddings: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attention(hidden_states, rel_embeddings, positions)
        intermediate_out = self.intermediate(attn_out)
        return self.output(intermediate_out, attn_out)


class DebertaV2Encoder(nn.Module):
    """Stack of DeBERTa transformer layers with a shared relative-position table."""

    def __init__(
        self,
        config: DebertaV2Config,
        max_relative_positions: int,
        max_len: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [
                DebertaV2Layer(
                    config,
                    max_relative_positions,
                    max_len,
                    prefix=f"{prefix}.layer.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # Shared relative position embedding table
        self.rel_embeddings = nn.Embedding(
            2 * max_relative_positions, config.hidden_size
        )

        # Optional LayerNorm applied to rel_embeddings before each forward pass
        norm_rel_ebd = getattr(config, "norm_rel_ebd", "none")
        if norm_rel_ebd.lower() != "none":
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _get_rel_embeddings(self) -> torch.Tensor:
        rel_emb = self.rel_embeddings.weight
        if hasattr(self, "LayerNorm"):
            rel_emb = self.LayerNorm(rel_emb)
        return rel_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        rel_embeddings = self._get_rel_embeddings()
        for layer in self.layer:
            hidden_states = layer(hidden_states, rel_embeddings, positions)
        return hidden_states


# ---------------------------------------------------------------------------
# Backbone model
# ---------------------------------------------------------------------------


class DebertaV2Model(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: DebertaV2Config = vllm_config.model_config.hf_config
        self.config = config

        # Resolve max_relative_positions from config
        max_relative_positions = getattr(config, "max_relative_positions", -1)
        if max_relative_positions < 1:
            position_buckets = getattr(config, "position_buckets", -1)
            max_relative_positions = (
                position_buckets
                if position_buckets > 0
                else config.max_position_embeddings
            )
        self.max_relative_positions = max_relative_positions

        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(
            config,
            max_relative_positions,
            vllm_config.model_config.max_model_len,
            prefix=f"{prefix}.encoder",
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            hidden_states = self.embeddings(input_ids, positions, token_type_ids)
        else:
            hidden_states = inputs_embeds
        return self.encoder(hidden_states, positions)
