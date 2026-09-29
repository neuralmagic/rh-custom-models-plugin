"""GLiNER2 classification (https://github.com/fastino-ai/GLiNER2).

The prompt holds one ``( [P] task ( [L] label [L] label ... ) )`` group per
classification task, then ``[SEP_TEXT]`` and the text. ``token_classify``
returns one row per ``[L]`` marker ahead of ``[SEP_TEXT]``: the label's raw
logit. Softmax or sigmoid, thresholds and grouping by task are applied by the
client (see ``client.py``), since they are chosen per request.
"""

from collections.abc import Iterable, Set

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.models.interfaces_base import (
    attn_type,
    default_pooling_type,
)
from vllm.model_executor.models.modernbert import ModernBertModel
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.sequence import IntermediateTensors
from vllm.tasks import PoolingTask
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata


class GLiNER2LabelPooler(Pooler):
    def __init__(self, hidden_size: int, label_id: int, sep_text_id: int):
        super().__init__()
        self.label_id = label_id
        self.sep_text_id = sep_text_id
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, 1),
        )

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        cursor = pooling_metadata.get_pooling_cursor()
        offsets = torch.cumsum(cursor.num_scheduled_tokens_cpu, 0)
        offsets = torch.cat([offsets.new_zeros(1), offsets[:-1]])

        marker_idx, num_markers = [], []
        for token_ids, offset in zip(
            pooling_metadata.get_prompt_token_ids_cpu(), offsets
        ):
            sep = (token_ids == self.sep_text_id).nonzero()
            schema_ids = token_ids[: int(sep[0])] if len(sep) else token_ids
            idx = (schema_ids == self.label_id).nonzero(as_tuple=True)[0]
            marker_idx.append(idx + offset)
            num_markers.append(len(idx))

        flat_idx = async_tensor_h2d(torch.cat(marker_idx), hidden_states.device)
        logits = self.classifier(hidden_states[flat_idx]).float()
        return list(logits.split(num_markers))


@attn_type("encoder_only")
@default_pooling_type(tok_pooling_type="ALL")
class GLiNER2ForClassification(nn.Module):
    is_pooling_model = True

    # Span extraction and counting heads are not served.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "classifier.": "pooler.classifier.",
            "span_rep.": None,
            "count_embed.": None,
            "count_pred.": None,
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        marker_ids = config.gliner2_config["marker_token_ids"]
        self.encoder = ModernBertModel(vllm_config=vllm_config, prefix="encoder")
        self.pooler = GLiNER2LabelPooler(
            config.hidden_size, marker_ids["[L]"], marker_ids["[SEP_TEXT]"]
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(
            input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
