"""Build GLiNER2 classification prompts and decode vLLM's label logits.

Uses the ``gliner2`` package's own processor and result decoding, so prompts
and answers match ``AutoExtractor.classify_text``. Needs ``pip install
rh-custom-models-plugin[gliner2]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from gliner2.inference.runtime import ExtractorRuntimeMixin
from gliner2.processor import PreprocessedBatch, SchemaTransformer
from transformers import AutoTokenizer


@dataclass
class GLiNER2Request:
    prompt_token_ids: list[int]
    batch: PreprocessedBatch
    metadata: dict[str, Any]


class GLiNER2Client(ExtractorRuntimeMixin):
    """``classify_text`` split around a vLLM ``token_classify`` call."""

    def __init__(self, model: str, token_pooling: str = "first"):
        self.processor = SchemaTransformer(
            tokenizer=AutoTokenizer.from_pretrained(model),
            token_pooling=token_pooling,
        )
        self.processor.change_mode(is_training=False)
        self._label_id = self.processor.tokenizer.convert_tokens_to_ids("[L]")
        self._sep_text_id = self.processor.tokenizer.convert_tokens_to_ids("[SEP_TEXT]")

    @staticmethod
    def classifier(logits: torch.Tensor) -> torch.Tensor:
        # _extract_classification_result feeds the stacked label rows through
        # the classifier; here those rows already are vLLM's logits.
        return logits

    def build(self, text: str, tasks: dict[str, Any]) -> GLiNER2Request:
        """Tasks as for ``classify_text``: ``{name: labels | {labels, ...}}``."""
        if not tasks:
            raise ValueError("tasks must name at least one classification")
        for name, config in tasks.items():
            labels = config.get("labels") if isinstance(config, dict) else config
            if not isinstance(labels, (list, dict)) or not labels:
                raise ValueError(
                    f"task {name!r}: labels must be a non-empty list, or a dict "
                    "of label to description"
                )
        schema = self._classification_schema(tasks)
        (schema_dict,), (metadata,) = self._build_schema_dicts_and_metadata([schema])
        batch = self.processor.collate_fn_inference([(text, schema_dict)])
        ids = batch.input_ids[0, : batch.original_lengths[0]].tolist()

        # The server finds labels by token id; GLiNER2 routes them by position.
        # They differ only if a task prompt or description spells "[L]".
        sep = ids.index(self._sep_text_id)
        by_id = [i for i, t in enumerate(ids[:sep]) if t == self._label_id]
        by_position = [
            p for group in batch.schema_special_indices[0] for p in group[1:]
        ]
        if by_id != by_position:
            raise ValueError("task prompts and labels must not contain '[L]'")
        return GLiNER2Request(ids, batch, metadata)

    def decode(
        self,
        request: GLiNER2Request,
        logits: torch.Tensor,
        include_confidence: bool = False,
    ) -> dict[str, Any]:
        """``logits`` is the ``token_classify`` output: one row per label."""
        batch, schema_embs, start = request.batch, [], 0
        for group in batch.schema_special_indices[0]:
            rows = logits[start : start + len(group) - 1].float().reshape(-1, 1)
            schema_embs.append([torch.zeros(1), *rows])
            start += len(group) - 1
        if start != len(logits):
            raise ValueError(f"expected {start} label logits, got {len(logits)}")

        result = self._extract_sample(
            token_embs=torch.empty(0),
            schema_embs=schema_embs,
            schema_tokens_list=batch.schema_tokens_list[0],
            task_types=batch.task_types[0],
            text_tokens=batch.text_tokens[0],
            original_text=batch.original_texts[0],
            schema=batch.original_schemas[0],
            start_mapping=batch.start_mappings[0],
            end_mapping=batch.end_mappings[0],
            threshold=0.5,
            metadata=request.metadata,
            include_confidence=include_confidence,
            include_spans=False,
        )
        return self.format_results(
            result,
            include_confidence,
            request.metadata.get("relation_order", []),
            request.metadata.get("classification_tasks", []),
        )
