"""Config parser for GLiNER2 checkpoints (``--config-format gliner2``).

GLiNER2 checkpoints keep the encoder's config in ``encoder_config/config.json``
next to a root ``config.json`` of ``model_type: "extractor"``, which
Transformers does not know. vLLM serves the encoder config, with the root
config and GLiNER2's marker token ids under ``gliner2_config``.
"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoConfig, PretrainedConfig
from vllm.transformers_utils.config_parser_base import ConfigParserBase
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

SUPPORTED_ENCODERS = ("modernbert",)
MARKER_TOKENS = ("[P]", "[L]", "[SEP_TEXT]")


class GLiNER2ConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        root = get_hf_file_to_dict("config.json", model, revision)
        encoder = get_hf_file_to_dict("encoder_config/config.json", model, revision)
        if root is None or encoder is None:
            raise ValueError(
                f"{model} is not a GLiNER2 checkpoint: it needs config.json and "
                "encoder_config/config.json"
            )
        if encoder.get("model_type") not in SUPPORTED_ENCODERS:
            raise ValueError(
                f"GLiNER2 with a {encoder.get('model_type')!r} encoder is not "
                f"supported; supported encoders: {SUPPORTED_ENCODERS}"
            )

        tokenizer = get_hf_file_to_dict("tokenizer.json", model, revision) or {}
        added = {t["content"]: t["id"] for t in tokenizer.get("added_tokens", [])}
        missing = [t for t in MARKER_TOKENS if t not in added]
        if missing:
            raise ValueError(f"{model} tokenizer lacks GLiNER2 tokens {missing}")

        config_dict = {
            **encoder,
            "architectures": ["GLiNER2ForClassification"],
            "gliner2_config": {
                **root,
                "marker_token_ids": {t: added[t] for t in MARKER_TOKENS},
            },
        }
        model_type = config_dict.pop("model_type")
        config = AutoConfig.for_model(model_type, **config_dict)
        return config_dict, config
