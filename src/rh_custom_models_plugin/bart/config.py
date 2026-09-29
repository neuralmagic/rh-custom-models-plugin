"""vLLM config hooks for BART-family models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    VerifyAndUpdateConfig,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# Architectures with a BART decoder, across the bart and florence2 families.
BART_ARCHITECTURES = (
    "BartForConditionalGeneration",
    "Florence2ForConditionalGeneration",
)


class BartMRV2Config(VerifyAndUpdateConfig):
    """Require MRV2 for BART-family architectures."""

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        if not vllm_config.use_v2_model_runner:
            raise ValueError(
                "BART-family models require the V2 model runner. "
                "Unset VLLM_USE_V2_MODEL_RUNNER or set it to 1."
            )


def register_bart_config() -> None:
    for architecture in BART_ARCHITECTURES:
        MODELS_CONFIG_MAP[architecture] = BartMRV2Config
