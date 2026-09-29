"""Tests for the BART MRV2 config check."""

from types import SimpleNamespace

import pytest
from vllm.model_executor.models.config import MODELS_CONFIG_MAP

from rh_custom_models_plugin.bart.config import (
    BART_ARCHITECTURES,
    BartMRV2Config,
    register_bart_config,
)


def test_register_bart_config_adds_mrv2_check():
    saved = {
        architecture: MODELS_CONFIG_MAP.pop(architecture, None)
        for architecture in BART_ARCHITECTURES
    }
    try:
        register_bart_config()

        for architecture in BART_ARCHITECTURES:
            assert MODELS_CONFIG_MAP[architecture] is BartMRV2Config
    finally:
        for architecture, config_cls in saved.items():
            if config_cls is None:
                MODELS_CONFIG_MAP.pop(architecture, None)
            else:
                MODELS_CONFIG_MAP[architecture] = config_cls


def test_bart_config_rejects_non_mrv2_runner():
    with pytest.raises(ValueError, match="require the V2 model runner"):
        BartMRV2Config.verify_and_update_config(
            SimpleNamespace(use_v2_model_runner=False)
        )


def test_bart_config_accepts_mrv2_runner():
    BartMRV2Config.verify_and_update_config(SimpleNamespace(use_v2_model_runner=True))
