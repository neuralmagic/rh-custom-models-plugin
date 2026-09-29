"""GLiNER2 zero-shot classification (ModernBERT encoders).

Serve with ``--config-format gliner2``; build prompts and decode answers with
``rh_custom_models_plugin.gliner2.client.GLiNER2Client``.
"""

ARCHITECTURES = {
    "GLiNER2ForClassification": (
        "rh_custom_models_plugin.gliner2.model:GLiNER2ForClassification"
    ),
}


def register() -> None:
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.config import (
        get_config_parser,
        register_config_parser,
    )

    from rh_custom_models_plugin.gliner2.config import GLiNER2ConfigParser

    try:
        get_config_parser("gliner2")
    except ValueError:
        register_config_parser("gliner2")(GLiNER2ConfigParser)
    for architecture, model_ref in ARCHITECTURES.items():
        ModelRegistry.register_model(architecture, model_ref)
