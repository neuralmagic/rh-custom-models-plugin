"""Florence-2 (DaViT vision encoder, BART language backbone).

Imported from https://github.com/vllm-project/bart-plugin at 4da3192. Uses the
BART modules, config check and OpenAI prompt adapter from the ``bart`` family.
"""

ARCHITECTURES = {
    "Florence2ForConditionalGeneration": (
        "rh_custom_models_plugin.florence2.model:Florence2ForConditionalGeneration"
    ),
}


def register() -> None:
    from vllm.model_executor.models.registry import ModelRegistry

    from rh_custom_models_plugin.bart.config import register_bart_config
    from rh_custom_models_plugin.bart.openai_serving import (
        install_openai_prompt_adapter,
    )

    register_bart_config()
    for architecture, model_ref in ARCHITECTURES.items():
        ModelRegistry.register_model(architecture, model_ref)
    install_openai_prompt_adapter()
