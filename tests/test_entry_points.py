"""Every model family entry point follows the repository's conventions.

vLLM runs general plugins in the API server, engine core and each worker, so
registration must work from a bare import and survive running more than once.
"""

import importlib
from importlib.metadata import entry_points

import pytest
from vllm.model_executor.models.registry import ModelRegistry

PACKAGE = "rh_custom_models_plugin"

FAMILIES = sorted(
    (
        ep
        for ep in entry_points(group="vllm.general_plugins")
        if ep.value.startswith(PACKAGE)
    ),
    key=lambda ep: ep.name,
)


def test_families_declared():
    assert FAMILIES, (
        "no rh_custom_models_plugin entry points; is the package installed?"
    )


@pytest.mark.parametrize("ep", FAMILIES, ids=lambda ep: ep.name)
def test_family_entry_point(ep):
    module_name, _, attr = ep.value.partition(":")
    family = module_name.removeprefix(f"{PACKAGE}.")
    assert ep.name == f"rh_{family}"
    assert attr == "register"

    module = importlib.import_module(module_name)
    ep.load()()
    ep.load()()

    supported = ModelRegistry.get_supported_archs()
    for architecture, model_ref in module.ARCHITECTURES.items():
        assert architecture in supported
        cls_module, _, cls_name = model_ref.partition(":")
        assert hasattr(importlib.import_module(cls_module), cls_name), model_ref
