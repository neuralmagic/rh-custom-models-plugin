# rh-custom-models-plugin

A [vLLM plugin](https://docs.vllm.ai/en/latest/design/plugin_system.html) for model definitions that are not carried in vLLM upstream. Install it next to vLLM and the models load like built-in ones.

## Model families

| Entry point | Architectures | Source |
|---|---|---|
| `rh_bart` | `BartForConditionalGeneration`, `Florence2ForConditionalGeneration` | [vllm-project/bart-plugin](https://github.com/vllm-project/bart-plugin) @ `4da3192` |

Requires vLLM 0.30 or newer.

### Status

- **BART** runs on vLLM 0.30.0. It was ported from `bart-plugin`, which targets older vLLM, for three vLLM API changes: the removed MRV2 architecture allowlist, `AutoWeightsLoader` skip lists, and the multimodal processor hook.
- **Florence-2** still uses the removed `_call_hf_processor` hook and does not load on vLLM 0.30 yet.
- `tests/bart/test_model_initialization.py` is still written for the older vLLM API: its tests build the model outside a vLLM config context, and two of them use a `small_model_name` fixture that does not exist.

## Install

```bash
uv pip install git+https://github.com/neuralmagic/rh-custom-models-plugin.git
# or, from a checkout
uv pip install -e ".[test,lint]"
```

vLLM loads every installed `vllm.general_plugins` entry point. To load only some families, list their entry points:

```bash
VLLM_PLUGINS=rh_bart vllm serve facebook/bart-large-cnn
```

Uninstall `vllm-bart-plugin` first if it is installed: both register the same architectures.

## Usage

### BART

BART is an encoder-decoder model. The encoder text goes in `multi_modal_data`:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/bart-large-cnn", max_model_len=1024)
outputs = llm.generate(
    {
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {"text": "The president of the United States is"},
        },
        "decoder_prompt": "<s>",
    },
    SamplingParams(temperature=0.0, max_tokens=20),
)
```

Plain `/v1/completions` prompts are routed to the encoder by the plugin. See [`examples/bart`](examples/bart) for BART and Florence-2.

## Adding a model family

1. Add `src/rh_custom_models_plugin/<family>/` with:
   - the model code, with imports under `rh_custom_models_plugin.<family>`;
   - an `__init__.py` exposing `ARCHITECTURES`, a map from architecture name to `"module:Class"`, and a `register()` that calls `ModelRegistry.register_model` for each one, along with any config hooks the family needs.
2. Add the entry point to `pyproject.toml`: `rh_<family> = "rh_custom_models_plugin.<family>:register"`.
3. Put tests in `tests/<family>/` and mark tests that load weights or need a GPU with `@pytest.mark.slow`. Add a row to the table above.

Keep `register()` cheap and repeatable. vLLM runs it in the API server, the engine core and every worker, sometimes more than once per process. Register models as `"module:Class"` strings so the model code is only imported when that model is used. `tests/test_entry_points.py` checks these conventions for every family.

## Tests

```bash
pytest tests -m "not slow"   # registration and CPU-only tests
pytest tests                 # everything; needs a GPU
```

CI runs `ruff` only.
