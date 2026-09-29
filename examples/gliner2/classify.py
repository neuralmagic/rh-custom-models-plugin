"""Zero-shot classification with GLiNER2 on vLLM.

    python examples/gliner2/classify.py

Online, serve the model and send the client's prompt ids to ``/pooling``:

    vllm serve fastino/GLiNER2.5-Decide-1B --config-format gliner2
    curl localhost:8000/pooling -d '{"input": [<prompt ids>], "task": "token_classify"}'
"""

import json

import torch
from vllm import LLM

from rh_custom_models_plugin.gliner2.client import GLiNER2Client

MODEL = "fastino/GLiNER2.5-Decide-1B"

TEXT = (
    "Guest in room 1408 says the AC has been out since yesterday and they want "
    "to move tonight or leave. They also asked for the incidentals hold to be "
    "released."
)
TASKS = {
    "intent": ["maintenance", "room_change", "checkout", "billing", "complaint"],
    "priority": ["low", "normal", "high", "urgent"],
    "needs_human": ["yes", "no"],
    "topics": {
        "labels": ["hvac", "billing", "housekeeping", "noise", "safety"],
        "multi_label": True,
        "cls_threshold": 0.4,
    },
}


def main() -> None:
    client = GLiNER2Client(MODEL)
    llm = LLM(MODEL, config_format="gliner2", runner="pooling")

    request = client.build(TEXT, TASKS)
    (output,) = llm.encode(
        [{"prompt_token_ids": request.prompt_token_ids}],
        pooling_task="token_classify",
    )
    answers = client.decode(request, torch.as_tensor(output.outputs.data))
    print(json.dumps(answers, indent=2))


if __name__ == "__main__":
    main()
