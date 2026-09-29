"""GLiNER2 on vLLM against the gliner2 package's own ``classify_text``."""

import pytest
import torch

MODEL = "fastino/GLiNER2.5-Decide-1B"

pytestmark = pytest.mark.slow

CASES = [
    (
        "My subscription renewed after the service was already down. "
        "Can I get that charge refunded?",
        {"intent": ["order_status", "refund_request", "cancel_subscription", "other"]},
    ),
    (
        "Guest in room 1408 says the AC has been out since yesterday and they "
        "want to move tonight. They also asked for the incidentals hold to be "
        "released.",
        {
            "intent": ["maintenance", "room_change", "checkout", "billing"],
            "priority": ["low", "normal", "high", "urgent"],
            "topics": {
                "labels": ["hvac", "billing", "housekeeping", "noise"],
                "multi_label": True,
                "cls_threshold": 0.4,
            },
        },
    ),
    (
        "Please reset the card PIN. The new one never arrived.",
        {
            "intent": {
                "labels": {
                    "card_pin_change": "The customer wants a new PIN",
                    "card_lost": "The physical card is missing",
                },
            }
        },
    ),
    (
        "The treaty was signed in Paris in 1992 and entered into force the "
        "following year.",
        {
            "answer": {
                "labels": ["yes", "no"],
                "prompt": "Did the treaty enter into force in 1992?",
            }
        },
    ),
]


def _assert_close(got, want):
    if isinstance(want, list):
        assert [g["label"] for g in got] == [w["label"] for w in want]
        for g, w in zip(got, want):
            assert g["confidence"] == pytest.approx(w["confidence"], abs=1e-2)
    else:
        assert got["label"] == want["label"]
        assert got["confidence"] == pytest.approx(want["confidence"], abs=1e-2)


@torch.inference_mode()
def test_matches_gliner2(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from gliner2 import AutoExtractor
    from vllm import LLM

    from rh_custom_models_plugin.gliner2.client import GLiNER2Client

    client = GLiNER2Client(MODEL)
    requests = [client.build(text, tasks) for text, tasks in CASES]
    llm = LLM(
        MODEL,
        config_format="gliner2",
        runner="pooling",
        dtype="float32",
        enforce_eager=True,
        gpu_memory_utilization=0.3,
    )
    outputs = llm.encode(
        [{"prompt_token_ids": r.prompt_token_ids} for r in requests],
        pooling_task="token_classify",
    )
    got = [
        client.decode(r, o.outputs.data, include_confidence=True)
        for r, o in zip(requests, outputs)
    ]
    del llm

    reference = AutoExtractor.from_pretrained(MODEL).to("cuda").eval()
    for (text, tasks), answers in zip(CASES, got):
        want = reference.classify_text(text, tasks, include_confidence=True)
        assert answers.keys() == want.keys()
        for name in want:
            _assert_close(answers[name], want[name])
