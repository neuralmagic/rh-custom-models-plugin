"""``POST /v1/classify``: GLiNER2's ``classify_text`` on the vLLM server.

Enabled by ``VLLM_PLUGINS=rh_gliner2``, which also loads the model family.

    {"text": "..." | ["...", ...], "tasks": {...}, "include_confidence": false}

``tasks`` takes the same shape as ``classify_text``. The response is
``{"results": [...]}``, one answer dict per text.
"""

from __future__ import annotations

import asyncio
import uuid
from argparse import Namespace
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.pooling_params import PoolingParams


class ClassifyRequest(BaseModel):
    text: str | list[str]
    tasks: dict[str, Any]
    include_confidence: bool = False


class GLiNER2Endpoint:
    name = "rh_gliner2"
    required_tasks = ("token_classify",)

    def attach_router(self, app: FastAPI) -> None:
        app.add_api_route("/v1/classify", self.classify, methods=["POST"])

    async def init_state(
        self, engine_client: EngineClient | None, state: State, args: Namespace
    ) -> None:
        from rh_custom_models_plugin.gliner2.client import GLiNER2Client

        state.gliner2_engine = engine_client
        state.gliner2_client = await asyncio.to_thread(
            GLiNER2Client, args.tokenizer or args.model
        )

    async def classify(self, body: ClassifyRequest, raw: Request) -> dict:
        engine = raw.app.state.gliner2_engine
        client = raw.app.state.gliner2_client
        texts = [body.text] if isinstance(body.text, str) else body.text
        try:
            requests = await asyncio.to_thread(
                lambda: [client.build(text, body.tasks) for text in texts]
            )
        except Exception as e:  # malformed tasks fail inside gliner2
            raise HTTPException(status_code=400, detail=str(e)) from e

        async def run(request) -> Any:
            output = None
            async for output in engine.encode(
                {"prompt_token_ids": request.prompt_token_ids},
                PoolingParams(task="token_classify"),
                f"classify-{uuid.uuid4().hex}",
            ):
                pass
            return client.decode(request, output.outputs.data, body.include_confidence)

        return {"results": await asyncio.gather(*(run(r) for r in requests))}
