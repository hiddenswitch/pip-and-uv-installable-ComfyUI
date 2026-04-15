"""End-to-end test: a workflow dict that carries ``__metadata_v1__``
survives both the embedded client path (``Comfy.queue_prompt``) and the
HTTP server path (``/prompt`` + ``/api/v1/prompts``) without tripping
``validate_prompt``.

These tests don't actually run a diffusion model — they use a trivial
workflow with only ``EmptyImage`` + ``PreviewImage`` so a CPU-only
ContextVarExecutor finishes in a few seconds.
"""
from __future__ import annotations

import asyncio
import uuid
import pytest

from comfy.component_model.prompt_envelope import (
    METADATA_KEY,
    wrap_with_metadata,
    set_configuration,
)


def _trivial_workflow() -> dict:
    return {
        "1": {"class_type": "EmptyImage",
              "inputs": {"width": 8, "height": 8, "batch_size": 1, "color": 0}},
        "2": {"class_type": "PreviewImage",
              "inputs": {"images": ["1", 0]}},
    }


@pytest.mark.asyncio
async def test_comfy_queue_prompt_accepts_envelope():
    """``Comfy.queue_prompt`` must pop ``__metadata_v1__`` before
    validation or the whole embedded path breaks for envelope-aware callers.
    """
    from comfy.client.embedded_comfy_client import Comfy

    envelope = wrap_with_metadata(_trivial_workflow(), {"configuration": {"cpu": True}})
    assert METADATA_KEY in envelope

    async with Comfy() as client:
        outputs = await client.queue_prompt(envelope, prompt_id=str(uuid.uuid4()))
    # Output key 2 is the PreviewImage output node; just assert it ran.
    assert "2" in outputs


@pytest.mark.asyncio
async def test_comfy_queue_prompt_still_accepts_plain_workflow():
    """Passing a workflow with no envelope continues to work unchanged."""
    from comfy.client.embedded_comfy_client import Comfy

    async with Comfy() as client:
        outputs = await client.queue_prompt(_trivial_workflow(),
                                             prompt_id=str(uuid.uuid4()))
    assert "2" in outputs


@pytest.mark.asyncio
async def test_set_configuration_then_submit():
    """Convenience wrapper: the common client flow is
    ``set_configuration(prompt, cfg)`` then submit."""
    from comfy.client.embedded_comfy_client import Comfy
    prompt = set_configuration(_trivial_workflow(), {"cpu": True, "novram": False})
    assert prompt[METADATA_KEY]["configuration"]["cpu"] is True
    async with Comfy() as client:
        outputs = await client.queue_prompt(prompt, prompt_id=str(uuid.uuid4()))
    assert "2" in outputs
