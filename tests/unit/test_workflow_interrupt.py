import asyncio

import pytest

from comfy.cli_args import default_configuration
from comfy.client.embedded_comfy_client import Comfy
from comfy.entrypoints import workflow as workflow_entrypoint


@pytest.mark.asyncio
async def test_run_workflows_interrupts_processing_on_cancel(monkeypatch):
    calls = []

    async def fake_stream_json_objects(_workflow):
        yield {
            "1": {
                "class_type": "SaveImage",
                "inputs": {"images": ["2", 0], "filename_prefix": "unused"},
            },
            "2": {"class_type": "PreviewImage", "inputs": {}},
        }

    class FakeComfy:
        def __init__(self, configuration=None):
            self.configuration = configuration

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def queue_prompt_api(self, _prompt):
            raise asyncio.CancelledError

    monkeypatch.setattr(workflow_entrypoint, "Comfy", FakeComfy)
    monkeypatch.setattr(workflow_entrypoint, "stream_json_objects", fake_stream_json_objects)
    monkeypatch.setattr(
        workflow_entrypoint.interruption,
        "interrupt_current_processing",
        lambda value=True: calls.append(value),
    )

    config = default_configuration()
    config.disable_progress = True

    with pytest.raises(asyncio.CancelledError):
        await workflow_entrypoint.run_workflows(["dummy.json"], configuration=config)

    assert calls == [True, True]


@pytest.mark.asyncio
async def test_comfy_async_exit_interrupts_processing_on_cancel(monkeypatch):
    calls = []

    from comfy import interruption

    monkeypatch.setattr(
        interruption,
        "interrupt_current_processing",
        lambda value=True: calls.append(value),
    )

    config = default_configuration()
    client = Comfy(configuration=config)

    await client.__aenter__()
    await client.__aexit__(asyncio.CancelledError, asyncio.CancelledError(), None)

    assert calls == [True]
