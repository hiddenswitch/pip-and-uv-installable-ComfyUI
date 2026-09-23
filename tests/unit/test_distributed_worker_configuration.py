from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from comfy.cli_args_types import Configuration
from comfy.distributed import distributed_prompt_worker
from comfy.execution_context import comfyui_execution_context, current_execution_context


@pytest.mark.asyncio
async def test_worker_client_preserves_execution_configuration(monkeypatch):
    configuration = Configuration(disable_all_custom_nodes=True, disable_fast_disk=True)
    context = replace(current_execution_context(), configuration=configuration)
    connection = AsyncMock()
    rpc = AsyncMock()
    client = Mock(is_running=True)
    factory = Mock(return_value=client)
    monkeypatch.setattr(distributed_prompt_worker, "connect_robust", AsyncMock(return_value=connection))
    monkeypatch.setattr(distributed_prompt_worker.JsonRPC, "create", AsyncMock(return_value=rpc))
    monkeypatch.setattr(distributed_prompt_worker, "Comfy", factory)
    worker = distributed_prompt_worker.DistributedPromptWorker()
    monkeypatch.setattr(worker, "_start_health_check_server", AsyncMock())

    token = comfyui_execution_context.set(context)
    try:
        async with worker:
            assert factory.call_args.kwargs["configuration"] is configuration
    finally:
        comfyui_execution_context.reset(token)
