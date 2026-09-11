import pytest
import torch

from comfy.api.components.schema.prompt import Prompt
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import Comfy
from comfy.distributed.process_pool_executor import ProcessPoolExecutor

pytestmark = pytest.mark.server_process


def _probe_prompt() -> dict:
    return {
        "1": {
            "class_type": "EvalPython_1_1",
            "inputs": {
                "pycode": "from comfy import memory_management\nreturn str(memory_management.aimdo_enabled)\n",
            },
        },
        "2": {"class_type": "PreviewString", "inputs": {"value": ["1", 0]}},
    }


@pytest.mark.asyncio
async def test_process_pool_worker_enables_dynamic_vram_like_the_cli():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    config = Configuration()
    config.disable_all_custom_nodes = True
    config.enable_eval = True
    config.cuda_device = "0"
    prompt = Prompt.validate(_probe_prompt())
    with ProcessPoolExecutor(max_workers=1) as executor:
        async with Comfy(configuration=config, executor=executor) as client:
            outputs = await client.queue_prompt(prompt)
    assert outputs["2"]["string"][0] == "True"


@pytest.mark.asyncio
async def test_process_pool_worker_honours_disable_dynamic_vram():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    config = Configuration()
    config.disable_all_custom_nodes = True
    config.enable_eval = True
    config.cuda_device = "0"
    config.disable_dynamic_vram = True
    prompt = Prompt.validate(_probe_prompt())
    with ProcessPoolExecutor(max_workers=1) as executor:
        async with Comfy(configuration=config, executor=executor) as client:
            outputs = await client.queue_prompt(prompt)
    assert outputs["2"]["string"][0] == "False"
