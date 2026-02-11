import asyncio
import json
import logging
import warnings
from typing import Optional, Literal

import typer

from ..cli_args_types import Configuration
from ..component_model.asyncio_files import stream_json_objects
from ..client.embedded_comfy_client import Comfy

logger = logging.getLogger(__name__)


def _apply_overrides(obj: dict, configuration: Configuration) -> dict:
    """Apply CLI overrides to a workflow dict."""
    from ..component_model.prompt_utils import (  # pylint: disable=import-outside-toplevel
        replace_prompt_text, replace_negative_prompt_text,
        replace_steps, replace_seed,
        replace_images, replace_videos, replace_audios,
    )

    if configuration.prompt is not None:
        obj = replace_prompt_text(obj, configuration.prompt)
    if configuration.negative_prompt is not None:
        obj = replace_negative_prompt_text(obj, configuration.negative_prompt)
    if configuration.steps is not None:
        obj = replace_steps(obj, configuration.steps)
    if configuration.seed is not None:
        obj = replace_seed(obj, configuration.seed)
    if configuration.image is not None:
        obj = replace_images(obj, configuration.image)
    if configuration.video is not None:
        obj = replace_videos(obj, configuration.video)
    if configuration.audio is not None:
        obj = replace_audios(obj, configuration.audio)
    return obj


async def run_workflows(workflows: list[str | Literal["-"]], configuration: Optional[Configuration] = None):
    if configuration is None:
        from ..cli_args import args
        configuration = args
    async with Comfy(configuration=configuration) as comfy:
        for workflow in workflows:
            obj: dict
            async for obj in stream_json_objects(workflow):
                obj = _apply_overrides(obj, configuration)
                try:
                    res = await comfy.queue_prompt_api(obj)
                    typer.echo(json.dumps(res.outputs))
                except asyncio.CancelledError:
                    logger.info("Exiting gracefully.")
                    break


def entrypoint():
    """Legacy entrypoint. Delegates to ``comfyui post-workflow``."""
    warnings.warn(
        "comfyui-workflow is deprecated. Use: comfyui post-workflow",
        DeprecationWarning,
        stacklevel=1,
    )
    import sys
    from ..cmd.cli import app
    sys.argv = [sys.argv[0], "post-workflow"] + sys.argv[1:]
    app()


if __name__ == "__main__":
    entrypoint()
