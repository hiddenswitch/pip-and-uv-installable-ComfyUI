import asyncio
import json
import logging
import os
import warnings
from typing import Optional, Literal

import typer

from ..cli_args_types import Configuration
from ..component_model.asyncio_files import stream_json_objects
from ..component_model.uris import is_uri
from ..client.embedded_comfy_client import Comfy

logger = logging.getLogger(__name__)


def _is_ui_workflow(obj: dict) -> bool:
    """Return True if *obj* is a UI/LiteGraph workflow (not API format)."""
    return "nodes" in obj and "links" in obj


def _ensure_api_format(obj: dict) -> dict:
    """Convert a UI workflow to API format if needed, otherwise return as-is."""
    if not _is_ui_workflow(obj):
        return obj
    from ..component_model.workflow_convert import convert_ui_to_api
    logger.info("Converting UI workflow to API format")
    return convert_ui_to_api(obj)


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


def _resolve_workflow(workflow: str) -> str:
    """Resolve a workflow argument to a path/URI/literal that stream_json_objects understands.

    If the string looks like a file path, URI, stdin marker, or literal JSON, return
    it as-is.  Otherwise try to resolve it as a template name or ID.
    """
    if workflow == "-" or workflow.lstrip().startswith("{") or is_uri(workflow):
        return workflow
    if os.sep in workflow or workflow.endswith(".json"):
        return workflow
    from ..cmd.workflow_templates import resolve_template
    return resolve_template(workflow)


async def run_workflows(workflows: list[str | Literal["-"]], configuration: Optional[Configuration] = None):
    if configuration is None:
        from ..cli_args import args
        configuration = args
    resolved = [_resolve_workflow(w) for w in workflows]
    async with Comfy(configuration=configuration) as comfy:
        for workflow in resolved:
            obj: dict
            async for obj in stream_json_objects(workflow):
                obj = _ensure_api_format(obj)
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
