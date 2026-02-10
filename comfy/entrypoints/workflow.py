from ..cmd.main_pre import args

import asyncio
import json
import logging
from typing import Optional, Literal

import typer
from ..cli_args_types import Configuration
from ..component_model.asyncio_files import stream_json_objects
from ..client.embedded_comfy_client import Comfy
from ..component_model.entrypoints_common import configure_application_paths

logger = logging.getLogger(__name__)


async def main():
    workflows = args.workflows
    assert len(workflows) > 0, "specify at least one path to a workflow, a literal workflow json starting with `{` or `-` (for standard in) using --workflows cli arg"

    # --output / -o overrides the output directory before paths are configured
    if args.output is not None:
        args.output_directory = args.output

    configure_application_paths(args)

    await run_workflows(workflows)


def _apply_overrides(obj: dict, configuration: Configuration) -> dict:
    """Apply CLI overrides (--prompt, --negative-prompt, --steps, --image) to a workflow dict."""
    from ..component_model.prompt_utils import replace_prompt_text, replace_negative_prompt_text, replace_steps, replace_images

    if configuration.prompt is not None:
        obj = replace_prompt_text(obj, configuration.prompt)
    if configuration.negative_prompt is not None:
        obj = replace_negative_prompt_text(obj, configuration.negative_prompt)
    if configuration.steps is not None:
        obj = replace_steps(obj, configuration.steps)
    if configuration.image is not None:
        obj = replace_images(obj, configuration.image)
    return obj


async def run_workflows(workflows: list[str | Literal["-"]], configuration: Optional[Configuration] = None):
    if configuration is None:
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
    asyncio.run(main())


if __name__ == "__main__":
    entrypoint()
