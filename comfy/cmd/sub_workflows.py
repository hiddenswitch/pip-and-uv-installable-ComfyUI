"""workflows sub-app: list, run, submit, convert, show."""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional

import typer

from .cli import (
    _with_options, _ALL_SHARED_OPTS, _WORKFLOW_OVERRIDE_OPTS,
    _COMFYUI_ENV, _collect_params, _build_config, _set_config_context,
)

workflows_app = typer.Typer(name="workflows", no_args_is_help=False, add_completion=False)


@workflows_app.callback(invoke_without_command=True)
def workflows_default(
    ctx: typer.Context,
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    template_dir: Optional[list[str]] = typer.Option(None, "--template-dir", help="Extra directories to scan."),
    all_templates: bool = typer.Option(False, "-a", "--all", help="Include API-key-requiring templates."),
):
    """List available workflow templates."""
    if ctx.invoked_subcommand is not None:
        return
    from .workflow_templates import list_templates
    interactive = sys.stdout.isatty() and format == "table"
    list_templates(
        format=format,
        extra_dirs=template_dir or [],
        show_all=all_templates,
        interactive=interactive,
    )


@workflows_app.command(name="list")
def workflows_list(
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    template_dir: Optional[list[str]] = typer.Option(None, "--template-dir", help="Extra directories to scan."),
    all_templates: bool = typer.Option(False, "-a", "--all", help="Include API-key-requiring templates."),
):
    """List available workflow templates."""
    from .workflow_templates import list_templates
    interactive = sys.stdout.isatty() and format == "table"
    list_templates(
        format=format,
        extra_dirs=template_dir or [],
        show_all=all_templates,
        interactive=interactive,
    )


@workflows_app.command(name="run", context_settings={**_COMFYUI_ENV, "allow_extra_args": True, "ignore_unknown_options": True})
@_with_options(_ALL_SHARED_OPTS, _WORKFLOW_OVERRIDE_OPTS)
def workflows_run(
    workflows: list[str] = typer.Argument(..., help="Workflow files, URIs, template names, '-' for stdin, or literal JSON."),
    disable_progress: bool = typer.Option(False, "--disable-progress", help="Disable CLI progress bars."),
    block_runtime_package_installation: bool = typer.Option(False, "--block-runtime-package-installation", help="Block runtime package installations."),
    **kwargs,
):
    """Execute workflow(s) locally and exit."""
    from ..component_model.setup import setup_pre_torch, setup_post_torch

    params = _collect_params(locals(), kwargs)

    if params.get("output") is not None:
        params["output_directory"] = params["output"]

    if params.get("otel_service_version") is None:
        from .. import __version__
        params["otel_service_version"] = __version__

    config = _build_config(params)

    setup_pre_torch(config)
    _set_config_context(config)
    setup_post_torch(config)

    from ..component_model.entrypoints_common import configure_application_paths
    configure_application_paths(config)

    from ..execution_context import context_configuration
    from ..nodes.package import import_all_nodes_in_workspace
    with context_configuration(config):
        import_all_nodes_in_workspace(raise_on_failure=False)

    from ..entrypoints.workflow import run_workflows
    try:
        asyncio.run(run_workflows(config.workflows, configuration=config))
    except KeyboardInterrupt:
        pass


@workflows_app.command(name="submit")
def workflows_submit(
    workflows: list[str] = typer.Argument(..., help="Workflow JSON files to submit."),
    server: Optional[str] = typer.Option(None, "--server", envvar="COMFYUI_SERVER", help="Server URL."),
    set_overrides: Optional[list[str]] = typer.Option(None, "--set", help="Override node inputs: node_id.inputs.field=value"),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Override positive prompt."),
    negative_prompt: Optional[str] = typer.Option(None, "--negative-prompt", help="Override negative prompt."),
    steps: Optional[int] = typer.Option(None, "--steps", help="Override steps."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Override seed."),
    cfg: Optional[float] = typer.Option(None, "--cfg", help="Override CFG scale."),
    sampler: Optional[str] = typer.Option(None, "--sampler", help="Override sampler."),
    scheduler: Optional[str] = typer.Option(None, "--scheduler", help="Override scheduler."),
    denoise: Optional[float] = typer.Option(None, "--denoise", help="Override denoise."),
    width: Optional[int] = typer.Option(None, "--width", help="Override width."),
    height: Optional[int] = typer.Option(None, "--height", help="Override height."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override batch size."),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="Override checkpoint."),
):
    """Submit workflow(s) to a running server."""
    asyncio.run(_submit_workflows(
        workflows=workflows, server=server, set_overrides=set_overrides or [],
        prompt=prompt, negative_prompt=negative_prompt, steps=steps, seed=seed,
        cfg=cfg, sampler=sampler, scheduler=scheduler, denoise=denoise,
        width=width, height=height, batch_size=batch_size, checkpoint=checkpoint,
    ))


async def _submit_workflows(
    workflows: list[str], server: Optional[str], set_overrides: list[str],
    prompt, negative_prompt, steps, seed, cfg, sampler, scheduler, denoise,
    width, height, batch_size, checkpoint,
):
    from pathlib import Path
    from rich.console import Console
    from .server_connection import post_json
    from ..component_model.workflow_convert import is_ui_workflow, convert_ui_to_api
    from ..component_model.prompt_utils import (
        replace_prompt_text, replace_negative_prompt_text,
        replace_steps, replace_seed,
        replace_cfg, replace_sampler, replace_scheduler, replace_denoise,
        replace_width, replace_height, replace_batch_size, replace_checkpoint,
    )
    from ..entrypoints.workflow import _apply_sets

    console = Console()
    for wf_path in workflows:
        obj = json.loads(Path(wf_path).read_text())
        if is_ui_workflow(obj):
            obj = convert_ui_to_api(obj)

        if prompt is not None:
            obj = replace_prompt_text(obj, prompt)
        if negative_prompt is not None:
            obj = replace_negative_prompt_text(obj, negative_prompt)
        if steps is not None:
            obj = replace_steps(obj, steps)
        if seed is not None:
            obj = replace_seed(obj, seed)
        if cfg is not None:
            obj = replace_cfg(obj, cfg)
        if sampler is not None:
            obj = replace_sampler(obj, sampler)
        if scheduler is not None:
            obj = replace_scheduler(obj, scheduler)
        if denoise is not None:
            obj = replace_denoise(obj, denoise)
        if width is not None:
            obj = replace_width(obj, width)
        if height is not None:
            obj = replace_height(obj, height)
        if batch_size is not None:
            obj = replace_batch_size(obj, batch_size)
        if checkpoint is not None:
            obj = replace_checkpoint(obj, checkpoint)
        if set_overrides:
            obj = _apply_sets(obj, set_overrides)

        result = await post_json(server, "/api/v1/prompts", body=obj)
        console.print_json(json.dumps(result))


@workflows_app.command(name="convert")
def workflows_convert(
    file: str = typer.Argument(..., help="Workflow JSON file to convert."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output file. Defaults to stdout."),
):
    """Convert a UI workflow to API format."""
    from pathlib import Path
    from ..component_model.workflow_convert import convert_ui_to_api

    workflow = json.loads(Path(file).read_text())
    api_workflow = convert_ui_to_api(workflow)
    result = json.dumps(api_workflow, indent=2)
    if output:
        Path(output).write_text(result)
        typer.echo(f"Written to {output}")
    else:
        typer.echo(result)


@workflows_app.command(name="show")
def workflows_show(
    file: str = typer.Argument(..., help="Workflow JSON file or template name."),
    format: str = typer.Option("command", "--format", help="Output format: command or table."),
):
    """Show a copy-pasteable invocation command for a workflow."""
    from pathlib import Path
    from rich.console import Console
    from rich.table import Table
    from .workflow_templates import (
        _detect_supported_params, _build_example_invocation, TemplateInfo,
    )
    from ..component_model.workflow_convert import is_ui_workflow, convert_ui_to_api

    path = Path(file)
    if path.exists():
        workflow = json.loads(path.read_text())
    else:
        from .workflow_templates import resolve_template
        resolved = resolve_template(file)
        workflow = json.loads(Path(resolved).read_text())
        path = Path(resolved)

    if is_ui_workflow(workflow):
        api_workflow = convert_ui_to_api(workflow)
    else:
        api_workflow = workflow

    params = _detect_supported_params(workflow)
    tmpl = TemplateInfo(name=path.stem, source="file", path=str(path), supported_params=params)

    if format == "table":
        console = Console()
        table = Table(show_edge=False, pad_edge=False, box=None)
        table.add_column("Parameter", no_wrap=True)
        table.add_column("Current Value")

        _show_current_values(table, api_workflow, params)
        console.print(table)
        console.print()
        console.print("[bold]Command:[/bold]")
        console.print(_build_example_invocation(tmpl), highlight=False)
    else:
        typer.echo(_build_example_invocation(tmpl))


def _show_current_values(table, workflow: dict, params: list[str]):
    from ..component_model.prompt_utils import (
        _TEXT_ENCODE_FIELDS, _STEPS_CLASS_TYPES, _SEED_FIELDS,
        _CFG_CLASS_TYPES, _SAMPLER_CLASS_TYPES, _SCHEDULER_CLASS_TYPES,
        _DENOISE_CLASS_TYPES, _LATENT_SIZE_CLASS_TYPES, _CHECKPOINT_CLASS_TYPES,
        find_positive_text_encoder,
    )

    field_map = {
        "prompt": (frozenset(_TEXT_ENCODE_FIELDS.keys()), "text"),
        "steps": (_STEPS_CLASS_TYPES, "steps"),
        "seed": (frozenset(_SEED_FIELDS.keys()), None),
        "cfg": (_CFG_CLASS_TYPES, "cfg"),
        "sampler": (_SAMPLER_CLASS_TYPES, "sampler_name"),
        "scheduler": (_SCHEDULER_CLASS_TYPES, "scheduler"),
        "denoise": (_DENOISE_CLASS_TYPES, "denoise"),
        "width": (_LATENT_SIZE_CLASS_TYPES, "width"),
        "height": (_LATENT_SIZE_CLASS_TYPES, "height"),
        "batch-size": (_LATENT_SIZE_CLASS_TYPES, "batch_size"),
        "checkpoint": (_CHECKPOINT_CLASS_TYPES, "ckpt_name"),
    }

    for param in params:
        if param == "prompt":
            nid = find_positive_text_encoder(workflow)
            if nid:
                node = workflow[nid]
                ct = node["class_type"]
                fields = _TEXT_ENCODE_FIELDS.get(ct, ["text"])
                val = node.get("inputs", {}).get(fields[0], "")
                table.add_row("--prompt", str(val)[:80])
            continue
        if param in ("image", "video", "audio", "negative-prompt"):
            table.add_row(f"--{param}", "(supported)")
            continue

        entry = field_map.get(param)
        if not entry:
            continue
        class_types, field_name = entry
        for nid, node in workflow.items():
            if node.get("class_type", "") in class_types and field_name:
                val = node.get("inputs", {}).get(field_name, "")
                table.add_row(f"--{param}", str(val))
                break
