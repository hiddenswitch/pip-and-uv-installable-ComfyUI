"""workflows sub-app: list, run, submit, convert, show."""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional

import typer

from .cli import (
    _with_options, _ALL_SHARED_OPTS, _WORKFLOW_OVERRIDE_OPTS,
    _WORKFLOW_OVERRIDE_OPTS_NO_OUTPUT,
    _COMFYUI_ENV, _collect_params, _build_config,
    _discover_from_ref, _RunWorkflowCommand, _run_workflow_cli,
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


@workflows_app.command(name="run", context_settings={**_COMFYUI_ENV, "allow_extra_args": True, "ignore_unknown_options": True}, cls=_RunWorkflowCommand)
@_with_options(_ALL_SHARED_OPTS, _WORKFLOW_OVERRIDE_OPTS)
def workflows_run(
    workflows: list[str] = typer.Argument(..., help="Workflow files, URIs, template names, '-' for stdin, or literal JSON."),
    all: bool = typer.Option(False, "--all", "-a", help="Install missing custom nodes and download missing models before running."),
    disable_progress: bool = typer.Option(False, "--disable-progress", help="Disable CLI progress bars."),
    block_runtime_package_installation: bool = typer.Option(False, "--block-runtime-package-installation", help="Block runtime package installations."),
    **kwargs,
):
    """Execute workflow(s) locally and exit.

    With --all, automatically install missing custom nodes from
    nodes.appmana.com and download missing models before running.
    """
    _all = all
    params = _collect_params(locals(), kwargs)
    params.pop("all", None)
    params.pop("_all", None)

    if params.get("output") is not None:
        params["output_directory"] = params["output"]

    if params.get("otel_service_version") is None:
        from .. import __version__
        params["otel_service_version"] = __version__

    config = _build_config(params)
    _run_workflow_cli(config, all=_all, dry_run=False)


@workflows_app.command(name="submit")
@_with_options(_WORKFLOW_OVERRIDE_OPTS_NO_OUTPUT)
def workflows_submit(
    workflows: list[str] = typer.Argument(..., help="Workflow files, URIs, template names, or literal JSON."),
    server: Optional[str] = typer.Option(None, "--server", envvar="COMFYUI_SERVER", help="Server URL."),
    **kwargs,
):
    """Submit workflow(s) to a running server with the same overrides as run-workflow."""
    config = _build_config(kwargs)
    asyncio.run(_submit_workflows(workflows, server, config))


async def _submit_workflows(workflows: list[str], server: Optional[str], config):
    from rich.console import Console
    from .server_connection import post_json
    from ..component_model.asyncio_files import load_workflow_json
    from ..entrypoints.workflow import _resolve_workflow, expand_workflow_quantity

    console = Console()
    for wf_path in workflows:
        resolved = _resolve_workflow(wf_path)
        obj = load_workflow_json(resolved)
        for prompt in expand_workflow_quantity(obj, config):
            result = await post_json(server, "/api/v1/prompts", body=prompt)
            console.print_json(json.dumps(result))


@workflows_app.command(name="convert")
@_with_options(_WORKFLOW_OVERRIDE_OPTS_NO_OUTPUT)
def workflows_convert(
    file: str = typer.Argument(..., help="Workflow file, URI, template name, or literal JSON."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output file. Defaults to stdout."),
    **kwargs,
):
    """Convert a UI workflow to API format.

    Accepts the full set of `run-workflow` override flags (--prompt,
    --seed, --add-lora, --compile, --set, --image, ...). When any are
    supplied, the converted API workflow has the overrides applied
    before it's written, so the shared JSON is self-contained.
    """
    from pathlib import Path
    from ..component_model.asyncio_files import load_workflow_json
    from ..entrypoints.workflow import _resolve_workflow, expand_workflow_quantity

    workflow = load_workflow_json(_resolve_workflow(file))
    config = _build_config(kwargs)
    workflows = expand_workflow_quantity(workflow, config)
    output_obj = workflows[0] if len(workflows) == 1 else workflows
    result = json.dumps(output_obj, indent=2)
    if output:
        Path(output).write_text(result)
        typer.echo(f"Written to {output}")
    else:
        typer.echo(result)


@workflows_app.command(name="show")
def workflows_show(
    file: str = typer.Argument(..., help="Workflow file, URI, template name, or literal JSON."),
    format: str = typer.Option("command", "--format", help="Output format: command or table."),
):
    """Show a copy-pasteable invocation command for a workflow."""
    from pathlib import Path
    from rich.console import Console
    from rich.table import Table
    from .workflow_templates import (
        _detect_supported_params, _build_example_invocation, TemplateInfo,
    )
    from ..component_model.asyncio_files import load_workflow_json
    from ..component_model.workflow_convert import is_ui_workflow, convert_ui_to_api

    path = Path(file)
    try:
        workflow = load_workflow_json(file)
    except (FileNotFoundError, OSError):
        from .workflow_templates import resolve_template
        resolved = resolve_template(file)
        workflow = load_workflow_json(resolved)
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


@workflows_app.command(name="requirements")
def workflows_requirements(
    workflow_file: str = typer.Argument(..., help="Workflow file, URI, or literal JSON."),
    format: str = typer.Option("requirements_txt", "--format", "-f", help="Output format: requirements_txt, requirements_txt_versioned, requirements_txt_locked"),
    snapshot_uri: Optional[str] = typer.Option(None, "--pip-facade-snapshot-uri", help="Facade registry snapshot URI."),
):
    """Print custom node packages required by a workflow in pip requirements format.

    Analyzes a workflow to determine which custom node packages are needed,
    then prints them as pip-installable package names. Use this to set up a
    new environment before running a workflow.

    \b
    Output formats:
      requirements_txt           Package names only (default)
      requirements_txt_versioned Package names with >=version
      requirements_txt_locked    Package names with ==version

    \b
    Install all requirements for a workflow:
      uv pip install --extra-index-url https://nodes.appmana.com/simple/ \\
        -r <(comfyui workflows requirements workflow.json)

    \b
    Or save to a file and install:
      comfyui workflows requirements workflow.json > requirements-nodes.txt
      uv pip install --extra-index-url https://nodes.appmana.com/simple/ \\
        -r requirements-nodes.txt

    \b
    Also download the models the workflow needs:
      comfyui models from-workflow workflow.json

    \b
    Full setup for a new workflow:
      uv pip install --extra-index-url https://nodes.appmana.com/simple/ \\
        -r <(comfyui workflows requirements workflow.json)
      comfyui models from-workflow workflow.json
      comfyui run-workflow workflow.json --guess-settings

    \b
    Accepts file paths, URIs, or literal JSON:
      comfyui workflows requirements workflow.json
      comfyui workflows requirements https://example.com/workflow.json
    """
    from .cli import workflow_requirements as _workflow_requirements
    _workflow_requirements(workflow_file=workflow_file, format=format, snapshot_uri=snapshot_uri)


@workflows_app.command(name="ls", hidden=True)
def workflows_ls(
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    template_dir: Optional[list[str]] = typer.Option(None, "--template-dir", help="Extra directories to scan."),
    all_templates: bool = typer.Option(False, "-a", "--all", help="Include API-key-requiring templates."),
):
    """Alias for list."""
    workflows_list(format=format, template_dir=template_dir, all_templates=all_templates)


def _hosts_for_cli(with_host: Optional[list[str]], without_host: Optional[list[str]]):
    from ..component_model.workflow_hosts import resolve_host_filter
    hosts = resolve_host_filter(with_host or [], without_host or [])
    if not hosts:
        typer.echo("No hosts selected (check --with-host / --without-host).", err=True)
        raise typer.Exit(2)
    return hosts


def _emit_results(results: list, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps([
            {"host": r.host, "uri": r.uri, "title": r.title, "creator": r.creator,
             "stats": r.stats, "nsfw": r.nsfw}
            for r in results
        ], indent=2, default=str))
        return
    width_uri = max((len(r.uri) for r in results), default=10) + 2
    for r in results:
        downloads = r.stats.get("downloads", 0)
        nsfw = " [nsfw]" if r.nsfw else ""
        creator = f"  by {r.creator}" if r.creator else ""
        downloads_str = f"  ↓ {downloads:,}" if downloads else ""
        typer.echo(f"  {r.uri:<{width_uri}}  {r.title}{creator}{downloads_str}{nsfw}")


_FAMILY_QUERIES: dict[str, list[str]] = {
    "wan": ["wan", "wanvideo"],
    "flux": ["flux", "kontext"],
    "sdxl": ["sdxl", "pony", "illustrious"],
    "ltxv": ["ltx", "ltxv"],
}


@workflows_app.command(name="top", context_settings=_COMFYUI_ENV)
def workflows_top(
    limit: int = typer.Option(100, "--limit", "-n", help="Top N per host (per family if --family is given)."),
    with_host: Optional[list[str]] = typer.Option(None, "--with-host", help="Only these hosts (csv or repeat)."),
    without_host: Optional[list[str]] = typer.Option(None, "--without-host", help="Exclude these hosts (csv or repeat)."),
    period: str = typer.Option("AllTime", "--period", help="Time window: AllTime | Year (≈360d) | Month (≈30d) | Week | Day. Numeric forms (30d, 180, 360) round to the closest enum."),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Substring filter applied to title/description."),
    family: Optional[list[str]] = typer.Option(None, "--family", help="Pre-canned model-family queries: wan, flux, sdxl, ltxv. Repeat to combine."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a grouped table."),
):
    """List the most popular workflows on each enabled host.

    Examples:
      comfyui workflows top --limit 10 --period 30d --family wan
      comfyui workflows top --limit 25 --period Year --family flux --family ltxv --with-host civitai
    """
    hosts = _hosts_for_cli(with_host, without_host)

    # Resolve --family into ((label, [aliases])); --query is treated as a family
    # of one. Each label produces ONE section per host, deduplicated by URI.
    if family:
        sections: list[tuple[str, list[str]]] = []
        seen_labels: set[str] = set()
        for fam in family:
            for f in (fam.lower().split(",") if "," in fam else [fam.lower()]):
                f = f.strip()
                if not f or f in seen_labels:
                    continue
                seen_labels.add(f)
                aliases = _FAMILY_QUERIES.get(f, [f])
                sections.append((f, aliases))
    else:
        sections = [("", [query] if query else [""])]

    def _filter_by_aliases(results, aliases: list[str]) -> list:
        if not any(a for a in aliases):
            return list(results)
        needles = [a.lower() for a in aliases if a]
        kept = []
        for r in results:
            hay = (r.title + " " + r.description).lower()
            if any(n in hay for n in needles):
                kept.append(r)
        return kept

    all_results = []
    for fam_label, aliases in sections:
        for h in hosts:
            try:
                # Pull a broader set when filtering by aliases so we end up with
                # ~limit hits after filtering.
                pull_n = limit if not aliases or not any(aliases) else max(limit * 3, limit)
                if hasattr(h, "top") and "period" in getattr(
                    h.top, "__code__", type("", (), {"co_varnames": ()})(),
                ).co_varnames:
                    raw = h.top(pull_n, period=period, query=None)
                else:
                    raw = h.top(pull_n)
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"  ({h.id}: {exc})", err=True)
                continue
            results = _filter_by_aliases(raw, aliases)[:limit]
            label = f"{h.id} / family={fam_label}" if fam_label else h.id
            if json_output:
                for r in results:
                    r.extra = {**r.extra, "family": fam_label, "period": period}
                all_results.extend(results)
                continue
            typer.echo(f"\n# {label}  period={period}  ({len(results)} results)")
            _emit_results(results, json_output=False)

    if json_output:
        _emit_results(all_results, json_output=True)


@workflows_app.command(name="search", context_settings=_COMFYUI_ENV)
def workflows_search(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results per host."),
    with_host: Optional[list[str]] = typer.Option(None, "--with-host", help="Only these hosts (csv or repeat)."),
    without_host: Optional[list[str]] = typer.Option(None, "--without-host", help="Exclude these hosts (csv or repeat)."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a grouped table."),
):
    """Search workflows across enabled hosts."""
    hosts = _hosts_for_cli(with_host, without_host)
    all_results = []
    for h in hosts:
        try:
            results = h.search(query, limit)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  ({h.id}: {exc})", err=True)
            continue
        if json_output:
            all_results.extend(results)
            continue
        if results:
            typer.echo(f"\n# {h.id}  ({len(results)} results)")
            _emit_results(results, json_output=False)
    if json_output:
        _emit_results(all_results, json_output=True)


@workflows_app.command(name="params", context_settings=_COMFYUI_ENV)
def workflows_params(
    workflow: str = typer.Argument(..., help="Workflow file, URI, '-' for stdin, or literal JSON."),
    show_all: bool = typer.Option(False, "--all", "-a", help="Include advanced-tier params (every non-disabled widget)."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON instead of a grouped table."),
):
    """Print the parameters discoverable in WORKFLOW.

    Headline params come from Set_<Name> blessings, primitive nodes, and
    easy-pack convenience nodes. Common params are class-type-tagged knobs
    (seed/steps/cfg/sampler/prompt/loaders/...). Advanced params are every
    other non-disabled widget — hidden unless --all.
    """
    from ..entrypoints.workflow_params import TIER_ADVANCED, format_params_text

    params = _discover_from_ref(workflow)

    if json_output:
        out = [
            {
                "node_id": p.node_id,
                "class_type": p.class_type,
                "widget_name": p.widget_name,
                "value": p.value,
                "type": p.type,
                "roles": sorted(p.roles),
                "tier": p.tier,
                "label": p.label,
                "source_predicates": p.source_predicates,
            }
            for p in params
            if show_all or p.tier != TIER_ADVANCED
        ]
        typer.echo(json.dumps(out, indent=2, default=str))
        return

    typer.echo(format_params_text(params, show_all=show_all))


def _show_current_values(table, workflow: dict, params: list[str]):
    from ..component_model.prompt_utils import (
        _TEXT_ENCODE_FIELDS, _STEPS_CLASS_TYPES, _SEED_FIELDS,
        _CFG_CLASS_TYPES, _SAMPLER_CLASS_TYPES, _SCHEDULER_CLASS_TYPES,
        _DENOISE_CLASS_TYPES, _LATENT_SIZE_CLASS_TYPES, _CHECKPOINT_CLASS_TYPES,
        _DIFFUSION_MODEL_CLASS_TYPES,
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
        "diffusion-model": (_DIFFUSION_MODEL_CLASS_TYPES, "unet_name"),
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
