"""models sub-app: list, download, and inspect models."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..cli_args_types import Configuration

models_app = typer.Typer(name="models", no_args_is_help=False, add_completion=False)

_COMFYUI_ENV = {"auto_envvar_prefix": "COMFYUI"}


def _boot_paths(cwd: Optional[str] = None, base_directory: Optional[str] = None,
                base_paths: Optional[list[str]] = None,
                extra_model_paths_config: Optional[list[str]] = None):
    from ..component_model.setup import setup_pre_torch
    from .cli import _build_config, _set_config_context

    params: dict = {"base_paths": base_paths or [], "extra_model_paths_config": extra_model_paths_config or []}
    if cwd is not None:
        params["cwd"] = cwd
    if base_directory is not None:
        params["base_directory"] = base_directory
    config = _build_config(params)
    setup_pre_torch(config)
    _set_config_context(config)
    return config


@models_app.callback(invoke_without_command=True, context_settings=_COMFYUI_ENV)
def models_default(
    ctx: typer.Context,
    folder: Optional[str] = typer.Option(None, "--folder", help="Filter by model folder (checkpoints, loras, vae, etc)."),
    source: Optional[str] = typer.Option(None, "--source", help="Filter downloadable catalog by source: known or manager."),
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    check_exists: bool = typer.Option(False, "--check-exists", help="Check if downloadable models exist locally."),
    local: bool = typer.Option(False, "--local", help="List files in registered local model folders instead of the downloadable catalog."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """List downloadable models."""
    if ctx.invoked_subcommand is not None:
        return
    _models_ls_impl(
        folder=folder,
        source=source,
        format=format,
        check_exists=check_exists,
        local=local,
        cwd=cwd,
        base_directory=base_directory,
        base_paths=base_paths,
        extra_model_paths_config=extra_model_paths_config,
    )


def _local_model_rows(
    folder: Optional[str] = None,
    cwd: Optional[str] = None,
    base_directory: Optional[str] = None,
    base_paths: Optional[list[str]] = None,
    extra_model_paths_config: Optional[list[str]] = None,
) -> list[tuple[str, str, str]]:
    _boot_paths(cwd=cwd, base_directory=base_directory, base_paths=base_paths,
                extra_model_paths_config=extra_model_paths_config)

    from . import folder_paths
    from ..execution_context import context_configuration
    config = Configuration()
    with context_configuration(config):
        fnp = folder_paths._folder_names_and_paths()

        rows: list[tuple[str, str, str]] = []
        seen_names: set[str] = set()
        for item in fnp.contents:
            for name in item.folder_names:
                if name in seen_names:
                    continue
                seen_names.add(name)
                if folder and name != folder:
                    continue
                try:
                    for filename in folder_paths.get_filename_list(name):
                        full = folder_paths.get_full_path(name, filename)
                        rows.append((name, filename, str(full) if full else ""))
                except Exception:
                    pass
    return rows


def _print_local_models(rows: list[tuple[str, str, str]], format: str) -> None:
    if format == "json":
        records = [{"folder": r[0], "filename": r[1], "path": r[2]} for r in rows]
        Console().print_json(json.dumps(records))
        return

    console = Console()
    if not rows:
        console.print("No models found.")
        return
    table = Table(show_edge=False, pad_edge=False, box=None, width=max(console.width, 160))
    table.add_column("Folder", no_wrap=True)
    table.add_column("Filename", no_wrap=True)
    table.add_column("Path", no_wrap=True, overflow="ellipsis", ratio=1)
    for r in rows:
        table.add_row(*r)
    console.print(table, soft_wrap=True)


def _downloadable_models(source: Optional[str] = None, check_exists: bool = False):
    from .list_models import _models_from_known, _models_from_manager, _check_exists

    models = []
    if source != "manager":
        for m in _models_from_known():
            m.exists = _check_exists(m.folder, m.filename, m.uri) if check_exists else None
            models.append(("known", m))
    if source != "known":
        for m in _models_from_manager():
            m.exists = _check_exists(m.folder, m.filename, m.uri) if check_exists else None
            models.append(("manager", m))
    return models


def _print_downloadable_models(models, format: str, check_exists: bool) -> None:
    if format == "json":
        records = [{**asdict(m), "source": s} for s, m in models]
        Console().print_json(json.dumps(records))
        return

    console = Console()
    if not models:
        console.print("No models found.")
        return
    table = Table(show_edge=False, pad_edge=False, box=None, width=max(console.width, 200))
    table.add_column("Folder", no_wrap=True)
    table.add_column("Filename", no_wrap=True)
    table.add_column("URI", no_wrap=True, overflow="ellipsis", ratio=1)
    table.add_column("Source", no_wrap=True)
    if check_exists:
        table.add_column("Exists", no_wrap=True)
    for src, m in models:
        row = [m.folder, m.filename, m.uri, src]
        if check_exists:
            row.append("yes" if m.exists else "")
        table.add_row(*row)
    console.print(table, soft_wrap=True)


def _models_ls_impl(
    folder: Optional[str],
    source: Optional[str],
    format: str,
    check_exists: bool,
    local: bool,
    cwd: Optional[str],
    base_directory: Optional[str],
    base_paths: Optional[list[str]],
    extra_model_paths_config: Optional[list[str]],
) -> None:
    if local:
        rows = _local_model_rows(
            folder=folder,
            cwd=cwd,
            base_directory=base_directory,
            base_paths=base_paths,
            extra_model_paths_config=extra_model_paths_config,
        )
        _print_local_models(rows, format)
        return

    if check_exists:
        _boot_paths(cwd=cwd, base_directory=base_directory, base_paths=base_paths,
                    extra_model_paths_config=extra_model_paths_config)
    models = _downloadable_models(source=source, check_exists=check_exists)
    if folder:
        models = [(s, m) for s, m in models if m.folder == folder]
    _print_downloadable_models(models, format, check_exists)


@models_app.command(name="ls", context_settings=_COMFYUI_ENV)
def models_ls(
    folder: Optional[str] = typer.Option(None, "--folder", help="Filter by model folder."),
    source: Optional[str] = typer.Option(None, "--source", help="Filter downloadable catalog by source: known or manager."),
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    check_exists: bool = typer.Option(False, "--check-exists", help="Check if downloadable models exist locally."),
    local: bool = typer.Option(False, "--local", help="List files in registered local model folders instead of the downloadable catalog."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """List downloadable models from known sources and manager.

    Use --local for the older view of files currently visible through
    registered model folders.
    """
    _models_ls_impl(
        folder=folder,
        source=source,
        format=format,
        check_exists=check_exists,
        local=local,
        cwd=cwd,
        base_directory=base_directory,
        base_paths=base_paths,
        extra_model_paths_config=extra_model_paths_config,
    )


@models_app.command(name="find-local", context_settings=_COMFYUI_ENV)
def models_find_local(
    extensions: Optional[list[str]] = typer.Option(None, "--extension", "-e", help="Model file extension to scan for. Repeatable. Defaults to safetensors, ckpt, pt, gguf, onnx."),
    scan_timeout: float = typer.Option(30.0, "--scan-timeout", help="Timeout in seconds for each OS index query and fallback walk root."),
    no_walk: bool = typer.Option(False, "--no-walk", help="Disable fallback filesystem walks for indexer gaps."),
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
):
    """Find local model files and classify them into model folders.

    This is the same heuristic facility used by `run-workflow --all` before
    downloading: query the OS file index, walk likely roots when the index is
    missing coverage, classify discovered files by directory/filename, and
    print the `--add-model-folder-path` registrations that would make them
    visible to ComfyUI.
    """
    from .local_model_discovery import find_local_model_paths
    from .model_search import DEFAULT_EXTENSIONS

    discovery = find_local_model_paths(
        extensions=extensions or DEFAULT_EXTENSIONS,
        scan_timeout=scan_timeout,
        walk_uncovered=not no_walk,
        register=False,
    )

    if format == "json":
        records = {
            "summary": discovery.scan_summary,
            "registrations": [
                {"kind": kind, "path": path, "count": len(items)}
                for (kind, path), items in discovery.registrations.items()
            ],
            "files": [asdict(c) for c in discovery.classifications],
        }
        Console().print_json(json.dumps(records))
        return

    _print_discovery(discovery)


def _print_discovery(discovery) -> None:
    console = Console()
    hf_cache = [item for item in discovery.classifications if item.is_hf_cache]
    unclassified = [
        item
        for item in discovery.classifications
        if item.kind is None and not item.is_hf_cache
    ]

    console.print("Scan summary", style="bold")
    for line in discovery.scan_summary:
        console.print(f"  {line}")

    if discovery.registrations:
        table = Table(title="Discovered model folders", show_edge=False, pad_edge=False, box=None, width=max(console.width, 160))
        table.add_column("Kind", no_wrap=True)
        table.add_column("Path", overflow="ellipsis", ratio=1)
        table.add_column("Files", justify="right", no_wrap=True)
        table.add_column("Flag", overflow="ellipsis", ratio=1)
        for (kind, path), items in sorted(discovery.registrations.items()):
            table.add_row(kind, path, str(len(items)), f"--add-model-folder-path {kind}={path}")
        console.print(table)
    else:
        console.print("No classifiable model folders discovered.")

    if hf_cache:
        console.print(f"HF cache files skipped: {len(hf_cache)}")
    if unclassified:
        console.print(f"Unclassified model-like files: {len(unclassified)}")


@models_app.command(name="download", context_settings=_COMFYUI_ENV)
def models_download(
    uri: str = typer.Argument(..., help="Model URI to download (hf://, https://, etc)."),
    folder: Optional[str] = typer.Option(None, "--folder", help="Target model folder."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """Download a model by URI."""
    _boot_paths(cwd=cwd, base_directory=base_directory, base_paths=base_paths,
                extra_model_paths_config=extra_model_paths_config)

    from ..model_downloader import get_or_download

    console = Console()
    path = get_or_download(uri, folder)
    if path:
        console.print(f"Model available at: {path}")
    else:
        console.print("Download failed or model not found.", style="bold red")
        raise typer.Exit(1)


@models_app.command(name="from-workflow", context_settings=_COMFYUI_ENV)
def models_from_workflow(
    workflow_file: str = typer.Argument(..., help="Workflow file, URI, template name, or literal JSON."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Check availability without downloading."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """Download models referenced by a workflow.

    Scans workflow node inputs for model filenames and matches them against
    the known models database (HuggingFace, CivitAI, and other sources).
    Models already on disk are skipped.

    \b
    With --dry-run, prints found models to stdout and missing models to stderr
    without downloading anything. Useful for checking what a workflow needs:
      comfyui models from-workflow workflow.json --dry-run
      comfyui models from-workflow image_anima_preview --dry-run

    \b
    Full setup for a new workflow:
      uv pip install --extra-index-url https://nodes.appmana.com/simple/ \\
        -r <(comfyui workflows requirements workflow.json)
      comfyui models from-workflow workflow.json
      comfyui run-workflow workflow.json --guess-settings
    """
    _boot_paths(cwd=cwd, base_directory=base_directory, base_paths=base_paths,
                extra_model_paths_config=extra_model_paths_config)

    from ..component_model.asyncio_files import load_workflow_json
    from ..component_model.workflow_convert import is_ui_workflow, convert_ui_to_api
    from ..entrypoints.workflow import _resolve_workflow
    from ..model_downloader import (
        _known_models_db, get_or_download, canonicalize_path,
    )
    from . import folder_paths

    workflow = load_workflow_json(_resolve_workflow(workflow_file))
    if is_ui_workflow(workflow):
        workflow = convert_ui_to_api(workflow)

    # Build filename -> (folder_name, downloadable) index from known models
    filename_index: dict[str, list[tuple[str, object]]] = {}
    for db in _known_models_db:
        for folder_name in db.folder_names:
            for item in db:
                for name in [str(item), item.filename, item.save_with_filename] + list(item.alternate_filenames):
                    key = canonicalize_path(name)
                    if key:
                        filename_index.setdefault(key, []).append((folder_name, item))

    # Extract all string input values from the workflow
    model_refs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for node_data in workflow.values():
        if not isinstance(node_data, dict):
            continue
        for value in (node_data.get("inputs") or {}).values():
            if not isinstance(value, str) or not value:
                continue
            key = canonicalize_path(value)
            if key in seen:
                continue
            seen.add(key)
            matches = filename_index.get(key)
            if matches:
                folder_name = matches[0][0]
                model_refs.append((folder_name, value))

    if not model_refs:
        typer.echo("No model references found in workflow.", err=True)
        return

    for folder_name, filename in sorted(model_refs):
        if dry_run:
            found = folder_paths.get_full_path(folder_name, filename)
        else:
            found = get_or_download(folder_name, filename)
        typer.echo(f"{folder_name}/{filename}", err=not found)


@models_app.command(name="search", context_settings=_COMFYUI_ENV)
def models_search(
    query: str = typer.Argument(..., help="Search query (free text)."),
    kind: Optional[str] = typer.Option(None, "--kind", "-k", help="Asset kind: lora | checkpoint | embedding | vae | controlnet."),
    base_model: Optional[list[str]] = typer.Option(None, "--base-model", "-b", help="Filter by base model (Civitai). Repeat or csv. Examples: 'Flux.1 Klein', 'Flux.1 D', 'SDXL 1.0', 'Pony', 'Illustrious'."),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results per host."),
    with_host: Optional[list[str]] = typer.Option(None, "--with-host", help="Only these hosts (csv or repeat)."),
    without_host: Optional[list[str]] = typer.Option(None, "--without-host", help="Exclude these hosts."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
):
    """Search Civitai and Hugging Face for LoRAs, checkpoints, embeddings, VAEs, controlnets.

    Distinct from `comfyui workflows search`, which finds workflow JSONs.
    This finds the building blocks a workflow loads, so you can paste the
    URI straight into `--add-lora`, `--checkpoint`, or
    `--set <node>.inputs.<field>=<URI>`. Civitai results pin to a version id
    so a re-publish doesn't silently change which weights you grabbed.

    Output rows print:

      (kind) URI  title  by creator  ↓ downloads
          base: <base model>
          trigger: <comma-joined trigger words>

    \b
    Find a Studio Ghibli LoRA across base models (Flux, Illustrious, SDXL, LTXV):
      comfyui models search "ghibli style" --kind lora --limit 5

      # Real output (sorted Most Downloaded):
      #   (lora) civitai://v/2627385  Studio Ghibli LTX2 style lora      base: LTXV          ↓ 482
      #   (lora) civitai://v/2769334  Cee_One's Ghibli-Esque Blend       base: Illustrious   ↓ 275
      #   (lora) civitai://v/2786125  Modern Ghibli Hires Style          base: Illustrious   ↓ 138
      #   (lora) civitai://v/2734259  V67 Ghibli Look                    base: Flux.1 D      ↓  68
      #   (lora) civitai://v/2740724  RIP Weights Series - Pink Ghibli   base: Flux.1 D      ↓  44
      #
      # Pick the row whose `base:` matches the workflow you're running.
      # Pass the URI to --add-lora directly:
      #   comfyui run-workflow image_flux2_dev_text_to_image --all \\
      #       --prompt "..." --add-lora civitai://v/2734259:0.8

    \b
    Find a popular checkpoint by family:
      comfyui models search "pony" --kind checkpoint --limit 3

      # Real output:
      #   (checkpoint) civitai://v/290640   Pony Diffusion V6 XL    base: Pony  ↓ 936,856
      #   (checkpoint) civitai://v/2884631  CyberRealistic Pony     base: Pony  ↓ 696,919
      #   (checkpoint) civitai://v/914390   Pony Realism            base: Pony  ↓ 529,545

    \b
    Filter by exact Civitai base-model string:
      comfyui models search "ghibli" --kind lora --base-model "Flux.1 D" --limit 3

      # Common --base-model strings:
      #   "Flux.1 D"  "Flux.1 Klein"  "SDXL 1.0"  "SD 1.5"  "Pony"
      #   "Illustrious"  "NoobAI"  "LTXV"  "Wan Video 2.1"  "Anima"

    \b
    Search Hugging Face for a base model repo (HF tags do the kind inference):
      comfyui models search "flux" --with-host huggingface --limit 5

    \b
    Anime character checkpoints (the fallback path: client-side substring
    filter when Civitai's server-side query+type returns 0):
      comfyui models search "anime" --kind checkpoint --limit 3

      # Real output:
      #   (checkpoint) civitai://v/128713  DreamShaper          base: SD 1.5  ↓ 1,588,288
      #   (checkpoint) civitai://v/354657  DreamShaper XL       base: SDXL Lightning  ↓ 595,448
      #   (checkpoint) civitai://v/425083  ReV Animated         base: SD 1.5  ↓ 570,859

    \b
    JSON output for piping into jq / Python:
      comfyui models search "ghibli" --kind lora --limit 3 --json | \\
        jq -r '.[] | select(.base_model=="Flux.1 D") | .uri'

    Authenticate first if you want NSFW / early-access results:
      export CIVITAI_API_TOKEN=ci_xxxx
    """
    from ..component_model.workflow_hosts import resolve_host_filter

    if base_model:
        flat: list[str] = []
        for b in base_model:
            flat.extend(p.strip() for p in b.split(",") if p.strip())
        base_model = flat or None

    hosts = resolve_host_filter(with_host or [], without_host or [])
    capable = [h for h in hosts if hasattr(h, "search_models")]
    if not capable:
        typer.echo("No hosts in --with-host/--without-host implement model search "
                   "(supported: civitai, civitai_red, huggingface).", err=True)
        raise typer.Exit(2)

    all_results: list = []
    for h in capable:
        results = h.search_models(query, kind=kind, base_models=base_model, limit=limit)
        if json_output:
            all_results.extend(results)
            continue
        if not results:
            continue
        typer.echo(f"\n# {h.id}  ({len(results)} results)")
        width_uri = max((len(r.uri) for r in results), default=10) + 2
        for r in results:
            downloads = (r.stats or {}).get("downloads", 0)
            nsfw = " [nsfw]" if r.nsfw else ""
            creator = f"  by {r.creator}" if r.creator else ""
            downloads_str = f"  ↓ {downloads:,}" if downloads else ""
            typer.echo(f"  ({r.kind}) {r.uri:<{width_uri}}  {r.title}{creator}{downloads_str}{nsfw}")
            if r.base_model:
                typer.echo(f"      base: {r.base_model}")
            if r.trigger_words:
                typer.echo(f"      trigger: {', '.join(r.trigger_words[:8])}")
    if json_output:
        typer.echo(json.dumps([
            {
                "host": r.host, "kind": r.kind, "uri": r.uri, "title": r.title,
                "creator": r.creator, "description": r.description,
                "base_model": r.base_model, "trigger_words": r.trigger_words,
                "download_url": r.download_url, "stats": r.stats,
                "nsfw": r.nsfw, "extra": r.extra,
            }
            for r in all_results
        ], indent=2, default=str))


@models_app.command(name="paths", context_settings=_COMFYUI_ENV)
def models_paths(
    folder: Optional[str] = typer.Option(None, "--folder", help="Filter by model folder."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """Show model search directories."""
    _boot_paths(cwd=cwd, base_directory=base_directory, base_paths=base_paths,
                extra_model_paths_config=extra_model_paths_config)

    from . import folder_paths
    fnp = folder_paths._folder_names_and_paths()

    console = Console()
    table = Table(show_edge=False, pad_edge=False, box=None)
    table.add_column("Folder", no_wrap=True)
    table.add_column("Paths")

    seen: set[str] = set()
    for item in fnp.contents:
        for name in item.folder_names:
            if name in seen:
                continue
            seen.add(name)
            if folder and name != folder:
                continue
            dirs = [str(p) for p in fnp.directory_paths(name)]
            table.add_row(name, "\n".join(dirs) if dirs else "(none)")
    console.print(table)
