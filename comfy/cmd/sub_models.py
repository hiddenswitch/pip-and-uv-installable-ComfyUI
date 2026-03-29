"""models sub-app: list, download, and inspect models."""
from __future__ import annotations

import json
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
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """List locally downloaded models."""
    if ctx.invoked_subcommand is not None:
        return
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


@models_app.command(name="ls", context_settings=_COMFYUI_ENV)
def models_ls(
    ctx: typer.Context,
    folder: Optional[str] = typer.Option(None, "--folder", help="Filter by model folder."),
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """List locally downloaded models."""
    models_default(ctx, folder=folder, format=format, cwd=cwd, base_directory=base_directory,
                   base_paths=base_paths, extra_model_paths_config=extra_model_paths_config)


@models_app.command(name="available", context_settings=_COMFYUI_ENV)
def models_available(
    folder: Optional[str] = typer.Option(None, "--folder", help="Filter by model folder."),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source: known or manager."),
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    check_exists: bool = typer.Option(False, "--check-exists", help="Check if models exist locally."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """List downloadable models from known sources and manager."""
    if check_exists:
        _boot_paths(cwd=cwd, base_directory=base_directory, base_paths=base_paths,
                    extra_model_paths_config=extra_model_paths_config)

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

    if folder:
        models = [(s, m) for s, m in models if m.folder == folder]

    if format == "json":
        from dataclasses import asdict
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
