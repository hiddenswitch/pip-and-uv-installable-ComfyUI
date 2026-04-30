"""env sub-app: environment info, checks, and diagnostics."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

env_app = typer.Typer(name="env", no_args_is_help=False, add_completion=False)

_COMFYUI_ENV = {"auto_envvar_prefix": "COMFYUI"}


@env_app.callback(invoke_without_command=True, context_settings=_COMFYUI_ENV)
def env_default(
    ctx: typer.Context,
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """Show environment info (system, GPU, RAM)."""
    if ctx.invoked_subcommand is not None:
        return
    _env_info(format=format, cwd=cwd, base_directory=base_directory,
              extra_model_paths_config=extra_model_paths_config)


def _env_info(format: str, cwd: Optional[str] = None, base_directory: Optional[str] = None,
              extra_model_paths_config: Optional[list[str]] = None):
    from ..component_model.setup import setup_pre_torch
    from .cli import _build_config, _set_config_context

    params: dict = {"base_paths": [], "extra_model_paths_config": extra_model_paths_config or []}
    if cwd is not None:
        params["cwd"] = cwd
    if base_directory is not None:
        params["base_directory"] = base_directory
    config = _build_config(params)
    setup_pre_torch(config)
    _set_config_context(config)

    console = Console()
    from .integrity_check import _section_device, _section_package_versions

    if format == "json":
        import psutil
        info = {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 1),
        }
        try:
            from .. import model_management
            device = model_management.get_torch_device()
            info["device"] = str(device)
            info["device_name"] = model_management.get_torch_device_name(device)
            info["vram_total_gb"] = round(model_management.get_total_memory(device) / (1024 ** 3), 1)
            info["vram_free_gb"] = round(model_management.get_free_memory(device) / (1024 ** 3), 1)
        except Exception:
            pass
        console.print_json(json.dumps(info))
        return

    from .. import __version__
    from .integrity_check import _section_folder_paths

    console.print(f"ComfyUI version: {__version__}")
    console.print(f"Platform: {platform.platform()}")
    console.print(f"Python: {sys.version.split()[0]}")
    console.print()
    console.rule("Device")
    _section_device(console)
    console.print()
    console.rule("Folder Paths")
    _section_folder_paths(console)
    console.print()
    console.rule("Package Versions")
    _section_package_versions(console)


@env_app.command(name="info", context_settings=_COMFYUI_ENV)
def env_info(
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """Show environment info (system, GPU, RAM)."""
    _env_info(format=format, cwd=cwd, base_directory=base_directory,
              extra_model_paths_config=extra_model_paths_config)


@env_app.command(name="check", context_settings=_COMFYUI_ENV)
def env_check(
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """Run system integrity check."""
    from ..component_model.setup import setup_pre_torch
    from .cli import _build_config, _set_config_context
    from .integrity_check import run_integrity_check

    params: dict = {"base_paths": [], "extra_model_paths_config": extra_model_paths_config or []}
    if cwd is not None:
        params["cwd"] = cwd
    if base_directory is not None:
        params["base_directory"] = base_directory
    config = _build_config(params)
    setup_pre_torch(config)
    _set_config_context(config)
    run_integrity_check(config)


@env_app.command(name="packages")
def env_packages(
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
):
    """List installed Python packages (no torch import, fast)."""
    dists = sorted(importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower())

    if format == "json":
        records = [{"name": d.metadata["Name"], "version": d.metadata["Version"]} for d in dists]
        Console().print_json(json.dumps(records))
        return

    console = Console()
    table = Table(show_edge=False, pad_edge=False, box=None)
    table.add_column("Package", no_wrap=True)
    table.add_column("Version", no_wrap=True)
    for d in dists:
        table.add_row(d.metadata["Name"], d.metadata["Version"])
    console.print(table)


@env_app.command(name="paths", context_settings=_COMFYUI_ENV)
def env_paths(
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """Show configured folder paths."""
    from ..component_model.setup import setup_pre_torch
    from .cli import _build_config, _set_config_context
    from .integrity_check import _section_folder_paths

    params: dict = {"base_paths": [], "extra_model_paths_config": extra_model_paths_config or []}
    if cwd is not None:
        params["cwd"] = cwd
    if base_directory is not None:
        params["base_directory"] = base_directory
    config = _build_config(params)
    setup_pre_torch(config)
    _set_config_context(config)

    console = Console()
    _section_folder_paths(console)


@env_app.command(name="create-dirs", context_settings=_COMFYUI_ENV)
def env_create_dirs(
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    output_directory: Optional[str] = typer.Option(None, "--output-directory", help="Output directory."),
    input_directory: Optional[str] = typer.Option(None, "--input-directory", help="Input directory."),
    temp_directory: Optional[str] = typer.Option(None, "--temp-directory", help="Temp directory."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """Create default model/input/output/temp directories."""
    from ..component_model.setup import setup_pre_torch
    from .cli import _build_config, _set_config_context

    params = {k: v for k, v in locals().items() if v is not None}
    params.setdefault("base_paths", [])
    params.setdefault("extra_model_paths_config", [])
    config = _build_config(params)
    setup_pre_torch(config)
    _set_config_context(config)

    from ..execution_context import context_configuration
    from .folder_paths import create_directories
    from ..nodes.package import import_all_nodes_in_workspace
    with context_configuration(config):
        import_all_nodes_in_workspace(raise_on_failure=False)
        create_directories()
    typer.echo("Directories created.")
