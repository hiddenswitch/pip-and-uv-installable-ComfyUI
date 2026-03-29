"""nodes sub-app: list and inspect installed nodes."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

nodes_app = typer.Typer(name="nodes", no_args_is_help=False, add_completion=False)

_COMFYUI_ENV = {"auto_envvar_prefix": "COMFYUI"}


def _boot_nodes(**dir_kwargs):
    from ..component_model.setup import setup_pre_torch, setup_post_torch
    from .cli import _build_config, _set_config_context

    params: dict = {
        "base_paths": dir_kwargs.get("base_paths") or [],
        "extra_model_paths_config": dir_kwargs.get("extra_model_paths_config") or [],
    }
    if dir_kwargs.get("cwd") is not None:
        params["cwd"] = dir_kwargs["cwd"]
    if dir_kwargs.get("base_directory") is not None:
        params["base_directory"] = dir_kwargs["base_directory"]
    config = _build_config(params)
    setup_pre_torch(config)
    _set_config_context(config)
    setup_post_torch(config)

    from ..execution_context import context_configuration
    from ..nodes.package import import_all_nodes_in_workspace
    with context_configuration(config):
        exported_nodes = import_all_nodes_in_workspace(raise_on_failure=False)
    return exported_nodes


@nodes_app.callback(invoke_without_command=True, context_settings=_COMFYUI_ENV)
def nodes_default(
    ctx: typer.Context,
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """List installed node classes."""
    if ctx.invoked_subcommand is not None:
        return
    exported_nodes = _boot_nodes(cwd=cwd, base_directory=base_directory, base_paths=base_paths,
                                 extra_model_paths_config=extra_model_paths_config)

    rows = []
    for cls_name in sorted(exported_nodes.NODE_CLASS_MAPPINGS.keys()):
        obj_class = exported_nodes.NODE_CLASS_MAPPINGS[cls_name]
        display = exported_nodes.NODE_DISPLAY_NAME_MAPPINGS.get(cls_name, cls_name)
        category = getattr(obj_class, "CATEGORY", "sd")
        output_types = ", ".join(str(t) for t in getattr(obj_class, "RETURN_TYPES", []))
        rows.append((cls_name, display, category, output_types))

    if format == "json":
        records = [{"class_name": r[0], "display_name": r[1], "category": r[2], "output_types": r[3]} for r in rows]
        Console().print_json(json.dumps(records))
        return

    console = Console()
    if not rows:
        console.print("No nodes found.")
        return
    table = Table(show_edge=False, pad_edge=False, box=None, width=max(console.width, 160))
    table.add_column("Class", no_wrap=True)
    table.add_column("Display Name", no_wrap=True)
    table.add_column("Category", no_wrap=True)
    table.add_column("Outputs", no_wrap=True)
    for r in rows:
        table.add_row(*r)
    console.print(table, soft_wrap=True)


@nodes_app.command(name="ls", context_settings=_COMFYUI_ENV)
def nodes_ls(
    ctx: typer.Context,
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """List installed node classes."""
    nodes_default(ctx, format=format, cwd=cwd, base_directory=base_directory,
                  base_paths=base_paths, extra_model_paths_config=extra_model_paths_config)


@nodes_app.command(name="packages", context_settings=_COMFYUI_ENV)
def nodes_packages(
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """List custom node packages/directories."""
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

    from . import folder_paths

    rows = []
    try:
        custom_paths = folder_paths.get_folder_paths("custom_nodes")
    except Exception:
        custom_paths = []

    for cp in custom_paths:
        cp_path = Path(cp)
        if not cp_path.is_dir():
            continue
        for entry in sorted(cp_path.iterdir()):
            if not entry.is_dir():
                continue
            has_init = (entry / "__init__.py").exists()
            has_pyproject = (entry / "pyproject.toml").exists()
            if not has_init and not has_pyproject:
                continue
            rows.append((entry.name, str(entry), "yes" if has_init else "no", "yes" if has_pyproject else "no"))

    if format == "json":
        records = [{"name": r[0], "path": r[1], "has_init": r[2], "has_pyproject": r[3]} for r in rows]
        Console().print_json(json.dumps(records))
        return

    console = Console()
    if not rows:
        console.print("No custom node packages found.")
        return
    table = Table(show_edge=False, pad_edge=False, box=None)
    table.add_column("Package", no_wrap=True)
    table.add_column("Path", no_wrap=True)
    table.add_column("__init__.py", no_wrap=True)
    table.add_column("pyproject.toml", no_wrap=True)
    for r in rows:
        table.add_row(*r)
    console.print(table)


@nodes_app.command(name="info", context_settings=_COMFYUI_ENV)
def nodes_info_cmd(
    node_class: str = typer.Argument(..., help="Node class name to inspect."),
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
):
    """Show detailed info about a node class."""
    exported_nodes = _boot_nodes(cwd=cwd, base_directory=base_directory, base_paths=base_paths,
                                 extra_model_paths_config=extra_model_paths_config)
    from .node_info import node_info

    if node_class not in exported_nodes.NODE_CLASS_MAPPINGS:
        typer.echo(f"Node class '{node_class}' not found.", err=True)
        raise typer.Exit(1)

    info = node_info(node_class, exported_nodes.NODE_CLASS_MAPPINGS, exported_nodes.NODE_DISPLAY_NAME_MAPPINGS)

    if format == "json":
        Console().print_json(json.dumps(info, default=str))
        return

    console = Console()
    console.print(f"[bold]Name:[/bold] {info.get('name', '')}")
    console.print(f"[bold]Display Name:[/bold] {info.get('display_name', '')}")
    console.print(f"[bold]Category:[/bold] {info.get('category', '')}")
    console.print(f"[bold]Description:[/bold] {info.get('description', '')}")
    console.print(f"[bold]Output Node:[/bold] {info.get('output_node', False)}")
    console.print(f"[bold]Outputs:[/bold] {', '.join(str(o) for o in info.get('output', []))}")
    if info.get('output_name'):
        console.print(f"[bold]Output Names:[/bold] {', '.join(str(n) for n in info['output_name'])}")
    console.print()

    inputs = info.get("input", {})
    if inputs:
        table = Table(show_edge=False, pad_edge=False, box=None, title="Inputs")
        table.add_column("Name", no_wrap=True)
        table.add_column("Type")
        table.add_column("Required", no_wrap=True)
        for group_name in ("required", "optional", "hidden"):
            group = inputs.get(group_name, {})
            for input_name, input_spec in group.items():
                type_info = str(input_spec[0]) if isinstance(input_spec, (list, tuple)) and input_spec else str(input_spec)
                table.add_row(input_name, type_info, group_name)
        console.print(table)
