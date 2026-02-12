from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from comfyui_workflow_templates import get_asset_path, iter_templates
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

_INDEX_PREFIXES = ("index", "fuse_options")


@dataclass
class TemplateInfo:
    name: str
    source: str
    path: Optional[str] = None
    template_id: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    media_type: Optional[str] = None


def _load_index_metadata() -> dict[str, dict]:
    path = get_asset_path("index", "index.json")
    with open(path) as f:
        categories = json.load(f)
    return {
        t["name"]: t
        for cat in categories
        for t in cat.get("templates", [])
    }


def _templates_from_package() -> list[TemplateInfo]:
    index = _load_index_metadata()
    templates = []
    for entry in iter_templates():
        tid = str(entry.template_id)
        if tid.startswith(_INDEX_PREFIXES):
            continue
        meta = index.get(tid, {})
        path = get_asset_path(tid, f"{tid}.json")
        templates.append(TemplateInfo(
            name=meta.get("title") or tid,
            source="package",
            path=path,
            template_id=tid,
            description=meta.get("description"),
            tags=meta.get("tags", []),
            media_type=meta.get("mediaType"),
        ))
    return templates


def _templates_from_custom_nodes() -> list[TemplateInfo]:
    from .folder_paths import get_folder_paths
    from ..app.custom_node_manager import CustomNodeManager

    return [
        TemplateInfo(name=wf_name, source=f"custom_node:{node_name}", path=filepath)
        for node_name, wf_name, filepath
        in CustomNodeManager.scan_example_workflows(get_folder_paths("custom_nodes"))
    ]


def _templates_from_dirs(dirs: list[str]) -> list[TemplateInfo]:
    templates = []
    for d in dirs:
        if not os.path.isdir(d):
            logger.warning("Template dir does not exist: %s", d)
            continue
        for filepath in glob.glob(os.path.join(d, "**/*.json"), recursive=True):
            wf_name = os.path.splitext(os.path.basename(filepath))[0]
            templates.append(TemplateInfo(
                name=wf_name,
                source=f"dir:{d}",
                path=filepath,
            ))
    return templates


def get_all_templates(extra_dirs: list[str] = None) -> list[TemplateInfo]:
    extra_dirs = extra_dirs or []
    all_templates = []
    all_templates.extend(_templates_from_package())
    all_templates.extend(_templates_from_custom_nodes())
    all_templates.extend(_templates_from_dirs(extra_dirs))
    return all_templates


def resolve_template(name_or_id: str, extra_dirs: list[str] = None) -> str:
    """Resolve a template name or ID to a workflow JSON file path.

    Raises ``ValueError`` if no match is found.
    """
    templates = get_all_templates(extra_dirs)

    for t in templates:
        if t.template_id == name_or_id:
            return t.path

    lower = name_or_id.lower()
    for t in templates:
        if t.name.lower() == lower:
            return t.path

    matches = [t for t in templates if lower in t.name.lower() or (t.template_id and lower in t.template_id.lower())]
    if len(matches) == 1:
        return matches[0].path
    if len(matches) > 1:
        names = ", ".join(m.template_id or m.name for m in matches[:5])
        raise ValueError(f"Ambiguous template '{name_or_id}', matches: {names}")
    raise ValueError(f"No template found matching '{name_or_id}'")


def list_templates(format: str = "table", extra_dirs: list[str] = None, convert: bool = False):
    all_templates = get_all_templates(extra_dirs)

    if convert:
        from ..component_model.workflow_convert import convert_ui_to_api
        for tmpl in all_templates:
            if tmpl.path and os.path.exists(tmpl.path):
                with open(tmpl.path) as f:
                    workflow = json.load(f)
                if "nodes" in workflow:
                    convert_ui_to_api(workflow)
                    tmpl.description = (tmpl.description or "") + " [converted to API format]"

    if format == "json":
        print(json.dumps([asdict(t) for t in all_templates], indent=2))
    else:
        _print_table(all_templates)


def _print_table(templates: list[TemplateInfo]):
    if not templates:
        print("No workflow templates found.")
        return

    console = Console()
    table = Table(show_edge=False, pad_edge=False, box=None, width=max(console.width, 200))
    table.add_column("ID", no_wrap=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("Description", no_wrap=True, overflow="ellipsis", ratio=1)
    for t in templates:
        table.add_row(
            t.template_id or "",
            t.name,
            t.description or "",
        )
    console.print(table, soft_wrap=True)
