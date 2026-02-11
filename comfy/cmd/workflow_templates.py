"""
List available workflow templates.

Reuses template enumeration from:
  - ``comfy.app.frontend_management.FrontendManager`` (installed package templates)
  - ``comfy.app.custom_node_manager`` (custom node example workflows)
"""
from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TemplateInfo:
    name: str
    source: str
    path: Optional[str] = None
    template_id: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    media_type: Optional[str] = None


def _templates_from_package() -> list[TemplateInfo]:
    templates = []
    try:
        from comfyui_workflow_templates import iter_templates
        for entry in iter_templates():
            templates.append(TemplateInfo(
                name=getattr(entry, "title", None) or getattr(entry, "name", str(entry.template_id)),
                source="package",
                template_id=str(entry.template_id),
                description=getattr(entry, "description", None),
                tags=list(getattr(entry, "tags", [])),
                media_type=getattr(entry, "media_type", None) or getattr(entry, "mediaType", None),
            ))
    except ImportError:
        logger.debug("comfyui_workflow_templates not installed")
    except Exception as exc:
        logger.warning("Failed to enumerate package templates: %s", exc)
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


def list_templates(format: str = "table", extra_dirs: list[str] = None, convert: bool = False):
    extra_dirs = extra_dirs or []

    all_templates = []
    all_templates.extend(_templates_from_package())
    all_templates.extend(_templates_from_custom_nodes())
    all_templates.extend(_templates_from_dirs(extra_dirs))

    if convert:
        from ..component_model.workflow_convert import convert_ui_to_api
        for tmpl in all_templates:
            if tmpl.path and os.path.exists(tmpl.path):
                try:
                    with open(tmpl.path) as f:
                        workflow = json.load(f)
                    if "nodes" in workflow:
                        convert_ui_to_api(workflow)
                        tmpl.description = (tmpl.description or "") + " [converted to API format]"
                except Exception as exc:
                    logger.warning("Failed to convert %s: %s", tmpl.name, exc)

    if format == "json":
        print(json.dumps([asdict(t) for t in all_templates], indent=2))
    else:
        if not all_templates:
            print("No workflow templates found.")
            return
        name_width = max(max(len(t.name) for t in all_templates), 4)
        source_width = max(max(len(t.source) for t in all_templates), 6)
        print(f"{'Name':<{name_width}}  {'Source':<{source_width}}  Description")
        print(f"{'-' * name_width}  {'-' * source_width}  {'-' * 40}")
        for t in all_templates:
            desc = t.description or ""
            if len(desc) > 60:
                desc = desc[:57] + "..."
            print(f"{t.name:<{name_width}}  {t.source:<{source_width}}  {desc}")
