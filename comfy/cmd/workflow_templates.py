from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Final, Optional

from comfyui_workflow_templates import get_asset_path, iter_templates
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from ..component_model.workflow_convert import convert_ui_to_api
from ..component_model.prompt_utils import (
    _TEXT_ENCODE_FIELDS,
    _STEPS_CLASS_TYPES,
    _SEED_FIELDS,
    _IMAGE_LOAD_CLASS_TYPES,
    _VIDEO_LOAD_CLASS_TYPES,
    _AUDIO_LOAD_CLASS_TYPES,
)

logger = logging.getLogger(__name__)

_INDEX_PREFIXES: Final[tuple[str, str]] = ("index", "fuse_options")

_NEGATIVE_CAPABLE_CLASS_TYPES: Final[frozenset[str]] = frozenset({
    "KSampler",
    "KSamplerAdvanced",
    "CFGGuider",
})


@dataclass
class TemplateInfo:
    name: str
    source: str
    path: Optional[str] = None
    template_id: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    media_type: Optional[str] = None
    bundle: Optional[str] = None
    supported_params: list[str] = field(default_factory=list)


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
            bundle=entry.bundle,
        ))
    return templates


def _templates_from_custom_nodes() -> list[TemplateInfo]:
    from .folder_paths import get_folder_paths  # pylint: disable=import-error
    from ..app.custom_node_manager import CustomNodeManager

    return [
        TemplateInfo(name=wf_name, source=f"custom_node:{node_name}", path=filepath)
        for node_name, wf_name, filepath
        in CustomNodeManager.scan_example_workflows(get_folder_paths("custom_nodes"))
    ]


def _templates_from_dirs(dirs: list[str]) -> list[TemplateInfo]:
    templates = []
    for d in dirs:
        d_path = Path(d)
        if not d_path.is_dir():
            logger.warning(f"Template dir does not exist: {d}")
            continue
        for filepath in d_path.rglob("*.json"):
            templates.append(TemplateInfo(
                name=filepath.stem,
                source=f"dir:{d}",
                path=str(filepath),
            ))
    return templates


def get_all_templates(extra_dirs: list[str] = None) -> list[TemplateInfo]:
    extra_dirs = extra_dirs or []
    all_templates = []
    all_templates.extend(_templates_from_package())
    all_templates.extend(_templates_from_custom_nodes())
    all_templates.extend(_templates_from_dirs(extra_dirs))
    return all_templates


def _collect_class_types(workflow: dict) -> set[str]:
    types: set[str] = set()

    if all(isinstance(v, dict) and "class_type" in v for v in workflow.values() if isinstance(v, dict)):
        for node in workflow.values():
            if isinstance(node, dict) and "class_type" in node:
                types.add(node["class_type"])

    extra_prompt = workflow.get("extra", {}).get("prompt", {})
    if isinstance(extra_prompt, dict):
        for node in extra_prompt.values():
            if isinstance(node, dict) and "class_type" in node:
                types.add(node["class_type"])

    for node in workflow.get("nodes", []):
        node_type = node.get("type", "")
        if node_type:
            types.add(node_type)

    for sg in workflow.get("definitions", {}).get("subgraphs", []):
        for node in sg.get("nodes", []):
            node_type = node.get("type", "")
            if node_type:
                types.add(node_type)

    return types


_PARAM_CHECKS: Final[tuple[tuple[str, frozenset[str]], ...]] = (
    ("prompt", frozenset(_TEXT_ENCODE_FIELDS.keys())),
    ("steps", _STEPS_CLASS_TYPES),
    ("seed", frozenset(_SEED_FIELDS.keys())),
    ("image", _IMAGE_LOAD_CLASS_TYPES),
    ("video", _VIDEO_LOAD_CLASS_TYPES),
    ("audio", _AUDIO_LOAD_CLASS_TYPES),
)


def _detect_supported_params(workflow: dict) -> list[str]:
    types = _collect_class_types(workflow)
    params = [name for name, class_types in _PARAM_CHECKS if types & class_types]
    if "prompt" in params and types & _NEGATIVE_CAPABLE_CLASS_TYPES:
        params.insert(params.index("prompt") + 1, "negative-prompt")
    return params


def _populate_supported_params(templates: list[TemplateInfo]) -> None:
    for tmpl in templates:
        if not tmpl.path or not Path(tmpl.path).exists():
            continue
        try:
            workflow = json.loads(Path(tmpl.path).read_text())
            tmpl.supported_params = _detect_supported_params(workflow)
        except (json.JSONDecodeError, OSError):
            logger.debug(f"Could not read template {tmpl.path}")


def _build_example_invocation(tmpl: TemplateInfo) -> str:
    name = tmpl.template_id or tmpl.name
    parts = [f"comfyui post-workflow {name}"]

    placeholders = {
        "prompt": '--prompt "your text here"',
        "negative-prompt": '--negative-prompt "things to avoid"',
        "steps": "--steps 20",
        "seed": "--seed 42",
        "image": "--image https://example.com/image.png",
        "video": "--video https://example.com/video.mp4",
        "audio": "--audio https://example.com/audio.wav",
    }

    for param in tmpl.supported_params:
        if param in placeholders:
            parts.append(placeholders[param])

    return " ".join(parts)


def resolve_template(name_or_id: str, extra_dirs: list[str] = None) -> str:
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


def list_templates(format: str = "table", extra_dirs: list[str] = None, convert: bool = False, show_all: bool = False, interactive: bool = False):
    all_templates = get_all_templates(extra_dirs)

    if convert:
        for tmpl in all_templates:
            if tmpl.path and Path(tmpl.path).exists():
                workflow = json.loads(Path(tmpl.path).read_text())
                if "nodes" in workflow:
                    convert_ui_to_api(workflow)
                    tmpl.description = (tmpl.description or "") + " [converted to API format]"

    if not show_all:
        all_templates = [t for t in all_templates if "API" not in t.tags]

    _populate_supported_params(all_templates)

    if format == "json":
        records = []
        for t in all_templates:
            d = asdict(t)
            d["example_invocation"] = _build_example_invocation(t)
            records.append(d)
        console = Console()
        console.print_json(json.dumps(records))
    elif interactive:
        _interactive_select(all_templates)
    else:
        _print_table(all_templates)


def _interactive_select(templates: list[TemplateInfo]):
    if not templates:
        Console().print("No workflow templates found.")
        return

    import questionary

    choices = []
    for t in templates:
        tid = t.template_id or t.name
        desc = t.description or ""
        if len(desc) > 60:
            desc = desc[:57] + "..."
        params = " ".join(t.supported_params) if t.supported_params else ""
        label = f"{tid:<40s} {params:<30s} {desc}"
        choices.append(questionary.Choice(title=label, value=t))

    selected = questionary.select(
        "Select a workflow template:",
        choices=choices,
        use_search_filter=True,
        use_jk_keys=False,
    ).ask()

    if selected is None:
        return

    _print_detail_panel(selected)


def _print_detail_panel(tmpl: TemplateInfo):
    stderr = Console(stderr=True)
    stdout = Console()
    lines = []
    lines.append(f"[bold]Name:[/bold] {tmpl.name}")
    if tmpl.template_id:
        lines.append(f"[bold]ID:[/bold] {tmpl.template_id}")
    if tmpl.description:
        lines.append(f"[bold]Description:[/bold] {tmpl.description}")
    if tmpl.tags:
        lines.append(f"[bold]Tags:[/bold] {', '.join(tmpl.tags)}")
    if tmpl.bundle:
        lines.append(f"[bold]Bundle:[/bold] {tmpl.bundle}")
    if tmpl.supported_params:
        lines.append(f"[bold]Supported params:[/bold] --{'  --'.join(tmpl.supported_params)}")
    stderr.print("\n".join(lines))

    stderr.print()
    stderr.print("[bold]Example:[/bold]")
    stdout.print(_build_example_invocation(tmpl), highlight=False)


def _print_table(templates: list[TemplateInfo]):
    console = Console()
    if not templates:
        console.print("No workflow templates found.")
        return
    table = Table(show_edge=False, pad_edge=False, box=None, width=max(console.width, 200))
    table.add_column("ID", no_wrap=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("Params", no_wrap=True)
    table.add_column("Description", no_wrap=True, overflow="ellipsis", ratio=1)
    for t in templates:
        table.add_row(
            t.template_id or "",
            t.name,
            " ".join(t.supported_params),
            t.description or "",
        )
    console.print(table, soft_wrap=True)
