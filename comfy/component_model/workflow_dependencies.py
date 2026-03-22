"""Resolve workflow node types to installable package names.

Given a ComfyUI workflow (UI or API format), determine which custom node
packages are required.  The mapping uses:

1. The built-in ``NODE_CLASS_MAPPINGS`` (if the node system is loaded).
2. ``comfyui_manager``'s ``extension-node-map.json``  (class_type -> repo URL).
3. ``comfyui_manager``'s ``custom-node-list.json``   (repo URL -> node id).
4. The pip facade snapshot DB                        (repo URL -> canonical name).

The canonical package name returned is the one usable with
``uv pip install --extra-index-url .../simple/ <name>``.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import sqlite3
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

import fsspec

logger = logging.getLogger(__name__)

# Node types that are built into the frontend / virtual and never need a package.
VIRTUAL_NODE_TYPES: frozenset[str] = frozenset({
    "Reroute",
    "PrimitiveNode",
    "Note",
    "MarkdownNote",
})

# UUID pattern for subgraph / component node IDs.
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)

_DEFAULT_SNAPSHOT_URI = "pkg://comfy.custom_nodes/pip_facade_registry_snapshot.sqlite.xz"


def _normalize_repo(url: str) -> str:
    """Strip trailing slashes and .git suffix for consistent matching."""
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url.lower()


def _load_extension_node_map() -> dict[str, str]:
    """Return *class_type -> repo_url* from comfyui_manager's extension-node-map."""
    path = files("comfyui_manager").joinpath("extension-node-map.json")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for repo_url, value in data.items():
        if isinstance(value, list) and value:
            for class_type in value[0]:
                mapping.setdefault(class_type, repo_url)
    return mapping


def _load_repo_to_node_id() -> dict[str, str]:
    """Return *normalized_repo_url -> node_id* from comfyui_manager's custom-node-list."""
    path = files("comfyui_manager").joinpath("custom-node-list.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for entry in data.get("custom_nodes", ()):
        ref = entry.get("reference", "")
        node_id = entry.get("id", "")
        if ref and node_id and node_id != "?":
            mapping[_normalize_repo(ref)] = node_id
    return mapping


def _load_repo_to_canonical_from_snapshot(snapshot_uri: str | None = None) -> dict[str, str]:
    """Return *normalized_repo_url -> canonical_name* from a facade snapshot DB."""
    if snapshot_uri is None:
        snapshot_uri = _DEFAULT_SNAPSHOT_URI
    with tempfile.NamedTemporaryFile(
        prefix="comfyui_deps_snapshot_", suffix=".sqlite", delete=False,
    ) as tmp:
        temp_path = Path(tmp.name)
    try:
        with fsspec.open(snapshot_uri, mode="rb", compression="infer") as source:
            with open(temp_path, "wb") as dest:
                shutil.copyfileobj(source, dest)
        conn = sqlite3.connect(temp_path)
        try:
            rows = conn.execute("SELECT repo_url, canonical_name FROM projects").fetchall()
            return {_normalize_repo(url): name for url, name in rows}
        finally:
            conn.close()
    finally:
        temp_path.unlink(missing_ok=True)


def build_class_type_to_package(
    *,
    snapshot_uri: str | None = None,
    builtin_class_types: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Build a mapping from class_type to canonical package name.

    Built-in class types (those shipped with ComfyUI itself) are excluded.
    """
    ext_map = _load_extension_node_map()
    repo_to_id = _load_repo_to_node_id()
    repo_to_canonical = _load_repo_to_canonical_from_snapshot(snapshot_uri)

    result: dict[str, str] = {}
    for class_type, repo_url in ext_map.items():
        if class_type in builtin_class_types:
            continue
        norm = _normalize_repo(repo_url)
        canonical = repo_to_canonical.get(norm)
        if canonical:
            result[class_type] = canonical
            continue
        node_id = repo_to_id.get(norm)
        if node_id:
            result[class_type] = node_id
    return result


def extract_class_types_from_workflow(workflow: dict[str, Any]) -> set[str]:
    """Extract all node class_type values from a workflow (UI or API format)."""
    class_types: set[str] = set()

    # UI format: {"nodes": [{"type": "KSampler", ...}, ...]}
    if "nodes" in workflow:
        for node in workflow.get("nodes", []):
            node_type = node.get("type")
            if node_type:
                class_types.add(node_type)

    # API format: {"1": {"class_type": "KSampler", "inputs": {...}}, ...}
    else:
        for node_data in workflow.values():
            if isinstance(node_data, dict):
                ct = node_data.get("class_type")
                if ct:
                    class_types.add(ct)

    return class_types


def resolve_workflow_packages(
    workflow: dict[str, Any],
    *,
    snapshot_uri: str | None = None,
    builtin_class_types: frozenset[str] | None = None,
) -> list[str]:
    """Return sorted list of package names required by *workflow*.

    Excludes built-in nodes, virtual/UI-only nodes, and subgraph component
    nodes (UUID-style type names).
    """
    if builtin_class_types is None:
        from ..nodes.package import import_all_nodes_in_workspace
        nodes = import_all_nodes_in_workspace()
        builtin_class_types = frozenset(nodes.NODE_CLASS_MAPPINGS.keys())

    class_types = extract_class_types_from_workflow(workflow)
    mapping = build_class_type_to_package(
        snapshot_uri=snapshot_uri,
        builtin_class_types=builtin_class_types,
    )

    packages: set[str] = set()
    unresolved: list[str] = []
    for ct in sorted(class_types):
        if ct in VIRTUAL_NODE_TYPES:
            continue
        if _UUID_RE.match(ct):
            continue
        if ct in builtin_class_types:
            continue
        pkg = mapping.get(ct)
        if pkg:
            packages.add(pkg)
        else:
            unresolved.append(ct)

    if unresolved:
        logger.warning("Unresolved node types: %s", ", ".join(unresolved))

    return sorted(packages)
