"""Resolve workflow node types to installable package names.

Given a ComfyUI workflow (UI or API format), determine which custom node
packages are required. The mapping uses the snapshot DB's ``class_types``
table (class_type -> canonical package name), built during snapshot creation
from comfyui_manager's extension-node-map.json joined against projects.

The canonical package name returned is the one usable with
``uv pip install --extra-index-url .../simple/ <name>``.
"""
from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import fsspec

logger = logging.getLogger(__name__)

VIRTUAL_NODE_TYPES: frozenset[str] = frozenset({
    "Reroute",
    "PrimitiveNode",
    "Note",
    "MarkdownNote",
})

CORE_PACKAGES: frozenset[str] = frozenset({
    "comfyui",
    "comfy",
})

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)

_DEFAULT_SNAPSHOT_URI = "pkg://comfy.custom_nodes/pip_facade_registry_snapshot.sqlite.xz"


def _open_snapshot(snapshot_uri: str | None = None) -> tuple[sqlite3.Connection, Path]:
    """Open a facade snapshot DB, returning ``(connection, temp_path)``."""
    if snapshot_uri is None:
        snapshot_uri = _DEFAULT_SNAPSHOT_URI
    with tempfile.NamedTemporaryFile(
        prefix="comfyui_deps_snapshot_", suffix=".sqlite", delete=False,
    ) as tmp:
        temp_path = Path(tmp.name)
    with fsspec.open(snapshot_uri, mode="rb", compression="infer") as source:
        with open(temp_path, "wb") as dest:
            shutil.copyfileobj(source, dest)
    return sqlite3.connect(temp_path), temp_path


def _load_class_type_to_package_from_snapshot(snapshot_uri: str | None = None) -> dict[str, str]:
    """Return *class_type -> canonical_name* from the snapshot's ``class_types`` table."""
    conn, temp_path = _open_snapshot(snapshot_uri)
    try:
        try:
            rows = conn.execute("SELECT class_type, canonical_name FROM class_types").fetchall()
        except sqlite3.OperationalError:
            return {}
        return dict(rows)
    finally:
        conn.close()
        temp_path.unlink(missing_ok=True)


def _load_package_versions_from_snapshot(snapshot_uri: str | None = None) -> dict[str, str]:
    """Return *canonical_name -> latest_version* from a facade snapshot DB."""
    conn, temp_path = _open_snapshot(snapshot_uri)
    try:
        rows = conn.execute(
            "SELECT canonical_name, latest_version FROM projects WHERE latest_version IS NOT NULL"
        ).fetchall()
        return {name: version for name, version in rows if version}
    finally:
        conn.close()
        temp_path.unlink(missing_ok=True)


def build_class_type_to_package(
    *,
    snapshot_uri: str | None = None,
    builtin_class_types: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Build a mapping from class_type to canonical package name.

    Built-in class types (those shipped with ComfyUI itself) are excluded.
    """
    result = _load_class_type_to_package_from_snapshot(snapshot_uri)
    for ct in builtin_class_types:
        result.pop(ct, None)
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
        if pkg and pkg not in CORE_PACKAGES:
            packages.add(pkg)
        elif not pkg:
            unresolved.append(ct)

    if unresolved:
        logger.warning("Unresolved node types: %s", ", ".join(unresolved))

    return sorted(packages)


def resolve_workflow_packages_versioned(
    workflow: dict[str, Any],
    *,
    snapshot_uri: str | None = None,
    builtin_class_types: frozenset[str] | None = None,
) -> list[tuple[str, str | None]]:
    """Return sorted list of ``(package_name, version_or_None)`` for *workflow*."""
    packages = resolve_workflow_packages(
        workflow,
        snapshot_uri=snapshot_uri,
        builtin_class_types=builtin_class_types,
    )
    versions = _load_package_versions_from_snapshot(snapshot_uri)
    return [(pkg, versions.get(pkg)) for pkg in packages]
