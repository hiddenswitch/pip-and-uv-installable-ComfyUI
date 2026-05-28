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
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import fsspec

from ..custom_node_facade.registry import is_excluded_facade_project

logger = logging.getLogger(__name__)

VIRTUAL_NODE_TYPES: frozenset[str] = frozenset({
    "Reroute",
    "PrimitiveNode",
    "Int",
    "Float",
    "String",
    "StringMultiline",
    "Boolean",
    "Note",
    "MarkdownNote",
    "Label (rgthree)",
})

CORE_PACKAGES: frozenset[str] = frozenset({
    "comfyui",
    "comfy",
})

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)

_DEFAULT_SNAPSHOT_URI = "pkg://comfy.custom_nodes/pip_facade_registry_snapshot.sqlite.xz"

# Snapshot-DB corrections: class_type -> canonical pip facade package name.
#
# Some entries in the bundled ``pip_facade_registry_snapshot.sqlite.xz`` are
# wrong (the snapshot is built by joining comfyui_manager's extension-node-map
# against pip facade projects, and the join occasionally picks the wrong
# project when two packages declare overlapping classes). Until the snapshot
# is rebuilt, fix mismappings here so ``--all`` resolves to a real package
# on nodes.appmana.com.
_CLASS_TYPE_PACKAGE_OVERRIDES: dict[str, str] = {
    # UltimateSDUpscale → ssitu/ComfyUI_UltimateSDUpscale (not the non-existent
    # "comfyui-umeairt-toolkit" package the snapshot maps it to).
    "UltimateSDUpscale": "comfyui-ultimatesdupscale",
    "UltimateSDUpscaleNoUpscale": "comfyui-ultimatesdupscale",
    "UltimateSDUpscaleCustomSample": "comfyui-ultimatesdupscale",
    # numz/ComfyUI-SeedVR2_VideoUpscaler — also wrongly attributed to umeairt.
    "SeedVR2": "comfyui-seedvr2-videoupscaler",
    "SeedVR2VideoUpscaler": "comfyui-seedvr2-videoupscaler",
    "SeedVR2LoadDiTModel": "comfyui-seedvr2-videoupscaler",
    "SeedVR2LoadVAEModel": "comfyui-seedvr2-videoupscaler",
    "SeedVR2BlockSwap": "comfyui-seedvr2-videoupscaler",
    "SeedVR2TorchCompileSettings": "comfyui-seedvr2-videoupscaler",
}


@contextmanager
def _open_snapshot(snapshot_uri: str | None = None):
    """Context manager that yields a SQLite connection to the facade snapshot."""
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
            yield conn
        finally:
            conn.close()
    finally:
        temp_path.unlink(missing_ok=True)


def _load_snapshot_data(
    snapshot_uri: str | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(class_type_to_package, package_versions)`` from a single snapshot open."""
    with _open_snapshot(snapshot_uri) as conn:
        try:
            ct_rows = conn.execute("SELECT class_type, canonical_name FROM class_types").fetchall()
        except sqlite3.OperationalError:
            ct_rows = []
        ver_rows = conn.execute(
            "SELECT canonical_name, latest_version FROM projects WHERE latest_version IS NOT NULL"
        ).fetchall()
    return dict(ct_rows), {name: version for name, version in ver_rows if version}


def extract_class_types_from_workflow(workflow: dict[str, Any]) -> set[str]:
    """Extract all node class_type values from a workflow (UI or API format)."""
    class_types: set[str] = set()

    if "nodes" in workflow:
        for node in workflow.get("nodes", []):
            node_type = node.get("type")
            if node_type:
                class_types.add(node_type)
    else:
        for node_data in workflow.values():
            if isinstance(node_data, dict):
                ct = node_data.get("class_type")
                if ct:
                    class_types.add(ct)

    return class_types


def resolve_workflow_packages_versioned(
    workflow: dict[str, Any],
    *,
    snapshot_uri: str | None = None,
    builtin_class_types: frozenset[str] | None = None,
) -> list[tuple[str, str | None]]:
    """Return sorted list of ``(package_name, version_or_None)`` for *workflow*.

    Excludes built-in nodes, virtual/UI-only nodes, and subgraph component
    nodes (UUID-style type names).
    """
    if builtin_class_types is None:
        from ..nodes.package import import_all_nodes_in_workspace
        nodes = import_all_nodes_in_workspace()
        builtin_class_types = frozenset(nodes.NODE_CLASS_MAPPINGS.keys())

    from .workflow_rewrites import rewrite_class_type
    class_types = {rewrite_class_type(ct) for ct in extract_class_types_from_workflow(workflow)}
    ct_to_pkg, versions = _load_snapshot_data(snapshot_uri)
    ct_to_pkg.update(_CLASS_TYPE_PACKAGE_OVERRIDES)

    for ct in builtin_class_types:
        ct_to_pkg.pop(ct, None)

    packages: set[str] = set()
    unresolved: list[str] = []
    for ct in sorted(class_types):
        if ct in VIRTUAL_NODE_TYPES or _UUID_RE.match(ct) or ct in builtin_class_types:
            continue
        pkg = ct_to_pkg.get(ct)
        if pkg and pkg not in CORE_PACKAGES:
            if is_excluded_facade_project(pkg):
                continue
            packages.add(pkg)
        elif not pkg:
            unresolved.append(ct)

    if unresolved:
        logger.warning("Unresolved node types: %s", ", ".join(unresolved))

    return [(pkg, versions.get(pkg)) for pkg in sorted(packages)]
