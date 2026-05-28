from __future__ import annotations

from ..cmd.main_pre import tracer

import asyncio
import json
import logging
import lzma
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from ..component_model.configuration import Configuration
from .registry import (
    FacadeProject,
    FacadeRegistry,
    FacadeVersion,
    is_excluded_facade_project,
    normalize_repo_url,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1"


def _compact_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=False)


def _resolve_compression(output_path: Path, compression: str) -> str:
    if compression == "auto":
        return "xz" if output_path.suffix == ".xz" else "none"
    return compression


_EXTENSION_NODE_MAP_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json"


def _parse_extension_node_map(data: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for repo_url, value in data.items():
        if isinstance(value, list) and value:
            for class_type in value[0]:
                mapping.setdefault(class_type, repo_url)
    return mapping


async def _load_class_type_to_repo(session: aiohttp.ClientSession) -> dict[str, str]:
    """Return *class_type -> repo_url* from the latest extension-node-map, with bundled fallback."""
    try:
        async with session.get(_EXTENSION_NODE_MAP_URL) as response:
            response.raise_for_status()
            data = await response.json(content_type=None)
            return _parse_extension_node_map(data)
    except Exception:
        logger.debug("Failed to fetch live extension-node-map, using bundled fallback")
        from importlib.resources import files as resource_files
        path = resource_files("comfyui_manager").joinpath("extension-node-map.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        return _parse_extension_node_map(data)


def _build_class_type_rows(
    projects: list[FacadeProject],
    class_type_to_repo: dict[str, str],
) -> list[tuple[str, str]]:
    """Build ``(class_type, canonical_name)`` rows by joining extension-node-map against projects."""
    from .registry import repo_basename

    repo_to_canonical: dict[str, str] = {}
    for project in projects:
        if project.repo_url:
            repo_to_canonical[normalize_repo_url(project.repo_url)] = project.canonical_name

    rows: list[tuple[str, str]] = []
    for class_type, repo_url in class_type_to_repo.items():
        norm = normalize_repo_url(repo_url)
        canonical = repo_to_canonical.get(norm)
        if canonical is None:
            canonical = canonicalize_project_name(repo_basename(repo_url))
        if is_excluded_facade_project(canonical):
            continue
        rows.append((class_type, canonical))
    return rows


def _write_snapshot_sqlite(
    sqlite_path: Path,
    *,
    projects: list[FacadeProject],
    versions_by_node_id: dict[str, list[FacadeVersion]],
    class_type_rows: list[tuple[str, str]],
    base_url: str,
    only_known_nodes: bool,
) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(sqlite_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA page_size=4096")
        connection.execute("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID
        """)
        connection.execute("""
            CREATE TABLE projects (
                canonical_name TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                node_id TEXT NOT NULL,
                repo_url TEXT NOT NULL,
                repo_name TEXT NOT NULL,
                description TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                extra_requirements_json TEXT NOT NULL,
                skip_requirements_json TEXT NOT NULL,
                depends_on_json TEXT NOT NULL,
                latest_version TEXT
            ) WITHOUT ROWID
        """)
        connection.execute("""
            CREATE TABLE versions (
                node_id TEXT NOT NULL,
                version TEXT NOT NULL,
                download_url TEXT NOT NULL,
                dependencies_json TEXT NOT NULL,
                deprecated INTEGER NOT NULL,
                PRIMARY KEY (node_id, version)
            ) WITHOUT ROWID
        """)
        connection.execute("CREATE INDEX versions_by_node_id ON versions (node_id)")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS class_types (
                class_type TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL
            ) WITHOUT ROWID
        """)

        project_rows = [
            (
                project.canonical_name,
                project.display_name,
                project.node_id,
                project.repo_url,
                project.repo_name,
                project.description,
                _compact_json(list(project.aliases)),
                _compact_json(list(project.extra_requirements)),
                _compact_json(sorted(project.skip_requirements)),
                _compact_json(list(project.depends_on)),
                project.latest_version,
            )
            for project in projects
        ]
        version_rows = [
            (
                project.node_id,
                version.version,
                version.download_url,
                _compact_json(list(version.dependencies)),
                int(version.deprecated),
            )
            for project in projects
            for version in versions_by_node_id.get(project.node_id, [])
        ]

        connection.executemany(
            """
            INSERT INTO projects (
                canonical_name, display_name, node_id, repo_url, repo_name, description,
                aliases_json, extra_requirements_json, skip_requirements_json, depends_on_json,
                latest_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            project_rows,
        )
        connection.executemany(
            """
            INSERT INTO versions (
                node_id, version, download_url, dependencies_json, deprecated
            ) VALUES (?, ?, ?, ?, ?)
            """,
            version_rows,
        )

        connection.executemany(
            "INSERT OR IGNORE INTO class_types (class_type, canonical_name) VALUES (?, ?)",
            class_type_rows,
        )

        metadata_rows = [
            ("format", "appmana-comfyui-pip-facade-registry-snapshot"),
            ("schema_version", _SCHEMA_VERSION),
            ("created_at", datetime.now(timezone.utc).isoformat()),
            ("registry_base_url", base_url),
            ("only_known_nodes", "1" if only_known_nodes else "0"),
            ("project_count", str(len(projects))),
            ("version_count", str(len(version_rows))),
        ]
        connection.executemany("INSERT INTO metadata (key, value) VALUES (?, ?)", metadata_rows)
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()


def write_facade_registry_snapshot(
    output_path: str | Path,
    *,
    projects: list[FacadeProject],
    versions_by_node_id: dict[str, list[FacadeVersion]],
    class_type_rows: list[tuple[str, str]] | None = None,
    base_url: str,
    only_known_nodes: bool,
    compression: str = "auto",
    overwrite: bool = False,
) -> Path:
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Snapshot output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_compression = _resolve_compression(destination, compression)

    with tempfile.TemporaryDirectory(prefix="comfyui_facade_snapshot_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        sqlite_path = temp_dir / "registry.sqlite"
        _write_snapshot_sqlite(
            sqlite_path,
            projects=projects,
            versions_by_node_id=versions_by_node_id,
            class_type_rows=class_type_rows or [],
            base_url=base_url,
            only_known_nodes=only_known_nodes,
        )
        if resolved_compression == "xz":
            with sqlite_path.open("rb") as source, lzma.open(destination, "wb", preset=9) as compressed:
                shutil.copyfileobj(source, compressed)
        else:
            shutil.copyfile(sqlite_path, destination)
    return destination


async def snapshot_facade_registry(configuration: Configuration) -> Path:
    output_path = configuration.pip_facade_snapshot_output
    if not output_path:
        raise ValueError("pip_facade_snapshot_output must be set")

    timeout = aiohttp.ClientTimeout(total=10 * 60.0, connect=60.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        registry = FacadeRegistry(
            session,
            base_url=configuration.pip_facade_registry_base_url,
            only_known_nodes=configuration.pip_facade_only_known_nodes,
        )
        with tracer.start_as_current_span("Snapshot Facade Registry") as span:
            span.set_attribute("facade.registry_base_url", configuration.pip_facade_registry_base_url)
            span.set_attribute("facade.only_known_nodes", configuration.pip_facade_only_known_nodes)
            projects = await registry.list_projects()
            versions_by_node_id = await _collect_versions(registry, projects)
            class_type_to_repo = await _load_class_type_to_repo(session)
            class_type_rows = _build_class_type_rows(projects, class_type_to_repo)
            span.set_attribute("facade.class_type_count", len(class_type_rows))
            snapshot_path = await asyncio.to_thread(
                write_facade_registry_snapshot,
                output_path,
                projects=projects,
                versions_by_node_id=versions_by_node_id,
                class_type_rows=class_type_rows,
                base_url=configuration.pip_facade_registry_base_url,
                only_known_nodes=configuration.pip_facade_only_known_nodes,
                compression=configuration.pip_facade_snapshot_compression,
                overwrite=configuration.pip_facade_snapshot_overwrite,
            )
            span.set_attribute("facade.project_count", len(projects))
            span.set_attribute("facade.version_count", sum(len(items) for items in versions_by_node_id.values()))
            span.set_attribute("facade.snapshot_path", str(snapshot_path))
            return snapshot_path


async def _collect_versions(
    registry: FacadeRegistry,
    projects: list[FacadeProject],
) -> dict[str, list[FacadeVersion]]:
    semaphore = asyncio.Semaphore(12)

    async def load(project: FacadeProject) -> tuple[str, list[FacadeVersion]]:
        async with semaphore:
            return project.node_id, await registry.list_versions(project)

    results = await asyncio.gather(*(load(project) for project in projects))
    return dict(results)
