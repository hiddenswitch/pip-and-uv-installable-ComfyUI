from __future__ import annotations

import json
import lzma
import sqlite3
import sys
from pathlib import Path

from comfy.custom_node_facade.registry import (
    FacadeProject,
    FacadeVersion,
    SnapshotFacadeRegistry,
    _filter_pep440_versions,
    _sort_versions,
)
from comfy.custom_node_facade.snapshot import write_facade_registry_snapshot


def _sample_project() -> FacadeProject:
    return FacadeProject(
        canonical_name="comfyui-custom-scripts",
        display_name="ComfyUI-Custom-Scripts",
        node_id="comfyui-custom-scripts",
        repo_url="https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
        repo_name="ComfyUI-Custom-Scripts",
        description="Scripts",
        aliases=("comfyui-custom-scripts", "pysssss"),
        extra_requirements=("deepdiff",),
        skip_requirements=frozenset({"torch", "numpy"}),
        depends_on=("comfyui-kjnodes",),
        latest_version="1.2.5",
    )


def _sample_version() -> FacadeVersion:
    return FacadeVersion(
        version="1.2.5",
        download_url="https://example.invalid/node.zip",
        dependencies=("requests>=2",),
        deprecated=False,
    )


def test_write_facade_registry_snapshot_sqlite(tmp_path: Path):
    output = tmp_path / "registry.sqlite"
    project = _sample_project()
    version = _sample_version()

    written = write_facade_registry_snapshot(
        output,
        projects=[project],
        versions_by_node_id={project.node_id: [version]},
        base_url="https://registry.example.invalid",
        only_known_nodes=True,
    )

    assert written == output
    assert output.exists()

    with sqlite3.connect(output) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        assert metadata["format"] == "appmana-comfyui-pip-facade-registry-snapshot"
        assert metadata["registry_base_url"] == "https://registry.example.invalid"
        assert metadata["only_known_nodes"] == "1"

        project_row = connection.execute(
            """
            SELECT display_name, aliases_json, extra_requirements_json, skip_requirements_json, depends_on_json, latest_version
            FROM projects
            WHERE canonical_name = ?
            """,
            (project.canonical_name,),
        ).fetchone()
        assert project_row is not None
        assert project_row[0] == project.display_name
        assert json.loads(project_row[1]) == list(project.aliases)
        assert json.loads(project_row[2]) == list(project.extra_requirements)
        assert json.loads(project_row[3]) == sorted(project.skip_requirements)
        assert json.loads(project_row[4]) == list(project.depends_on)
        assert project_row[5] == project.latest_version

        version_row = connection.execute(
            "SELECT version, download_url, dependencies_json, deprecated FROM versions WHERE node_id = ?",
            (project.node_id,),
        ).fetchone()
        assert version_row == (
            version.version,
            version.download_url,
            json.dumps(list(version.dependencies), separators=(",", ":")),
            0,
        )


def test_write_facade_registry_snapshot_xz(tmp_path: Path):
    output = tmp_path / "registry.sqlite.xz"
    project = _sample_project()
    version = _sample_version()

    written = write_facade_registry_snapshot(
        output,
        projects=[project],
        versions_by_node_id={project.node_id: [version]},
        base_url="https://registry.example.invalid",
        only_known_nodes=False,
        compression="auto",
    )

    assert written == output
    assert output.exists()

    extracted = tmp_path / "extracted.sqlite"
    with lzma.open(output, "rb") as source, extracted.open("wb") as destination:
        destination.write(source.read())

    with sqlite3.connect(extracted) as connection:
        assert connection.execute("SELECT COUNT(*) FROM projects").fetchone() == (1,)


def test_sort_versions_tolerates_non_pep440_versions():
    versions = [
        FacadeVersion(version="0.8.17-bugfix", download_url="https://example.invalid/b", dependencies=(), deprecated=False),
        FacadeVersion(version="1.2.0", download_url="https://example.invalid/a", dependencies=(), deprecated=False),
        FacadeVersion(version="1.1.0", download_url="https://example.invalid/c", dependencies=(), deprecated=False),
    ]

    sorted_versions = _sort_versions(versions)

    assert [item.version for item in sorted_versions] == ["1.2.0", "1.1.0", "0.8.17-bugfix"]


def test_filter_pep440_versions_omits_invalid_entries():
    versions = [
        FacadeVersion(version="1.2.0", download_url="https://example.invalid/a", dependencies=(), deprecated=False),
        FacadeVersion(version="0.8.17-bugfix", download_url="https://example.invalid/b", dependencies=(), deprecated=False),
    ]

    filtered = _filter_pep440_versions(versions)

    assert [item.version for item in filtered] == ["1.2.0"]


async def test_snapshot_registry_reads_plain_sqlite(tmp_path: Path):
    output = tmp_path / "registry.sqlite"
    project = _sample_project()
    version = _sample_version()
    write_facade_registry_snapshot(
        output,
        projects=[project],
        versions_by_node_id={project.node_id: [version]},
        base_url="https://registry.example.invalid",
        only_known_nodes=True,
    )

    registry = SnapshotFacadeRegistry(snapshot_uri=str(output))
    projects = await registry.list_projects()

    assert project.canonical_name in [item.canonical_name for item in projects]
    loaded_version = await registry.get_version(project.canonical_name, version.version)
    assert loaded_version == version


async def test_snapshot_registry_reads_pkg_xz_uri(tmp_path: Path, monkeypatch):
    package_root = tmp_path / "snapshot_pkg"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")

    snapshot_path = package_root / "registry.sqlite.xz"
    project = _sample_project()
    version = _sample_version()
    write_facade_registry_snapshot(
        snapshot_path,
        projects=[project],
        versions_by_node_id={project.node_id: [version]},
        base_url="https://registry.example.invalid",
        only_known_nodes=False,
        compression="auto",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("snapshot_pkg", None)

    registry = SnapshotFacadeRegistry(snapshot_uri="pkg://snapshot_pkg/registry.sqlite.xz")
    projects = await registry.list_projects()

    assert project.canonical_name in [item.canonical_name for item in projects]
    versions = await registry.list_versions(project.canonical_name)
    assert version in versions


async def test_snapshot_registry_omits_non_pep440_versions(tmp_path: Path):
    output = tmp_path / "registry.sqlite"
    project = _sample_project()
    valid = _sample_version()
    invalid = FacadeVersion(
        version="0.8.17-bugfix",
        download_url="https://example.invalid/node-invalid.zip",
        dependencies=(),
        deprecated=False,
    )
    write_facade_registry_snapshot(
        output,
        projects=[project],
        versions_by_node_id={project.node_id: [valid, invalid]},
        base_url="https://registry.example.invalid",
        only_known_nodes=True,
    )

    registry = SnapshotFacadeRegistry(snapshot_uri=str(output))
    versions = await registry.list_versions(project.canonical_name)

    assert versions == [valid]
