from __future__ import annotations

import importlib.resources
import re
from pathlib import Path

import yaml


def _source_root() -> Path:
    return Path(importlib.resources.files("comfy")).parent


def _workflow(name: str) -> dict:
    contents = (_source_root() / ".github" / "workflows" / name).read_text(
        encoding="utf-8"
    )
    return yaml.safe_load(contents)


def test_accelerator_validation_consumes_baked_candidate_images():
    cases = (
        ("docker-build.yml", "validate-candidates", "promote"),
        ("docker-build-amd.yml", "validate-rx7600", "promote"),
    )
    for filename, validation_name, promotion_name in cases:
        jobs = _workflow(filename)["jobs"]
        validation = jobs[validation_name]
        assert ".dev" in validation["container"]["image"]
        assert all("uses" not in step for step in validation["steps"])
        commands = "\n".join(step.get("run", "") for step in validation["steps"])
        assert "uv pip install" not in commands
        assert validation_name in jobs[promotion_name]["needs"]


def test_cuda_image_preserves_the_ngc_python_environment():
    dockerfile = (_source_root() / "Dockerfile").read_text(encoding="utf-8")
    assert "ngc-preserved.txt" in dockerfile
    assert "torchaudio==2.11.0+cpu" in dockerfile
    assert "UV_NO_CACHE" not in dockerfile
    assert not re.search(r"(?<!uv )\bpip install\b", dockerfile)


def test_source_is_added_after_the_reusable_dependency_layer():
    for name in ("Dockerfile", "amd.Dockerfile"):
        dockerfile = (_source_root() / name).read_text(encoding="utf-8")
        assert dockerfile.index("-r /workspace/project/pyproject.toml") < dockerfile.index(
            "ADD . /workspace/src"
        )


def test_accelerator_builds_reuse_promoted_images_without_duplicate_gha_export():
    for name in ("docker-build.yml", "docker-build-amd.yml"):
        workflow = (_source_root() / ".github" / "workflows" / name).read_text(
            encoding="utf-8"
        )
        assert "cache-to: type=inline" in workflow
        assert "type=registry" in workflow
        assert "type=gha" not in workflow
