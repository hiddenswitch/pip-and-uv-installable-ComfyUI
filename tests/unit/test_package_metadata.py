import tomllib
from pathlib import Path


def test_cli_runtime_dependencies_are_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert any(dependency == "click" for dependency in dependencies)
    assert any(dependency.startswith("transformers>=4.57.3,<5") for dependency in dependencies)
