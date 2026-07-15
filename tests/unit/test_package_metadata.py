from pathlib import Path


def test_cli_runtime_dependencies_are_declared():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '    "click",' in pyproject
    assert '    "transformers>=4.57.3,<5",' in pyproject
