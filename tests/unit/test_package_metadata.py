import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def test_cli_runtime_dependencies_are_declared():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '    "click",' in pyproject
    assert '    "transformers>=4.57.3,<5",' in pyproject


def test_vtracer_keeps_convert_pixels_to_svg_api():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    vtracer = next(
        Requirement(dependency)
        for dependency in pyproject["project"]["dependencies"]
        if Requirement(dependency).name == "vtracer"
    )

    assert Version("0.6.15") in vtracer.specifier
    assert Version("1.0.0a1") not in vtracer.specifier
