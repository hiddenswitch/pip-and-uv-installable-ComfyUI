import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def test_cli_runtime_dependencies_are_declared():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '    "click",' in pyproject
    dependency = re.search(r'^\s*"(?P<requirement>transformers[^\"]*)",$', pyproject, re.MULTILINE)
    assert dependency is not None
    transformers = Requirement(dependency.group("requirement"))
    assert Version("4.57.3") in transformers.specifier
    assert Version("5.10.0") in transformers.specifier


def test_vtracer_keeps_convert_pixels_to_svg_api():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    dependency = re.search(r'^\s*"(?P<requirement>vtracer[^"]*)",$', pyproject, re.MULTILINE)
    assert dependency is not None
    vtracer = Requirement(dependency.group("requirement"))

    assert Version("0.6.15") in vtracer.specifier
    assert Version("1.0.0a1") not in vtracer.specifier
