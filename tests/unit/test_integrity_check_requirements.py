import importlib.metadata

from comfy.cmd import integrity_check


def test_check_installed_requirement_passes_when_version_satisfies(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "requires",
        lambda package: ["comfyui-workflow-templates>=0.10.0,<0.11"] if package == "comfyui" else [],
    )
    monkeypatch.setattr(integrity_check, "_pkg_version", lambda package: "0.10.0")

    name, result, detail = integrity_check._check_installed_requirement("comfyui-workflow-templates")

    assert name == "comfyui-workflow-templates constraint"
    assert result is True
    assert "satisfies" in detail


def test_check_installed_requirement_fails_when_version_is_too_old(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "requires",
        lambda package: ["comfyui-workflow-templates>=0.10.0,<0.11"] if package == "comfyui" else [],
    )
    monkeypatch.setattr(integrity_check, "_pkg_version", lambda package: "0.9.98")

    _, result, detail = integrity_check._check_installed_requirement("comfyui-workflow-templates")

    assert result is False
    assert "does not satisfy" in detail


def test_check_installed_requirement_fails_when_package_missing(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "requires",
        lambda package: ["comfyui-frontend-package>=1.45.15,<1.46"] if package == "comfyui" else [],
    )
    monkeypatch.setattr(integrity_check, "_pkg_version", lambda package: "(not installed)")

    _, result, detail = integrity_check._check_installed_requirement("comfyui-frontend-package")

    assert result is False
    assert "missing" in detail
