"""Print system diagnostics and verify installation integrity."""
from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path

import psutil
from rich.console import Console
from rich.table import Table

from ..cli_args_types import Configuration


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "(not installed)"


def _section_config_files(console: Console):
    from .cli import _find_default_config_file, _load_config_file

    path = _find_default_config_file()
    if path is None:
        console.print("  No default config file found.")
        return

    console.print(f"  File: {path}")
    data = _load_config_file(path)
    for k, v in data.items():
        console.print(f"    {k}: {v}")


def _section_guess_settings(console: Console):
    from ..component_model.guess_settings import (
        _total_ram_gb, _has_nvidia_gpu, _has_amd_gpu, _competing_gpu_processes,
        _amd_gfx_version, _has_package, apply_guess_settings,
    )

    is_nvidia = _has_nvidia_gpu()
    is_amd = _has_amd_gpu()
    ram_gb = _total_ram_gb()
    gfx = _amd_gfx_version() if is_amd else None
    procs = _competing_gpu_processes() if is_nvidia else []

    det = Table(show_edge=False, pad_edge=False, box=None, title="Detected Hardware")
    det.add_column("Check", no_wrap=True)
    det.add_column("Value")
    det.add_row("Total RAM", f"{ram_gb:.1f} GB")
    det.add_row("NVIDIA GPU", str(is_nvidia))
    det.add_row("AMD GPU", str(is_amd))
    if is_amd:
        det.add_row("AMD GFX version", gfx or "unknown")
    det.add_row("Competing GPU processes", ", ".join(procs) if procs else "(none)")
    det.add_row("sageattention", "installed" if _has_package("sageattention") else "not installed")
    det.add_row("xformers", "installed" if _has_package("xformers") else "not installed")
    console.print(det)
    console.print()

    fresh = Configuration()
    apply_guess_settings(fresh)

    decisions: list[tuple[str, str, str]] = []

    if fresh.disable_pinned_memory:
        decisions.append(("disable_pinned_memory", "True", f"RAM < 32 GB ({ram_gb:.1f} GB)"))
    if fresh.fast:
        fast_str = ", ".join(str(f) for f in fresh.fast)
        decisions.append(("fast", fast_str, "NVIDIA GPU detected"))
    if fresh.novram:
        decisions.append(("novram", "True", f"competing GPU processes: {', '.join(procs)}"))
    if fresh.fp16_vae:
        decisions.append(("fp16_vae", "True", f"AMD RDNA 4 ({gfx})"))
    if fresh.fp32_vae:
        decisions.append(("fp32_vae", "True", f"AMD GPU ({gfx or 'unknown'})"))
    if fresh.use_quad_cross_attention:
        decisions.append(("use_quad_cross_attention", "True", "AMD GPU on Windows"))
    if fresh.use_sage_attention:
        decisions.append(("use_sage_attention", "True", "sageattention package found"))
    if fresh.disable_xformers is False and not fresh.use_sage_attention and not fresh.use_pytorch_cross_attention:
        decisions.append(("disable_xformers", "False", "xformers package found"))
    if fresh.use_pytorch_cross_attention:
        decisions.append(("use_pytorch_cross_attention", "True", "no preferred attention backend found"))

    if decisions:
        cfg = Table(show_edge=False, pad_edge=False, box=None, title="Resulting Configuration (--guess-settings)")
        cfg.add_column("Setting", no_wrap=True)
        cfg.add_column("Value", no_wrap=True)
        cfg.add_column("Reason")
        for setting, value, reason in decisions:
            cfg.add_row(setting, value, reason)
        console.print(cfg)
    else:
        console.print("  No settings would be changed by --guess-settings.")


_OPENCV_PACKAGES = (
    "opencv-contrib-python", "opencv-contrib-python-headless",
    "opencv-python", "opencv-python-headless",
)

_PASS = "[bold green]PASS[/bold green]"
_FAIL = "[bold red]FAIL[/bold red]"
_SKIP = "[bold yellow]SKIP[/bold yellow]"


# check result: True=pass, False=fail, None=skip
_CheckResult = tuple[str, bool | None, str]


def _run_compatibility_checks() -> list[_CheckResult]:
    from packaging.version import Version

    checks: list[_CheckResult] = []

    # opencv + numpy 2 compatibility
    numpy_ver = _pkg_version("numpy")
    if numpy_ver != "(not installed)" and Version(numpy_ver) >= Version("2"):
        bad = []
        for opencv_pkg in _OPENCV_PACKAGES:
            cv_ver = _pkg_version(opencv_pkg)
            if cv_ver != "(not installed)" and Version(cv_ver) < Version("4.8"):
                bad.append(f"{opencv_pkg} {cv_ver}")
        if bad:
            checks.append((
                "opencv + numpy 2",
                False,
                f"{', '.join(bad)} incompatible with numpy {numpy_ver} (requires opencv >= 4.8)",
            ))
        else:
            checks.append(("opencv + numpy 2", True, f"numpy {numpy_ver}"))

    # multiple opencv packages installed
    installed_cv = [pkg for pkg in _OPENCV_PACKAGES if _pkg_version(pkg) != "(not installed)"]
    if len(installed_cv) > 1:
        versions = ", ".join(f"{p} {_pkg_version(p)}" for p in installed_cv)
        checks.append((
            "single opencv package",
            False,
            f"multiple opencv packages installed: {versions} (only one should be installed)",
        ))
    elif len(installed_cv) == 1:
        checks.append(("single opencv package", True, installed_cv[0]))
    else:
        checks.append(("single opencv package", False, "no opencv package installed"))

    # torch ecosystem build suffix alignment
    import torch
    torch_ver = torch.__version__
    torch_suffix = _build_suffix(torch_ver)
    for companion in ("torchvision", "torchaudio"):
        ver = _pkg_version(companion)
        if ver == "(not installed)":
            continue
        suffix = _build_suffix(ver)
        if suffix != torch_suffix:
            checks.append((
                f"{companion} build match",
                False,
                f"{companion} build '{suffix or 'cpu/default'}' != torch build '{torch_suffix or 'cpu/default'}'",
            ))
        else:
            checks.append((f"{companion} build match", True, suffix or "cpu/default"))

        req_warning = _check_torch_requirement(companion, torch_ver)
        if req_warning:
            checks.append((f"{companion} torch constraint", False, req_warning))
        else:
            checks.append((f"{companion} torch constraint", True, f"compatible with torch {torch_ver.split('+')[0]}"))

    # triton runtime check
    checks.append(_check_triton())

    # attention backend runtime checks
    checks.append(_check_sageattention())
    checks.append(_check_xformers())

    return checks


def _check_triton() -> _CheckResult:
    # triton ships as "triton" for both CUDA and ROCm
    if _pkg_version("triton") == "(not installed)":
        return ("triton runtime", None, "not installed")
    import torch
    if not torch.cuda.is_available():
        return ("triton runtime", None, "no GPU device (CUDA/ROCm)")
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401

        @triton.jit
        def _add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr):
            idx = tl.arange(0, n)
            x = tl.load(x_ptr + idx)
            y = tl.load(y_ptr + idx)
            tl.store(out_ptr + idx, x + y)

        dev = torch.device("cuda")
        x = torch.ones(32, device=dev)
        y = torch.ones(32, device=dev)
        out = torch.empty(32, device=dev)
        _add_kernel[(1,)](x, y, out, 32)
        assert torch.allclose(out, torch.full((32,), 2.0, device=dev))
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        return ("triton runtime", True, f"triton {_pkg_version('triton')} kernel executed ({backend})")
    except Exception as exc:
        return ("triton runtime", False, str(exc))


def _check_sageattention() -> _CheckResult:
    if _pkg_version("sageattention") == "(not installed)":
        return ("sageattention runtime", None, "not installed")
    import torch
    if not torch.cuda.is_available():
        return ("sageattention runtime", None, "no CUDA device")
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_cuda  # pylint: disable=import-error
        q = torch.randn(1, 8, 64, 64, dtype=torch.float16, device="cuda")
        k = torch.randn(1, 8, 64, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(1, 8, 64, 64, dtype=torch.float16, device="cuda")
        sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout="HND", is_causal=False)
        return ("sageattention runtime", True, f"sageattention {_pkg_version('sageattention')} kernel executed")
    except Exception as exc:
        return ("sageattention runtime", False, str(exc))


def _check_xformers() -> _CheckResult:
    import importlib.util
    if importlib.util.find_spec("xformers") is None:
        return ("xformers runtime", None, "not installed")
    import torch
    if not torch.cuda.is_available():
        return ("xformers runtime", None, "no GPU device")
    try:
        from xformers.ops import memory_efficient_attention  # pylint: disable=import-error
        q = torch.randn(1, 64, 8, 64, dtype=torch.float16, device="cuda")
        k = torch.randn(1, 64, 8, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(1, 64, 8, 64, dtype=torch.float16, device="cuda")
        memory_efficient_attention(q, k, v)
        return ("xformers runtime", True, f"xformers {_pkg_version('xformers')} kernel executed")
    except Exception as exc:
        return ("xformers runtime", False, str(exc))


def _section_package_versions(console: Console):
    packages = [
        "comfyui-frontend-package",
        "comfyui_kitchen",
        "comfyui-workflow-templates",
        "comfy-aimdo",
        "torch",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
        "opencv-python",
        "opencv-python-headless",
        "numpy",
        "triton",
        "transformers",
        "diffusers",
        "sageattention",
        "xformers",
        "uv",
        "setuptools",
    ]

    table = Table(show_edge=False, pad_edge=False, box=None)
    table.add_column("Package", no_wrap=True)
    table.add_column("Version")

    table.add_row("Python", sys.version.split()[0])
    for pkg in packages:
        table.add_row(pkg, _pkg_version(pkg))

    console.print(table)


def _build_suffix(version_str: str) -> str:
    """Extract the build suffix from a version like '2.9.1+cu130' -> '+cu130'."""
    if "+" in version_str:
        return version_str.split("+", 1)[1]
    return ""


def _check_torch_requirement(pkg: str, torch_version: str) -> str | None:
    """Check if *pkg*'s metadata requires a different torch version. Returns warning or None."""
    from packaging.requirements import Requirement
    from packaging.version import Version

    reqs = importlib.metadata.requires(pkg)
    if reqs is None:
        return None

    torch_base = torch_version.split("+")[0]
    for req_str in reqs:
        req = Requirement(req_str)
        if req.name == "torch" and not req.specifier.contains(Version(torch_base)):
            return f"{pkg} requires {req}, but torch {torch_base} is installed"
    return None


def _section_torch_alignment(console: Console):
    import torch

    torch_ver = torch.__version__
    torch_suffix = _build_suffix(torch_ver)

    table = Table(show_edge=False, pad_edge=False, box=None)
    table.add_column("Package", no_wrap=True)
    table.add_column("Version", no_wrap=True)
    table.add_column("Build", no_wrap=True)

    table.add_row("torch", torch_ver, torch_suffix or "(cpu/default)")

    for companion in ("torchvision", "torchaudio"):
        ver = _pkg_version(companion)
        if ver == "(not installed)":
            table.add_row(companion, ver, "")
            continue
        suffix = _build_suffix(ver)
        table.add_row(companion, ver, suffix or "(cpu/default)")

    console.print(table)
    console.print()

    details = Table(show_edge=False, pad_edge=False, box=None)
    details.add_column("Property", no_wrap=True)
    details.add_column("Value")
    details.add_row("torch.version.cuda", str(torch.version.cuda or "n/a"))
    details.add_row("torch.version.hip", str(getattr(torch.version, "hip", None) or "n/a"))
    cudnn = "n/a"
    if torch.backends.cudnn.is_available():
        cudnn = str(torch.backends.cudnn.version())
    details.add_row("cuDNN version", cudnn)
    console.print(details)


def _status_label(result: bool | None) -> str:
    if result is True:
        return _PASS
    if result is False:
        return _FAIL
    return _SKIP


def _section_compatibility_checks(console: Console):
    checks = _run_compatibility_checks()
    table = Table(show_edge=False, pad_edge=False, box=None)
    table.add_column("Status", no_wrap=True)
    table.add_column("Check", no_wrap=True)
    table.add_column("Detail")
    for name, result, detail in checks:
        table.add_row(_status_label(result), name, detail)
    console.print(table)


def _section_device(console: Console):
    from .. import model_management

    device = model_management.get_torch_device()
    name = model_management.get_torch_device_name(device)

    table = Table(show_edge=False, pad_edge=False, box=None)
    table.add_column("Property", no_wrap=True)
    table.add_column("Value")

    table.add_row("Device", str(device))
    table.add_row("Name", name)

    total_vram = model_management.get_total_memory(device)
    free_vram = model_management.get_free_memory(device)
    total_vram_gb = total_vram / (1024 ** 3)
    free_vram_gb = free_vram / (1024 ** 3)
    used_vram_gb = total_vram_gb - free_vram_gb
    table.add_row("VRAM total", f"{total_vram_gb:.1f} GB")
    table.add_row("VRAM used", f"{used_vram_gb:.1f} GB")
    table.add_row("VRAM free", f"{free_vram_gb:.1f} GB")

    mem = psutil.virtual_memory()
    table.add_row("RAM total", f"{mem.total / (1024 ** 3):.1f} GB")
    table.add_row("RAM used", f"{mem.used / (1024 ** 3):.1f} GB")
    table.add_row("RAM free", f"{mem.available / (1024 ** 3):.1f} GB")

    console.print(table, highlight=False)


def _section_folder_paths(console: Console):
    from . import folder_paths

    fnp = folder_paths._folder_names_and_paths()

    table = Table(show_edge=False, pad_edge=False, box=None)
    table.add_column("Folder", no_wrap=True)
    table.add_column("Paths")

    seen_names: set[str] = set()
    for item in fnp.contents:
        for name in item.folder_names:
            if name in seen_names:
                continue
            seen_names.add(name)
            dirs = [str(p) for p in fnp.directory_paths(name)]
            table.add_row(name, "\n".join(dirs) if dirs else "(none)")

    app_paths = fnp.application_paths
    if app_paths:
        table.add_row("output", str(Path(app_paths.output_directory).resolve()))
        table.add_row("input", str(Path(app_paths.input_directory).resolve()))
        table.add_row("temp", str(Path(app_paths.temp_directory).resolve()))
        table.add_row("user", str(Path(app_paths.user_directory).resolve()))

    table.add_row("base_paths", "\n".join(str(p) for p in fnp.base_paths) if fnp.base_paths else "(none)")

    console.print(table)


def run_integrity_check(config: Configuration):
    console = Console()

    from .. import __version__
    console.rule("ComfyUI Integrity Check")
    console.print(f"  ComfyUI version: {__version__}")
    console.print(f"  Platform: {platform.platform()}")
    console.print()

    console.rule("Config Files")
    _section_config_files(console)
    console.print()

    console.rule("Hardware Detection (guess-settings)")
    _section_guess_settings(console)
    console.print()

    console.rule("Package Versions")
    _section_package_versions(console)
    console.print()

    console.rule("Torch Version Alignment")
    _section_torch_alignment(console)
    console.print()

    console.rule("Compatibility Checks")
    _section_compatibility_checks(console)
    console.print()

    console.rule("Device")
    _section_device(console)
    console.print()

    console.rule("Folder Paths")
    _section_folder_paths(console)
    console.print()
