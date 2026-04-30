"""Print system diagnostics and verify installation integrity."""
from __future__ import annotations

import importlib.metadata
import os
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

    # macOS-specific checks
    checks.append(_check_fp8_mps())

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
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
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
        from xformers.ops import memory_efficient_attention
        q = torch.randn(1, 64, 8, 64, dtype=torch.float16, device="cuda")
        k = torch.randn(1, 64, 8, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(1, 64, 8, 64, dtype=torch.float16, device="cuda")
        memory_efficient_attention(q, k, v)
        return ("xformers runtime", True, f"xformers {_pkg_version('xformers')} kernel executed")
    except Exception as exc:
        return ("xformers runtime", False, str(exc))


def _check_fp8_mps() -> _CheckResult:
    if sys.platform != "darwin":
        return ("fp4-fp8-for-torch-mps", None, "not macOS")
    if _pkg_version("fp4-fp8-for-torch-mps") == "(not installed)":
        return ("fp4-fp8-for-torch-mps", False,
                "not installed — required for FP8 models on Apple Silicon. "
                "Install with: uv pip install fp4-fp8-for-torch-mps")
    return ("fp4-fp8-for-torch-mps", True, f"fp4-fp8-for-torch-mps {_pkg_version('fp4-fp8-for-torch-mps')}")


def _section_package_versions(console: Console):
    packages = [
        "comfyui-frontend-package",
        "comfy-kitchen",
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


def _parse_arch_entry(entry: str) -> tuple[str, int, int] | None:
    """Parse a torch arch-list entry into (kind, major, minor).

    CUDA entries look like ``sm_75``, ``sm_90a`` (Hopper architectural variants),
    or ``compute_90`` (PTX). Returns ``None`` for HIP entries (``gfx906`` etc.)
    or anything unrecognised.
    """
    for prefix, kind in (("sm_", "sm"), ("compute_", "compute")):
        if not entry.startswith(prefix):
            continue
        tail = entry[len(prefix):]
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            return None
        if len(digits) >= 2:
            return kind, int(digits[:-1]), int(digits[-1])
        return kind, int(digits), 0
    return None


def _norm_gfx(entry: str) -> str:
    """Strip ROCm feature suffixes: ``gfx90a:sramecc+:xnack-`` -> ``gfx90a``."""
    return entry.split(":", 1)[0] if entry else ""


def _cuda_device_supported(major: int, minor: int, arch_list: list[str]) -> bool:
    """Apply NVIDIA's intra-major forward-compat rule plus PTX JIT compat.

    A cubin compiled for compute capability ``M.k`` runs on devices with the
    same major ``M`` and minor ``n >= k`` (per the CUDA Programming Guide's
    binary-compatibility section). PTX (``compute_X.Y``) JIT-compiles to any
    device with capability ``>= (X, Y)``.
    """
    cc = (major, minor)
    for entry in arch_list:
        parsed = _parse_arch_entry(entry)
        if parsed is None:
            continue
        kind, e_major, e_minor = parsed
        if kind == "sm" and e_major == major and e_minor <= minor:
            return True
        if kind == "compute" and (e_major, e_minor) <= cc:
            return True
    return False


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
        try:
            cudnn = str(torch.backends.cudnn.version())
        except RuntimeError as exc:
            cudnn = f"unavailable ({exc.__class__.__name__}: {exc})"
    details.add_row("cuDNN version", cudnn)
    console.print(details)

    is_hip = getattr(torch.version, "hip", None) is not None
    try:
        arch_list = list(torch.cuda.get_arch_list())
    except Exception:
        arch_list = []
    try:
        device_count = torch.cuda.device_count()
    except Exception:
        device_count = 0

    if not arch_list and device_count == 0:
        return

    console.print()
    align = Table(show_edge=False, pad_edge=False, box=None)
    align.add_column("Status", no_wrap=True)
    align.add_column("Device", no_wrap=True)
    align.add_column("Arch", no_wrap=True)
    align.add_column("Detail")

    arches_str = " ".join(arch_list) if arch_list else "(none)"

    if device_count == 0:
        align.add_row(_SKIP, "(no accelerator)", "-", f"compiled for: {arches_str}")
    else:
        for i in range(device_count):
            try:
                props = torch.cuda.get_device_properties(i)
            except Exception as exc:
                align.add_row(_FAIL, f"device {i}", "?", f"could not read properties: {exc}")
                continue
            name = getattr(props, "name", f"device {i}")
            if is_hip:
                raw = getattr(props, "gcnArchName", "") or ""
                device_arch = _norm_gfx(raw)
                hip_arches = {_norm_gfx(a) for a in arch_list if a}
                compatible = bool(device_arch) and device_arch in hip_arches
                detail = (f"compiled for: {arches_str}"
                          if compatible
                          else f"torch was not built for {device_arch}; compiled for: {arches_str}")
            else:
                major = int(getattr(props, "major", 0))
                minor = int(getattr(props, "minor", 0))
                device_arch = f"sm_{major}{minor}"
                compatible = _cuda_device_supported(major, minor, arch_list)
                if compatible:
                    detail = f"compiled for: {arches_str}"
                else:
                    detail = (f"torch was not built for {device_arch} "
                              f"(no SASS in same major and no compatible PTX); "
                              f"compiled for: {arches_str}")
            align.add_row(_PASS if compatible else _FAIL, name, device_arch or "?", detail)

    console.print(align)


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


def _format_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _summarize_directory(path: str) -> dict:
    """Return ``{files, size, symlinks, hardlinks}`` for one directory.

    Counts only regular files at the top level (not recursed) since a model
    folder is typically flat. ``symlinks`` and ``hardlinks`` are counted
    independently; a regular file with ``st_nlink == 1`` is neither.
    """
    summary = {"files": 0, "size": 0, "symlinks": 0, "hardlinks": 0}
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    lst = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if entry.is_symlink():
                    summary["symlinks"] += 1
                    summary["files"] += 1
                    try:
                        summary["size"] += entry.stat(follow_symlinks=True).st_size
                    except OSError:
                        pass
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                summary["files"] += 1
                summary["size"] += lst.st_size
                if lst.st_nlink > 1:
                    summary["hardlinks"] += 1
    except OSError:
        pass
    return summary


def _check_symlink_support(path: str) -> tuple[bool, str]:
    """Try to create then remove a symlink under *path*. Returns ``(ok, detail)``.

    Windows non-admin processes can't create symlinks unless Developer Mode is
    on, and Synology-style overlays sometimes return EACCES even when the
    parent fs supports them — we want the *operational* answer for this exact
    directory, not what the filesystem theoretically supports.
    """
    import tempfile
    if not os.path.isdir(path):
        return False, "directory missing"
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".symlink_probe_", delete=False) as f:
            target = f.name
        link = target + ".lnk"
        try:
            os.symlink(target, link)
        except OSError as exc:
            return False, type(exc).__name__
        finally:
            for p in (link, target):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return True, "ok"
    except OSError as exc:
        return False, type(exc).__name__


def _section_folder_paths(console: Console):
    from . import folder_paths

    fnp = folder_paths._folder_names_and_paths()

    table = Table(show_edge=False, pad_edge=False, box=None)
    table.add_column("Folder", no_wrap=True)
    table.add_column("Paths")
    table.add_column("Files", justify="right", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Links", no_wrap=True)
    table.add_column("symlink ok", no_wrap=True)

    def _row(label: str, dirs: list[str]) -> None:
        if not dirs:
            table.add_row(label, "(none)", "-", "-", "-", "-")
            return
        total = {"files": 0, "size": 0, "symlinks": 0, "hardlinks": 0}
        sym_results: list[str] = []
        for d in dirs:
            s = _summarize_directory(d)
            for k in total:
                total[k] += s[k]
            ok, detail = _check_symlink_support(d)
            sym_results.append("ok" if ok else f"no ({detail})")
        link_str = f"{total['symlinks']} sym / {total['hardlinks']} hard"
        sym_str = "\n".join(sym_results)
        table.add_row(label, "\n".join(dirs), str(total["files"]),
                      _format_size(total["size"]), link_str, sym_str)

    seen_names: set[str] = set()
    for item in fnp.contents:
        for name in item.folder_names:
            if name in seen_names:
                continue
            seen_names.add(name)
            dirs = [str(p) for p in fnp.directory_paths(name)]
            _row(name, dirs)

    app_paths = fnp.application_paths
    if app_paths:
        for label, attr in (("output", "output_directory"), ("input", "input_directory"),
                            ("temp", "temp_directory"), ("user", "user_directory")):
            _row(label, [str(Path(getattr(app_paths, attr)).resolve())])

    _row("base_paths", [str(p) for p in (fnp.base_paths or [])])

    console.print(table)


def _section_comfy_kitchen_capabilities(console: Console):
    """PASS/FAIL matrix of comfy_kitchen ops × backends × representative dtypes.

    Each cell actually invokes the op on a tiny tensor through the registry's
    capability check, so a PASS reflects what would actually dispatch on this
    machine. FAIL/SKIP reflects either constraint mismatch or a backend that's
    been disabled (e.g. by --disable-comfy-kitchen-backends or guess-settings).
    """
    try:
        import torch
        import comfy_kitchen as ck
        from .. import model_management
    except ImportError:
        console.print("  comfy_kitchen not installed.")
        return

    backends = sorted(ck.list_backends().keys())
    if not backends:
        console.print("  No comfy_kitchen backends registered.")
        return

    device = model_management.get_torch_device()

    table = Table(show_edge=False, pad_edge=False, box=None,
                  title=f"comfy_kitchen op × backend × dtype matrix (device={device})")
    table.add_column("Operation", no_wrap=True)
    table.add_column("Dtype", no_wrap=True)
    for backend in backends:
        table.add_column(backend, no_wrap=True)

    # torch.device() is a context manager since 2.0 — every new tensor inside
    # the block defaults to this device, so kwargs factories can stay terse.
    with torch.device(device):
        for op_name, dtype_label, kwargs_factory in _comfy_kitchen_capability_cases():
            row = [op_name, dtype_label]
            try:
                kwargs = kwargs_factory()
            except Exception as exc:
                for _ in backends:
                    row.append(f"SKIP ({type(exc).__name__})")
                table.add_row(*row)
                continue
            for backend in backends:
                row.append(_check_backend_capability(ck, backend, op_name, kwargs))
            table.add_row(*row)
    console.print(table)


def _comfy_kitchen_capability_cases() -> list[tuple[str, str, "callable"]]:
    """Return ``(op_name, dtype_label, kwargs_factory)`` test cases.

    The factories rely on the caller wrapping in ``with torch.device(...)``
    so tensors land on the appropriate accelerator without per-call plumbing.
    """
    import torch

    fp8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    fp8_e5m2 = getattr(torch, "float8_e5m2", None)

    cases: list[tuple[str, str, "callable"]] = [
        ("quantize_per_tensor_fp8", "bf16→fp8_e4m3fn", lambda: dict(
            x=torch.zeros(8, 8, dtype=torch.bfloat16),
            scale=torch.ones(1, dtype=torch.float32),
            output_type=fp8_e4m3fn,
        )) if fp8_e4m3fn is not None else None,
        ("quantize_per_tensor_fp8", "bf16→fp8_e5m2", lambda: dict(
            x=torch.zeros(8, 8, dtype=torch.bfloat16),
            scale=torch.ones(1, dtype=torch.float32),
            output_type=fp8_e5m2,
        )) if fp8_e5m2 is not None else None,
        ("quantize_per_tensor_fp8", "fp16→fp8_e4m3fn", lambda: dict(
            x=torch.zeros(8, 8, dtype=torch.float16),
            scale=torch.ones(1, dtype=torch.float32),
            output_type=fp8_e4m3fn,
        )) if fp8_e4m3fn is not None else None,
        ("dequantize_per_tensor_fp8", "fp8_e4m3fn→bf16", lambda: dict(
            x=torch.zeros(8, 8, dtype=fp8_e4m3fn),
            scale=torch.ones(1, dtype=torch.float32),
            output_type=torch.bfloat16,
        )) if fp8_e4m3fn is not None else None,
        ("dequantize_per_tensor_fp8", "fp8_e5m2→fp16", lambda: dict(
            x=torch.zeros(8, 8, dtype=fp8_e5m2),
            scale=torch.ones(1, dtype=torch.float32),
            output_type=torch.float16,
        )) if fp8_e5m2 is not None else None,
        ("quantize_nvfp4", "bf16→nvfp4", lambda: dict(
            x=torch.zeros(32, 32, dtype=torch.bfloat16),
            per_tensor_scale=torch.ones(1, dtype=torch.float32),
        )),
        ("quantize_mxfp8", "bf16→mxfp8", lambda: dict(
            x=torch.zeros(32, 32, dtype=torch.bfloat16),
        )),
    ]
    return [c for c in cases if c is not None]


def _check_backend_capability(ck, backend: str, op_name: str, kwargs: dict) -> str:
    """Return a short PASS/FAIL/SKIP cell for a single matrix entry.

    Performs both the constraint check *and* an actual invocation through
    ``registry.use_backend``. The execution probe is what catches things like
    triton's fp8e4nv kernel failing to compile on Ampere — constraints alone
    don't surface JIT compile errors.
    """
    info = ck.list_backends().get(backend, {})
    if info.get("disabled"):
        return "DISABLED"
    if not info.get("available"):
        reason = info.get("unavailable_reason")
        if reason:
            return f"N/A ({reason.split(':')[0][:20]})"
        return "N/A"
    if op_name not in (info.get("capabilities") or ()):
        return "N/A"
    result = ck.registry.validate_backend_for_call(backend, op_name, kwargs)
    if not result.success:
        return f"FAIL ({result.failed_param}: {str(result.failure_reason)[:20]})"

    op_fn = getattr(ck, op_name, None)
    if op_fn is None:
        return "PASS (no-call)"
    try:
        with ck.registry.use_backend(backend):
            op_fn(**kwargs)
    except Exception as exc:
        msg = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return f"FAIL ({msg[:32]})"
    return "PASS"


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

    console.rule("comfy_kitchen Backend Capabilities")
    _section_comfy_kitchen_capabilities(console)
    console.print()
