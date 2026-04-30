"""OS-indexed search for model files.

Dispatches to the platform's native file-search index:
  - macOS:   ``mdfind`` (Spotlight; same engine as Finder)
  - Windows: ``Search.CollatorDSO`` via pywin32 (Windows Search)
  - Linux:   ``plocate`` / ``mlocate`` / ``locate`` (locate database)

There is no Python binding for the locate database on Linux, and ``mdfind``
on macOS is the same engine as ``NSMetadataQuery`` with a fraction of the
boilerplate, so the macOS and Linux paths shell out. Windows uses the
ADO/COM provider through pywin32 because there is no command-line equivalent
of the Windows Search SQL provider.

On Windows, fixed drives are walked with a per-drive timeout in addition to
the index query — secondary drives are usually outside the indexer's scope
and the walk is what catches them. The walk is depth-bounded and skips the
usual system noise (``$RECYCLE.BIN``, ``System Volume Information``, etc.).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS: tuple[str, ...] = (
    "safetensors", "ckpt", "pt", "gguf", "onnx",
)
# .bin is omitted from the default — it's used by enough non-ML toolchains
# (cargo, gradle, browsers, Unity, etc.) that scanning every .bin file on
# the machine produces megabytes of noise. Pass extensions=("bin",) to
# include it explicitly when looking for a specific .bin model.

WINDOWS_SKIP_DIRS: frozenset[str] = frozenset({
    "$recycle.bin", "system volume information", "windows",
    "program files", "program files (x86)", "programdata",
    "node_modules", ".git", "appdata", "msocache", "perflogs",
})

POSIX_SKIP_DIRS: frozenset[str] = frozenset({
    "node_modules", ".git", ".cache", "snap", "proc", "sys", "dev",
    ".cargo", ".gradle", ".npm", ".pyenv", ".rustup", ".nvm",
    ".local", ".m2", ".ivy2", ".sbt", "venv", ".venv", "__pycache__",
})

DEFAULT_WALK_MAX_DEPTH: int = 6


class ScanResult:
    """Search results plus per-backend timing/coverage metadata.

    Returned by :func:`find_files`. Callers that just want the paths can
    iterate the instance; ``.summary`` exposes what each backend did so
    the dry-run output can show "Spotlight: 1247 files in 0.4s".
    """

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.summary: list[str] = []  # human-readable lines

    def __iter__(self):
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)


def find_files(
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    *,
    index_timeout: float = 30.0,
    walk_timeout_per_drive: float = 30.0,
    walk_uncovered: bool = True,
) -> ScanResult:
    """Query the OS file index (and fall-back walk on Windows) for model files.

    :param extensions: bare extensions without leading dot (``"safetensors"``).
    :param index_timeout: seconds to wait on the indexed query.
    :param walk_timeout_per_drive: seconds to spend walking each Windows fixed
        drive (only applied when ``walk_uncovered`` is true and we're on Windows).
    :param walk_uncovered: walk Windows fixed drives in addition to querying
        the index. The drives are usually outside the index scope; without
        this, secondary drives won't appear.
    :returns: a :class:`ScanResult` whose ``.paths`` is a deduplicated,
        sorted list of absolute paths.
    """
    exts = tuple(e.lstrip(".").lower() for e in extensions)
    result = ScanResult()
    paths: set[str] = set()

    t0 = time.monotonic()
    if sys.platform == "darwin":
        found = _spotlight(exts, timeout=index_timeout)
        paths.update(found)
        result.summary.append(
            f"Spotlight (mdfind): {len(found)} files in {time.monotonic() - t0:.1f}s"
        )
    elif sys.platform == "win32":
        found = _windows_search(exts, timeout=index_timeout)
        paths.update(found)
        result.summary.append(
            f"Windows Search: {len(found)} files in {time.monotonic() - t0:.1f}s"
        )
        if walk_uncovered:
            walk_paths, walk_summary = _walk_windows_drives(
                exts, per_drive_timeout=walk_timeout_per_drive,
            )
            paths.update(walk_paths)
            result.summary.extend(walk_summary)
    else:
        found = _locate(exts, timeout=index_timeout)
        paths.update(found)
        backend = found.backend if hasattr(found, "backend") else "locate"
        result.summary.append(
            f"{backend}: {len(found)} files in {time.monotonic() - t0:.1f}s"
        )
        if not found and walk_uncovered:
            walk_paths, walk_summary = _walk_posix_roots(
                exts, per_root_timeout=walk_timeout_per_drive,
            )
            paths.update(walk_paths)
            result.summary.extend(walk_summary)

    result.paths = sorted(paths)
    return result


# ---------------------------------------------------------------------------
# macOS — Spotlight
# ---------------------------------------------------------------------------

def _spotlight(extensions: tuple[str, ...], *, timeout: float) -> list[str]:
    if not shutil.which("mdfind"):
        return []
    clauses = " || ".join(
        f'kMDItemFSName == "*.{ext}"cd' for ext in extensions
    )
    proc = subprocess.run(
        ["mdfind", clauses],
        capture_output=True, text=True, timeout=timeout,
    )
    return [line for line in proc.stdout.splitlines() if line]


# ---------------------------------------------------------------------------
# Linux — plocate / mlocate / locate
# ---------------------------------------------------------------------------

class _LocateResult(list):
    backend: str = "locate"


def _locate(extensions: tuple[str, ...], *, timeout: float) -> _LocateResult:
    binary = (
        shutil.which("plocate")
        or shutil.which("mlocate")
        or shutil.which("locate")
    )
    if binary is None:
        return _LocateResult()
    pattern = r"\.(" + "|".join(extensions) + r")$"
    proc = subprocess.run(
        [binary, "-i", "--regex", pattern],
        capture_output=True, text=True, timeout=timeout,
    )
    res = _LocateResult(line for line in proc.stdout.splitlines() if line)
    res.backend = Path(binary).name
    return res


# ---------------------------------------------------------------------------
# Windows — Search.CollatorDSO via pywin32, plus walk for uncovered drives
# ---------------------------------------------------------------------------

def _windows_search(extensions: tuple[str, ...], *, timeout: float) -> list[str]:
    try:
        import win32com.client  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "pywin32 is not installed; Windows Search query is unavailable. "
            "Install pywin32 for indexed search; secondary drives still scanned via walk."
        )
        return []
    ext_list = ",".join(f"'.{e}'" for e in extensions)
    query = (
        "SELECT System.ItemPathDisplay FROM SYSTEMINDEX "
        f"WHERE System.FileExtension IN ({ext_list})"
    )
    conn = win32com.client.Dispatch("ADODB.Connection")
    conn.CommandTimeout = int(timeout)
    conn.Open(
        "Provider=Search.CollatorDSO;"
        "Extended Properties='Application=Windows';"
    )
    rs = conn.Execute(query)
    out: list[str] = []
    while not rs.EOF:
        out.append(rs.Fields.Item(0).Value)
        rs.MoveNext()
    conn.Close()
    return out


def _windows_fixed_drives() -> list[str]:
    drives: list[str] = []
    if sys.platform != "win32":
        return drives
    import string
    import ctypes  # noqa: PLR1714
    GetDriveTypeW = ctypes.windll.kernel32.GetDriveTypeW
    DRIVE_FIXED = 3
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if GetDriveTypeW(root) == DRIVE_FIXED:
            drives.append(root)
    return drives


def _posix_walk_roots() -> list[str]:
    home = os.path.expanduser("~")
    candidates = [
        home,
        "/opt", "/srv", "/mnt", f"/media/{os.environ.get('USER', '')}",
        "/Volumes",  # macOS-mounted volumes when locate-only fallback is hit
    ]
    return [r for r in candidates if r and os.path.isdir(r)]


def _walk_posix_roots(
    extensions: tuple[str, ...], *, per_root_timeout: float,
) -> tuple[list[str], list[str]]:
    out: list[str] = []
    summary: list[str] = []
    ext_set = {f".{e}" for e in extensions}
    for root in _posix_walk_roots():
        t0 = time.monotonic()
        deadline = t0 + per_root_timeout
        count_before = len(out)
        timed_out = _walk(root, ext_set, POSIX_SKIP_DIRS, deadline, out)
        elapsed = time.monotonic() - t0
        suffix = " (timeout)" if timed_out else ""
        summary.append(
            f"walked {root}: {len(out) - count_before} matches in {elapsed:.1f}s{suffix}"
        )
    return out, summary


def _walk_windows_drives(
    extensions: tuple[str, ...], *, per_drive_timeout: float,
) -> tuple[list[str], list[str]]:
    out: list[str] = []
    summary: list[str] = []
    ext_set = {f".{e}" for e in extensions}
    for drive in _windows_fixed_drives():
        t0 = time.monotonic()
        deadline = t0 + per_drive_timeout
        count_before = len(out)
        timed_out = _walk(drive, ext_set, WINDOWS_SKIP_DIRS, deadline, out)
        elapsed = time.monotonic() - t0
        suffix = " (timeout)" if timed_out else ""
        summary.append(
            f"walked {drive}: {len(out) - count_before} matches in {elapsed:.1f}s{suffix}"
        )
    return out, summary


def _walk(
    root: str,
    ext_set: set[str],
    skip_dir_names: frozenset[str],
    deadline: float,
    out: list[str],
    max_depth: int = DEFAULT_WALK_MAX_DEPTH,
) -> bool:
    """Iterative scandir walk. Returns True if the walk hit the deadline."""
    root_depth = root.rstrip(os.sep).count(os.sep)
    stack: list[str] = [root]
    while stack:
        if time.monotonic() >= deadline:
            return True
        path = stack.pop()
        if path.count(os.sep) - root_depth > max_depth:
            continue
        try:
            it = os.scandir(path)
        except PermissionError:
            continue
        with it:
            for entry in it:
                if time.monotonic() >= deadline:
                    return True
                name_lower = entry.name.lower()
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if is_dir:
                    if name_lower in skip_dir_names:
                        continue
                    stack.append(entry.path)
                else:
                    ext = os.path.splitext(name_lower)[1]
                    if ext in ext_set:
                        out.append(entry.path)
    return False
