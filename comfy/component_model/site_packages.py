"""Utilities for adding extra package directories to the Python path.

Custom node dependencies are installed into a ``node_site/`` directory via
``pip install --target``.  Unlike virtualenv's ``site-packages``, these
directories are not automatically on ``sys.path``.  The helpers here ensure
they are importable in the current process **and** in child processes
spawned via :class:`~comfy.distributed.process_pool_executor.ProcessPoolExecutor`
(which uses the ``spawn`` multiprocessing context and inherits ``PYTHONPATH``
but not in-process ``sys.path`` modifications).
"""

from __future__ import annotations

import os
import sys


def add_site_dir(path: str | os.PathLike[str]) -> None:
    """Add *path* to ``sys.path`` and ``PYTHONPATH`` if it is a directory.

    This is idempotent — calling it multiple times with the same path is safe.
    """
    site_dir = str(path)
    if not os.path.isdir(site_dir):
        return
    if site_dir not in sys.path:
        sys.path.insert(0, site_dir)
        # A second copy of fsspec in site_dir may shadow the venv copy,
        # losing our pkg:// filesystem registration.  Re-register it.
        from .package_filesystem import ensure_registered
        ensure_registered()
    pypath = os.environ.get("PYTHONPATH", "")
    entries = [p for p in pypath.split(os.pathsep) if p]
    if site_dir not in entries:
        entries.insert(0, site_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(entries)


def add_node_site(base_directory: str | os.PathLike[str]) -> None:
    """Add ``<base_directory>/node_site`` to the Python path."""
    add_site_dir(os.path.join(str(base_directory), "node_site"))
