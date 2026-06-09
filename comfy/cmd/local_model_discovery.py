"""Find local model files and optionally register their parent folders."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .model_classifier import Classification, classify_many
from .model_search import DEFAULT_EXTENSIONS, find_files


@dataclass
class LocalModelDiscovery:
    classifications: list[Classification]
    scan_summary: list[str]
    registrations: dict[tuple[str, str], list[Classification]]


def find_local_model_paths(
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    *,
    scan_timeout: float = 30.0,
    walk_uncovered: bool = True,
    register: bool = False,
) -> LocalModelDiscovery:
    """Search, classify, and optionally register local model folders."""
    scan = find_files(
        extensions=extensions,
        index_timeout=scan_timeout,
        walk_timeout_per_drive=scan_timeout,
        walk_uncovered=walk_uncovered,
    )
    classifications = classify_many(scan.paths)
    registrations = _group_registrations(classifications)

    if register:
        from . import folder_paths

        for kind, directory in registrations:
            folder_paths.add_model_folder_path(kind, directory)

    return LocalModelDiscovery(
        classifications=classifications,
        scan_summary=scan.summary,
        registrations=registrations,
    )


def _group_registrations(
    classifications: list[Classification],
) -> dict[tuple[str, str], list[Classification]]:
    grouped: dict[tuple[str, str], list[Classification]] = defaultdict(list)
    for item in classifications:
        if item.kind is None or item.register_dir is None:
            continue
        grouped[(item.kind, item.register_dir)].append(item)
    return dict(grouped)
