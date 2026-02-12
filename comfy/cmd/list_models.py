from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    folder: str
    filename: str
    uri: str
    description: Optional[str] = None
    exists: Optional[bool] = None


def _models_from_known() -> list[ModelInfo]:
    from ..model_downloader import _known_models_db
    from ..model_downloader_types import HuggingFile, CivitFile, UrlFile

    models = []
    for db in _known_models_db:
        folders = db.folder_names if hasattr(db, "folder_names") else [db.folder_name]
        folder = folders[0]
        for item in db.data:
            if isinstance(item, HuggingFile):
                models.append(ModelInfo(
                    folder=folder,
                    filename=item.save_with_filename or item.filename,
                    uri=f"hf://{item.repo_id}/{item.filename}",
                ))
            elif isinstance(item, CivitFile):
                models.append(ModelInfo(
                    folder=folder,
                    filename=item.filename,
                    uri=f"https://civitai.com/models/{item.model_id}?modelVersionId={item.model_version_id}",
                ))
            elif isinstance(item, UrlFile):
                models.append(ModelInfo(
                    folder=folder,
                    filename=item.save_with_filename,
                    uri=item.url,
                ))
    return models


def _models_from_manager() -> list[ModelInfo]:
    from ..manager_model_cache import _load_from_package, _resolve_folder

    data = _load_from_package()
    if data is None:
        return []

    models = []
    for m in data.get("models", []):
        folder = _resolve_folder(m.get("save_path", ""), m.get("type", ""))
        models.append(ModelInfo(
            folder=folder,
            filename=m.get("filename", ""),
            uri=m.get("url", ""),
            description=m.get("description"),
        ))
    return models


def _check_exists(folder: str, filename: str, uri: str) -> bool:
    from . import folder_paths

    path = folder_paths.get_full_path(folder, filename)
    if path is not None:
        return True

    if uri.startswith("hf://"):
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import LocalEntryNotFoundError
            from ..model_downloader import _get_hf_token

            # parse repo_id and hf_filename from hf://repo_id/filename
            stripped = uri[len("hf://"):]
            parts = stripped.split("/", 2)
            if len(parts) >= 3:
                repo_id = f"{parts[0]}/{parts[1]}"
                hf_filename = parts[2]
            else:
                return False

            hf_hub_download(
                repo_id,
                hf_filename,
                local_files_only=True,
                token=_get_hf_token(),
            )
            return True
        except Exception:
            return False

    return False


def get_all_models(include_manager: bool = True) -> list[ModelInfo]:
    known = _models_from_known()
    known_filenames = frozenset(m.filename for m in known)
    all_models = list(known)
    if include_manager:
        for m in _models_from_manager():
            if m.filename not in known_filenames:
                all_models.append(m)
    return all_models


def list_models(
    format: str = "table",
    folder: Optional[str] = None,
    include_manager: bool = True,
    check_exists: bool = False,
):
    all_models = get_all_models(include_manager=include_manager)
    if folder:
        all_models = [m for m in all_models if m.folder == folder]

    if check_exists:
        for m in all_models:
            m.exists = _check_exists(m.folder, m.filename, m.uri)

    if format == "json":
        print(json.dumps([asdict(m) for m in all_models], indent=2))
    else:
        _print_table(all_models, show_exists=check_exists)


def _print_table(models: list[ModelInfo], show_exists: bool = False):
    if not models:
        print("No models found.")
        return

    console = Console()
    table = Table(show_edge=False, pad_edge=False, box=None, width=max(console.width, 200))
    table.add_column("Folder", no_wrap=True)
    table.add_column("Filename", no_wrap=True)
    table.add_column("URI", no_wrap=True, overflow="ellipsis", ratio=1)
    table.add_column("Description", no_wrap=True, overflow="ellipsis")
    if show_exists:
        table.add_column("Exists", no_wrap=True)
    for m in models:
        row = [
            m.folder,
            m.filename,
            m.uri,
            m.description or "",
        ]
        if show_exists:
            row.append("yes" if m.exists else "")
        table.add_row(*row)
    console.print(table, soft_wrap=True)
