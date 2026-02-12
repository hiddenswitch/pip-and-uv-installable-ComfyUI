"""
Manager model cache for comfyui_manager's model database.

This module provides a fallback mechanism for model lookup and validation
by integrating with comfyui_manager's model-list.json database.

Cache invalidation: Not needed during runtime. The model-list.json is bundled
with comfyui_manager and only changes on package update (requires restart).
"""
from __future__ import annotations

import dataclasses
import functools
import importlib.resources as resources
import json
import logging
from typing import Optional, Dict, FrozenSet, Tuple

from can_ada import can_parse, parse as urlparse  # pylint: disable=no-name-in-module

import requests

from .model_downloader_types import HuggingFile, UrlFile, Downloadable

logger = logging.getLogger(__name__)

GITHUB_MODEL_LIST_URL = "https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/main/model-list.json"

FOLDER_TO_MANAGER_TYPES: Dict[str, list[str]] = {
    "checkpoints": ["checkpoint", "checkpoints", "unclip"],
    "loras": ["lora"],
    "vae": ["vae"],
    "controlnet": ["controlnet", "t2i-adapter", "t2i-style"],
    "clip_vision": ["clip_vision"],
    "text_encoders": ["clip", "text_encoders"],
    "diffusion_models": ["unet", "diffusion_model"],
    "upscale_models": ["upscale"],
    "embeddings": ["embedding", "embeddings"],
    "gligen": ["gligen"],
}


@dataclasses.dataclass(frozen=True)
class ManagerModelEntry:
    """Represents a model from comfyui_manager's model-list.json"""
    name: str
    type: str
    base: str
    save_path: str
    filename: str
    url: str


def parse_huggingface_url(url: str, filename: str) -> Optional[HuggingFile]:
    """Convert HuggingFace HTTPS URL to HuggingFile."""
    if not can_parse(url):
        return None
    parsed = urlparse(url)
    if parsed.hostname != "huggingface.co":
        return None
    parts = [p for p in parsed.pathname.split("/") if p]
    try:
        resolve_idx = parts.index("resolve")
    except ValueError:
        return None
    if resolve_idx < 1 or resolve_idx + 2 > len(parts):
        return None
    repo_id = "/".join(parts[:resolve_idx])
    revision = parts[resolve_idx + 1]
    filepath = "/".join(parts[resolve_idx + 2:])
    return HuggingFile(
        repo_id=repo_id,
        filename=filepath,
        save_with_filename=filename,
        revision=revision if revision != "main" else None,
        show_in_ui=False,
    )


def _resolve_folder(save_path: str, type_name: str) -> str:
    """Resolve the folder name from save_path or type."""
    if save_path and save_path != "default":
        return save_path.split("/")[0]
    for folder, types in FOLDER_TO_MANAGER_TYPES.items():
        if type_name in types:
            return folder
    return type_name


def _fetch_from_github() -> Optional[dict]:
    try:
        response = requests.get(GITHUB_MODEL_LIST_URL, timeout=30)
        response.raise_for_status()
        logger.info("Fetched fresh model list from GitHub")
        return response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch model list from GitHub: {e}")
        return None


def _load_from_package() -> Optional[dict]:
    try:
        traversable = resources.files("comfyui_manager") / "model-list.json"
        return json.loads(traversable.read_text())
    except (ImportError, ModuleNotFoundError):
        logger.debug("comfyui_manager not installed")
        return None
    except FileNotFoundError:
        logger.debug("comfyui_manager model-list.json not found")
        return None
    except Exception as e:
        logger.debug(f"Failed to load manager model list: {e}")
        return None


IndexedModels = Tuple[Dict[str, Dict[str, ManagerModelEntry]], Dict[str, FrozenSet[str]]]


@functools.cache
def _load_and_index_models(refresh_from_github: bool) -> Optional[IndexedModels]:
    """Load and index models from manager. Cached for process lifetime."""
    from .component_model.files import canonicalize_path

    data = None
    if refresh_from_github:
        data = _fetch_from_github()
    if data is None:
        data = _load_from_package()
    if data is None:
        return None

    by_folder: Dict[str, Dict[str, ManagerModelEntry]] = {}
    for m in data.get("models", []):
        entry = ManagerModelEntry(
            name=m.get("name", ""),
            type=m.get("type", ""),
            base=m.get("base", ""),
            save_path=m.get("save_path", ""),
            filename=m.get("filename", ""),
            url=m.get("url", "")
        )
        folder = _resolve_folder(entry.save_path, entry.type)
        if folder not in by_folder:
            by_folder[folder] = {}
        by_folder[folder][canonicalize_path(entry.filename)] = entry

    filenames_by_folder = {folder: frozenset(entries.keys()) for folder, entries in by_folder.items()}

    total = sum(len(e) for e in by_folder.values())
    logger.info(f"Indexed {total} models from manager database across {len(by_folder)} folders")

    return by_folder, filenames_by_folder


# Module state
_enabled: bool = False
_refresh_from_github: bool = False


def init_manager_model_cache(refresh_from_github: bool = False):
    """Initialize the manager model cache. Called once at startup."""
    global _enabled, _refresh_from_github
    from . import manager_integration
    from .cli_args import args

    if args.disable_manager_model_fallback:
        logger.debug("Manager model fallback disabled via CLI flag")
        return

    if not manager_integration.is_available():
        return

    _enabled = True
    _refresh_from_github = refresh_from_github
    # Eagerly load to log count at startup
    _load_and_index_models(refresh_from_github)


def get_filenames_for_folder(folder_name: str) -> FrozenSet[str]:
    """Get all known filenames for a folder."""
    if not _enabled:
        return frozenset()
    indexed = _load_and_index_models(_refresh_from_github)
    if indexed is None:
        return frozenset()
    return indexed[1].get(folder_name, frozenset())


def get_model_entry(folder_name: str, filename: str) -> Optional[ManagerModelEntry]:
    """Get a model entry by folder and filename."""
    if not _enabled:
        return None
    from .component_model.files import canonicalize_path
    indexed = _load_and_index_models(_refresh_from_github)
    if indexed is None:
        return None
    folder_entries = indexed[0].get(folder_name, {})
    return folder_entries.get(canonicalize_path(filename))


def entry_to_downloadable(entry: ManagerModelEntry) -> Optional[Downloadable]:
    """Convert a manager model entry to a Downloadable object."""
    hf = parse_huggingface_url(entry.url, entry.filename)
    if hf:
        return hf
    if entry.url:
        return UrlFile(entry.url, _save_with_filename=entry.filename, show_in_ui=False)
    return None
