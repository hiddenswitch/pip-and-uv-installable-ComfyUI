"""Civitai model lookup, mirroring manager_model_cache's API.

Civitai's search index doesn't full-text-match file names, so live
filename->model resolution is unreliable. Instead we prefetch the file lists
of the top-N most-downloaded models per type at startup (the same pattern
``comfyui_manager``'s ``model-list.json`` follows, just sourced from Civitai
rather than a hand-curated GitHub file).

Auth via ``CIVITAI_API_TOKEN`` (or ``CIVITAI_API_KEY``). The cache is
process-local; on disk it lives at ``~/.cache/comfyui/civitai_model_index.json``
to avoid the multi-second prefetch on every startup.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import requests

from .model_downloader_types import Downloadable, UrlFile

logger = logging.getLogger(__name__)

# Civitai model `type` -> ComfyUI folder name.
_CIVITAI_TYPE_TO_FOLDER: dict[str, str] = {
    "Checkpoint": "checkpoints",
    "TextualInversion": "embeddings",
    "Hypernetwork": "hypernetworks",
    "AestheticGradient": "embeddings",
    "LORA": "loras",
    "LoCon": "loras",
    "DoRA": "loras",
    "Controlnet": "controlnet",
    "Upscaler": "upscale_models",
    "MotionModule": "animatediff_models",
    "VAE": "vae",
    "TextEncoder": "text_encoders",
    "UNet": "diffusion_models",
    "CLIPVision": "clip_vision",
    "Poses": "poses",
    "Wildcards": "wildcards",
    "Detection": "detection",
}


def _api_token() -> Optional[str]:
    return os.environ.get("CIVITAI_API_TOKEN") or os.environ.get("CIVITAI_API_KEY")


def _auth_headers() -> dict[str, str]:
    token = _api_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


_DEFAULT_TYPES_TO_PREFETCH: tuple[str, ...] = (
    "Checkpoint", "LORA", "LoCon", "DoRA", "VAE", "UNet",
    "Controlnet", "Upscaler", "TextEncoder", "CLIPVision",
    "MotionModule", "TextualInversion",
)
_DEFAULT_PER_TYPE_LIMIT = 100


def _cache_path() -> str:
    return os.path.expanduser("~/.cache/comfyui/civitai_model_index.json")


def _fetch_top_models_for_type(model_type: str, *, limit: int = 100) -> list[dict]:
    out: list[dict] = []
    cursor: Optional[str] = None
    while len(out) < limit:
        params: dict[str, object] = {
            "types": model_type,
            "sort": "Most Downloaded",
            "period": "AllTime",
            "limit": min(100, limit - len(out)),
        }
        if cursor:
            params["cursor"] = cursor
        try:
            r = requests.get(
                "https://civitai.com/api/v1/models",
                params=params,
                headers=_auth_headers(),
                timeout=30,
            )
            r.raise_for_status()
        except Exception as exc:
            logger.debug("Civitai prefetch type=%s failed: %s", model_type, exc)
            return out
        data = r.json()
        out.extend(data.get("items") or [])
        cursor = (data.get("metadata") or {}).get("nextCursor")
        if not cursor:
            break
    return out[:limit]


def _build_index_from_api() -> dict[str, dict]:
    """Return ``{canonical_filename: {folder, url, model_id, version_id}}``."""
    from .component_model.files import canonicalize_path
    index: dict[str, dict] = {}
    for model_type in _DEFAULT_TYPES_TO_PREFETCH:
        folder = _CIVITAI_TYPE_TO_FOLDER.get(model_type)
        if folder is None:
            continue
        for item in _fetch_top_models_for_type(model_type, limit=_DEFAULT_PER_TYPE_LIMIT):
            for version in item.get("modelVersions") or ():
                for f in version.get("files") or ():
                    name = f.get("name") or ""
                    url = f.get("downloadUrl")
                    if not name or not url:
                        continue
                    key = canonicalize_path(name)
                    if not key or key in index:
                        continue
                    index[key] = {
                        "folder": folder,
                        "url": url,
                        "model_id": item.get("id"),
                        "version_id": version.get("id"),
                        "name": name,
                    }
    return index


def _load_disk_cache() -> Optional[dict[str, dict]]:
    try:
        with open(_cache_path(), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _save_disk_cache(index: dict[str, dict]) -> None:
    path = _cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(index, fh)
    except OSError as exc:
        logger.debug("Could not save civitai cache to %s: %s", path, exc)


_enabled: bool = False
_index: dict[str, dict] = {}


def init_civitai_model_cache(
    *,
    disabled: bool | None = None,
    refresh: bool = False,
) -> None:
    """Enable Civitai-backed model lookup. Loads disk cache if present;
    refreshes from the API on first call if missing or *refresh* is True.
    No-op if disabled or no API token."""
    global _enabled, _index
    if disabled is None:
        from .cli_args import args
        disabled = getattr(args, "disable_civitai_model_fallback", False)
    if disabled:
        return
    if not _api_token():
        logger.debug("Civitai model fallback disabled: CIVITAI_API_TOKEN not set")
        return

    if not refresh:
        cached = _load_disk_cache()
        if cached:
            _index = cached
            _enabled = True
            logger.info("Loaded %d Civitai-indexed models from %s", len(_index), _cache_path())
            return

    logger.info("Prefetching Civitai top-%d-per-type model index ...", _DEFAULT_PER_TYPE_LIMIT)
    _index = _build_index_from_api()
    if _index:
        _save_disk_cache(_index)
    _enabled = bool(_index)
    if _enabled:
        logger.info("Indexed %d Civitai models", len(_index))


def get_model_entry(folder_name: str, filename: str) -> Optional[tuple[str, str]]:
    """Return ``(folder_name, download_url)`` if our prefetched index or a
    runtime live-search lookup finds the file.

    Workflows often store filenames with a subdirectory prefix
    (``SD1.5\\Hyper-SD15-8steps-lora.safetensors``) — look up by full path
    first, then fall back to the basename. If neither hit, search Civitai's
    ``/api/v1/models?query=<basename>`` live so community-trained
    checkpoints/loras Just Work.
    """
    if not _enabled or not filename:
        return None
    from .component_model.files import canonicalize_path
    key = canonicalize_path(filename)
    if not key:
        return None
    entry = _index.get(key)
    if entry is None:
        # Fall back to basename — workflows commonly include subdir prefixes.
        basename = key.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if basename and basename != key:
            entry = _index.get(canonicalize_path(basename))
    if entry is None:
        # Live search Civitai for community checkpoints not in the prefetched
        # top-100. Cache hits to ``_index`` so repeat lookups are free.
        entry = _live_search_civitai(filename)
        if entry is not None:
            # Cache by both full key and basename so re-lookups skip the API.
            _index[key] = entry
            bn = key.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            if bn and bn != key:
                _index[canonicalize_path(bn) or bn] = entry
            _save_disk_cache(_index)
    if entry is None:
        return None
    return (entry["folder"], entry["url"])


# Live-search cache for misses too — Civitai search is slow and many workflows
# reference the same bogus filename (typos, "None", etc.). Cache misses for the
# session so we don't re-query.
_live_miss_cache: set[str] = set()


def _live_search_civitai(filename: str) -> Optional[dict]:
    """No-op for now — Civitai's public ``/api/v1/models?query=`` ignores the
    filter and returns the global most-popular list, so generic name search
    cannot find arbitrary community checkpoints. Use
    :func:`prefetch_civitai_models_for_user` to populate the index from
    workflow-author context instead.
    """
    if not filename:
        return None
    return None


_prefetched_users: set[str] = set()


def prefetch_civitai_models_for_user(username: str, *, limit: int = 200) -> int:
    """Fetch *username*'s Civitai uploads and merge into the model index.

    Civitai's public ``query=`` parameter is broken (returns global popular
    models regardless of input), but ``username=`` works correctly. We use
    that to hydrate the cache on demand: when a workflow URI from
    ``civitai://m/<id>`` is loaded, the author's other uploads (checkpoints,
    LoRAs, VAEs, custom models) get pulled into ``_index`` so any of those
    files referenced by the workflow resolve at lookup time.

    Returns the number of new files added to the index.
    """
    if not username or not _enabled:
        return 0
    if username in _prefetched_users:
        return 0
    _prefetched_users.add(username)
    if not _api_token():
        return 0
    from .component_model.files import canonicalize_path
    added = 0
    cursor: Optional[str] = None
    fetched = 0
    while fetched < limit:
        params: dict[str, object] = {
            "username": username,
            "limit": min(100, limit - fetched),
            "sort": "Most Downloaded",
        }
        if cursor:
            params["cursor"] = cursor
        try:
            r = requests.get(
                "https://civitai.com/api/v1/models",
                params=params,
                headers=_auth_headers(),
                timeout=20,
            )
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Civitai user prefetch %s failed: %s", username, exc)
            break
        data = r.json()
        items = data.get("items") or []
        if not items:
            break
        for item in items:
            mtype = item.get("type") or ""
            folder = _CIVITAI_TYPE_TO_FOLDER.get(mtype)
            if folder is None:
                continue
            for version in item.get("modelVersions") or ():
                for f in version.get("files") or ():
                    name = f.get("name") or ""
                    url = f.get("downloadUrl")
                    if not name or not url:
                        continue
                    key = canonicalize_path(name)
                    if not key or key in _index:
                        continue
                    _index[key] = {
                        "folder": folder,
                        "url": url,
                        "model_id": item.get("id"),
                        "version_id": version.get("id"),
                        "name": name,
                    }
                    added += 1
        fetched += len(items)
        cursor = (data.get("metadata") or {}).get("nextCursor")
        if not cursor:
            break
    if added:
        logger.info("Civitai: indexed %d files for user %r", added, username)
        _save_disk_cache(_index)
    return added


def prefetch_civitai_models_for_workflow_uri(uri: str) -> int:
    """If *uri* is a Civitai workflow URI, look up its author and prefetch
    that user's uploads. No-op for non-Civitai URIs.
    """
    if not _enabled or not uri.startswith(("civitai://", "civitai-red://")):
        return 0
    import re
    m = re.match(r"^civitai(?:-red)?://m/(\d+)", uri)
    if not m:
        return 0
    model_id = m.group(1)
    hostname = "civitai.red" if uri.startswith("civitai-red://") else "civitai.com"
    try:
        r = requests.get(
            f"https://{hostname}/api/v1/models/{model_id}",
            headers=_auth_headers(), timeout=20,
        )
        r.raise_for_status()
        creator = (r.json().get("creator") or {})
        username = creator.get("username")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Civitai author lookup for %s failed: %s", uri, exc)
        return 0
    if not username:
        return 0
    return prefetch_civitai_models_for_user(username)


def entry_to_downloadable(entry: tuple[str, str], filename: str) -> Optional[Downloadable]:
    """Convert a Civitai entry into a :class:`UrlFile`."""
    _, url = entry
    return UrlFile(url, _save_with_filename=filename, show_in_ui=False)
