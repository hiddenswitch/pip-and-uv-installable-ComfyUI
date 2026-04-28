"""Add fsspec URI support to ComfyUI-VideoHelperSuite's video/audio loaders.

VHS upstream supports `http://` / `https://` URLs (only via yt-dlp) for the
`*Path` variants. The `*Upload` variants take a filename from `input/` only.
This patch widens both paths to accept any fsspec URI (`s3://`, `gs://`,
`hf://`, `https://`, ...) by:

* extending `videohelpersuite.utils.is_url` to recognize fsspec schemes,
* falling back from yt-dlp to fsspec in `try_download_video` for non-yt-dlp
  schemes (and as a fallback when yt-dlp fails),
* short-circuiting `LoadVideoUpload`/`LoadVideoFFmpegUpload`/`LoadAudioUpload`
  to download URI-shaped inputs to the temp directory before the upstream
  ``folder_paths.get_annotated_filepath`` lookup that would otherwise reject
  a non-local path.

Apply via ``apply_vhs_fsspec_patch()`` after VHS has loaded (i.e. after
``import_all_nodes_in_workspace``).
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Schemes that should be routed through fsspec rather than treated as local paths.
_URI_SCHEMES = frozenset({
    "http", "https", "ftp", "sftp",
    "s3", "gs", "gcs", "az", "abfs", "adl", "webhdfs",
    "hf", "file",
})


def _has_uri_scheme(value: str) -> bool:
    if not isinstance(value, str) or "://" not in value:
        return False
    return value.split("://", 1)[0].lower() in _URI_SCHEMES


def _find_vhs_module(suffix: str):
    """Return the loaded VHS submodule whose dotted name ends with *suffix*."""
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if "videohelpersuite" not in name.lower():
            continue
        if name.endswith(suffix):
            return mod
    return None


def _download_via_fsspec(uri: str, temp_dir: str) -> str:
    import fsspec  # type: ignore
    os.makedirs(temp_dir, exist_ok=True)
    suffix = os.path.splitext(urlparse(uri).path)[1] or ".bin"
    tmp = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=suffix, delete=False)
    tmp.close()
    with fsspec.open(uri, "rb") as src, open(tmp.name, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return tmp.name


def apply_vhs_fsspec_patch() -> bool:
    """Patch VHS's loaders to accept fsspec URIs. Returns True if applied."""
    utils = _find_vhs_module(".videohelpersuite.utils") or _find_vhs_module("videohelpersuite.utils")
    load_video = _find_vhs_module(".videohelpersuite.load_video_nodes") or _find_vhs_module("videohelpersuite.load_video_nodes")
    if utils is None or load_video is None:
        return False
    if getattr(utils, "_appmana_fsspec_patched", False):
        return True

    _orig_is_url = utils.is_url
    _orig_try_download = utils.try_download_video

    def is_url_patched(value: str) -> bool:
        return _has_uri_scheme(value) or _orig_is_url(value)

    def try_download_video_patched(url: str) -> Optional[str]:
        if not _has_uri_scheme(url):
            return _orig_try_download(url)
        scheme = url.split("://", 1)[0].lower()
        if scheme in ("http", "https"):
            try:
                local = _orig_try_download(url)
            except Exception:  # noqa: BLE001
                local = None
            if local:
                return local
        try:
            from comfy.cmd import folder_paths
            return _download_via_fsspec(url, folder_paths.get_temp_directory())
        except Exception as exc:  # noqa: BLE001
            logger.warning("fsspec download of %s failed: %s", url, exc)
            return None

    utils.is_url = is_url_patched
    utils.try_download_video = try_download_video_patched

    # The *Upload classes call folder_paths.get_annotated_filepath() before
    # load_video(), which raises for URIs. Wrap their load_video to download
    # first when the input looks like a URI.
    def _wrap_upload_loader(cls, base_attr_name: str = "load_video") -> None:
        if cls is None:
            return
        orig = getattr(cls, base_attr_name, None)
        if orig is None or getattr(orig, "_appmana_fsspec_wrapped", False):
            return

        def wrapper(self, **kwargs):
            video = kwargs.get("video") or kwargs.get("audio")
            if isinstance(video, str) and _has_uri_scheme(video):
                local = try_download_video_patched(video)
                if local:
                    if "video" in kwargs:
                        kwargs["video"] = local
                    elif "audio" in kwargs:
                        kwargs["audio"] = local
                    return orig(self, **kwargs)
            return orig(self, **kwargs)

        wrapper._appmana_fsspec_wrapped = True  # type: ignore[attr-defined]
        setattr(cls, base_attr_name, wrapper)

        # VALIDATE_INPUTS rejects non-local paths; relax for URIs.
        # Preserve the original signature exactly — ComfyUI introspects
        # VALIDATE_INPUTS to decide which kwargs to pass; widening the
        # signature to **kwargs makes it pass *everything*, which the
        # underlying upstream method then chokes on (e.g. force_rate).
        validate = getattr(cls, "VALIDATE_INPUTS", None)
        if validate is not None and not getattr(validate, "_appmana_fsspec_wrapped", False):
            import functools as _ft
            import inspect as _inspect
            inner = validate.__func__ if hasattr(validate, "__func__") else validate
            try:
                sig = _inspect.signature(inner)
            except (TypeError, ValueError):
                sig = None

            @_ft.wraps(inner)
            def validate_wrapper(*args, **kw):
                target = (
                    args[1] if len(args) >= 2 else
                    (kw.get("video") or kw.get("audio") or kw.get("path"))
                )
                if isinstance(target, str) and _has_uri_scheme(target):
                    return True
                return inner(*args, **kw)

            if sig is not None:
                validate_wrapper.__signature__ = sig  # type: ignore[attr-defined]
            validate_wrapper._appmana_fsspec_wrapped = True  # type: ignore[attr-defined]
            setattr(cls, "VALIDATE_INPUTS", classmethod(validate_wrapper))

    for cls_name in ("LoadVideoUpload", "LoadVideoFFmpegUpload"):
        _wrap_upload_loader(getattr(load_video, cls_name, None))

    load_audio = _find_vhs_module(".videohelpersuite.load_audio_nodes") or _find_vhs_module("videohelpersuite.load_audio_nodes")
    if load_audio is not None:
        _wrap_upload_loader(getattr(load_audio, "LoadAudioUpload", None), base_attr_name="load_audio")

    utils._appmana_fsspec_patched = True
    logger.info("VHS fsspec URI support patched")
    return True
