"""fsspec backend for ``youtube://<video_id>`` (read-only) backed by yt-dlp.

Used by ``--video https://youtube.com/...`` (which the URL canonicalizer
rewrites to ``youtube://<video_id>``). Downloads the video on first open,
caches the resulting path in-process, and returns a file handle.

yt-dlp is imported lazily so this module is cheap to register.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

from fsspec import AbstractFileSystem


_DOWNLOAD_CACHE: dict[str, str] = {}


def _download_youtube(video_id: str, *, format_selector: str = "best") -> str:
    cached = _DOWNLOAD_CACHE.get(video_id)
    if cached and os.path.isfile(cached):
        return cached

    try:
        from yt_dlp import YoutubeDL  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "youtube:// support requires yt-dlp; install with `uv pip install yt-dlp`"
        ) from exc

    out_dir = tempfile.mkdtemp(prefix="comfyui_youtube_")
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")
    opts = {
        "format": format_selector,
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        path = ydl.prepare_filename(info)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"yt-dlp did not produce a file for {video_id}")
    _DOWNLOAD_CACHE[video_id] = path
    return path


class YouTubeFileSystem(AbstractFileSystem):
    """Read-only fsspec backend that downloads YouTube videos via yt-dlp."""

    protocol = "youtube"
    root_marker = "/"

    def _open(self, path, mode="rb", block_size=None, **kwargs):
        if mode != "rb":
            raise NotImplementedError("youtube filesystem is read-only")
        video_id = path.strip("/").split("/")[0]
        if not video_id:
            raise ValueError(f"empty youtube:// path: {path!r}")
        local = _download_youtube(video_id)
        return open(local, "rb")

    def _info(self, path, **kwargs):
        return {"name": path, "size": None, "type": "file"}
