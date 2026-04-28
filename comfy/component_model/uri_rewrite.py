"""Canonicalize HTTPS URLs to their fsspec scheme equivalents.

Examples:

* ``https://civitai.com/models/12345``                  → ``civitai://m/12345``
* ``https://civitai.com/models/12345?modelVersionId=67`` → ``civitai://v/67``
* ``https://civitai.red/models/12345``                  → ``civitai-red://m/12345``
* ``https://civitai.com/api/download/models/67``        → ``civitai://v/67``
* ``https://huggingface.co/owner/repo/blob/main/p/x.json`` → ``hf://owner/repo/p/x.json``
* ``https://huggingface.co/owner/repo/resolve/main/p/x.json`` → ``hf://owner/repo/p/x.json``
* ``https://www.youtube.com/watch?v=abcXYZ``            → ``youtube://abcXYZ``
* ``https://youtu.be/abcXYZ``                           → ``youtube://abcXYZ``

Anything we don't recognize is returned as-is.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse


def canonicalize_uri(uri: str) -> str:
    if not isinstance(uri, str) or "://" not in uri:
        return uri
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        return uri

    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parse_qs(parsed.query)

    if host in ("civitai.com", "www.civitai.com"):
        return _civitai(path, query, scheme="civitai")
    if host in ("civitai.red", "www.civitai.red"):
        return _civitai(path, query, scheme="civitai-red")
    if host in ("huggingface.co", "www.huggingface.co"):
        return _huggingface(path)
    if host in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        v = (query.get("v") or [None])[0]
        if v:
            return f"youtube://{v}"
        # /shorts/<id>
        m = re.match(r"^/shorts/([^/]+)", path)
        if m:
            return f"youtube://{m.group(1)}"
    if host == "youtu.be":
        m = re.match(r"^/([^/?&]+)", path)
        if m:
            return f"youtube://{m.group(1)}"

    return uri


def _civitai(path: str, query: dict, *, scheme: str) -> str:
    m = re.match(r"^/api/download/models/(\d+)/?$", path)
    if m:
        return f"{scheme}://v/{m.group(1)}"
    m = re.match(r"^/models/(\d+)", path)
    if m:
        version_id = (query.get("modelVersionId") or [None])[0]
        if version_id:
            return f"{scheme}://v/{version_id}"
        return f"{scheme}://m/{m.group(1)}"
    return f"{scheme}://" + path.lstrip("/")


def _huggingface(path: str) -> str:
    # path forms: /<owner>/<repo>/{blob,resolve,raw}/<branch>/<rest>
    #             /<owner>/<repo>
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 5 and parts[2] in ("blob", "resolve", "raw"):
        owner, repo = parts[0], parts[1]
        rest = "/".join(parts[4:])
        return f"hf://{owner}/{repo}/{rest}"
    if len(parts) >= 2:
        return f"hf://{parts[0]}/{parts[1]}"
    return f"https://huggingface.co{path}"
