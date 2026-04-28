"""Multi-host workflow search/top registry.

Each provider implements ``search(query, limit)`` and ``top(limit)`` returning a
list of ``WorkflowResult``. Hosts have a stable string id that the CLI uses for
``--with-host``/``--without-host`` filtering, and a URI scheme for emitting
canonical references (e.g. ``civitai://m/12345``, ``comfyui-org://t/<template_id>``).
"""
from __future__ import annotations

import logging
import os
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    host: str                         # provider id, e.g. "civitai"
    uri: str                          # canonical e.g. "civitai://m/618578"
    title: str
    creator: str | None = None
    description: str = ""
    download_url: str | None = None   # direct fetch URL, may need auth
    stats: dict[str, Any] = field(default_factory=dict)
    nsfw: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class WorkflowHost(Protocol):
    id: str
    scheme: str

    def top(self, limit: int = 100) -> list[WorkflowResult]: ...
    def search(self, query: str, limit: int = 50) -> list[WorkflowResult]: ...


_REGISTRY: dict[str, WorkflowHost] = {}


def register_host(host: WorkflowHost) -> None:
    _REGISTRY[host.id] = host


def get_host(host_id: str) -> WorkflowHost | None:
    _ensure_registered()
    return _REGISTRY.get(host_id)


def list_hosts() -> list[WorkflowHost]:
    _ensure_registered()
    return list(_REGISTRY.values())


def resolve_host_filter(with_host: list[str] | None, without_host: list[str] | None) -> list[WorkflowHost]:
    _ensure_registered()
    selected = list(_REGISTRY.values())
    if with_host:
        wanted = set(_split_csv(with_host))
        selected = [h for h in selected if h.id in wanted]
    if without_host:
        excluded = set(_split_csv(without_host))
        selected = [h for h in selected if h.id not in excluded]
    return selected


def _split_csv(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for v in values or ():
        out.extend(p.strip() for p in v.split(",") if p.strip())
    return out


# ── Civitai ───────────────────────────────────────────────────────────────────

class _CivitaiHost:
    def __init__(self, *, id: str, hostname: str, scheme: str):
        self.id = id
        self.hostname = hostname
        self.scheme = scheme

    def _api_get(self, path: str, params: dict[str, Any]) -> dict:
        import requests
        headers: dict[str, str] = {}
        token = os.environ.get("CIVITAI_API_TOKEN") or os.environ.get("CIVITAI_API_KEY")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        url = f"https://{self.hostname}/api/v1{path}"
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def _to_result(self, item: dict) -> WorkflowResult:
        mid = item.get("id")
        versions = item.get("modelVersions") or []
        download = None
        if versions:
            files = versions[0].get("files") or []
            if files:
                download = files[0].get("downloadUrl")
        nsfw_level = item.get("nsfwLevel") or 0
        stats = item.get("stats") or {}
        return WorkflowResult(
            host=self.id,
            uri=f"{self.scheme}://m/{mid}",
            title=str(item.get("name") or ""),
            creator=(item.get("creator") or {}).get("username"),
            description=str(item.get("description") or "")[:500],
            download_url=download,
            stats={
                "downloads": stats.get("downloadCount", 0),
                "thumbs_up": stats.get("thumbsUpCount", 0),
                "comments": stats.get("commentCount", 0),
            },
            nsfw=bool(item.get("nsfw")) or nsfw_level >= 8,
            extra={"nsfw_level": nsfw_level, "version_id": versions[0].get("id") if versions else None},
        )

    def top(self, limit: int = 100) -> list[WorkflowResult]:
        params = {"types": "Workflows", "sort": "Most Downloaded",
                  "period": "AllTime", "limit": min(max(limit, 1), 100)}
        data = self._api_get("/models", params)
        return [self._to_result(it) for it in (data.get("items") or [])][:limit]

    def search(self, query: str, limit: int = 50) -> list[WorkflowResult]:
        # The API's `query=` parameter doesn't combine with `types=Workflows`
        # (returns 0 hits); pull a larger top-N and filter client-side.
        q = query.lower().strip()
        params = {"types": "Workflows", "sort": "Most Downloaded",
                  "period": "AllTime", "limit": 100}
        data = self._api_get("/models", params)
        out: list[WorkflowResult] = []
        for it in data.get("items") or []:
            haystack = " ".join([
                str(it.get("name") or ""),
                str(it.get("description") or "")[:1000],
                " ".join(it.get("tags") or []),
            ]).lower()
            if q in haystack:
                out.append(self._to_result(it))
                if len(out) >= limit:
                    break
        return out


# ── ComfyUI Org bundled templates ────────────────────────────────────────────

class _ComfyUIOrgHost:
    id = "comfyui-org"
    scheme = "comfyui-org"

    def _entries(self):
        try:
            from comfyui_workflow_templates import iter_templates
        except ImportError:
            return []
        return list(iter_templates())

    def _to_result(self, t) -> WorkflowResult:
        return WorkflowResult(
            host=self.id,
            uri=f"{self.scheme}://t/{t.template_id}",
            title=str(t.template_id),
            creator="comfyui-org",
            description=f"Bundled template ({t.bundle})" if hasattr(t, "bundle") else "",
            download_url=None,
            stats={},
            extra={"bundle": getattr(t, "bundle", None)},
        )

    def top(self, limit: int = 100) -> list[WorkflowResult]:
        # No popularity signal for bundled templates; return alphabetically.
        return [self._to_result(t) for t in self._entries()[:limit]]

    def search(self, query: str, limit: int = 50) -> list[WorkflowResult]:
        q = query.lower()
        out: list[WorkflowResult] = []
        for t in self._entries():
            if q in str(t.template_id).lower():
                out.append(self._to_result(t))
                if len(out) >= limit:
                    break
        return out


# ── Hugging Face ─────────────────────────────────────────────────────────────

class _HuggingFaceHost:
    id = "huggingface"
    scheme = "hf"

    def _search_repos(self, query: str | None, limit: int) -> list:
        try:
            from huggingface_hub import HfApi  # type: ignore
        except ImportError:
            return []
        api = HfApi()
        out = []
        # Datasets are the typical home for ComfyUI workflow JSON collections.
        try:
            for ds in api.list_datasets(
                search=query or "comfyui workflow",
                limit=limit, full=False,
            ):
                out.append(("dataset", ds))
                if len(out) >= limit:
                    break
        except Exception:  # noqa: BLE001
            pass
        # Models tagged with comfyui-workflow are also common.
        try:
            for m in api.list_models(
                search=query or "comfyui-workflow",
                limit=max(0, limit - len(out)), full=False,
            ):
                out.append(("model", m))
                if len(out) >= limit:
                    break
        except Exception:  # noqa: BLE001
            pass
        return out

    def _to_result(self, kind: str, item) -> WorkflowResult:
        repo_id = getattr(item, "id", None) or getattr(item, "modelId", None) or ""
        downloads = getattr(item, "downloads", 0) or 0
        likes = getattr(item, "likes", 0) or 0
        return WorkflowResult(
            host=self.id,
            uri=f"hf://{repo_id}",
            title=str(repo_id),
            creator=repo_id.split("/", 1)[0] if "/" in repo_id else None,
            description=f"HF {kind}",
            stats={"downloads": downloads, "thumbs_up": likes},
            extra={"kind": kind},
        )

    def top(self, limit: int = 100) -> list[WorkflowResult]:
        return [self._to_result(k, it) for k, it in self._search_repos(None, limit)][:limit]

    def search(self, query: str, limit: int = 50) -> list[WorkflowResult]:
        return [self._to_result(k, it) for k, it in self._search_repos(query, limit)][:limit]


# ── Tensor.Art (stub) ────────────────────────────────────────────────────────

class _TensorArtHost:
    id = "tensorart"
    scheme = "tensorart"

    def _unimplemented(self) -> list[WorkflowResult]:
        # The TAMS public API requires an application id + key (per
        # tams-docs.tensor.art) — there is no anonymous list endpoint.
        # Returning empty rather than raising keeps the multi-host CLI
        # usable. Set TENSOR_ART_APP_ID + TENSOR_ART_API_KEY and extend
        # this provider when needed.
        return []

    def top(self, limit: int = 100) -> list[WorkflowResult]:
        return self._unimplemented()

    def search(self, query: str, limit: int = 50) -> list[WorkflowResult]:
        return self._unimplemented()


# ── Bootstrap ────────────────────────────────────────────────────────────────

_REGISTERED = False


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    register_host(_CivitaiHost(id="civitai", hostname="civitai.com", scheme="civitai"))
    register_host(_CivitaiHost(id="civitai_red", hostname="civitai.red", scheme="civitai-red"))
    register_host(_ComfyUIOrgHost())
    register_host(_HuggingFaceHost())
    register_host(_TensorArtHost())
