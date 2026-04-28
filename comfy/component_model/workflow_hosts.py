"""Multi-host workflow / model search registry."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    host: str
    uri: str
    title: str
    creator: str | None = None
    description: str = ""
    download_url: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    nsfw: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResult:
    host: str
    kind: str
    uri: str
    title: str
    creator: str | None = None
    description: str = ""
    base_model: str | None = None
    trigger_words: list[str] = field(default_factory=list)
    download_url: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    nsfw: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class WorkflowHost(Protocol):
    id: str
    scheme: str

    def top(self, limit: int = 100) -> list[WorkflowResult]: ...
    def search(self, query: str, limit: int = 50) -> list[WorkflowResult]: ...


class ModelHost(Protocol):
    id: str
    scheme: str

    def search_models(
        self,
        query: str,
        *,
        kind: str | None = None,
        base_models: list[str] | None = None,
        limit: int = 50,
    ) -> list[ModelResult]: ...


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


_CIVITAI_PERIOD_ALIASES: dict[str, str] = {
    "day": "Day", "1d": "Day",
    "week": "Week", "7d": "Week",
    "month": "Month", "30d": "Month", "30": "Month",
    "year": "Year", "180d": "Year", "180": "Year", "365d": "Year", "360d": "Year", "360": "Year",
    "alltime": "AllTime", "all": "AllTime", "all-time": "AllTime",
}


def _civitai_period(value: str | None) -> str:
    if not value:
        return "AllTime"
    return _CIVITAI_PERIOD_ALIASES.get(str(value).lower().strip(), "AllTime")


_KIND_TO_CIVITAI = {
    "lora": "LORA",
    "loras": "LORA",
    "checkpoint": "Checkpoint",
    "checkpoints": "Checkpoint",
    "ckpt": "Checkpoint",
    "embedding": "TextualInversion",
    "embeddings": "TextualInversion",
    "ti": "TextualInversion",
    "vae": "VAE",
    "controlnet": "Controlnet",
    "control": "Controlnet",
    "lycoris": "LoCon",
    "locon": "LoCon",
    "hypernet": "Hypernetwork",
}

_CIVITAI_TYPE_TO_KIND = {
    "lora": "lora", "checkpoint": "checkpoint",
    "textualinversion": "embedding", "vae": "vae",
    "controlnet": "controlnet", "locon": "lora",
    "hypernetwork": "other", "workflows": "other",
}

_KNOWN_KINDS = {"lora", "checkpoint", "embedding", "vae", "controlnet"}


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

    def top(self, limit: int = 100, *, period: str = "AllTime", query: str | None = None) -> list[WorkflowResult]:
        api_period = _civitai_period(period)
        params: dict[str, object] = {
            "types": "Workflows",
            "sort": "Most Downloaded",
            "period": api_period,
            "limit": min(max(limit, 1), 100),
        }
        out: list[WorkflowResult] = []
        while len(out) < limit:
            data = self._api_get("/models", params)
            items = data.get("items") or []
            for it in items:
                if query and query.lower() not in (it.get("name") or "").lower() and \
                   query.lower() not in (it.get("description") or "").lower()[:1000]:
                    continue
                out.append(self._to_result(it))
                if len(out) >= limit:
                    break
            cursor = (data.get("metadata") or {}).get("nextCursor")
            if not cursor or len(items) == 0 or query is None and len(out) >= limit:
                break
            params["cursor"] = cursor
        return out[:limit]

    def search(self, query: str, limit: int = 50) -> list[WorkflowResult]:
        # Civitai's `query=` parameter returns 0 hits when combined with
        # types=Workflows; pull a larger top-N and filter client-side.
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

    def _to_model_result(self, item: dict, *, fallback_kind: str) -> ModelResult:
        mid = item.get("id")
        api_type = (item.get("type") or "").lower()
        kind = _CIVITAI_TYPE_TO_KIND.get(api_type, fallback_kind)
        versions = item.get("modelVersions") or []
        v0 = versions[0] if versions else {}
        files = v0.get("files") or []
        download = files[0].get("downloadUrl") if files else None
        trigger_words = [str(w) for w in (v0.get("trainedWords") or []) if w]
        version_id = v0.get("id") if v0 else None
        # Pin to version id so re-publishes don't silently change weights.
        uri = f"{self.scheme}://v/{version_id}" if version_id else f"{self.scheme}://m/{mid}"
        nsfw_level = item.get("nsfwLevel") or 0
        stats = item.get("stats") or {}
        return ModelResult(
            host=self.id,
            kind=kind,
            uri=uri,
            title=str(item.get("name") or ""),
            creator=(item.get("creator") or {}).get("username"),
            description=str(item.get("description") or "")[:500],
            base_model=v0.get("baseModel"),
            trigger_words=trigger_words,
            download_url=download,
            stats={
                "downloads": stats.get("downloadCount", 0),
                "thumbs_up": stats.get("thumbsUpCount", 0),
                "comments": stats.get("commentCount", 0),
            },
            nsfw=bool(item.get("nsfw")) or nsfw_level >= 8,
            extra={
                "model_id": mid,
                "version_id": version_id,
                "nsfw_level": nsfw_level,
                "tags": list(item.get("tags") or []),
            },
        )

    def search_models(
        self,
        query: str,
        *,
        kind: str | None = None,
        base_models: list[str] | None = None,
        limit: int = 50,
    ) -> list[ModelResult]:
        api_type = _KIND_TO_CIVITAI.get((kind or "").lower().strip()) if kind else None
        fallback = (kind or "").lower().strip() or "other"
        if fallback not in _KNOWN_KINDS:
            fallback = "other"

        params: dict[str, Any] = {
            "query": query,
            "sort": "Most Downloaded",
            "period": "AllTime",
            "limit": min(max(limit, 1), 100),
        }
        if api_type:
            params["types"] = api_type
        if base_models:
            params["baseModels"] = list(base_models)
        data = self._api_get("/models", params)
        out = [self._to_model_result(it, fallback_kind=fallback)
               for it in (data.get("items") or [])]
        if out:
            return out[:limit]

        # Civitai's `query=` returns 0 for some `types=` combinations
        # (notably Checkpoint). Fall back to top-N + client-side substring
        # filter, matching how `search()` handles workflows.
        q = query.lower().strip()
        broad: dict[str, Any] = {
            "sort": "Most Downloaded",
            "period": "AllTime",
            "limit": 100,
        }
        if api_type:
            broad["types"] = api_type
        if base_models:
            broad["baseModels"] = list(base_models)
        data = self._api_get("/models", broad)
        results: list[ModelResult] = []
        for it in data.get("items") or []:
            haystack = " ".join([
                str(it.get("name") or ""),
                str(it.get("description") or "")[:1000],
                " ".join(it.get("tags") or []),
            ]).lower()
            if q in haystack:
                results.append(self._to_model_result(it, fallback_kind=fallback))
                if len(results) >= limit:
                    break
        return results


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


_HF_KIND_HINTS = {
    "lora": "lora",
    "checkpoint": "text-to-image",
    "embedding": "textual-inversion",
    "vae": "vae",
    "controlnet": "controlnet",
}


class _HuggingFaceHost:
    id = "huggingface"
    scheme = "hf"

    def _hf_api(self):
        from huggingface_hub import HfApi  # type: ignore
        return HfApi()

    def _search_repos(self, query: str | None, limit: int) -> list:
        api = self._hf_api()
        out: list = []
        for ds in api.list_datasets(
            search=query or "comfyui workflow",
            limit=limit, full=False,
        ):
            out.append(("dataset", ds))
            if len(out) >= limit:
                return out
        for m in api.list_models(
            search=query or "comfyui-workflow",
            limit=max(0, limit - len(out)), full=False,
        ):
            out.append(("model", m))
            if len(out) >= limit:
                break
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

    def search_models(
        self,
        query: str,
        *,
        kind: str | None = None,
        base_models: list[str] | None = None,
        limit: int = 50,
    ) -> list[ModelResult]:
        api = self._hf_api()
        kind_norm = (kind or "").lower().strip()
        hint = _HF_KIND_HINTS.get(kind_norm)
        search_term = query
        if hint and hint not in (query or "").lower():
            search_term = f"{query} {hint}"
        out: list[ModelResult] = []
        for m in api.list_models(search=search_term, limit=limit, full=False):
            repo_id = getattr(m, "id", "") or getattr(m, "modelId", "") or ""
            if not repo_id:
                continue
            tags = list(getattr(m, "tags", []) or [])
            inferred = kind_norm or "other"
            if not kind_norm:
                if "lora" in tags:
                    inferred = "lora"
                elif "controlnet" in tags:
                    inferred = "controlnet"
                elif "textual-inversion" in tags or "embedding" in tags:
                    inferred = "embedding"
                elif "vae" in tags:
                    inferred = "vae"
            base = next(
                (t.split("base_model:", 1)[1] for t in tags if t.startswith("base_model:")),
                None,
            )
            out.append(ModelResult(
                host=self.id,
                kind=inferred,
                uri=f"hf://{repo_id}",
                title=str(repo_id),
                creator=repo_id.split("/", 1)[0] if "/" in repo_id else None,
                description="",
                base_model=base,
                trigger_words=[],
                download_url=None,
                stats={
                    "downloads": getattr(m, "downloads", 0) or 0,
                    "thumbs_up": getattr(m, "likes", 0) or 0,
                },
                extra={"tags": tags},
            ))
            if len(out) >= limit:
                break
        return out


class _TensorArtHost:
    """TAMS public API requires app id + key; no anonymous list endpoint."""
    id = "tensorart"
    scheme = "tensorart"

    def top(self, limit: int = 100) -> list[WorkflowResult]:
        return []

    def search(self, query: str, limit: int = 50) -> list[WorkflowResult]:
        return []


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
