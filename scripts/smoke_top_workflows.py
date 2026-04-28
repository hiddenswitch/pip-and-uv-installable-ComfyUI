"""Smoke-test top-N civitai workflows through the full --all pre-flight.

For each workflow URI:
  1. load_workflow_json: parse + extract from zip/PNG/foreign-format
  2. resolve_workflow_packages_versioned: name the custom node packages
  3. check missing packages resolve on nodes.appmana.com (HEAD via pip index)
  4. convert UI → API (covers the schema validation path)
  5. inventory required model filenames + check known-models DB / civitai cache

Reports a one-line summary per workflow, then a total grade.
Skips actual model download and GPU execution.

Usage:
  python scripts/smoke_top_workflows.py --limit 10 --period AllTime
  python scripts/smoke_top_workflows.py --family wan --period 30d
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke")


@dataclass
class Result:
    uri: str
    title: str
    load_ok: bool = False
    load_err: str | None = None
    format_kind: str | None = None
    pkg_total: int = 0
    pkg_unresolved: list[str] = field(default_factory=list)
    convert_ok: bool = False
    convert_err: str | None = None
    model_total: int = 0
    model_known: int = 0
    model_unknown: list[str] = field(default_factory=list)

    @property
    def grade(self) -> str:
        if not self.load_ok:
            return "LOAD"
        if not self.convert_ok:
            return "CONVERT"
        if self.pkg_unresolved:
            return "NODES"
        if self.model_unknown:
            return "MODELS"
        return "OK"


def _smoke(uri: str, title: str) -> Result:
    r = Result(uri=uri, title=title)

    # 1. Load
    try:
        from comfy.component_model.asyncio_files import load_workflow_json
        from comfy.component_model.foreign_workflow import detect_workflow_format
        wf = load_workflow_json(uri)
        r.load_ok = True
        r.format_kind = detect_workflow_format(wf)
    except Exception as e:  # noqa: BLE001
        r.load_err = f"{type(e).__name__}: {e}"[:240]
        return r

    # 2. Resolve custom node packages
    try:
        from comfy.component_model.workflow_dependencies import resolve_workflow_packages_versioned
        pkgs = list(resolve_workflow_packages_versioned(wf))
        r.pkg_total = len(pkgs)
        # Cross-check installable: query the simple index for each name.
        unresolved = _check_pip_index({n for n, _ in pkgs})
        r.pkg_unresolved = sorted(unresolved)
    except Exception as e:  # noqa: BLE001
        r.pkg_unresolved = [f"<resolve-error: {type(e).__name__}: {e}>"]

    # 3. Convert UI → API
    try:
        from comfy.component_model.workflow_convert import is_ui_workflow, convert_ui_to_api
        if is_ui_workflow(wf):
            api = convert_ui_to_api(wf)
        else:
            api = wf
        r.convert_ok = True
    except Exception as e:  # noqa: BLE001
        r.convert_err = f"{type(e).__name__}: {e}"[:240]
        api = None

    # 4. Inventory model filenames
    if api is not None:
        try:
            filenames = _collect_model_filenames(api)
            r.model_total = len(filenames)
            unknown = _check_models_resolvable(filenames)
            r.model_known = r.model_total - len(unknown)
            r.model_unknown = sorted(unknown)[:10]
        except Exception as e:  # noqa: BLE001
            r.model_unknown = [f"<inventory-error: {type(e).__name__}: {e}>"]
    return r


_MODEL_INPUT_KEYS = {
    "ckpt_name", "lora_name", "vae_name", "model_name", "control_net_name",
    "clip_name", "clip_name1", "clip_name2", "style_model_name",
    "ipadapter_file", "preset", "unet_name", "diffusion_model",
}


def _collect_model_filenames(api: dict) -> list[str]:
    out: list[str] = []
    for nd in api.values():
        if not isinstance(nd, dict):
            continue
        inputs = nd.get("inputs") or {}
        for k, v in inputs.items():
            if k in _MODEL_INPUT_KEYS and isinstance(v, str) and v:
                out.append(v)
    # Dedup preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


def _check_models_resolvable(filenames: list[str]) -> list[str]:
    from comfy.model_downloader import _known_models_db
    from comfy.model_downloader_types import canonicalize_path
    try:
        from comfy import civitai_model_cache
        civitai_model_cache.init_civitai_model_cache()
    except Exception:  # noqa: BLE001
        pass

    # Build name index of known models
    known: set[str] = set()
    for db in _known_models_db:
        for item in db:
            for name in [str(item), getattr(item, "filename", None),
                         getattr(item, "save_with_filename", None)] + list(getattr(item, "alternate_filenames", []) or []):
                if name:
                    key = canonicalize_path(name)
                    if key:
                        known.add(key)
                        # Also basename
                        bn = key.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                        if bn:
                            known.add(bn)

    unknown: list[str] = []
    for fn in filenames:
        if not fn or fn in {"None", "Custom", "none"}:
            continue  # placeholder values aren't real model references
        # Workflow widgets sometimes carry display labels like "VIT-G (medium strength)"
        # in slots that the schema *also* uses for filenames. Skip values that
        # contain spaces or parens — real filenames don't.
        if " " in fn or "(" in fn:
            continue
        key = canonicalize_path(fn)
        bn = (key or fn).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        in_known = bool(key and (key in known or bn in known))
        # Try civitai cache (also tries basename fallback internally)
        try:
            from comfy.civitai_model_cache import get_model_entry
            in_civitai = any(get_model_entry(folder, fn) is not None
                             for folder in ("checkpoints", "loras", "vae", "controlnet",
                                            "clip", "unet", "embeddings", "upscale_models",
                                            "clip_vision", "ipadapter", "diffusion_models",
                                            "text_encoders", "animatediff_models"))
        except Exception:  # noqa: BLE001
            in_civitai = False
        if not (in_known or in_civitai):
            unknown.append(fn)
    return unknown


_PIP_INDEX_CACHE: dict[str, bool] = {}


def _check_pip_index(names: set[str]) -> set[str]:
    """Return names that don't appear on nodes.appmana.com/simple/."""
    if not names:
        return set()
    import urllib.request
    base = "https://nodes.appmana.com/simple/"
    unresolved: set[str] = set()
    for n in names:
        norm = n.lower().replace("_", "-")
        if norm in _PIP_INDEX_CACHE:
            if not _PIP_INDEX_CACHE[norm]:
                unresolved.add(n)
            continue
        url = f"{base}{norm}/"
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                ok = 200 <= resp.status < 300
        except Exception:  # noqa: BLE001
            ok = False
        _PIP_INDEX_CACHE[norm] = ok
        if not ok:
            unresolved.add(n)
    return unresolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--period", default="AllTime")
    parser.add_argument("--family", default=None)
    parser.add_argument("--with-host", default="civitai")
    parser.add_argument("--uri", action="append", default=[])
    args = parser.parse_args()

    if args.uri:
        candidates = [(u, u) for u in args.uri]
    else:
        from comfy.component_model.workflow_hosts import resolve_host_filter
        from comfy.cmd.sub_workflows import _FAMILY_QUERIES
        hosts = resolve_host_filter([args.with_host], None)
        candidates: list[tuple[str, str]] = []
        for host in hosts:
            kwargs: dict[str, Any] = {"limit": args.limit}
            try:
                # Civitai host accepts `period`
                if hasattr(host, "top") and "period" in host.top.__code__.co_varnames:
                    kwargs["period"] = args.period
            except Exception:  # noqa: BLE001
                pass
            queries = [None]
            if args.family:
                queries = _FAMILY_QUERIES.get(args.family, [args.family])
            for q in queries:
                if q is not None and "query" in host.top.__code__.co_varnames:
                    results = host.top(query=q, **kwargs)
                else:
                    results = host.top(**kwargs)
                for w in results:
                    candidates.append((w.uri, w.title))
                    if len(candidates) >= args.limit:
                        break
                if len(candidates) >= args.limit:
                    break
            if len(candidates) >= args.limit:
                break
        candidates = candidates[:args.limit]

    print(f"Smoke-testing {len(candidates)} workflow(s) "
          f"(period={args.period}, family={args.family})\n")
    results: list[Result] = []
    for i, (uri, title) in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {uri} — {title[:70]}", flush=True)
        try:
            r = _smoke(uri, title)
        except Exception as e:  # noqa: BLE001
            r = Result(uri=uri, title=title, load_err=f"smoke crashed: {e}")
            traceback.print_exc()
        results.append(r)
        # One-line per-workflow report
        if r.grade == "OK":
            print(f"    ✓ OK  pkg={r.pkg_total} models={r.model_total}/{r.model_total}")
        elif r.grade == "LOAD":
            print(f"    ✗ LOAD failed: {r.load_err}")
        elif r.grade == "CONVERT":
            print(f"    ✗ CONVERT failed: {r.convert_err}")
        elif r.grade == "NODES":
            print(f"    ⚠ NODES unresolved ({len(r.pkg_unresolved)}/{r.pkg_total}): {r.pkg_unresolved[:3]}")
        elif r.grade == "MODELS":
            print(f"    ⚠ MODELS unknown ({len(r.model_unknown)}/{r.model_total}): {r.model_unknown[:3]}")

    # Summary
    print("\n" + "=" * 70)
    grades: dict[str, int] = {}
    for r in results:
        grades[r.grade] = grades.get(r.grade, 0) + 1
    print("Summary:", " ".join(f"{k}={v}" for k, v in sorted(grades.items())))

    # JSON dump for programmatic inspection
    out_path = "/tmp/smoke_top_workflows_results.json"
    with open(out_path, "w") as f:
        json.dump([{"uri": r.uri, "title": r.title, "grade": r.grade,
                    "load_err": r.load_err, "format": r.format_kind,
                    "pkg_total": r.pkg_total, "pkg_unresolved": r.pkg_unresolved,
                    "convert_err": r.convert_err,
                    "model_total": r.model_total, "model_known": r.model_known,
                    "model_unknown": r.model_unknown}
                   for r in results], f, indent=2)
    print(f"Wrote: {out_path}")
    return 0 if all(r.grade == "OK" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
