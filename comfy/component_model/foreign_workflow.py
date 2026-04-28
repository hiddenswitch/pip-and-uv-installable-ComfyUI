"""Detect and translate non-ComfyUI workflow formats hosted on Civitai et al.

Civitai's ``types=Workflows`` bucket is generic: most uploads are ComfyUI but
roughly 5–10% are A1111/Forge ``.txt`` parameter dumps, Fooocus presets,
SwarmUI session JSON, InvokeAI graphs, or Krita-AI plugin files.

For A1111 / Forge / Fooocus we can synthesize a generic ComfyUI text-to-image
workflow because those tools are essentially "checkpoint + LoRAs + sampler +
prompts" — fields that map cleanly to a CheckpointLoaderSimple → LoRA chain
→ KSampler → VAEDecode → SaveImage graph.

For SwarmUI / InvokeAI / Krita-AI / unknown shapes we cannot translate; we
raise :class:`UnsupportedWorkflowFormatError` (a subclass of ``ApiValueError``)
with a clear explanation instead of letting an opaque schema validation fail
deeper in the pipeline.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from ..api.exceptions import ApiValueError

logger = logging.getLogger(__name__)


WorkflowKind = Literal[
    "comfyui-ui",
    "comfyui-api",
    "a1111",
    "fooocus",
    "swarmui",
    "invokeai",
    "krita-ai",
    "unknown",
]


class UnsupportedWorkflowFormatError(ApiValueError):
    """A workflow source was loaded but its format is not a usable ComfyUI workflow.

    Subclass of ``ApiValueError`` so existing API-validation handlers still
    catch it, but with a clearer message that names the detected format and
    the supported list (instead of a low-level ``oneOf`` mismatch).
    """

    def __init__(self, kind: WorkflowKind, *, source: str | None = None, reason: str | None = None):
        self.kind = kind
        self.source = source
        bits = [f"workflow format {kind!r} is not a valid ComfyUI workflow"]
        if source:
            bits.append(f"source={source}")
        if reason:
            bits.append(reason)
        bits.append(
            "supported: ComfyUI UI graph (nodes+links), ComfyUI API graph "
            "(class_type), A1111/Forge .txt parameter dump, Fooocus JSON preset"
        )
        super().__init__(" — ".join(bits))


# ── Sniffer ───────────────────────────────────────────────────────────────────

_A1111_FIELD_RE = re.compile(
    r"\b(Steps|Sampler|CFG\s*scale|Seed|Size|Model(?:\s*hash)?|Lora\s*hashes|Schedule\s*type|Denoising\s*strength|VAE)\s*:\s*",
    re.IGNORECASE,
)


def detect_workflow_format(data: bytes | str | dict) -> WorkflowKind:
    """Classify a workflow source into one of the known shapes.

    Accepts bytes (file body), str (decoded text), or a parsed dict.
    """
    if isinstance(data, dict):
        return _classify_parsed_dict(data)
    if isinstance(data, bytes):
        # Strip BOM and try UTF-8.
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return "unknown"
    else:
        text = data
    text_stripped = text.lstrip("﻿").strip()
    if not text_stripped:
        return "unknown"
    if text_stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(text_stripped)
        except Exception:  # noqa: BLE001
            return "unknown"
        if isinstance(parsed, dict):
            return _classify_parsed_dict(parsed)
        return "unknown"
    if _looks_like_a1111(text_stripped):
        return "a1111"
    return "unknown"


def _classify_parsed_dict(parsed: dict) -> WorkflowKind:
    if not parsed:
        return "unknown"
    # ComfyUI UI form
    if "nodes" in parsed and "links" in parsed:
        return "comfyui-ui"
    # ComfyUI API form: every value is a dict with class_type
    values = list(parsed.values())
    if values and all(isinstance(v, dict) and "class_type" in v for v in values):
        return "comfyui-api"
    # Fooocus presets save under top-level keys like "prompt", "negative_prompt",
    # "performance_selection", "base_model", "loras". The "performance_selection"
    # key is the most distinctive.
    if any(k in parsed for k in ("performance_selection", "base_model_name", "default_loras")):
        return "fooocus"
    # SwarmUI session/preset JSON has a "params" or "rawInput" with a
    # "model"/"prompt" hierarchy under top-level keys like "session_id",
    # "rawinput_overrides", or "presets".
    if any(k in parsed for k in ("rawInput", "rawinput", "session_id", "swarm_session")):
        return "swarmui"
    # InvokeAI graph schema uses "graph" → "nodes" with "type": "<node>".
    if "graph" in parsed and isinstance(parsed["graph"], dict) and "nodes" in parsed["graph"]:
        return "invokeai"
    # Krita-AI plugin saves under "version" + "kind" + "checkpoint" with
    # "loras" array and "scheduler"/"sampler".
    if "kind" in parsed and "checkpoint" in parsed and "version" in parsed:
        return "krita-ai"
    return "unknown"


def _looks_like_a1111(text: str) -> bool:
    """Heuristic for an A1111/Forge parameter dump.

    A1111 PNGInfo / .txt format:

        positive prompt
        Negative prompt: ...
        Steps: 30, Sampler: DPM++ 2M, CFG scale: 7, Seed: 12345, Size: 512x512, ...

    We require the metadata line (``Steps:`` or ``Sampler:`` or ``CFG scale:``)
    AND at least one prompt section to avoid matching arbitrary text.
    """
    has_metadata_line = bool(_A1111_FIELD_RE.search(text))
    has_negative_prompt = "Negative prompt:" in text
    # Either a metadata line + prompt-shaped content, or both a positive and
    # negative prompt header.
    return has_metadata_line and (has_negative_prompt or len(text.splitlines()) >= 2)


# ── A1111 / Forge parser ──────────────────────────────────────────────────────


def parse_a1111_dump(text: str) -> dict[str, Any]:
    """Parse an A1111/Forge parameter dump into a structured dict.

    Returns keys: positive, negative, steps, sampler, scheduler, cfg_scale,
    seed, width, height, model, model_hash, vae, loras (list of (name, weight)),
    denoising_strength.
    """
    text = text.lstrip("﻿")
    lines = text.splitlines()

    # Find the negative-prompt header and the metadata line (last line that
    # parses as comma-separated K: V pairs).
    neg_idx = None
    meta_idx = None
    for i, line in enumerate(lines):
        if neg_idx is None and line.startswith("Negative prompt:"):
            neg_idx = i
        if _A1111_FIELD_RE.search(line) and line.count(":") >= 2:
            meta_idx = i

    if meta_idx is None:
        meta_idx = len(lines)

    positive_end = neg_idx if neg_idx is not None else meta_idx
    positive = "\n".join(lines[:positive_end]).strip()

    if neg_idx is not None:
        negative = lines[neg_idx][len("Negative prompt:"):].strip()
        if meta_idx > neg_idx + 1:
            negative = (negative + "\n" + "\n".join(lines[neg_idx + 1:meta_idx])).strip()
    else:
        negative = ""

    meta_line = lines[meta_idx] if meta_idx < len(lines) else ""
    meta = _split_a1111_metadata_line(meta_line)

    out: dict[str, Any] = {
        "positive": positive,
        "negative": negative,
        "steps": _maybe_int(meta.get("Steps")),
        "sampler": meta.get("Sampler"),
        "scheduler": meta.get("Schedule type"),
        "cfg_scale": _maybe_float(meta.get("CFG scale")),
        "seed": _maybe_int(meta.get("Seed")),
        "model": meta.get("Model"),
        "model_hash": meta.get("Model hash"),
        "vae": meta.get("VAE"),
        "denoising_strength": _maybe_float(meta.get("Denoising strength")),
        "loras": _parse_lora_hashes(meta.get("Lora hashes") or ""),
    }
    if meta.get("Size"):
        m = re.match(r"\s*(\d+)\s*x\s*(\d+)", meta["Size"])
        if m:
            out["width"], out["height"] = int(m.group(1)), int(m.group(2))
    return out


def _split_a1111_metadata_line(line: str) -> dict[str, str]:
    """Split A1111's metadata line into K: V pairs respecting quoted values.

    The line uses commas as separators but values can be quoted strings that
    themselves contain commas (e.g. ``Lora hashes: "a:1, b:2"``).
    """
    out: dict[str, str] = {}
    i, n = 0, len(line)
    while i < n:
        # Read key up to ':'
        key_start = i
        while i < n and line[i] != ":":
            i += 1
        if i >= n:
            break
        key = line[key_start:i].strip().lstrip(",").strip()
        i += 1  # skip ':'
        # Skip whitespace
        while i < n and line[i] == " ":
            i += 1
        # Read value: either quoted or up to next top-level comma
        if i < n and line[i] == '"':
            i += 1
            val_start = i
            while i < n and line[i] != '"':
                i += 1
            value = line[val_start:i]
            i += 1  # skip closing quote
            # Skip until next comma
            while i < n and line[i] != ",":
                i += 1
        else:
            val_start = i
            while i < n and line[i] != ",":
                i += 1
            value = line[val_start:i].strip()
        if key:
            out[key] = value
        if i < n and line[i] == ",":
            i += 1
    return out


def _parse_lora_hashes(s: str) -> list[tuple[str, float]]:
    """Parse ``Lora hashes: "name1: hash1, name2: hash2"`` into a name list.

    A1111 doesn't record per-LoRA strength here (that's in the prompt as
    ``<lora:name:weight>``). Strength defaults to 1.0; the prompt-embedded
    references will be parsed separately.
    """
    if not s:
        return []
    out: list[tuple[str, float]] = []
    for chunk in s.split(","):
        if ":" in chunk:
            name = chunk.split(":", 1)[0].strip()
            if name:
                out.append((name, 1.0))
    return out


def _extract_prompt_loras(prompt: str) -> tuple[str, list[tuple[str, float]]]:
    """Extract ``<lora:name:weight>`` references from a prompt.

    Returns the prompt with the references stripped, plus a list of
    ``(name, weight)`` tuples. A1111 records strengths inline like this.
    """
    pat = re.compile(r"<lora:([^:>]+)(?::([\-+0-9.]+))?(?::[^>]*)?>", re.IGNORECASE)
    loras: list[tuple[str, float]] = []

    def _sub(m: re.Match) -> str:
        name = m.group(1).strip()
        weight = float(m.group(2)) if m.group(2) else 1.0
        loras.append((name, weight))
        return ""

    cleaned = pat.sub(_sub, prompt)
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip(" ,\t\n")
    return cleaned, loras


def _maybe_int(s: Any) -> int | None:
    if s is None:
        return None
    try:
        return int(str(s).strip())
    except (ValueError, TypeError):
        return None


def _maybe_float(s: Any) -> float | None:
    if s is None:
        return None
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return None


# ── Fooocus parser ────────────────────────────────────────────────────────────


def parse_fooocus_preset(parsed: dict) -> dict[str, Any]:
    """Translate a Fooocus preset/JSON into the same dict shape as A1111.

    Fooocus presets carry a wider set of params (style sets, refiner) but the
    core txt2img path is the same: prompt, negative, steps, sampler, cfg,
    seed, dimensions, base model + LoRAs.
    """
    out: dict[str, Any] = {
        "positive": parsed.get("prompt") or "",
        "negative": parsed.get("negative_prompt") or "",
        "steps": _maybe_int(parsed.get("performance_selection_to_steps") or parsed.get("steps")),
        "sampler": parsed.get("sampler") or parsed.get("sampler_name"),
        "scheduler": parsed.get("scheduler") or parsed.get("scheduler_name"),
        "cfg_scale": _maybe_float(parsed.get("guidance_scale") or parsed.get("cfg_scale")),
        "seed": _maybe_int(parsed.get("seed")),
        "width": _maybe_int(parsed.get("aspect_ratios_selection_width") or parsed.get("width")),
        "height": _maybe_int(parsed.get("aspect_ratios_selection_height") or parsed.get("height")),
        "model": parsed.get("base_model_name") or parsed.get("base_model"),
        "vae": parsed.get("vae_name"),
    }
    loras_raw = parsed.get("default_loras") or parsed.get("loras") or []
    loras: list[tuple[str, float]] = []
    if isinstance(loras_raw, list):
        for entry in loras_raw:
            if isinstance(entry, dict):
                name = entry.get("model") or entry.get("name")
                weight = entry.get("weight") or 1.0
                if name:
                    loras.append((str(name), float(weight)))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                # ["enabled", "name.safetensors", 1.0, 1.0] (Fooocus 2.x)
                if len(entry) >= 3 and (entry[0] in (True, "True", "true", 1)):
                    loras.append((str(entry[1]), float(entry[2])))
                elif len(entry) >= 2 and isinstance(entry[0], str) and entry[0].endswith(".safetensors"):
                    loras.append((str(entry[0]), float(entry[1])))
    out["loras"] = loras
    return out


# ── ComfyUI workflow synthesizer ──────────────────────────────────────────────

# Sane defaults for any field the source dump doesn't set. These line up with
# A1111's defaults so that a bare prompt produces a reasonable image.
_DEFAULT_SAMPLER = "euler"
_DEFAULT_SCHEDULER = "normal"
_DEFAULT_STEPS = 20
_DEFAULT_CFG = 7.0
_DEFAULT_WIDTH = 1024
_DEFAULT_HEIGHT = 1024
_DEFAULT_SEED = 0

# A1111 sampler name → ComfyUI KSampler (sampler_name, scheduler) tuple.
# A1111 packs the scheduler into the sampler name; ComfyUI splits them.
_A1111_SAMPLER_MAP: dict[str, tuple[str, str]] = {
    "euler": ("euler", "normal"),
    "euler a": ("euler_ancestral", "normal"),
    "euler ancestral": ("euler_ancestral", "normal"),
    "lms": ("lms", "normal"),
    "heun": ("heun", "normal"),
    "dpm2": ("dpm_2", "normal"),
    "dpm2 a": ("dpm_2_ancestral", "normal"),
    "dpm++ 2s a": ("dpmpp_2s_ancestral", "normal"),
    "dpm++ 2m": ("dpmpp_2m", "normal"),
    "dpm++ 2m karras": ("dpmpp_2m", "karras"),
    "dpm++ 2m sde": ("dpmpp_2m_sde", "normal"),
    "dpm++ 2m sde karras": ("dpmpp_2m_sde", "karras"),
    "dpm++ sde": ("dpmpp_sde", "normal"),
    "dpm++ sde karras": ("dpmpp_sde", "karras"),
    "dpm++ 3m sde": ("dpmpp_3m_sde", "normal"),
    "dpm++ 3m sde karras": ("dpmpp_3m_sde", "karras"),
    "ddim": ("ddim", "normal"),
    "uni pc": ("uni_pc", "normal"),
    "lcm": ("lcm", "normal"),
}


def _map_a1111_sampler(sampler: str | None, scheduler: str | None) -> tuple[str, str]:
    if not sampler:
        return _DEFAULT_SAMPLER, _DEFAULT_SCHEDULER
    key = sampler.strip().lower()
    if key in _A1111_SAMPLER_MAP:
        s, sc = _A1111_SAMPLER_MAP[key]
        # A separate "Schedule type" field overrides the suffix.
        if scheduler:
            sc_lower = scheduler.strip().lower()
            if sc_lower in {"normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"}:
                sc = sc_lower
        return s, sc
    return _DEFAULT_SAMPLER, _DEFAULT_SCHEDULER


def synthesize_comfyui_workflow(params: dict[str, Any]) -> dict:
    """Build a generic ComfyUI API-form workflow from parsed A1111/Fooocus params.

    Produces a CheckpointLoaderSimple → (LoraLoader)* → CLIPTextEncode×2
    → KSampler → VAEDecode → SaveImage chain.
    """
    positive_text, embedded_loras = _extract_prompt_loras(params.get("positive") or "")
    negative_text = params.get("negative") or ""
    loras: list[tuple[str, float]] = []
    seen_lora_names: set[str] = set()
    # Inline ``<lora:name:weight>`` references carry precise per-LoRA strengths;
    # the metadata-line ``Lora hashes:`` only records names with weight 1.0.
    # Process embedded first so dedup keeps the more-specific weight.
    for name, weight in embedded_loras + (params.get("loras") or []):
        nm = str(name).strip()
        if not nm:
            continue
        # Ensure .safetensors extension for ComfyUI loader compatibility.
        if not nm.lower().endswith((".safetensors", ".ckpt", ".pt", ".bin")):
            nm = nm + ".safetensors"
        if nm in seen_lora_names:
            continue
        seen_lora_names.add(nm)
        loras.append((nm, float(weight)))

    model = params.get("model") or "model.safetensors"
    if not model.lower().endswith((".safetensors", ".ckpt")):
        model = model + ".safetensors"

    sampler_name, scheduler = _map_a1111_sampler(params.get("sampler"), params.get("scheduler"))
    steps = params.get("steps") or _DEFAULT_STEPS
    cfg = params.get("cfg_scale") or _DEFAULT_CFG
    seed = params.get("seed") if params.get("seed") is not None else _DEFAULT_SEED
    width = params.get("width") or _DEFAULT_WIDTH
    height = params.get("height") or _DEFAULT_HEIGHT
    denoise = params.get("denoising_strength") or 1.0

    workflow: dict[str, dict] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model},
        },
    }

    # MODEL/CLIP outputs come from node 1 by default; chain LoraLoaders.
    model_src: list = ["1", 0]
    clip_src: list = ["1", 1]
    next_id = 2
    for lora_name, lora_weight in loras:
        nid = str(next_id)
        workflow[nid] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora_name,
                "strength_model": lora_weight,
                "strength_clip": lora_weight,
                "model": model_src,
                "clip": clip_src,
            },
        }
        model_src = [nid, 0]
        clip_src = [nid, 1]
        next_id += 1

    pos_id = str(next_id); next_id += 1
    neg_id = str(next_id); next_id += 1
    latent_id = str(next_id); next_id += 1
    sampler_id = str(next_id); next_id += 1
    decode_id = str(next_id); next_id += 1
    save_id = str(next_id); next_id += 1

    workflow[pos_id] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": positive_text, "clip": clip_src},
    }
    workflow[neg_id] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative_text, "clip": clip_src},
    }
    workflow[latent_id] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
    }
    workflow[sampler_id] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": float(denoise),
            "model": model_src,
            "positive": [pos_id, 0],
            "negative": [neg_id, 0],
            "latent_image": [latent_id, 0],
        },
    }
    workflow[decode_id] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [sampler_id, 0], "vae": ["1", 2]},
    }
    workflow[save_id] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI", "images": [decode_id, 0]},
    }
    return workflow


# ── Top-level helper ──────────────────────────────────────────────────────────


def translate_foreign_workflow(data: bytes | str | dict, *, source: str | None = None) -> dict:
    """Detect format and translate to ComfyUI API workflow, or raise.

    Returns a ComfyUI API-form workflow dict for translatable formats
    (a1111, fooocus). For any other non-ComfyUI format, raises
    :class:`UnsupportedWorkflowFormatError`.
    """
    kind = detect_workflow_format(data)
    if kind in ("comfyui-ui", "comfyui-api"):
        # Caller handles these natively; surface as a flag.
        raise ValueError(f"caller should handle native format: {kind}")
    if kind == "a1111":
        text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data if isinstance(data, str) else ""
        params = parse_a1111_dump(text)
        logger.info("translating A1111 dump → ComfyUI (%d LoRAs)", len(params.get("loras") or []))
        return synthesize_comfyui_workflow(params)
    if kind == "fooocus":
        if isinstance(data, dict):
            parsed = data
        else:
            text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
            parsed = json.loads(text)
        params = parse_fooocus_preset(parsed)
        logger.info("translating Fooocus preset → ComfyUI (%d LoRAs)", len(params.get("loras") or []))
        return synthesize_comfyui_workflow(params)
    raise UnsupportedWorkflowFormatError(kind, source=source)
