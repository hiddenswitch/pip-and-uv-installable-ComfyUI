"""Mine Note / MarkdownNote nodes for model download URLs.

Civitai workflow authors put the download URLs for the files their workflow
needs *inside* the workflow itself, in markdown-formatted Note nodes. The
recurring pattern looks like:

    ## Diffusion Model
    **LTX-2 19B Q4_K_M**
    *Place in:* `diffusion_models`

    - [ltx-2-19b-dev_Q4_K_M.gguf](https://huggingface.co/QuantStack/LTX-2-GGUF/...)

This module extracts those URLs along with a folder hint (from ``Place in:``)
and the filename (from the link text or URL basename), so the ``--all``
download path can resolve workflow-specific files even when they aren't in
any static registry.

Output: a list of :class:`NoteModel` records, each suitable for synthesizing
a runtime ``KnownDownloadable`` (UrlFile / HuggingFile / CivitFile) keyed by
filename so existing ``model_downloader.get_or_download`` lookup just works.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoteModel:
    filename: str
    url: str
    folder: str | None = None
    alternate_names: tuple[str, ...] = field(default_factory=tuple)


# Recognized model file extensions. Anything ending in these is a download
# target; anything else is just an external link.
_MODEL_EXTENSIONS: frozenset[str] = frozenset({
    ".safetensors", ".sft",
    ".gguf",
    ".ckpt", ".pt", ".bin",
    ".pth", ".onnx",
})

# ComfyUI folder names commonly named in note text. Mapped to the canonical
# folder_paths key.
_FOLDER_ALIASES: dict[str, str] = {
    "diffusion_models": "diffusion_models",
    "diffusion model": "diffusion_models",
    "unet": "diffusion_models",
    "checkpoints": "checkpoints",
    "checkpoint": "checkpoints",
    "models": "checkpoints",
    "loras": "loras",
    "lora": "loras",
    "vae": "vae",
    "vae approx": "vae_approx",
    "vae_approx": "vae_approx",
    "clip": "text_encoders",
    "text_encoders": "text_encoders",
    "text encoder": "text_encoders",
    "text encoders": "text_encoders",
    "clip_vision": "clip_vision",
    "clip vision": "clip_vision",
    "controlnet": "controlnet",
    "controlnets": "controlnet",
    "control_nets": "controlnet",
    "upscale_models": "upscale_models",
    "upscale models": "upscale_models",
    "upscaler": "upscale_models",
    "upscalers": "upscale_models",
    "latent_upscale_models": "latent_upscale_models",
    "latent upscale models": "latent_upscale_models",
    "ipadapter": "ipadapter",
    "embeddings": "embeddings",
    "embedding": "embeddings",
    "animatediff_models": "animatediff_models",
    "ultralytics_bbox": "ultralytics_bbox",
    "ultralytics_segm": "ultralytics_segm",
    "bbox": "ultralytics_bbox",
    "segm": "ultralytics_segm",
}


# Inferred from the file extension when no "Place in:" hint is present.
_EXT_TO_FOLDER: dict[str, str] = {
    ".safetensors": "checkpoints",  # too generic; corrected by other heuristics
    ".sft": "diffusion_models",
    ".gguf": "diffusion_models",
    ".ckpt": "checkpoints",
    ".pt": "ultralytics_bbox",
    ".pth": "upscale_models",
    ".bin": "ipadapter",
    ".onnx": "ultralytics_bbox",
}


# Match a "Place in: <folder>" line, tolerant of markdown emphasis
# (``*Place in:*``, ``**Place in**:``) and backtick / quote wrapping of the
# folder name (`` `diffusion_models` ``, ``"diffusion_models"``).
_PLACE_IN_RE = re.compile(
    r"(?:place\s+in|put\s+in|save\s+to|save\s+in|copy\s+to)"
    r"[\*_:：\s]*"  # any combination of emphasis chars + colon + whitespace
    r"[`\"']?"
    r"([A-Za-z0-9_./\\\- ]+?)"
    r"[`\"']?(?:[\s.,)<]|$)",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
# Standalone URLs (not already inside a markdown link).
_BARE_URL_RE = re.compile(r"(?<!\]\()https?://\S+")


def _looks_like_model_url(url: str) -> bool:
    """Whether *url* points at a downloadable model file.

    Counts huggingface ``/resolve/`` and ``/blob/`` paths whose terminal
    segment has a model extension, civitai ``/api/download/models/<id>``
    URLs (which always serve a model), and any direct URL whose path ends
    in a model extension.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    last = path.rsplit("/", 1)[-1]
    suffix = "." + last.rsplit(".", 1)[-1].lower() if "." in last else ""
    if suffix in _MODEL_EXTENSIONS:
        return True
    if "civitai.com" in host or "civitai.red" in host:
        return "/api/download/models/" in path
    return False


def _basename_from_url(url: str) -> str:
    parsed = urlparse(url)
    last = (parsed.path or "").rstrip("/").rsplit("/", 1)[-1]
    return last or ""


def _normalize_huggingface_url(url: str) -> str:
    """Convert HF ``/blob/main/...`` to ``/resolve/main/...`` for direct download."""
    parsed = urlparse(url)
    if (parsed.hostname or "").lower() not in ("huggingface.co", "www.huggingface.co"):
        return url
    return url.replace("/blob/main/", "/resolve/main/", 1).replace("/blob/", "/resolve/", 1)


def _infer_folder(filename: str, link_text: str, hint: str | None) -> str:
    if hint:
        canonical = _FOLDER_ALIASES.get(hint.strip().lower())
        if canonical:
            return canonical
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    # Heuristic: filenames containing typical role hints disambiguate ext.
    lc = (filename + " " + link_text).lower()
    if any(t in lc for t in ("vae",)):
        return "vae"
    if any(t in lc for t in ("clip_l", "clip-l", "umt5", "t5xxl", "gemma", "qwen", "llama", "text_encoder", "embedding")):
        return "text_encoders"
    if any(t in lc for t in ("clip_vision", "clip-vision", "siglip")):
        return "clip_vision"
    if any(t in lc for t in ("controlnet", "control_net", "control-net")):
        return "controlnet"
    if any(t in lc for t in ("ipadapter", "ip-adapter", "ip_adapter")):
        return "ipadapter"
    if any(t in lc for t in ("upscaler", "esrgan", "swinir", "atd", "skincontrast", "skin-contrast", "realesrgan", "real-esrgan")):
        return "upscale_models"
    if any(t in lc for t in ("lora",)):
        return "loras"
    if any(t in lc for t in ("animatediff", "motion_module")):
        return "animatediff_models"
    return _EXT_TO_FOLDER.get(suffix, "checkpoints")


def _iter_note_texts(workflow: dict) -> Iterable[str]:
    if not isinstance(workflow, dict):
        return
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("type") not in ("Note", "MarkdownNote"):
            continue
        wv = node.get("widgets_values")
        if isinstance(wv, list):
            for w in wv:
                if isinstance(w, str) and w.strip():
                    yield w
        elif isinstance(wv, dict):
            for w in wv.values():
                if isinstance(w, str) and w.strip():
                    yield w


def extract_models_from_notes(workflow: dict) -> list[NoteModel]:
    """Walk *workflow*'s Note/MarkdownNote nodes and return models referenced
    by markdown-link or bare URL.

    Each note is split into "sections" by horizontal rules / blank-line gaps.
    The most-recent ``Place in:`` line in the same section is attributed as
    the folder hint for any links that follow it.
    """
    out: list[NoteModel] = []
    seen_filenames: set[str] = set()
    for text in _iter_note_texts(workflow):
        # Split on horizontal-rule lines (---, ***, ___). Within a section,
        # the most-recent "Place in:" hint applies to all links that follow,
        # even if separated by blank lines or extra prose.
        sections = re.split(r"\n\s*[-*_]{3,}\s*\n", text)
        for section in sections:
            current_hint: str | None = None
            # Walk by line so each link picks up the latest hint preceding it.
            for line in section.split("\n"):
                hint_match = _PLACE_IN_RE.search(line)
                if hint_match:
                    current_hint = hint_match.group(1).strip()

                for m in _MARKDOWN_LINK_RE.finditer(line):
                    label = m.group(1).strip()
                    url = m.group(2).rstrip(".,;)>")
                    if not _looks_like_model_url(url):
                        continue
                    url = _normalize_huggingface_url(url)
                    bn = _basename_from_url(url)
                    fn_candidates: list[str] = []
                    if "." in label and any(label.lower().endswith(ext) for ext in _MODEL_EXTENSIONS):
                        fn_candidates.append(label)
                    if bn and bn not in fn_candidates:
                        fn_candidates.append(bn)
                    if not fn_candidates:
                        continue
                    primary = fn_candidates[0]
                    alternates = tuple(fn_candidates[1:])
                    key = primary.lower()
                    if key in seen_filenames:
                        continue
                    seen_filenames.add(key)
                    folder = _infer_folder(primary, label, current_hint)
                    out.append(NoteModel(filename=primary, url=url, folder=folder, alternate_names=alternates))

                for m in _BARE_URL_RE.finditer(line):
                    url = m.group(0).rstrip(".,;)>")
                    if not _looks_like_model_url(url):
                        continue
                    url = _normalize_huggingface_url(url)
                    bn = _basename_from_url(url)
                    if not bn:
                        continue
                    key = bn.lower()
                    if key in seen_filenames:
                        continue
                    seen_filenames.add(key)
                    folder = _infer_folder(bn, "", current_hint)
                    out.append(NoteModel(filename=bn, url=url, folder=folder))

    return out
