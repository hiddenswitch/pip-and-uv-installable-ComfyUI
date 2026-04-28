"""Mining model download URLs from workflow Note / MarkdownNote nodes."""
from __future__ import annotations

import pytest

from comfy.component_model.workflow_note_models import (
    NoteModel,
    extract_models_from_notes,
    _normalize_huggingface_url,
    _looks_like_model_url,
    _infer_folder,
)


def _wf(note_text: str) -> dict:
    """Wrap *note_text* in a UI workflow with a single MarkdownNote node."""
    return {
        "nodes": [{"id": 1, "type": "MarkdownNote", "widgets_values": [note_text]}],
        "links": [],
    }


def test_basic_markdown_link_with_place_in_hint():
    wf = _wf(
        "## Diffusion Model\n"
        "*Place in:* `diffusion_models`\n\n"
        "- [my_model.safetensors](https://huggingface.co/foo/bar/resolve/main/my_model.safetensors)\n"
    )
    out = extract_models_from_notes(wf)
    assert len(out) == 1
    assert out[0].filename == "my_model.safetensors"
    assert out[0].folder == "diffusion_models"
    assert "my_model.safetensors" in out[0].url


def test_huggingface_blob_normalized_to_resolve():
    url = "https://huggingface.co/foo/bar/blob/main/x.safetensors"
    assert _normalize_huggingface_url(url) == "https://huggingface.co/foo/bar/resolve/main/x.safetensors"


def test_skips_non_model_urls():
    wf = _wf(
        "See https://example.com/docs for setup\n"
        "And https://github.com/user/repo (a code repo)\n"
    )
    assert extract_models_from_notes(wf) == []


def test_recognizes_civitai_download_url():
    wf = _wf("Get it from [civitai](https://civitai.com/api/download/models/12345)")
    # No filename — bare URL detection only kicks in for URLs whose path ends
    # in a model extension. The civitai download URL has no extension, but
    # _looks_like_model_url returns True; basename is the version_id which
    # isn't a useful filename.
    assert _looks_like_model_url("https://civitai.com/api/download/models/12345")


def test_link_text_filename_takes_precedence_over_url_basename():
    # Workflow authors often link with a friendly name; we want the friendly
    # name as the resolved filename if it's a real model file.
    wf = _wf(
        "*Place in:* `loras`\n\n"
        "- [my_lora.safetensors](https://huggingface.co/foo/bar/resolve/main/some_other_name.safetensors)\n"
    )
    out = extract_models_from_notes(wf)
    assert len(out) == 1
    assert out[0].filename == "my_lora.safetensors"
    # URL basename is registered as an alternate so workflow values referencing
    # the original HF filename still resolve.
    assert out[0].alternate_names == ("some_other_name.safetensors",)


def test_multiple_sections_get_distinct_place_in_hints():
    wf = _wf(
        "## Diffusion\n"
        "*Place in:* `diffusion_models`\n\n"
        "- [a.safetensors](https://huggingface.co/x/y/resolve/main/a.safetensors)\n\n"
        "---\n\n"
        "## LoRAs\n"
        "*Place in:* `loras`\n\n"
        "- [b.safetensors](https://huggingface.co/x/y/resolve/main/b.safetensors)\n"
    )
    out = extract_models_from_notes(wf)
    assert len(out) == 2
    by_name = {m.filename: m for m in out}
    assert by_name["a.safetensors"].folder == "diffusion_models"
    assert by_name["b.safetensors"].folder == "loras"


def test_extension_inference_when_no_place_in():
    wf = _wf("- [up.pth](https://example.com/up.pth)\n")
    out = extract_models_from_notes(wf)
    assert len(out) == 1
    # .pth defaults to upscale_models per _EXT_TO_FOLDER
    assert out[0].folder == "upscale_models"


def test_dedup_same_filename_across_sections():
    wf = _wf(
        "*Place in:* `diffusion_models`\n\n"
        "- [m.safetensors](https://huggingface.co/a/b/resolve/main/m.safetensors)\n\n"
        "---\n\n"
        "*Place in:* `loras`\n\n"
        "- [m.safetensors](https://huggingface.co/c/d/resolve/main/m.safetensors)\n"
    )
    out = extract_models_from_notes(wf)
    assert len(out) == 1
    # First occurrence wins
    assert out[0].folder == "diffusion_models"


def test_handles_dict_widgets_values():
    wf = {
        "nodes": [{
            "id": 1, "type": "MarkdownNote",
            "widgets_values": {"text": "[m.gguf](https://example.com/m.gguf)"},
        }],
        "links": [],
    }
    out = extract_models_from_notes(wf)
    assert len(out) == 1
    assert out[0].filename == "m.gguf"


def test_skips_non_workflow_input():
    assert extract_models_from_notes({}) == []
    assert extract_models_from_notes({"nodes": []}) == []


def test_infer_folder_from_filename_hints():
    # VAE keyword wins regardless of extension
    assert _infer_folder("some_vae_thing.safetensors", "", None) == "vae"
    assert _infer_folder("t5xxl_fp16.safetensors", "", None) == "text_encoders"
    assert _infer_folder("ip-adapter_sdxl.safetensors", "", None) == "ipadapter"
    assert _infer_folder("4xNomos8k.pth", "", None) == "upscale_models"
    assert _infer_folder("my_lora_v1.safetensors", "lora", None) == "loras"


def test_real_world_ltx2_pattern():
    """Reproduce the actual LTX-2 19B Q4_K_M MarkdownNote shape."""
    wf = _wf("""# LTXV-2 Model Files & Dependencies

## Diffusion Model
**LTXV-2 DEV GGUF Q4_K_M**
*Place in:* `diffusion_models`

- [ltx-2-19b-dev_Q4_K_M.gguf](https://huggingface.co/QuantStack/LTX-2-GGUF/blob/main/LTX-2-dev/LTX-2-dev-Q4_K_M.gguf)

---

## Distilled LoRA
**LTX-2 19B DISTILLED LORA**
*Place in:* `loras`

- [ltx-2-19b-distilled-lora](https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/loras/ltx-2-19b-distilled-lora_resized_dynamic_fro09_avg_rank_175_fp8.safetensors)
""")
    out = extract_models_from_notes(wf)
    assert len(out) == 2
    diffusion = next(m for m in out if "Q4_K_M" in m.filename)
    assert diffusion.filename == "ltx-2-19b-dev_Q4_K_M.gguf"
    assert diffusion.folder == "diffusion_models"
    # blob → resolve normalization
    assert "/resolve/main/" in diffusion.url
    lora = next(m for m in out if m.folder == "loras")
    # Link label has no extension so URL basename wins
    assert lora.filename == "ltx-2-19b-distilled-lora_resized_dynamic_fro09_avg_rank_175_fp8.safetensors"
