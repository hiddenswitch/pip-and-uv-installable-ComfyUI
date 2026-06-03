import json
from pathlib import Path

from comfy.cmd.workflow_templates import get_all_templates
from comfy.model_downloader import _known_models_db


_SUPPORTED_URL_MARKERS = (
    "/split_files/diffusion_models/",
    "/split_files/loras/",
    "/split_files/vae/",
    "/split_files/text_encoders/",
    "/split_files/clip_vision/",
    "/split_files/controlnet/",
    "/split_files/latent_upscale_models/",
    "/split_files/audio_encoders/",
    "/diffusion_models/",
    "/text_encoders/",
    "/vae/",
    "/diffusion_models/ltx-2-19b-distilled_transformer_only_bf16.safetensors",
    "/checkpoints/ace_step_1.5_turbo_aio.safetensors",
    "/checkpoints/sdpose_wholebody_fp16.safetensors",
    "/lotus-depth-d-v1-1.safetensors",
    "/RealESRGAN_x4plus.safetensors",
)

_SUPPORTED_REPO_MARKERS = (
    "huggingface.co/Comfy-Org/",
    "huggingface.co/Kijai/",
    "huggingface.co/onnx-community/",
    "huggingface.co/JunkyByte/",
)

_WIDGET_ONLY_EXPECTED = {
    "video_wanmove_480p": {
        "Wan21-WanMove_fp8_scaled_e4m3fn_KJ.safetensors",
        "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
    },
    "video_wan2_1_infinitetalk": {
        "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
    },
}


def _known_names() -> set[str]:
    known: set[str] = set()
    for db in _known_models_db:
        for item in db.data:
            known.add(str(item))
            known.update(getattr(item, "alternate_filenames", ()))
            save_with_filename = getattr(item, "save_with_filename", None)
            if save_with_filename:
                known.add(save_with_filename)
    return known


def _iter_machine_readable_assets(workflow: object):
    if isinstance(workflow, dict):
        if isinstance(workflow.get("name"), str) and isinstance(workflow.get("url"), str):
            yield workflow["name"], workflow["url"]
        for value in workflow.values():
            yield from _iter_machine_readable_assets(value)
    elif isinstance(workflow, list):
        for value in workflow:
            yield from _iter_machine_readable_assets(value)


def _iter_widget_values(workflow: object):
    if isinstance(workflow, dict):
        widgets_values = workflow.get("widgets_values")
        if isinstance(widgets_values, list):
            for value in widgets_values:
                if isinstance(value, str):
                    yield value
        for value in workflow.values():
            yield from _iter_widget_values(value)
    elif isinstance(workflow, list):
        for value in workflow:
            yield from _iter_widget_values(value)


def test_supported_package_workflow_assets_are_known_models():
    known = _known_names()
    missing: list[tuple[str, str, str]] = []

    for template in get_all_templates():
        if template.source != "package" or not template.path:
            continue

        workflow = json.loads(Path(template.path).read_text(encoding="utf-8"))
        for name, url in _iter_machine_readable_assets(workflow):
            if (
                any(marker in url for marker in _SUPPORTED_REPO_MARKERS)
                and any(marker in url for marker in _SUPPORTED_URL_MARKERS)
                and name not in known
            ):
                missing.append((template.template_id or template.name, name, url))

    assert missing == []


def test_widget_only_workflow_model_names_are_known():
    templates = {
        template.template_id: json.loads(Path(template.path).read_text(encoding="utf-8"))
        for template in get_all_templates()
        if template.source == "package" and template.path and template.template_id in _WIDGET_ONLY_EXPECTED
    }
    known = _known_names()

    for template_id, expected_names in _WIDGET_ONLY_EXPECTED.items():
        assert template_id in templates

        widgets_values = set(_iter_widget_values(templates[template_id]))

        for expected_name in expected_names:
            assert expected_name in widgets_values
            assert expected_name in known
