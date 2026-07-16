from comfy.model_downloader import _known_models_db


QWEN_IMAGE_LAYERED_MODEL_NAMES = {
    "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "qwen_image_layered_bf16.safetensors",
    "qwen_image_layered_fp8mixed.safetensors",
    "qwen_image_layered_int8convrot.safetensors",
    "qwen_image_layered_vae.safetensors",
}


def test_qwen_image_layered_files_are_known_models():
    known = {
        str(item)
        for database in _known_models_db
        for item in database.data
    }

    assert QWEN_IMAGE_LAYERED_MODEL_NAMES <= known
