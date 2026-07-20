from comfy.model_downloader import _known_models_db


JOYIMAGE_MODEL_NAMES = {
    "joyai_image_edit_bf16.safetensors",
    "joyai_image_edit_int8_convrot.safetensors",
    "qwen3vl_8b_joyimage_edit_bf16.safetensors",
    "qwen3vl_8b_joyimage_edit_int8_convrot.safetensors",
    "wan_2.1_vae.safetensors",
}


def test_joyimage_files_are_known_models():
    known = {
        str(item)
        for database in _known_models_db
        for item in database.data
    }

    assert JOYIMAGE_MODEL_NAMES <= known
