from comfy.model_downloader import KNOWN_UNET_MODELS


FLUX2_DEV_INT8_CONVROT = "flux2-dev-int8-convrot-simple.safetensors"
FLUX2_DEV_INT8_CONVROT_REVISION = "14cd543dbc65aa53b21220e0c67c58d6c64bbdb9"


def test_flux2_dev_int8_convrot_is_a_pinned_known_model():
    matches = [
        model
        for model in KNOWN_UNET_MODELS.data
        if str(model) == FLUX2_DEV_INT8_CONVROT
    ]

    assert len(matches) == 1
    model = matches[0]
    assert model.repo_id == "Comfy-Org/flux2-dev"
    assert model.filename == (
        "split_files/diffusion_models/"
        "flux2-dev-int8-convrot-simple.safetensors"
    )
    assert model.revision == FLUX2_DEV_INT8_CONVROT_REVISION
    assert model.size == 33055836040
