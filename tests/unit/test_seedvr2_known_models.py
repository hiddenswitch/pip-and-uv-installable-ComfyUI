from comfy.model_downloader import KNOWN_VAES


def test_seedvr2_workflow_vae_and_legacy_name_are_known():
    seedvr2_vae = next(
        item
        for item in KNOWN_VAES.data
        if item.repo_id == "Comfy-Org/SeedVR2"
        and item.filename == "vae/seedvr2_ema_vae_fp16.safetensors"
    )

    assert str(seedvr2_vae) == "seedvr2_ema_vae_fp16.safetensors"
    assert "ema_vae_fp16.safetensors" in seedvr2_vae.alternate_filenames
