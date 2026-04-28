"""Civitai prefetched model index — offline tests using a stubbed _index."""
from comfy import civitai_model_cache as c


def setup_function(_):
    c._enabled = True
    c._index = {
        "realisticVisionV60B1_v51VAE.safetensors": {
            "folder": "checkpoints",
            "url": "https://civitai.com/api/download/models/130072",
            "model_id": 4201, "version_id": 130072,
            "name": "realisticVisionV60B1_v51VAE.safetensors",
        },
        "sdxl_vae.safetensors": {
            "folder": "checkpoints",
            "url": "https://civitai.com/api/download/models/290640?type=VAE&format=SafeTensor",
            "model_id": 1, "version_id": 290640,
            "name": "sdxl_vae.safetensors",
        },
    }


def teardown_function(_):
    c._enabled = False
    c._index = {}


def test_get_model_entry_hits_index():
    e = c.get_model_entry("checkpoints", "realisticVisionV60B1_v51VAE.safetensors")
    assert e is not None
    folder, url = e
    assert folder == "checkpoints"
    assert "civitai.com/api/download/models/130072" in url


def test_get_model_entry_basename_match():
    # Workflow values often include directory prefixes; the basename should match.
    e = c.get_model_entry("checkpoints", "subdir/realisticVisionV60B1_v51VAE.safetensors")
    assert e is not None
    folder, url = e
    assert folder == "checkpoints"
    assert "civitai.com/api/download/models/130072" in url


def test_get_model_entry_windows_path_prefix():
    # Civitai workflow uploads often use Windows-style backslashes for subdirs.
    e = c.get_model_entry("checkpoints", "SD1.5\\realisticVisionV60B1_v51VAE.safetensors")
    assert e is not None
    folder, _ = e
    assert folder == "checkpoints"


def test_get_model_entry_misses_unknown_filename():
    assert c.get_model_entry("checkpoints", "nope.safetensors") is None


def test_get_model_entry_disabled_returns_none():
    c._enabled = False
    assert c.get_model_entry("checkpoints", "sdxl_vae.safetensors") is None


def test_entry_to_downloadable_returns_url_file():
    from comfy.model_downloader_types import UrlFile
    e = ("checkpoints", "https://civitai.com/api/download/models/130072")
    d = c.entry_to_downloadable(e, "myname.safetensors")
    assert isinstance(d, UrlFile)
    assert d.save_with_filename == "myname.safetensors"
