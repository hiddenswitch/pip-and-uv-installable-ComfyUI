from comfy.model_downloader import KNOWN_HUGGINGFACE_MODEL_REPOS, _known_models_db, get_huggingface_repo_list
from comfy.nodes.base_nodes import DiffusersLoader


IDEOGRAM4_MODEL_NAMES = {
    "ideogram4_fp8_scaled.safetensors",
    "ideogram4_nvfp4_mixed.safetensors",
    "ideogram4_unconditional_fp8_scaled.safetensors",
    "ideogram4_unconditional_nvfp4_mixed.safetensors",
    "qwen3vl_8b_fp8_scaled.safetensors",
}

IDEOGRAM4_DIFFUSERS_REPOS = {
    "ideogram-ai/ideogram-4-nf4",
    "ideogram-ai/ideogram-4-fp8",
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


def test_ideogram4_split_files_are_known_models():
    assert IDEOGRAM4_MODEL_NAMES <= _known_names()


def test_ideogram4_diffusers_repos_are_known_models():
    assert IDEOGRAM4_DIFFUSERS_REPOS <= KNOWN_HUGGINGFACE_MODEL_REPOS


def test_ideogram4_diffusers_repos_are_exposed_to_repo_list():
    assert IDEOGRAM4_DIFFUSERS_REPOS <= set(get_huggingface_repo_list())


def test_ideogram4_diffusers_repos_are_exposed_to_loader():
    input_types = DiffusersLoader.INPUT_TYPES()
    model_paths = set(input_types["required"]["model_path"][0])

    assert IDEOGRAM4_DIFFUSERS_REPOS <= model_paths
