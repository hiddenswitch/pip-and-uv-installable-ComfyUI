from comfy.model_downloader import KNOWN_HUGGINGFACE_MODEL_REPOS, get_huggingface_repo_list
from comfy.nodes.base_nodes import DiffusersLoader


IDEOGRAM4_REPOS = {
    "ideogram-ai/ideogram-4-nf4",
    "ideogram-ai/ideogram-4-fp8",
}


def test_ideogram4_diffusers_repos_are_known_models():
    assert IDEOGRAM4_REPOS <= KNOWN_HUGGINGFACE_MODEL_REPOS


def test_ideogram4_diffusers_repos_are_exposed_to_repo_list():
    assert IDEOGRAM4_REPOS <= set(get_huggingface_repo_list())


def test_ideogram4_diffusers_repos_are_exposed_to_loader():
    input_types = DiffusersLoader.INPUT_TYPES()
    model_paths = set(input_types["required"]["model_path"][0])

    assert IDEOGRAM4_REPOS <= model_paths
