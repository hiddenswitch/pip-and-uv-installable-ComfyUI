"""URL → fsspec URI canonicalizer."""
from comfy.component_model.uri_rewrite import canonicalize_uri


def test_civitai_model_url():
    assert canonicalize_uri("https://civitai.com/models/12345") == "civitai://m/12345"


def test_civitai_model_url_with_version():
    assert canonicalize_uri(
        "https://civitai.com/models/12345?modelVersionId=67890"
    ) == "civitai://v/67890"


def test_civitai_red_model_url():
    assert canonicalize_uri("https://civitai.red/models/12345") == "civitai-red://m/12345"


def test_civitai_api_download_url():
    assert canonicalize_uri(
        "https://civitai.com/api/download/models/67890"
    ) == "civitai://v/67890"


def test_huggingface_blob_url():
    assert canonicalize_uri(
        "https://huggingface.co/owner/repo/blob/main/path/file.json"
    ) == "hf://owner/repo/path/file.json"


def test_huggingface_resolve_url():
    assert canonicalize_uri(
        "https://huggingface.co/owner/repo/resolve/main/file.safetensors"
    ) == "hf://owner/repo/file.safetensors"


def test_huggingface_repo_url():
    assert canonicalize_uri("https://huggingface.co/owner/repo") == "hf://owner/repo"


def test_youtube_watch_url():
    assert canonicalize_uri("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "youtube://dQw4w9WgXcQ"


def test_youtube_short_url():
    assert canonicalize_uri("https://youtu.be/dQw4w9WgXcQ") == "youtube://dQw4w9WgXcQ"


def test_unknown_url_passthrough():
    assert canonicalize_uri("https://example.com/foo") == "https://example.com/foo"


def test_already_canonical_passthrough():
    assert canonicalize_uri("civitai://m/12345") == "civitai://m/12345"


def test_non_url_passthrough():
    assert canonicalize_uri("local/path.json") == "local/path.json"


def test_civitai_fsspec_registers():
    import fsspec
    from comfy.component_model import civitai_fsspec  # noqa: F401  (registers)
    fs_civ = fsspec.filesystem("civitai")
    fs_red = fsspec.filesystem("civitai-red")
    assert type(fs_civ).__name__ == "CivitaiFileSystem"
    assert type(fs_red).__name__ == "CivitaiFileSystem"
    assert fs_red._scheme == "civitai-red"
