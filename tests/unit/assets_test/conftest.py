"""
Fixtures for assets API tests.

Uses comfy_background_server_from_config from the top-level conftest.py
with a Configuration object instead of raw CLI args.
"""
import contextlib
import json
import socket
import tempfile
from pathlib import Path
from typing import Callable, Iterator, Optional, Generator, Any
from multiprocessing import Process

import pytest
import requests

from comfy.cli_args import default_configuration
from comfy.cli_args_types import Configuration

# Import the server runner from top-level conftest
from tests.conftest import comfy_background_server_from_config


def _find_free_port() -> int:
    """Find a free port by binding to port 0 and reading the assigned port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def _make_base_dirs(root: Path) -> None:
    """Create the standard ComfyUI directory structure."""
    for sub in ("models", "custom_nodes", "input", "output", "temp", "user"):
        (root / sub).mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def comfy_tmp_base_dir() -> Generator[Path, Any, None]:
    """Create a temporary base directory for ComfyUI."""
    tmp = Path(tempfile.mkdtemp(prefix="comfyui-assets-tests-"))
    _make_base_dirs(tmp)
    yield tmp
    with contextlib.suppress(Exception):
        for p in sorted(tmp.rglob("*"), reverse=True):
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
        for p in sorted(tmp.glob("**/*"), reverse=True):
            with contextlib.suppress(Exception):
                p.rmdir()
        tmp.rmdir()


@pytest.fixture(scope="session")
def assets_server_config(comfy_tmp_base_dir: Path) -> Configuration:
    """
    Create a Configuration object for the assets test server.
    Uses default_configuration() and modifies it for assets testing.
    """
    config = default_configuration()

    db_path = comfy_tmp_base_dir / "assets-test.sqlite3"
    db_url = f"sqlite:///{db_path}"

    # Use temp dir as the only base path for models/custom_nodes
    config.cwd = str(comfy_tmp_base_dir)
    config.base_directory = str(comfy_tmp_base_dir)
    config.base_paths = [str(comfy_tmp_base_dir)]
    config.database_url = db_url
    config.disable_assets_autoscan = True
    config.listen = "127.0.0.1"
    config.port = _find_free_port()
    config.cpu = True
    config.output_directory = str(comfy_tmp_base_dir / "output")
    config.input_directory = str(comfy_tmp_base_dir / "input")

    return config


@pytest.fixture(scope="session")
def comfy_url_and_proc(
    assets_server_config: Configuration,
) -> Generator[tuple[str, Process], Any, None]:
    """
    Boot ComfyUI using comfy_background_server_from_config.
    Returns (base_url, process).
    """
    for config, proc in comfy_background_server_from_config(assets_server_config):
        base_url = f"http://{config.listen}:{config.port}"
        yield base_url, proc


@pytest.fixture
def http() -> Iterator[requests.Session]:
    """Provide a requests Session with a default timeout."""
    with requests.Session() as s:
        s.timeout = 120
        yield s


@pytest.fixture
def api_base(comfy_url_and_proc: tuple[str, Process]) -> str:
    """Extract the base URL from comfy_url_and_proc."""
    base_url, _proc = comfy_url_and_proc
    return base_url


def _post_multipart_asset(
    session: requests.Session,
    base: str,
    *,
    name: str,
    tags: list[str],
    meta: dict,
    data: bytes,
    extra_fields: Optional[dict] = None,
) -> tuple[int, dict]:
    """Helper to POST a multipart asset upload."""
    files = {"file": (name, data, "application/octet-stream")}
    form_data = {
        "tags": json.dumps(tags),
        "name": name,
        "user_metadata": json.dumps(meta),
    }
    if extra_fields:
        for k, v in extra_fields.items():
            form_data[k] = v
    r = session.post(base + "/api/assets", files=files, data=form_data, timeout=120)
    return r.status_code, r.json()


@pytest.fixture
def make_asset_bytes() -> Callable[[str, int], bytes]:
    """Factory for creating deterministic test asset bytes."""
    def _make(name: str, size: int = 8192) -> bytes:
        seed = sum(ord(c) for c in name) % 251
        return bytes((i * 31 + seed) % 256 for i in range(size))
    return _make


@pytest.fixture
def asset_factory(http: requests.Session, api_base: str):
    """
    Factory fixture for creating assets.
    Returns create(name, tags, meta, data) -> response dict.
    Tracks created ids and deletes them after the test.
    """
    created: list[str] = []

    def create(name: str, tags: list[str], meta: dict, data: bytes) -> dict:
        status, body = _post_multipart_asset(http, api_base, name=name, tags=tags, meta=meta, data=data)
        assert status in (200, 201), body
        created.append(body["id"])
        return body

    yield create

    for aid in created:
        with contextlib.suppress(Exception):
            http.delete(f"{api_base}/api/assets/{aid}", timeout=30)


@pytest.fixture
def seeded_asset(request: pytest.FixtureRequest, http: requests.Session, api_base: str) -> dict:
    """
    Upload one asset with ".safetensors" extension.
    Returns response dict with id, asset_hash, tags, etc.
    Can be parameterized with {"tags": [...]} to customize tags.
    """
    name = "unit_1_example.safetensors"
    p = getattr(request, "param", {}) or {}
    tags: Optional[list[str]] = p.get("tags")
    if tags is None:
        tags = ["models", "checkpoints", "unit-tests", "alpha"]
    meta = {"purpose": "test", "epoch": 1, "flags": ["x", "y"], "nullable": None}
    files = {"file": (name, b"A" * 4096, "application/octet-stream")}
    form_data = {
        "tags": json.dumps(tags),
        "name": name,
        "user_metadata": json.dumps(meta),
    }
    r = http.post(api_base + "/api/assets", files=files, data=form_data, timeout=120)
    body = r.json()
    assert r.status_code == 201, body
    return body


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets(http: requests.Session, api_base: str):
    """Ensure isolation by removing all AssetInfo rows tagged with 'unit-tests' after each test."""
    yield

    while True:
        r = http.get(
            api_base + "/api/assets",
            params={"include_tags": "unit-tests", "limit": "500", "sort": "name"},
            timeout=30,
        )
        if r.status_code != 200:
            break
        body = r.json()
        ids = [a["id"] for a in body.get("assets", [])]
        if not ids:
            break
        for aid in ids:
            with contextlib.suppress(Exception):
                http.delete(f"{api_base}/api/assets/{aid}", timeout=30)


# Helper functions are in helpers.py for direct import by test files
