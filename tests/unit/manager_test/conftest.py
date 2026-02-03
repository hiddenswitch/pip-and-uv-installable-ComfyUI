"""
Fixtures for comfyui_manager integration tests.

Tests that the manager's REST API endpoints are properly injected
when --enable-manager is set.
"""
import contextlib
import socket
import tempfile
from pathlib import Path
from typing import Generator, Any
from multiprocessing import Process

import pytest
import requests

from comfy.cli_args import default_configuration
from comfy.cli_args_types import Configuration
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
def manager_tmp_base_dir() -> Generator[Path, Any, None]:
    """Create a temporary base directory for ComfyUI with manager enabled."""
    tmp = Path(tempfile.mkdtemp(prefix="comfyui-manager-tests-"))
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
def manager_enabled_config(manager_tmp_base_dir: Path) -> Configuration:
    """
    Create a Configuration with manager enabled.
    """
    config = default_configuration()
    config.base_directory = str(manager_tmp_base_dir)
    config.listen = "127.0.0.1"
    config.port = _find_free_port()
    config.cpu = True
    config.enable_manager = True
    config.disable_manager_ui = False
    config.output_directory = str(manager_tmp_base_dir / "output")
    config.input_directory = str(manager_tmp_base_dir / "input")
    return config


@pytest.fixture(scope="session")
def manager_disabled_config(manager_tmp_base_dir: Path) -> Configuration:
    """
    Create a Configuration with manager disabled.
    """
    config = default_configuration()
    config.base_directory = str(manager_tmp_base_dir)
    config.listen = "127.0.0.1"
    config.port = _find_free_port()
    config.cpu = True
    config.enable_manager = False
    config.output_directory = str(manager_tmp_base_dir / "output")
    config.input_directory = str(manager_tmp_base_dir / "input")
    return config


@pytest.fixture(scope="session")
def manager_enabled_server(
    manager_enabled_config: Configuration,
) -> Generator[tuple[str, Process], Any, None]:
    """
    Boot ComfyUI with manager enabled.
    Returns (base_url, process).
    """
    for config, proc in comfy_background_server_from_config(manager_enabled_config):
        base_url = f"http://{config.listen}:{config.port}"
        yield base_url, proc


@pytest.fixture(scope="session")
def manager_disabled_server(
    manager_disabled_config: Configuration,
) -> Generator[tuple[str, Process], Any, None]:
    """
    Boot ComfyUI with manager disabled.
    Returns (base_url, process).
    """
    for config, proc in comfy_background_server_from_config(manager_disabled_config):
        base_url = f"http://{config.listen}:{config.port}"
        yield base_url, proc


@pytest.fixture(scope="session")
def manager_with_disabled_custom_nodes_config(manager_tmp_base_dir: Path) -> Configuration:
    """
    Create a Configuration with manager enabled but custom nodes disabled.
    """
    config = default_configuration()
    config.base_directory = str(manager_tmp_base_dir)
    config.listen = "127.0.0.1"
    config.port = _find_free_port()
    config.cpu = True
    config.enable_manager = True
    config.disable_all_custom_nodes = True
    config.output_directory = str(manager_tmp_base_dir / "output")
    config.input_directory = str(manager_tmp_base_dir / "input")
    return config


@pytest.fixture(scope="session")
def manager_with_disabled_custom_nodes_server(
    manager_with_disabled_custom_nodes_config: Configuration,
) -> Generator[tuple[str, Process], Any, None]:
    """
    Boot ComfyUI with manager enabled but custom nodes disabled.
    Returns (base_url, process).
    """
    for config, proc in comfy_background_server_from_config(manager_with_disabled_custom_nodes_config):
        base_url = f"http://{config.listen}:{config.port}"
        yield base_url, proc


@pytest.fixture
def http() -> Generator[requests.Session, Any, None]:
    """Provide a requests Session with a default timeout."""
    with requests.Session() as s:
        s.timeout = 30
        yield s
