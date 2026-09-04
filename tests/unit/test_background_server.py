import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests import conftest


def test_background_server_startup_timeout_is_configurable(monkeypatch):
    monkeypatch.setenv("COMFYUI_TEST_SERVER_STARTUP_TIMEOUT", "180")

    assert conftest.server_startup_timeout_seconds() == 180


def test_background_server_startup_timeout_must_be_positive(monkeypatch):
    monkeypatch.setenv("COMFYUI_TEST_SERVER_STARTUP_TIMEOUT", "0")

    with pytest.raises(ValueError, match="must be positive"):
        conftest.server_startup_timeout_seconds()


@pytest.mark.parametrize(
    ("fixtures", "expected"),
    [
        ({"tmp_path"}, False),
        ({"tmp_path", "comfy_url_and_proc"}, True),
        ({"process_startup_timeout_seconds"}, True),
        ({"manager_enabled_server"}, True),
    ],
)
def test_process_isolated_fixtures_are_serialized(fixtures, expected):
    assert conftest.requires_serial_process_group(fixtures) is expected


def test_background_server_uses_bounded_readiness_probe_and_cleans_up(monkeypatch):
    process = MagicMock()
    process.poll.return_value = None
    connection = MagicMock()
    connection.__enter__.return_value = connection
    create_connection = MagicMock(return_value=connection)

    monkeypatch.setattr(conftest.subprocess, "Popen", MagicMock(return_value=process))
    monkeypatch.setattr(conftest.socket, "create_connection", create_connection)

    server = conftest.comfy_background_server_from_config(
        SimpleNamespace(listen="127.0.0.1", port=8188)
    )
    _configuration, returned_process = next(server)
    server.close()

    assert returned_process is process
    create_connection.assert_called_once_with(("127.0.0.1", 8188), timeout=1)
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=10)


def test_background_server_reports_early_exit_and_cleans_up(monkeypatch):
    process = MagicMock()
    process.poll.return_value = 17
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(conftest.subprocess, "Popen", popen)

    server = conftest.comfy_background_server_from_config(
        SimpleNamespace(listen="127.0.0.1", port=8188)
    )

    with pytest.raises(RuntimeError, match="exited during startup with code 17"):
        next(server)

    config_path = pathlib.Path(popen.call_args.args[0][-1])
    assert not config_path.exists()
    process.terminate.assert_not_called()
