import logging

import pytest

from comfy.app.logger import get_log_level


def test_get_log_level_supports_python_310_logging_api(monkeypatch):
    monkeypatch.delattr(logging, "getLevelNamesMapping", raising=False)

    assert get_log_level("INFO") == logging.INFO
    assert get_log_level("DEBUG") == logging.DEBUG
    with pytest.raises(KeyError):
        get_log_level("NOT_A_LEVEL")
