"""PR #67: faulthandler must work with a notebook's redirected stderr."""

import asyncio
import faulthandler
import io
import signal
import sys

import pytest

from comfy.cli_args_types import Configuration
from comfy.component_model.setup import setup_debug_hang


@pytest.mark.ipython(capture_fd_output=False)
@pytest.mark.parametrize("debug_hang", [False, True])
async def test_setup_debug_hang_in_notebook(debug_hang):
    assert asyncio.get_running_loop().is_running()
    assert sys.modules["IPython"].get_ipython().kernel is not None
    assert isinstance(sys.stderr, sys.modules["ipykernel.iostream"].OutStream)
    with pytest.raises(io.UnsupportedOperation):
        sys.stderr.fileno()

    setup_debug_hang(Configuration(debug_hang=debug_hang))
    assert faulthandler.is_enabled()


@pytest.mark.ipython(capture_fd_output=False)
def test_debug_hang_interrupt_in_notebook():
    setup_debug_hang(Configuration(debug_hang=True))
    with pytest.raises(KeyboardInterrupt):
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
