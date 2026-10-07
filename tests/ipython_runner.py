"""Execute marked, self-contained test functions as IPython kernel cells."""

import inspect
import json
import sys

import pytest
from jupyter_client import KernelManager


def run_in_ipython(pyfuncitem, marker):
    parameters = getattr(pyfuncitem, "callspec", None)
    parameters = parameters.params if parameters is not None else {}
    arguments = inspect.signature(pyfuncitem.obj).parameters
    if pyfuncitem.cls is not None or set(arguments) - parameters.keys():
        raise pytest.UsageError("ipython tests must be module-level functions using only JSON-serializable parameters, not fixtures")
    kwargs = json.dumps({name: parameters[name] for name in arguments})
    call = f"test_module.{pyfuncitem.obj.__name__}(**json.loads({kwargs!r}))"
    if inspect.iscoroutinefunction(inspect.unwrap(pyfuncitem.obj)):
        call = "await " + call
    code = (
        "import importlib, json\n"
        f"test_module = importlib.import_module({pyfuncitem.module.__name__!r})\n"
        f"{call}\n"
    )

    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    capture_fd = marker.kwargs.get("capture_fd_output", True)
    manager.start_kernel(
        cwd=str(pyfuncitem.config.rootpath),
        extra_arguments=[f"--IPKernelApp.capture_fd_output={capture_fd}"],
    )
    client = manager.blocking_client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=60)
        reply = client.execute_interactive(code, timeout=120, allow_stdin=False)
        if reply["content"]["status"] != "ok":
            pytest.fail("\n".join(reply["content"].get("traceback", [str(reply["content"])])), pytrace=False)
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
