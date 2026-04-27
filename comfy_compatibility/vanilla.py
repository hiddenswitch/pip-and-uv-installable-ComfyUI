from __future__ import annotations

import collections.abc
import contextvars
import logging
import os
import subprocess
import sys
import types
from contextlib import contextmanager, nullcontext
from functools import partial
from importlib.util import find_spec
from pathlib import Path
import threading
from threading import RLock
from typing import Dict

import wrapt

logger = logging.getLogger(__name__)

# there isn't a way to do this per-thread, it's only per process, so the global is valid
# we don't want some kind of multiprocessing lock, because this is munging the sys.modules
# wrapt.synchronized will be used to synchronize this


class _NodeClassMappingsShim(collections.abc.Mapping):
    def __init__(self):
        super().__init__()
        self._active = 0
        self._active_lock = RLock()

    def activate(self):
        with self._active_lock:
            self._active += 1

    def deactivate(self):
        with self._active_lock:
            self._active -= 1

    def __iter__(self):
        if self._active > 0:
            from comfy.nodes_context import get_nodes
            for key in get_nodes().NODE_CLASS_MAPPINGS:
                yield key
        else:
            from comfy.nodes.base_nodes import NODE_CLASS_MAPPINGS
            for key in NODE_CLASS_MAPPINGS:
                yield key

    def __getitem__(self, item):
        if self._active > 0:
            from comfy.nodes_context import get_nodes
            return get_nodes().NODE_CLASS_MAPPINGS[item]
        else:
            from comfy.nodes.base_nodes import NODE_CLASS_MAPPINGS
            return NODE_CLASS_MAPPINGS[item]

    def __len__(self):
        if self._active > 0:
            from comfy.nodes_context import get_nodes
            return len(get_nodes().NODE_CLASS_MAPPINGS)
        else:
            from comfy.nodes.base_nodes import NODE_CLASS_MAPPINGS
            return len(NODE_CLASS_MAPPINGS)

    # todo: does this need to be mutable?


class _NodeShim:
    def __init__(self):
        self.__name__ = 'nodes'
        self.__package__ = ''

        nodes_file = None
        try:
            # the 'nodes' module is expected to be in the directory above 'comfy'
            spec = find_spec('comfy')
            if spec and spec.origin:
                comfy_package_path = Path(spec.origin).parent
                nodes_module_dir = comfy_package_path.parent
                nodes_file = str(nodes_module_dir / 'nodes.py')
        except (ImportError, AttributeError):
            # don't do anything exotic
            pass

        self.__file__ = nodes_file
        self.__loader__ = None
        self.__spec__ = None

    def __node_class_mappings(self) -> _NodeClassMappingsShim:
        return getattr(self, "NODE_CLASS_MAPPINGS")

    def activate(self):
        self.__node_class_mappings().activate()

    def deactivate(self):
        self.__node_class_mappings().deactivate()


class _ComfyExtrasRedirectFinder:
    """Meta path finder that redirects ``comfy_extras.<name>`` to ``comfy_extras.nodes.<name>``.

    Vanilla custom nodes import modules like ``comfy_extras.nodes_custom_sampler``
    which in this fork live under ``comfy_extras.nodes.nodes_custom_sampler``.
    """

    def __init__(self):
        self._resolving = threading.local()

    def find_spec(self, fullname: str, path=None, target=None):
        if not fullname.startswith("comfy_extras."):
            return None
        parts = fullname.split(".")
        if len(parts) != 2:
            return None
        short_name = parts[1]
        if short_name == "nodes" or short_name.startswith("_"):
            return None
        canonical = f"comfy_extras.nodes.{short_name}"
        active = getattr(self._resolving, "active", None)
        if active is None:
            active = self._resolving.active = set()
        if canonical in active:
            return None
        active.add(canonical)
        try:
            import importlib.util
            spec = importlib.util.find_spec(canonical)
            if spec is None:
                return None
            return importlib.util.spec_from_loader(
                fullname,
                loader=_ComfyExtrasRedirectLoader(canonical),
                origin=spec.origin,
            )
        except (ModuleNotFoundError, ValueError):
            return None
        finally:
            active.discard(canonical)


class _ComfyExtrasRedirectLoader:
    def __init__(self, canonical: str):
        self._canonical = canonical

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        import importlib
        real = importlib.import_module(self._canonical)
        sys.modules[module.__name__] = real
        import comfy_extras
        setattr(comfy_extras, self._canonical.split(".")[-1], real)


@wrapt.synchronized
def prepare_vanilla_environment():
    # Dedup against the actual end-state, not a separate flag: tests that
    # tear down `sys.modules['nodes']` (e.g. test_nodes_context_shim) need
    # the next call to re-install the shim, not short-circuit.
    if isinstance(sys.modules.get('nodes'), _NodeShim):
        return
    try:
        import comfy  # noqa: F401
    except ModuleNotFoundError:
        logger.debug("comfy not installed, skipping vanilla environment prep")
        return
    from comfy.cmd import cuda_malloc, folder_paths, latent_preview, protocol

    from comfy.distributed.executors import ContextVarExecutor
    from comfy.nodes import base_nodes
    from comfy.nodes.vanilla_node_importing import _PromptServerStub
    from comfy import node_helpers
    from comfy import __version__
    import concurrent.futures
    import threading
    for module in (cuda_malloc, folder_paths, latent_preview, node_helpers, protocol):
        module_short_name = module.__name__.split(".")[-1]
        sys.modules[module_short_name] = module

    # easy-use needs a shim
    # this ensures NODE_CLASS_MAPPINGS is loaded lazily and contains all the nodes loaded so far, not just the base nodes
    # easy-use and other nodes expect NODE_CLASS_MAPPINGS to contain all the nodes in the environment
    # the shim must be activated after importing, which happens in a tightly coupled way
    # todo: it's not clear if we should skip the dunder methods or not
    nodes_shim_dir = {k: getattr(base_nodes, k) for k in dir(base_nodes) if not k.startswith("__")}
    nodes_shim_dir['NODE_CLASS_MAPPINGS'] = _NodeClassMappingsShim()
    nodes_shim_dir['EXTENSION_WEB_DIRS'] = {}

    nodes_shim = _NodeShim()
    for k, v in nodes_shim_dir.items():
        setattr(nodes_shim, k, v)

    sys.modules['nodes'] = nodes_shim

    comfyui_version = types.ModuleType('comfyui_version', '')
    setattr(comfyui_version, "__version__", __version__)
    sys.modules['comfyui_version'] = comfyui_version
    from comfy.cmd import execution, server
    for module in (execution, server):
        module_short_name = module.__name__.split(".")[-1]
        sys.modules[module_short_name] = module
    if server.PromptServer.instance is None:
        server.PromptServer.instance = _PromptServerStub()
    # Impact Pack wants to find model_patcher
    from comfy import model_patcher
    sys.modules['model_patcher'] = model_patcher
    # NormalCrafter and others import bare 'model_management'
    from comfy import model_management
    sys.modules['model_management'] = model_management
    import comfy_extras
    if not any(isinstance(f, _ComfyExtrasRedirectFinder) for f in sys.meta_path):
        sys.meta_path.append(_ComfyExtrasRedirectFinder())
    comfy_extras_mitigation: Dict[str, types.ModuleType] = {}
    for module_name, module in sys.modules.items():
        if not module_name.startswith("comfy_extras.nodes"):
            continue
        module_short_name = module_name.split(".")[-1]
        setattr(comfy_extras, module_short_name, module)
        comfy_extras_mitigation[f'comfy_extras.{module_short_name}'] = module
    sys.modules.update(comfy_extras_mitigation)
    _ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor
    original_thread_start = threading.Thread.start
    concurrent.futures.ThreadPoolExecutor = ContextVarExecutor

    # mitigate missing folder names and paths context
    def patched_start(self, *args, **kwargs):
        if not hasattr(self.run, '__wrapped_by_context__'):
            ctx = contextvars.copy_context()
            self.run = partial(ctx.run, self.run)
            setattr(self.run, '__wrapped_by_context__', True)
        original_thread_start(self, *args, **kwargs)

    if not getattr(threading.Thread.start, '__is_patched_by_us', False):
        threading.Thread.start = patched_start
        setattr(threading.Thread.start, '__is_patched_by_us', True)
        logger.debug("Patched `threading.Thread.start` to propagate contextvars.")


def _is_pip_install_command(command_list) -> tuple[bool, list[str]]:
    """Detect pip/uv-pip install commands regardless of arg structure.

    Matches all of:
      [python, -m, pip, install, ...]
      [python, -s, -m, pip, install, ...]
      [python, -m, uv, pip, install, ...]
      [python, -s, -m, uv, pip, install, ...]

    Returns (is_match, package_names).
    """
    if not isinstance(command_list, list) or len(command_list) < 4:
        return False, []
    if command_list[0] != sys.executable:
        return False, []

    # strip optional -s flag
    rest = command_list[1:]
    if rest and rest[0] == "-s":
        rest = rest[1:]

    # expect [-m, pip, install, ...] or [-m, uv, pip, install, ...]
    if len(rest) < 3 or rest[0] != "-m":
        return False, []

    if rest[1] == "pip" and rest[2] == "install":
        return True, rest[3:]
    if rest[1] == "uv" and len(rest) >= 4 and rest[2] == "pip" and rest[3] == "install":
        return True, rest[4:]

    return False, []


def _log_pip_intercept(intercept_type: str, command_list: list):
    _pip_log = os.environ.get("COMFY_PIP_INTERCEPT_LOG")
    if _pip_log:
        import json as _json
        with open(_pip_log, "a") as _f:
            _f.write(_json.dumps({"type": intercept_type, "command": command_list}) + "\n")


@contextmanager
def patch_pip_install_subprocess_run():
    from unittest.mock import patch, MagicMock
    original_subprocess_run = subprocess.run
    original_check_call = subprocess.check_call
    original_check_output = subprocess.check_output

    def _run_side_effect(*args, **kwargs):
        command_list = args[0] if args else []
        is_pip, packages = _is_pip_install_command(command_list)
        if is_pip:
            logger.warning(f"Blocked runtime pip install for: {' '.join(packages)}. Pre-install these packages or see comfy.nodes.custom_node_dependencies for known runtime deps.")
            _log_pip_intercept("subprocess_run", command_list)
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result
        return original_subprocess_run(*args, **kwargs)

    def _check_call_side_effect(*args, **kwargs):
        command_list = args[0] if args else []
        is_pip, packages = _is_pip_install_command(command_list)
        if is_pip:
            logger.warning(f"Blocked runtime pip install (check_call) for: {' '.join(packages)}. Pre-install these packages or see comfy.nodes.custom_node_dependencies for known runtime deps.")
            _log_pip_intercept("check_call", command_list)
            return 0
        return original_check_call(*args, **kwargs)

    def _check_output_side_effect(*args, **kwargs):
        command_list = args[0] if args else []
        is_pip, packages = _is_pip_install_command(command_list)
        if is_pip:
            logger.warning(f"Blocked runtime pip install (check_output) for: {' '.join(packages)}. Pre-install these packages or see comfy.nodes.custom_node_dependencies for known runtime deps.")
            _log_pip_intercept("check_output", command_list)
            return b""
        return original_check_output(*args, **kwargs)

    with (
        patch('subprocess.run') as mock_run,
        patch('subprocess.check_call') as mock_check_call,
        patch('subprocess.check_output') as mock_check_output,
    ):
        mock_run.side_effect = _run_side_effect
        mock_check_call.side_effect = _check_call_side_effect
        mock_check_output.side_effect = _check_output_side_effect
        yield


@contextmanager
def patch_pip_install_popen():
    from unittest.mock import patch, MagicMock
    original_subprocess_popen = subprocess.Popen

    def custom_side_effect(*args, **kwargs):
        command_list = args[0] if args else []
        is_pip, packages = _is_pip_install_command(command_list)

        if is_pip:
            logger.warning(f"Blocked runtime pip install (Popen) for: {' '.join(packages)}. Pre-install these packages or see comfy.nodes.custom_node_dependencies for known runtime deps.")
            _log_pip_intercept("popen", command_list)

            mock_popen_instance = MagicMock()
            # make stdout and stderr empty iterables so loops over them complete immediately.
            mock_popen_instance.stdout = []
            mock_popen_instance.stderr = []

            return mock_popen_instance
        else:
            return original_subprocess_popen(*args, **kwargs)

    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = custom_side_effect
        yield mock_popen


@contextmanager
def vanilla_environment_node_execution_hooks():
    # this handles activating the NODE_CLASS_MAPPINGS shim
    from comfy.execution_context import current_execution_context
    from comfy.nodes.download_interception import patch_folder_paths_functions
    ctx = current_execution_context()

    if 'nodes' in sys.modules and isinstance(sys.modules['nodes'], _NodeShim):
        nodes_shim: _NodeShim = sys.modules['nodes']
        try:
            nodes_shim.activate()

            block_installs = ctx and ctx.configuration and ctx.configuration.block_runtime_package_installation is True
            with (
                patch_folder_paths_functions(),
                patch_pip_install_subprocess_run() if block_installs else nullcontext(),
                patch_pip_install_popen() if block_installs else nullcontext(),
            ):
                yield
        finally:
            nodes_shim.deactivate()
    else:
        yield
