from __future__ import annotations

import fnmatch
import importlib
import importlib.util
import logging
import os
import sys
import time
import types
from contextlib import contextmanager, nullcontext
from os.path import join, basename, dirname, isdir, isfile, exists, abspath, split, splitext, realpath
from typing import Iterable, Any, Generator
from unittest.mock import patch, MagicMock

from comfy_compatibility.vanilla import prepare_vanilla_environment, patch_pip_install_subprocess_run, patch_pip_install_popen
from . import base_nodes
from .comfyui_v3_package_imports import _comfy_entrypoint_upstream_v3_imports
from .download_interception import (
    patch_hf_hub_download,
    patch_snapshot_download,
    patch_folder_paths_functions,
    patch_folder_names_dict,
    patch_torch_downloads,
)
from .package_typing import ExportedNodes
from .python_module_metadata import resolve_python_module_name
from ..cmd import folder_paths
from ..component_model.plugins import prompt_server_instance_routes
from ..distributed.server_stub import ServerStub
from ..execution_context import current_execution_context

logger = logging.getLogger(__name__)


def _stamp_relative_python_modules(node_class_mappings: dict[str, type]) -> None:
    for node_class in node_class_mappings.values():
        node_class.RELATIVE_PYTHON_MODULE = resolve_python_module_name(node_class)


class StreamToLogger:
    """
    File-like stream object that redirects writes to a logger instance.
    This is used to capture print() statements from modules during import.
    """

    def __init__(self, logger: logging.Logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        # Process each line from the buffer. Print statements usually end with a newline.
        for line in buf.rstrip().splitlines():
            # Log the line, removing any trailing whitespace
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        # The logger handles its own flushing, so this can be a no-op.
        pass

    @property
    def encoding(self):
        return "utf-8"


class _PromptQueueStub:
    """Stub for PromptServer.instance.prompt_queue used by VideoHelperSuite at import time."""

    def __init__(self):
        self.currently_running = {}

    def put(self, item):
        logger.warning("prompt_queue.put() called on stub — ignored")

    def get_current_queue(self):
        return [], []

    def get_current_queue_volatile(self):
        return [], []

    def get_history(self, **kwargs):
        return {}

    def size(self):
        return 0

    def get_tasks_remaining(self):
        return 0


class _PromptServerStub(ServerStub):
    def __init__(self):
        super().__init__()
        self.routes = prompt_server_instance_routes
        self.on_prompt_handlers = []
        self.prompt_queue = _PromptQueueStub()
        self.number = 0
        self._warned_events: set[str] = set()

    def add_on_prompt_handler(self, handler):
        # todo: these need to be added to a real prompt server if the loading order is behaving in a complex way
        self.on_prompt_handlers.append(handler)

    def send_sync(self, *args, **kwargs):
        # Suppress repetitive monitor messages (e.g. crystools.monitor fires every second)
        event_type = args[0] if args else None
        if event_type and event_type in self._warned_events:
            return
        if event_type:
            self._warned_events.add(event_type)
        logger.warning(f"Node tried to send a message over the websocket while importing, args={args} kwargs={kwargs}")


def _vanilla_load_importing_execute_prestartup_script(node_paths: Iterable[str]) -> None:
    def execute_script(script_path):
        module_name = splitext(script_path)[0]
        try:
            with _stdout_intercept(module_name):
                spec = importlib.util.spec_from_file_location(module_name, script_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            return True
        except Exception as e:
            logger.error(f"Failed to execute startup-script: {script_path}", exc_info=e)
        return False

    node_prestartup_times = []
    for custom_node_path in node_paths:
        # patched
        if not isdir(custom_node_path):
            continue
        # end patch
        possible_modules = os.listdir(custom_node_path)

        for possible_module in possible_modules:
            module_path = join(custom_node_path, possible_module)
            if isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            # Check if manager policy blocks this node
            from ..manager_integration import should_be_disabled
            if should_be_disabled(module_path):
                logger.info(f"Blocked by manager policy: {module_path}")
                continue

            script_path = join(module_path, "prestartup_script.py")
            if exists(script_path):
                if "comfyui-manager" in module_path.lower():
                    os.environ['COMFYUI_PATH'] = str(folder_paths.base_path)
                    os.environ['COMFYUI_FOLDERS_BASE_PATH'] = str(folder_paths.models_dir)
                    # Monkey-patch ComfyUI-Manager's security check to prevent it from crashing on startup
                    # and its logging handler to prevent it from taking over logging.
                    glob_path = join(module_path, "glob")
                    glob_path_added = False
                    original_add_handler = logging.Logger.addHandler

                    def no_op_add_handler(self, handler):
                        logger.info(f"Skipping addHandler for {type(handler).__name__} during ComfyUI-Manager prestartup.")

                    try:
                        sys.path.insert(0, glob_path)
                        glob_path_added = True
                        # Patch security_check
                        import security_check  # pylint: disable=import-error
                        original_check = security_check.security_check

                        def patched_security_check():
                            try:
                                return original_check()
                            except Exception as e:
                                logger.error(f"ComfyUI-Manager security_check failed but was caught gracefully: {e}", exc_info=e)

                        security_check.security_check = patched_security_check
                        logger.debug("Patched ComfyUI-Manager's security_check to fail gracefully.")

                        # Patch logging
                        logging.Logger.addHandler = no_op_add_handler
                        logger.debug("Patched logging.Logger.addHandler to prevent ComfyUI-Manager from adding a logging handler.")

                        time_before = time.perf_counter()
                        success = execute_script(script_path)
                        node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
                    except Exception as e:
                        logger.error(f"Failed to patch and execute ComfyUI-Manager's prestartup script: {e}", exc_info=e)
                    finally:
                        if glob_path_added and glob_path in sys.path:
                            sys.path.remove(glob_path)
                        logging.Logger.addHandler = original_add_handler
                else:
                    time_before = time.perf_counter()
                    success = execute_script(script_path)
                    node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))


_MITIGATED_MODULES = frozenset((
    "comfyui-manager",
    "comfyui_ryanonyheinside",
    "comfyui-easy-use",
    "comfyui_custom_nodes_alekpet",
))

# Map short model folder names to HuggingFace repo IDs.  Custom nodes often
# pass ``os.path.join(folder_paths.models_dir, name)`` to ``from_pretrained``
# which fails when the model isn't pre-downloaded.  We patch the call site to
# resolve to the HF repo ID so ``from_pretrained`` downloads from HuggingFace.
_MODEL_NAME_TO_HF_REPO: dict[str, str] = {
    "segformer_b2_clothes": "mattmdjaga/segformer_b2_clothes",
    "segformer_b3_clothes": "mattmdjaga/segformer_b3_clothes",
    "segformer_b3_fashion": "mattmdjaga/segformer_b3_fashion",
}



def _apply_post_import_patches(module_name: str) -> None:
    """Apply patches to custom-node submodules after they finish importing.

    These patches fix cases where custom nodes construct local model paths
    that may not exist, instead of using HuggingFace repo IDs that
    ``from_pretrained`` can resolve and cache automatically.
    """
    _patch_segformer_model_resolution(module_name)


def _patch_segformer_model_resolution(module_name: str) -> None:
    """Patch segformer_ultra.get_segmentation to resolve model names to HF repos.

    ComfyUI_LayerStyle's segformer_ultra builds a local path like
    ``models/segformer_b2_clothes`` and passes it to ``from_pretrained``.
    When the folder doesn't exist, we resolve the HuggingFace repo via
    ``model_downloader`` so the model is fetched to the HF cache and
    ``from_pretrained`` loads from the cached snapshot path.
    """
    if module_name.lower() != "comfyui_layerstyle":
        return

    for mod_name, mod in list(sys.modules.items()):
        if "segformer_ultra" not in mod_name or not hasattr(mod, "get_segmentation"):
            continue

        _orig_get_seg = mod.get_segmentation

        def _patched_get_seg(tensor_image, model_name='segformer_b2_clothes', _orig=_orig_get_seg):
            hf_repo = _MODEL_NAME_TO_HF_REPO.get(model_name)
            if hf_repo is not None:
                # Check if the model already exists at the local path the node expects
                try:
                    resolved = os.path.normpath(folder_paths.folder_names_and_paths[model_name][0][0])
                except Exception:
                    resolved = os.path.join(folder_paths.models_dir, model_name)
                if not os.path.isdir(resolved):
                    # Use the project's download infrastructure to get the repo
                    # into the HF cache; from_pretrained accepts the repo ID directly
                    from .. import model_downloader
                    logger.info("Model folder %s not found, resolving via model_downloader for %s", model_name, hf_repo)
                    model_downloader.get_or_download_huggingface_repo(hf_repo)
                    # Pass the HF repo ID so from_pretrained fetches from cache
                    return _orig(tensor_image, hf_repo)
            return _orig(tensor_image, model_name)

        mod.get_segmentation = _patched_get_seg
        logger.info("Patched segformer_ultra.get_segmentation to resolve model names to HuggingFace repos")
        break


@contextmanager
def _protect_sys_path():
    """Snapshot and restore sys.path after custom node import.

    Custom nodes often call ``sys.path.insert(0, ...)`` during import to make
    sibling modules importable.  We allow mutations during import (some nodes
    need them for intra-package imports) but restore the original path list
    afterward so that one node's path additions don't leak to later nodes.
    """
    snapshot = list(sys.path)
    try:
        yield
    finally:
        sys.path[:] = snapshot


@contextmanager
def _exec_mitigations(module: types.ModuleType, module_path: str) -> Generator[ExportedNodes, Any, None]:
    config = current_execution_context()
    block_installation = config and config.configuration and config.configuration.block_runtime_package_installation

    needs_file_mitigation = module.__name__.lower() in _MITIGATED_MODULES

    with (
        # download interception — always active during custom node import
        patch_hf_hub_download(),
        patch_snapshot_download(),
        patch_folder_paths_functions(),
        patch_folder_names_dict(),
        patch_torch_downloads(),
        # pip blocking
        patch_pip_install_subprocess_run() if block_installation else nullcontext(),
        patch_pip_install_popen() if block_installation else nullcontext(),
        # sys.path protection — prevent custom nodes from polluting the path
        _protect_sys_path(),
    ):
        if needs_file_mitigation:
            from ..cmd import folder_paths
            old_file = folder_paths.__file__
            try:
                new_path = join(abspath(join(dirname(old_file), "..", "..")), basename(old_file))
                with patch.object(folder_paths, "__file__", new_path):
                    yield ExportedNodes()
            finally:
                logger.info(f"Exec mitigations were applied for {module.__name__}, due to using the folder_paths.__file__ symbol and manipulating EXTENSION_WEB_DIRS")
        else:
            yield ExportedNodes()


@contextmanager
def _stdout_intercept(name: str):
    original_stdout = sys.stdout

    try:
        module_logger = logging.getLogger(name)
        sys.stdout = StreamToLogger(module_logger, logging.INFO)
        yield
    finally:
        sys.stdout = original_stdout


def _vanilla_load_custom_nodes_1(module_path, ignore: set = None) -> ExportedNodes:
    if ignore is None:
        ignore = set()
    exported_nodes = ExportedNodes()
    module_name = basename(module_path)
    if isfile(module_path):
        sp = splitext(module_path)
        module_name = sp[0]
    try:
        if isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            module_dir = split(module_path)[0]
        else:
            module_spec = importlib.util.spec_from_file_location(module_name, join(module_path, "__init__.py"))
            module_dir = module_path

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module

        if not isfile(module_path):
            _stale_prefix = module_name + "."
            for _key in [k for k in sys.modules if k.startswith(_stale_prefix)]:
                del sys.modules[_key]
            _mod_name = module_name

            def _submodule_getattr(name, _prefix=_mod_name):
                fullname = f'{_prefix}.{name}'
                sub = sys.modules.get(fullname)
                if sub is not None:
                    return sub
                raise AttributeError(f"module {_prefix!r} has no attribute {name!r}")

            module.__getattr__ = _submodule_getattr

        with _exec_mitigations(module, module_path) as mitigated_exported_nodes, _stdout_intercept(module_name):
            module_spec.loader.exec_module(module)
            exported_nodes.update(mitigated_exported_nodes)

        _apply_post_import_patches(module_name)

        if hasattr(module, "WEB_DIRECTORY") and getattr(module, "WEB_DIRECTORY") is not None:
            web_dir = abspath(join(module_dir, getattr(module, "WEB_DIRECTORY")))
            if isdir(web_dir):
                exported_nodes.EXTENSION_WEB_DIRS[module_name] = web_dir

        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
            _stamp_relative_python_modules(module.NODE_CLASS_MAPPINGS)
            for name in module.NODE_CLASS_MAPPINGS:
                if name not in ignore:
                    exported_nodes.NODE_CLASS_MAPPINGS[name] = module.NODE_CLASS_MAPPINGS[name]
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module,
                                                                         "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                exported_nodes.NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        else:
            logger.error(f"Skip {module_path} module for custom nodes due to the lack of NODE_CLASS_MAPPINGS.")

        exported_nodes.update(_comfy_entrypoint_upstream_v3_imports(module))
    except Exception as e:
        logger.error(f"Cannot import {module_path} module for custom nodes:", exc_info=e)
    return exported_nodes


def _vanilla_load_custom_nodes_2(node_paths: Iterable[str]) -> ExportedNodes:
    from ..cli_args import args
    base_node_names = set(base_nodes.NODE_CLASS_MAPPINGS.keys())
    node_import_times = []
    exported_nodes = ExportedNodes()
    for custom_node_path in node_paths:
        if not exists(custom_node_path) or not isdir(custom_node_path):
            continue
        possible_modules = os.listdir(realpath(custom_node_path))
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = join(custom_node_path, possible_module)
            if isfile(module_path) and splitext(module_path)[1] != ".py": continue
            if module_path.endswith(".disabled"): continue
            if args.disable_all_custom_nodes and possible_module not in args.whitelist_custom_nodes:
                logger.info(f"Skipping {possible_module} due to disable_all_custom_nodes and whitelist_custom_nodes")
                continue
            if any(fnmatch.fnmatch(possible_module, pattern) for pattern in args.blacklist_custom_nodes):
                logger.info(f"Skipping {possible_module} due to blacklist_custom_nodes")
                continue
            # Check if manager policy blocks this node
            from ..manager_integration import should_be_disabled
            if should_be_disabled(module_path):
                logger.info(f"Blocked by manager policy: {module_path}")
                continue
            time_before = time.perf_counter()
            possible_exported_nodes = _vanilla_load_custom_nodes_1(module_path, ignore=base_node_names)
            # comfyui-manager mitigation
            import_succeeded = len(possible_exported_nodes.NODE_CLASS_MAPPINGS) > 0 or "ComfyUI-Manager" in module_path
            node_import_times.append(
                (time.perf_counter() - time_before, module_path, import_succeeded))
            exported_nodes.update(possible_exported_nodes)

    if len(node_import_times) > 0:
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (IMPORT FAILED)"
            logger.debug(f"{n[0]:6.1f} seconds{import_message}: {n[1]}")
    return exported_nodes


def mitigated_import_of_vanilla_custom_nodes(extra_node_paths: Iterable[str] = ()) -> ExportedNodes:
    # only vanilla custom nodes will ever go into the custom_nodes directory
    # this mitigation puts files that custom nodes expects are at the root of the repository back where they should be
    # found. we're in the middle of executing the import of execution and server, in all likelihood, so like all things,
    # the way community custom nodes is pretty radioactive
    # there's a lot of subtle details here, and unfortunately, once this is called, there are some things that have
    # to be activated later, in different places, to make all the hacks necessary for custom nodes to work
    prepare_vanilla_environment()

    from ..cmd import folder_paths
    node_paths = list(folder_paths.get_folder_paths("custom_nodes"))
    node_paths.extend(str(path) for path in extra_node_paths)

    potential_git_dir_parent = join(dirname(__file__), "..", "..")
    is_git_repository = exists(join(potential_git_dir_parent, ".git"))
    if is_git_repository:
        node_paths += [abspath(join(potential_git_dir_parent, "custom_nodes"))]

    node_paths = frozenset(abspath(custom_node_path) for custom_node_path in node_paths)
    _vanilla_load_importing_execute_prestartup_script(node_paths)
    vanilla_custom_nodes = _vanilla_load_custom_nodes_2(node_paths)
    return vanilla_custom_nodes
