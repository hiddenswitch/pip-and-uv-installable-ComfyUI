"""
Centralized integration with comfyui_manager package.

This module provides a clean interface to comfyui_manager functionality,
handling import errors gracefully and providing stubs when manager is unavailable.

CLI Args Used:
- args.enable_manager: Main toggle for manager functionality
- args.disable_manager_ui: Disables UI but keeps background tasks
- args.enable_manager_legacy_ui: Uses legacy manager UI (passed through to manager)
- args.windows_standalone_build: Affects warning messages
"""
import importlib.util
import logging
import os
import sys
from typing import Optional, Callable

from .cli_args_types import Configuration

logger = logging.getLogger(__name__)

_manager_module = None
_manager_available = False
_args: Optional[Configuration] = None
_vanilla_prepared = False


def _prepare_manager_environment():
    """
    Set up the vanilla environment and env vars that comfyui_manager expects.
    Only called when manager is enabled.
    """
    global _vanilla_prepared
    if _vanilla_prepared:
        return

    # Set environment variables that comfyui_manager expects
    from .cmd import folder_paths
    os.environ['COMFYUI_PATH'] = str(folder_paths.base_path)
    os.environ['COMFYUI_FOLDERS_BASE_PATH'] = str(folder_paths.models_dir)

    # Prepare the vanilla module layout (folder_paths, nodes, etc. as top-level modules)
    from comfy_compatibility.vanilla import prepare_vanilla_environment
    prepare_vanilla_environment()

    _vanilla_prepared = True


def init_manager(args: Configuration) -> bool:
    """
    Initialize comfyui_manager if available and enabled.
    Returns True if manager was successfully initialized.

    Uses CLI args:
    - args.enable_manager: Must be True to initialize
    - args.windows_standalone_build: Affects warning messages
    """
    global _manager_module, _manager_available, _args
    _args = args

    if not args.enable_manager:
        logger.debug("Manager disabled via --enable-manager=false")
        return False

    if not importlib.util.find_spec("comfyui_manager"):
        _warn_manager_unavailable(args)
        return False

    try:
        # Prepare vanilla environment before importing comfyui_manager
        _prepare_manager_environment()

        import comfyui_manager
        if not comfyui_manager.__file__ or not comfyui_manager.__file__.endswith('__init__.py'):
            _warn_manager_unavailable(args)
            return False

        _manager_module = comfyui_manager
        _manager_available = True
        logger.info("ComfyUI Manager initialized successfully")
        return True
    except Exception as e:
        logger.warning(f"Failed to import comfyui_manager: {e}")
        return False


def _warn_manager_unavailable(args: Configuration):
    if not args.windows_standalone_build:
        logger.warning(
            f"\n\ncomfyui-manager is not properly installed. Install with:\n"
            f"\t{sys.executable} -m pip install --pre comfyui_manager\n"
        )
    args.enable_manager = False


def is_available() -> bool:
    """Check if manager is available and initialized."""
    return _manager_available


def should_be_disabled(module_path: str) -> bool:
    """
    Check if a custom node should be disabled by manager policy.

    Uses CLI args (via comfyui_manager internals):
    - args.enable_manager: Manager checks this internally
    """
    if not _manager_available or _manager_module is None:
        return False
    try:
        return _manager_module.should_be_disabled(module_path)
    except Exception as e:
        logger.debug(f"should_be_disabled check failed: {e}")
        return False


def prestartup():
    """Run manager prestartup if available."""
    if not _manager_available or _manager_module is None:
        return
    try:
        _manager_module.prestartup()
    except Exception as e:
        logger.warning(f"comfyui_manager.prestartup() failed: {e}")


def start():
    """
    Start manager UI if available.

    Uses CLI args:
    - args.disable_manager_ui: Skip if True (called at call site)
    - args.enable_manager_legacy_ui: Passed through to manager internally
    """
    if not _manager_available or _manager_module is None:
        return
    try:
        _manager_module.start()
    except Exception as e:
        logger.warning(f"comfyui_manager.start() failed: {e}")


def get_middleware() -> Optional[Callable]:
    """
    Get manager middleware if available.

    Uses CLI args (via comfyui_manager internals):
    - args.listen: Used for security policy decisions
    """
    if not _manager_available or _manager_module is None:
        return None
    try:
        return _manager_module.create_middleware()
    except Exception as e:
        logger.warning(f"comfyui_manager.create_middleware() failed: {e}")
        return None
