__version__ = "0.24.0.10"

# This deals with workspace issues
from comfy_compatibility.workspace import auto_patch_workspace_and_restart

auto_patch_workspace_and_restart()
