import sys

from comfy.nodes.package import import_all_nodes_in_workspace
from comfy_compatibility.vanilla import prepare_vanilla_environment


def test_issue_61_pip_package_frontend_registered_via_nodes_module():
    prepare_vanilla_environment()
    nodes_shim = sys.modules['nodes']

    entry_name = 'comfyui-manager-legacy-issue61'
    entry_path = '/tmp/issue61-manager/js'

    try:
        nodes_shim.EXTENSION_WEB_DIRS[entry_name] = entry_path
        loaded = import_all_nodes_in_workspace()
        assert entry_name in loaded.EXTENSION_WEB_DIRS
        assert loaded.EXTENSION_WEB_DIRS[entry_name] == entry_path
    finally:
        nodes_shim.EXTENSION_WEB_DIRS.pop(entry_name, None)
