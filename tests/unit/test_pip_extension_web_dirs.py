import sys

from comfy.nodes.package import import_all_nodes_in_workspace
from comfy_compatibility.vanilla import prepare_vanilla_environment


def test_upstream_extension_web_dirs_writes_propagate_to_server_nodes():
    prepare_vanilla_environment()
    nodes_shim = sys.modules['nodes']

    fake_entry = 'regression61-pip-frontend'
    fake_path = '/tmp/regression61-pip-frontend/js'

    try:
        nodes_shim.EXTENSION_WEB_DIRS[fake_entry] = fake_path
        result = import_all_nodes_in_workspace()
        assert fake_entry in result.EXTENSION_WEB_DIRS
        assert result.EXTENSION_WEB_DIRS[fake_entry] == fake_path
    finally:
        nodes_shim.EXTENSION_WEB_DIRS.pop(fake_entry, None)
