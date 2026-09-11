import sys

import pytest

from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1


@pytest.fixture(autouse=True)
def _drop_test_modules():
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name.endswith("test_v1_custom_node") or name.endswith("test_v3_custom_node"):
                sys.modules.pop(name, None)


def test_load_custom_node_skips_display_names_for_ignored_nodes(tmp_path, monkeypatch):
    v1_module = tmp_path / "test_v1_custom_node.py"
    v1_module.write_text(
        "class LeakTest:\n    pass\n\n\n"
        "NODE_CLASS_MAPPINGS = {\"LeakTest\": LeakTest}\n"
        "NODE_DISPLAY_NAME_MAPPINGS = {\"LeakTest\": \"Leak Test\"}\n",
    )

    v3_module = tmp_path / "test_v3_custom_node.py"
    v3_module.write_text(
        "from comfy_api.latest import ComfyExtension\n\n"
        "class LeakTestV3Node:\n"
        "    @classmethod\n"
        "    def GET_SCHEMA(cls):\n"
        "        class Schema:\n"
        "            node_id = \"LeakTestV3\"\n"
        "            display_name = \"Leak Test V3\"\n\n"
        "        return Schema()\n\n\n"
        "class TestExtension(ComfyExtension):\n"
        "    async def get_node_list(self):\n"
        "        return [LeakTestV3Node]\n\n\n"
        "async def comfy_entrypoint():\n"
        "    return TestExtension()\n",
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    v1 = _vanilla_load_custom_nodes_1(str(v1_module), ignore={"LeakTest"})
    v3 = _vanilla_load_custom_nodes_1(str(v3_module), ignore={"LeakTestV3"})

    assert "LeakTest" not in v1.NODE_CLASS_MAPPINGS
    assert "LeakTest" not in v1.NODE_DISPLAY_NAME_MAPPINGS
    assert "LeakTestV3" not in v3.NODE_CLASS_MAPPINGS
    assert "LeakTestV3" not in v3.NODE_DISPLAY_NAME_MAPPINGS


def test_load_custom_node_keeps_display_names_for_kept_nodes(tmp_path, monkeypatch):
    v1_module = tmp_path / "test_v1_custom_node.py"
    v1_module.write_text(
        "class KeepTest:\n    pass\n\n\n"
        "NODE_CLASS_MAPPINGS = {\"KeepTest\": KeepTest}\n"
        "NODE_DISPLAY_NAME_MAPPINGS = {\"KeepTest\": \"Keep Test\"}\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    exported = _vanilla_load_custom_nodes_1(str(v1_module))

    assert exported.NODE_CLASS_MAPPINGS["KeepTest"].__name__ == "KeepTest"
    assert exported.NODE_DISPLAY_NAME_MAPPINGS["KeepTest"] == "Keep Test"
