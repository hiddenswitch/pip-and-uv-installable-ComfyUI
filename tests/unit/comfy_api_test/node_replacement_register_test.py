import asyncio

from comfy.cmd.server import PromptServer
from comfy.nodes.vanilla_node_importing import _PromptServerStub
from comfy_api.latest import ComfyAPI_latest, io


def _replace():
    return io.NodeReplace(new_node_id="NewNode", old_node_id="OldNode")


def test_register_is_a_no_op_while_custom_nodes_import_against_the_stub_server(monkeypatch):
    # The vanilla custom-node loader installs _PromptServerStub as PromptServer.instance; built-in
    # extensions' on_load registers replacements during that import (comfy_extras nodes_replacements).
    monkeypatch.setattr(PromptServer, "instance", _PromptServerStub())

    asyncio.run(ComfyAPI_latest.NodeReplacement().register(_replace()))


def test_register_is_a_no_op_without_a_server(monkeypatch):
    monkeypatch.setattr(PromptServer, "instance", None)

    asyncio.run(ComfyAPI_latest.NodeReplacement().register(_replace()))
