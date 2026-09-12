from __future__ import annotations

import logging
from pathlib import Path

from comfy.nodes import vanilla_node_importing
from comfy.nodes.vanilla_node_importing import _vanilla_load_importing_execute_prestartup_script


class _RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _make_pack(root: Path, name: str) -> Path:
    pack = root / name
    pack.mkdir(parents=True)
    (pack / "prestartup_script.py").write_text("VALUE = 1\n")
    return pack


def test_execute_prestartup_script_handles_empty_custom_nodes_paths(tmp_path):
    _vanilla_load_importing_execute_prestartup_script([])
    _vanilla_load_importing_execute_prestartup_script([str(tmp_path / "missing")])


def test_execute_prestartup_script_keeps_all_timing_entries(tmp_path):
    first_custom_nodes = tmp_path / "custom_nodes_1"
    second_custom_nodes = tmp_path / "custom_nodes_2"
    pack_one = _make_pack(first_custom_nodes, "pack_one")
    pack_two = _make_pack(second_custom_nodes, "pack_two")

    handler = _RecordingHandler()
    logger = vanilla_node_importing.logger
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        _vanilla_load_importing_execute_prestartup_script([str(first_custom_nodes), str(second_custom_nodes)])
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    joined = "\n".join(handler.messages)
    assert "Prestartup times for custom nodes" in joined
    assert str(pack_one) in joined
    assert str(pack_two) in joined
    assert "PRESTARTUP FAILED" not in joined
