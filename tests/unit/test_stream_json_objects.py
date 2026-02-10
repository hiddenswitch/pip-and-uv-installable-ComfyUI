import asyncio
import importlib.resources
import json
from decimal import Decimal
from pathlib import Path

import pytest

from comfy.component_model.asyncio_files import stream_json_objects
from comfy.api.components.schema.prompt import Prompt
from tests.inference import workflows


WORKFLOW_PATH = str(Path(__file__).resolve().parents[1] / "inference" / "workflows" / "z_image-0.json")


async def _collect(source: str) -> list[dict]:
    results = []
    async for obj in stream_json_objects(source):
        results.append(obj)
    return results


def _assert_no_decimals(obj):
    """Recursively check that no Decimal values snuck through."""
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_no_decimals(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _assert_no_decimals(v)
    else:
        assert not isinstance(obj, Decimal), f"Decimal value found: {obj}"


def _all_workflow_files():
    return {
        f.name: f
        for f in importlib.resources.files(workflows).iterdir()
        if f.is_file() and f.name.endswith(".json")
    }


class TestStreamJsonObjectsFile:
    def test_workflow_file_no_decimals(self):
        results = asyncio.run(_collect(WORKFLOW_PATH))
        assert len(results) == 1
        _assert_no_decimals(results[0])

    def test_workflow_file_float_values_are_float(self):
        results = asyncio.run(_collect(WORKFLOW_PATH))
        prompt = results[0]
        # z_image-0.json has cfg: 1.0 and denoise: 1.0 in node "3"
        assert isinstance(prompt["3"]["inputs"]["cfg"], float)
        assert isinstance(prompt["3"]["inputs"]["denoise"], float)


class TestStreamJsonObjectsLiteral:
    def test_literal_json_no_decimals(self):
        literal = '{"a": {"inputs": {"val": 3.14}}, "b": {"inputs": {"val": 0.001}}}'
        results = asyncio.run(_collect(literal))
        assert len(results) == 1
        _assert_no_decimals(results[0])
        assert isinstance(results[0]["a"]["inputs"]["val"], float)

    def test_literal_json_multiple_objects(self):
        literal = '{"x": 1}{"y": 2}'
        results = asyncio.run(_collect(literal))
        assert len(results) == 2


class TestStreamJsonObjectsURI:
    def test_https_uri(self):
        url = "https://raw.githubusercontent.com/hiddenswitch/pip-and-uv-installable-ComfyUI/refs/heads/master/tests/inference/workflows/z_image-0.json"
        results = asyncio.run(_collect(url))
        assert len(results) == 1
        _assert_no_decimals(results[0])
        assert isinstance(results[0]["3"]["inputs"]["cfg"], float)


class TestStreamJsonObjectsEmpty:
    def test_none_returns_empty(self):
        results = asyncio.run(_collect(None))
        assert results == []

    def test_empty_string_returns_empty(self):
        results = asyncio.run(_collect(""))
        assert results == []


class TestPromptValidateWorkflow:
    def test_prompt_validate_with_decimals(self):
        """Prompt.validate should handle dicts that contain Decimal values (e.g. from ijson without use_float)."""
        prompt = {
            "1": {
                "inputs": {"cfg": Decimal("7.5"), "steps": Decimal("20")},
                "class_type": "KSampler",
            }
        }
        validated = Prompt.validate(prompt)
        assert validated is not None
        # Decimal("7.5") should become float, Decimal("20") should become int
        cfg_val = validated["1"]["inputs"]["cfg"]
        steps_val = validated["1"]["inputs"]["steps"]
        assert isinstance(cfg_val, float)
        assert isinstance(steps_val, int)
        assert cfg_val == 7.5
        assert steps_val == 20

    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_prompt_validate_all_workflows(self, workflow_name, workflow_file):
        """Prompt.validate should succeed on every bundled workflow JSON file."""
        workflow = json.loads(workflow_file.read_text(encoding="utf8"))
        validated = Prompt.validate(workflow)
        assert validated is not None
        assert len(validated) > 0
