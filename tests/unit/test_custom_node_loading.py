"""Tests that pip-installed custom nodes (ollama, sam2, custom-scripts) load and execute.

These tests verify the full lifecycle: the facade packages are importable,
their nodes register in ComfyUI's node system, and lightweight nodes can
execute through the embedded client using GraphBuilder workflows.
"""
from __future__ import annotations

import importlib.metadata

import pytest
import torch

from comfy_execution.graph_utils import GraphBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_installed(package: str) -> bool:
    try:
        importlib.metadata.distribution(package)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


_nodes_loaded = False


def _load_nodes():
    """Import all nodes (including pip-facade custom nodes) and return the registry."""
    global _nodes_loaded
    if not _nodes_loaded:
        from comfy.nodes.package import import_all_nodes_in_workspace
        import_all_nodes_in_workspace()
        _nodes_loaded = True


def _get_node_class(class_type: str):
    """Resolve a node class_type to its Python class through the node registry."""
    _load_nodes()
    from comfy.nodes_context import get_nodes
    cls = get_nodes().NODE_CLASS_MAPPINGS.get(class_type)
    if cls is None:
        pytest.skip(f"Node {class_type!r} not found in NODE_CLASS_MAPPINGS")
    return cls


def _call_node(class_type: str, **kwargs):
    """Instantiate a node and call its FUNCTION with the given kwargs."""
    cls = _get_node_class(class_type)
    func_name = getattr(cls, "FUNCTION", "execute")
    func = getattr(cls(), func_name)
    return func(**kwargs)


async def _run_graph(builder: GraphBuilder) -> dict:
    """Execute a GraphBuilder workflow through the embedded client."""
    from comfy.client.embedded_comfy_client import Comfy
    from comfy.cli_args import default_configuration

    config = default_configuration()
    config.database_url = "sqlite:///:memory:"

    async with Comfy(configuration=config) as client:
        return await client.queue_prompt(builder.finalize())


# ---------------------------------------------------------------------------
# Package installation checks
# ---------------------------------------------------------------------------

class TestPackagesInstalled:
    def test_ollama_installed(self):
        assert _is_installed("comfyui-ollama"), "comfyui-ollama not installed"

    def test_sam2_installed(self):
        assert _is_installed("comfyui-sam2"), "comfyui-sam2 not installed"

    def test_custom_scripts_installed(self):
        assert _is_installed("comfyui-custom-scripts"), "comfyui-custom-scripts not installed"


# ---------------------------------------------------------------------------
# Node registration checks
# ---------------------------------------------------------------------------

class TestNodeRegistration:
    """Verify that custom node classes appear in NODE_CLASS_MAPPINGS."""

    def test_ollama_nodes_registered(self):
        _load_nodes()
        from comfy.nodes_context import get_nodes
        mappings = get_nodes().NODE_CLASS_MAPPINGS
        for name in ("OllamaOptionsV2", "OllamaConnectivityV2", "OllamaGenerateV2", "OllamaChat"):
            assert name in mappings, f"{name} not registered"

    def test_sam2_nodes_registered(self):
        _load_nodes()
        from comfy.nodes_context import get_nodes
        mappings = get_nodes().NODE_CLASS_MAPPINGS
        for name in (
            "SAM2ModelLoader (segment anything2)",
            "GroundingDinoModelLoader (segment anything2)",
            "GroundingDinoSAM2Segment (segment anything2)",
            "IsMaskEmpty",
        ):
            assert name in mappings, f"{name} not registered"

    def test_custom_scripts_nodes_registered(self):
        _load_nodes()
        from comfy.nodes_context import get_nodes
        mappings = get_nodes().NODE_CLASS_MAPPINGS
        for name in ("MathExpression|pysssss", "ShowText|pysssss", "StringFunction|pysssss"):
            assert name in mappings, f"{name} not registered"


# ---------------------------------------------------------------------------
# Workflow execution: comfyui-custom-scripts
# ---------------------------------------------------------------------------

class TestCustomScriptsExecution:
    """Execute custom-scripts nodes via GraphBuilder + embedded client."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_math_expression(self):
        """MathExpression evaluates '2 + 3'."""
        g = GraphBuilder()
        g.node("MathExpression|pysssss", expression="2 + 3")
        outputs = await _run_graph(g)
        assert outputs is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_math_expression_with_variables(self):
        """MathExpression using a, b inputs."""
        g = GraphBuilder()
        g.node("MathExpression|pysssss", expression="a * b + 1", a=3, b=4)
        outputs = await _run_graph(g)
        assert outputs is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_string_function_append(self):
        """StringFunction appends two strings."""
        g = GraphBuilder()
        g.node(
            "StringFunction|pysssss",
            action="append",
            tidy_tags="no",
            text_a="hello ",
            text_b="world",
        )
        outputs = await _run_graph(g)
        assert outputs is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_string_function_replace(self):
        """StringFunction replaces text."""
        g = GraphBuilder()
        g.node(
            "StringFunction|pysssss",
            action="replace",
            tidy_tags="no",
            text_a="hello world",
            text_b="world",
            text_c="comfy",
        )
        outputs = await _run_graph(g)
        assert outputs is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_show_text(self):
        """ShowText receives string input and outputs it."""
        g = GraphBuilder()
        string_node = g.node(
            "StringFunction|pysssss",
            action="append",
            tidy_tags="no",
            text_a="test output",
        )
        g.node("ShowText|pysssss", text=string_node.out(0))
        outputs = await _run_graph(g)
        assert outputs is not None


# ---------------------------------------------------------------------------
# Direct execution: comfyui-sam2 (no-model nodes)
# ---------------------------------------------------------------------------

class TestSAM2Execution:
    """Execute SAM2 utility nodes directly (no model files needed)."""

    def test_invert_mask(self):
        """InvertMask from SAM2 inverts a zero mask to ones."""
        mask = torch.zeros(1, 64, 64)
        result = _call_node("InvertMask (segment anything)", mask=mask)
        inverted = result[0]
        assert torch.allclose(inverted, torch.ones(1, 64, 64))

    def test_invert_mask_roundtrip(self):
        """Double inversion returns the original mask."""
        mask = torch.rand(1, 64, 64)
        result1 = _call_node("InvertMask (segment anything)", mask=mask)
        result2 = _call_node("InvertMask (segment anything)", mask=result1[0])
        assert torch.allclose(result2[0], mask, atol=1e-6)

    def test_is_mask_empty_with_empty_mask(self):
        """IsMaskEmpty returns 1 for an all-zero mask."""
        mask = torch.zeros(1, 64, 64)
        result = _call_node("IsMaskEmpty", mask=mask)
        assert result[0] == 1

    def test_is_mask_empty_with_nonempty_mask(self):
        """IsMaskEmpty returns 0 for a non-zero mask."""
        mask = torch.ones(1, 64, 64)
        result = _call_node("IsMaskEmpty", mask=mask)
        assert result[0] == 0


# ---------------------------------------------------------------------------
# Direct execution: comfyui-ollama (no-server nodes)
# ---------------------------------------------------------------------------

class TestOllamaExecution:
    """Execute ollama nodes that don't require a running server."""

    def test_ollama_options_default(self):
        """OllamaOptionsV2 returns an options dict."""
        result = _call_node(
            "OllamaOptionsV2",
            enable_mirostat=False, mirostat=0,
            enable_mirostat_eta=False, mirostat_eta=0.1,
            enable_mirostat_tau=False, mirostat_tau=5.0,
            enable_num_ctx=False, num_ctx=2048,
            enable_repeat_last_n=False, repeat_last_n=64,
            enable_repeat_penalty=False, repeat_penalty=1.1,
            enable_temperature=False, temperature=0.8,
            enable_seed=False, seed=0,
            enable_stop=False, stop="",
            enable_tfs_z=False, tfs_z=1,
            enable_num_predict=False, num_predict=-1,
            enable_top_k=False, top_k=40,
            enable_top_p=False, top_p=0.9,
            enable_min_p=False, min_p=0.0,
            debug=False,
        )
        options = result[0]
        assert isinstance(options, dict)

    def test_ollama_options_with_temperature(self):
        """OllamaOptionsV2 with temperature enabled returns it in the dict."""
        result = _call_node(
            "OllamaOptionsV2",
            enable_mirostat=False, mirostat=0,
            enable_mirostat_eta=False, mirostat_eta=0.1,
            enable_mirostat_tau=False, mirostat_tau=5.0,
            enable_num_ctx=False, num_ctx=2048,
            enable_repeat_last_n=False, repeat_last_n=64,
            enable_repeat_penalty=False, repeat_penalty=1.1,
            enable_temperature=True, temperature=0.7,
            enable_seed=False, seed=0,
            enable_stop=False, stop="",
            enable_tfs_z=False, tfs_z=1,
            enable_num_predict=False, num_predict=-1,
            enable_top_k=False, top_k=40,
            enable_top_p=False, top_p=0.9,
            enable_min_p=False, min_p=0.0,
            debug=False,
        )
        options = result[0]
        assert isinstance(options, dict)
        assert "temperature" in options
        assert options["temperature"] == pytest.approx(0.7)

    def test_ollama_options_multiple_enabled(self):
        """OllamaOptionsV2 with multiple toggles returns all enabled options."""
        result = _call_node(
            "OllamaOptionsV2",
            enable_mirostat=False, mirostat=0,
            enable_mirostat_eta=False, mirostat_eta=0.1,
            enable_mirostat_tau=False, mirostat_tau=5.0,
            enable_num_ctx=True, num_ctx=4096,
            enable_repeat_last_n=False, repeat_last_n=64,
            enable_repeat_penalty=True, repeat_penalty=1.2,
            enable_temperature=True, temperature=0.5,
            enable_seed=True, seed=42,
            enable_stop=False, stop="",
            enable_tfs_z=False, tfs_z=1,
            enable_num_predict=False, num_predict=-1,
            enable_top_k=True, top_k=50,
            enable_top_p=False, top_p=0.9,
            enable_min_p=False, min_p=0.0,
            debug=False,
        )
        options = result[0]
        assert options["num_ctx"] == 4096
        assert options["repeat_penalty"] == pytest.approx(1.2)
        assert options["temperature"] == pytest.approx(0.5)
        assert options["seed"] == 42
        assert options["top_k"] == 50
