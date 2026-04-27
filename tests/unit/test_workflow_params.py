"""Tests for workflow parameter discovery and application.

Two layers:

* **Synthetic unit tests** exercise each predicate against minimal, controlled
  workflows.

* **Parameterized real-workflow tests** at the bottom of this file iterate over
  every directory under ``tests/data/workflows/`` that contains both a
  ``workflow.json`` (UI or API format) and an ``expected.json`` sidecar. Drop
  new workflows in there to extend coverage without touching test code.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from comfy.entrypoints.workflow_params import (
    Param,
    TIER_ADVANCED,
    TIER_HEADLINE,
    apply,
    discover,
    frontend_widget_pool,
    params_by_address,
    params_by_role,
)


# ── synthetic unit tests: frontend_widget_pool ────────────────────────────────


def _api(node_id: str, class_type: str, **inputs) -> tuple[str, dict]:
    return node_id, {"class_type": class_type, "inputs": inputs}


def test_frontend_widget_pool_emits_one_param_per_non_link_widget():
    api = dict(
        [
            _api("3", "KSampler", seed=42, steps=20, cfg=8.0, model=["1", 0]),
            _api("1", "CheckpointLoaderSimple", ckpt_name="model_a.safetensors"),
        ]
    )
    out = list(frontend_widget_pool(api, ui=None))
    by_addr = {p.address: p for p in out}
    assert ("3", "seed") in by_addr
    assert ("3", "steps") in by_addr
    assert ("3", "cfg") in by_addr
    assert ("1", "ckpt_name") in by_addr
    # `model` is a link, must not be promoted to a Param
    assert ("3", "model") not in by_addr


def test_frontend_widget_pool_infers_widget_types():
    api = dict(
        [
            _api(
                "1", "Anything",
                an_int=5, a_float=1.25, a_string="hello", a_bool=True,
                a_combo_choice="euler",
            ),
        ]
    )
    by_name = {p.widget_name: p for p in frontend_widget_pool(api, None)}
    assert by_name["an_int"].type == "INT"
    assert by_name["a_float"].type == "FLOAT"
    assert by_name["a_string"].type == "STRING"
    assert by_name["a_bool"].type == "BOOLEAN"
    # Plain string from a combo selection still types as STRING; only list
    # values type as COMBO. Frontend rule doesn't introspect choices.
    assert by_name["a_combo_choice"].type == "STRING"


def test_frontend_widget_pool_propagates_node_meta_title_to_label():
    api = {
        "5": {
            "class_type": "WanVideoSampler",
            "inputs": {"seed": 42},
            "_meta": {"title": "My Sampler"},
        }
    }
    out = list(frontend_widget_pool(api, None))
    assert out[0].label == "My Sampler"


def test_frontend_widget_pool_treats_string_node_id_links_as_links():
    """Link form is [src_id, slot]; src_id is a string in API output."""
    api = {
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["1", 2]},
        }
    }
    out = list(frontend_widget_pool(api, None))
    assert out == []


# ── synthetic unit tests: discover() orchestration ────────────────────────────


def test_discover_runs_predicates_and_dedupes_by_address():
    api = dict([_api("1", "Foo", x=1, y=2)])
    params = discover(api)
    addrs = {p.address for p in params}
    assert addrs == {("1", "x"), ("1", "y")}
    # Stage 1 has only the frontend predicate; every Param should record it
    for p in params:
        assert "frontend_widget_pool" in p.source_predicates


def test_discover_ranks_advanced_tier_by_default():
    api = dict([_api("1", "Foo", x=1)])
    params = discover(api)
    assert params[0].tier == TIER_ADVANCED


def test_discover_sorted_stably_by_node_then_widget():
    api = dict(
        [
            _api("10", "Foo", b=1, a=2),
            _api("2", "Foo", b=3, a=4),
        ]
    )
    addrs = [p.address for p in discover(api)]
    assert addrs == [("10", "a"), ("10", "b"), ("2", "a"), ("2", "b")]


# ── synthetic unit tests: helpers ─────────────────────────────────────────────


def test_params_by_address_finds_existing_param():
    api = dict([_api("3", "KSampler", seed=42, steps=20)])
    params = discover(api)
    seed = params_by_address(params, "3", "seed")
    assert seed is not None
    assert seed.value == 42


def test_params_by_address_returns_none_for_missing():
    api = dict([_api("3", "KSampler", seed=42)])
    assert params_by_address(discover(api), "99", "seed") is None


def test_params_by_role_filters_on_role_set():
    p1 = Param(node_id="1", class_type="K", widget_name="seed", value=0, roles={"seed"})
    p2 = Param(node_id="2", class_type="L", widget_name="x", value=0, roles={"prompt"})
    assert params_by_role([p1, p2], "seed") == [p1]


def test_params_by_role_empty_when_no_matches():
    api = dict([_api("1", "Foo", x=1)])
    # Stage 1 hasn't introduced any roles yet
    assert params_by_role(discover(api), "prompt") == []


# ── synthetic unit tests: apply() ─────────────────────────────────────────────


def test_apply_sets_widget_value_and_returns_a_copy():
    api = dict([_api("3", "KSampler", seed=42, steps=20)])
    params = discover(api)
    seed = params_by_address(params, "3", "seed")
    out = apply(api, seed, 1234)
    assert out["3"]["inputs"]["seed"] == 1234
    # Original untouched
    assert api["3"]["inputs"]["seed"] == 42


def test_apply_rejects_ui_format():
    ui = {"nodes": [], "links": []}
    p = Param(node_id="1", class_type="X", widget_name="x", value=0)
    with pytest.raises(ValueError, match="API-format"):
        apply(ui, p, 1)


def test_apply_raises_when_node_missing():
    api = dict([_api("3", "KSampler", seed=42)])
    p = Param(node_id="99", class_type="X", widget_name="x", value=0)
    with pytest.raises(KeyError):
        apply(api, p, 1)


# ── parameterized real-workflow tests ─────────────────────────────────────────

_WORKFLOWS_DIR = Path(__file__).parent.parent / "data" / "workflows"


def _workflow_cases() -> list:
    if not _WORKFLOWS_DIR.exists():
        return []
    cases = []
    for case_dir in sorted(_WORKFLOWS_DIR.iterdir()):
        if not case_dir.is_dir():
            continue
        if not (case_dir / "workflow.json").exists():
            continue
        if not (case_dir / "expected.json").exists():
            continue
        cases.append(pytest.param(case_dir, id=case_dir.name))
    return cases


def _load_workflow_for_discover(case_dir: Path) -> dict:
    """Load a fixture's workflow, booting the node system if it is UI-format.

    `convert_ui_to_api` requires `import_all_nodes_in_workspace` to have
    populated the node registry. Unknown classes (e.g. WanVideoWrapper)
    are preserved via `preserve_unknown_nodes=True`.
    """
    workflow = json.loads((case_dir / "workflow.json").read_text())
    if "nodes" in workflow and "links" in workflow:
        from comfy.nodes.package import import_all_nodes_in_workspace
        from comfy.nodes_context import get_nodes
        if len(get_nodes()) == 0:
            import_all_nodes_in_workspace()
    return workflow


@pytest.mark.parametrize("case_dir", _workflow_cases())
def test_real_workflow_discovery(case_dir: Path):
    workflow = _load_workflow_for_discover(case_dir)
    expected = json.loads((case_dir / "expected.json").read_text())

    params = discover(workflow)

    min_total = expected.get("min_total_params", 0)
    assert len(params) >= min_total, (
        f"discover() returned {len(params)} params; expected at least {min_total}"
    )

    # Every Param should be addressable in the API workflow it came from.
    from comfy.entrypoints.workflow_params import _to_api
    api, _ = _to_api(workflow)
    for p in params:
        assert p.node_id in api, f"param node_id {p.node_id!r} missing from API workflow"
        node = api[p.node_id]
        assert p.widget_name in (node.get("inputs") or {}), (
            f"param widget {p.widget_name!r} missing on node {p.node_id} ({p.class_type})"
        )

    # No Param should hold a link as its value (the frontend predicate
    # excludes them; later predicates must too).
    from comfy.entrypoints.workflow_params import _is_link
    for p in params:
        assert not _is_link(p.value), (
            f"param {p.address} has a link value {p.value!r}"
        )

    # Stage gating: assertions only fire if the case opts in for that stage.
    stage1 = (expected.get("stages") or {}).get("1") or {}
    for predicate_name in stage1.get("predicates_present", []):
        assert any(predicate_name in p.source_predicates for p in params), (
            f"no Param attributed to predicate {predicate_name!r}"
        )
