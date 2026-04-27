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
    TIER_COMMON,
    TIER_HEADLINE,
    apply,
    class_type_roles,
    discover,
    easy_pack_nodes,
    frontend_widget_pool,
    params_by_address,
    params_by_role,
    primitive_nodes,
    prompt_polarity,
    set_node_pairs,
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


# ── synthetic unit tests: class_type_roles ────────────────────────────────────


def test_class_type_roles_tags_ksampler_widgets():
    api = dict(
        [
            _api(
                "3", "KSampler",
                seed=42, steps=20, cfg=8.0,
                sampler_name="euler", scheduler="normal", denoise=1.0,
                model=["1", 0],
            ),
        ]
    )
    out = list(class_type_roles(api, None))
    by_widget = {p.widget_name: p for p in out}
    assert by_widget["seed"].roles == {"seed"}
    assert by_widget["steps"].roles == {"steps"}
    assert by_widget["cfg"].roles == {"cfg"}
    assert by_widget["sampler_name"].roles == {"sampler"}
    assert by_widget["scheduler"].roles == {"scheduler"}
    assert by_widget["denoise"].roles == {"denoise"}
    # All class_type_roles output sits at TIER_COMMON
    for p in out:
        assert p.tier == TIER_COMMON


def test_class_type_roles_tags_random_noise_seed_field():
    api = dict([_api("5", "RandomNoise", noise_seed=12345)])
    out = list(class_type_roles(api, None))
    assert len(out) == 1
    assert out[0].widget_name == "noise_seed"
    assert out[0].roles == {"seed"}


def test_class_type_roles_tags_loaders_and_latent():
    api = dict(
        [
            _api("1", "CheckpointLoaderSimple", ckpt_name="model_a.safetensors"),
            _api("2", "UNETLoader", unet_name="unet_b.safetensors"),
            _api("5", "EmptyLatentImage", width=512, height=768, batch_size=2),
            _api("9", "LoadImage", image="cat.png"),
            _api("10", "LoadVideo", value="walk.mp4"),
            _api("11", "LoadAudio", value="bgm.wav"),
        ]
    )
    out = list(class_type_roles(api, None))
    by_addr = {p.address: p for p in out}
    assert by_addr[("1", "ckpt_name")].roles == {"checkpoint"}
    assert by_addr[("2", "unet_name")].roles == {"unet"}
    assert by_addr[("5", "width")].roles == {"width"}
    assert by_addr[("5", "height")].roles == {"height"}
    assert by_addr[("5", "batch_size")].roles == {"batch_size"}
    assert by_addr[("9", "image")].roles == {"image_input"}
    assert by_addr[("10", "value")].roles == {"video_input"}
    assert by_addr[("11", "value")].roles == {"audio_input"}


def test_class_type_roles_skips_links():
    api = dict([_api("3", "KSampler", seed=42, model=["1", 0])])
    out = list(class_type_roles(api, None))
    addrs = {p.address for p in out}
    assert ("3", "seed") in addrs


def test_class_type_roles_text_encode_widgets_get_generic_tag():
    api = dict(
        [
            _api("6", "CLIPTextEncode", text="positive prompt", clip=["4", 1]),
        ]
    )
    out = list(class_type_roles(api, None))
    assert len(out) == 1
    assert out[0].roles == {"text_encode"}


# ── synthetic unit tests: prompt_polarity ─────────────────────────────────────


def test_prompt_polarity_disambiguates_positive_via_sampler_input():
    api = {
        "3": {
            "class_type": "KSampler",
            "inputs": {"positive": ["6", 0], "negative": ["7", 0], "seed": 1, "steps": 20},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "good things", "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "bad things", "clip": ["4", 1]},
        },
    }
    out = list(prompt_polarity(api, None))
    by_role = {next(iter(p.roles)): p for p in out}
    assert by_role["prompt"].node_id == "6"
    assert by_role["negative_prompt"].node_id == "7"


def test_prompt_polarity_falls_back_to_sole_encoder_for_positive():
    api = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "prompt only"}},
    }
    out = list(prompt_polarity(api, None))
    roles = {r for p in out for r in p.roles}
    assert "prompt" in roles
    assert "negative_prompt" not in roles


def test_prompt_polarity_emits_nothing_when_no_text_encoder():
    api = dict([_api("1", "Foo", x=1)])
    assert list(prompt_polarity(api, None)) == []


# ── synthetic unit tests: set_node_pairs ──────────────────────────────────────


def _ui(nodes: list[dict], links: list[list]) -> dict:
    return {"nodes": nodes, "links": links}


def test_set_node_pairs_tags_upstream_widgets_at_headline_tier():
    api = {
        "11": {
            "class_type": "VHS_LoadVideoFFmpeg",
            "inputs": {"video": "AstronautWalking2.mp4", "force_rate": 16},
        },
        "64": {"class_type": "SetNode", "inputs": {}},
    }
    ui = _ui(
        nodes=[
            {"id": 11, "type": "VHS_LoadVideoFFmpeg",
             "outputs": [{"name": "IMAGE", "links": [100]}]},
            {"id": 64, "type": "SetNode", "title": "Set_Input_Video",
             "inputs": [{"name": "*", "link": 100}], "widgets_values": ["Input_Video"]},
        ],
        links=[[100, 11, 0, 64, 0, "IMAGE"]],
    )
    out = list(set_node_pairs(api, ui))
    assert all(p.tier == TIER_HEADLINE for p in out)
    assert {p.widget_name for p in out} == {"video", "force_rate"}
    assert all(p.roles == {"set:input_video"} for p in out)
    assert all(p.node_id == "11" for p in out)


def test_set_node_pairs_skips_when_ui_missing():
    api = {"1": {"class_type": "Foo", "inputs": {"x": 1}}}
    assert list(set_node_pairs(api, ui=None)) == []


def test_set_node_pairs_ignores_non_set_titles():
    ui = _ui(
        nodes=[
            {"id": 1, "type": "VHS_LoadVideoFFmpeg",
             "outputs": [{"name": "IMAGE", "links": [9]}]},
            {"id": 2, "type": "SetNode", "title": "MyNode",
             "inputs": [{"name": "*", "link": 9}], "widgets_values": ["X"]},
        ],
        links=[[9, 1, 0, 2, 0, "IMAGE"]],
    )
    api = {"1": {"class_type": "VHS_LoadVideoFFmpeg", "inputs": {"video": "x.mp4"}}}
    assert list(set_node_pairs(api, ui)) == []


def test_set_node_pairs_skips_when_input_link_missing():
    ui = _ui(
        nodes=[
            {"id": 2, "type": "SetNode", "title": "Set_Foo",
             "inputs": [{"name": "*", "link": None}]},
        ],
        links=[],
    )
    assert list(set_node_pairs({}, ui)) == []


def test_set_node_pairs_skips_when_source_widget_is_a_link():
    """The source's own connected inputs (links) should not be promoted."""
    api = {
        "1": {
            "class_type": "VHS_LoadVideoFFmpeg",
            "inputs": {"video": "x.mp4", "image_input": ["99", 0]},
        },
        "2": {"class_type": "SetNode", "inputs": {}},
    }
    ui = _ui(
        nodes=[
            {"id": 1, "type": "VHS_LoadVideoFFmpeg",
             "outputs": [{"name": "IMAGE", "links": [10]}]},
            {"id": 2, "type": "SetNode", "title": "Set_Foo",
             "inputs": [{"name": "*", "link": 10}], "widgets_values": ["Foo"]},
        ],
        links=[[10, 1, 0, 2, 0, "IMAGE"]],
    )
    out = list(set_node_pairs(api, ui))
    assert {p.widget_name for p in out} == {"video"}


def test_set_node_pairs_slugifies_complex_names():
    ui = _ui(
        nodes=[
            {"id": 1, "type": "Foo", "outputs": [{"name": "x", "links": [1]}]},
            {"id": 2, "type": "SetNode", "title": "Set_My-Cool Name!",
             "inputs": [{"name": "*", "link": 1}], "widgets_values": ["x"]},
        ],
        links=[[1, 1, 0, 2, 0, "X"]],
    )
    api = {"1": {"class_type": "Foo", "inputs": {"v": 1}}}
    out = list(set_node_pairs(api, ui))
    assert out and out[0].roles == {"set:my_cool_name"}


# ── synthetic unit tests: primitive_nodes ─────────────────────────────────────


def test_primitive_nodes_typed_primitive_int_tagged_at_headline():
    api = {
        "1": {
            "class_type": "PrimitiveInt",
            "inputs": {"value": 42},
            "_meta": {"title": "My Steps"},
        },
    }
    out = list(primitive_nodes(api, None))
    assert len(out) == 1
    p = out[0]
    assert p.roles == {"primitive:my_steps"}
    assert p.tier == TIER_HEADLINE
    assert p.value == 42
    assert p.label == "My Steps"


def test_primitive_nodes_typed_primitive_uses_node_id_when_untitled():
    api = {"5": {"class_type": "PrimitiveString", "inputs": {"value": "hi"}}}
    out = list(primitive_nodes(api, None))
    assert out[0].roles == {"primitive:node_5"}


def test_primitive_nodes_legacy_primitivenode_tags_consumer_widget():
    """Legacy PrimitiveNode is virtual; its value lands on the consumer's widget."""
    api = {
        "20": {
            "class_type": "KSampler",
            "inputs": {"seed": 1234, "steps": 20},
        },
    }
    ui = _ui(
        nodes=[
            {"id": 10, "type": "PrimitiveNode", "title": "User Seed",
             "outputs": [{"name": "INT", "links": [50]}],
             "widgets_values": [1234]},
            {"id": 20, "type": "KSampler",
             "inputs": [{"name": "seed", "link": 50, "widget": {"name": "seed"}}]},
        ],
        links=[[50, 10, 0, 20, 0, "INT"]],
    )
    out = list(primitive_nodes(api, ui))
    assert len(out) == 1
    p = out[0]
    assert p.node_id == "20"
    assert p.widget_name == "seed"
    assert p.roles == {"primitive:user_seed"}
    assert p.tier == TIER_HEADLINE


def test_primitive_nodes_skips_typed_primitives_when_value_is_a_link():
    api = {
        "1": {"class_type": "PrimitiveInt", "inputs": {"value": ["99", 0]}},
    }
    assert list(primitive_nodes(api, None)) == []


# ── synthetic unit tests: easy_pack_nodes ─────────────────────────────────────


def test_easy_pack_nodes_tags_easy_seed_as_seed_role():
    api = {"3": {"class_type": "easy seed", "inputs": {"seed": 42}}}
    out = list(easy_pack_nodes(api, None))
    assert len(out) == 1
    assert out[0].roles == {"seed"}
    assert out[0].tier == TIER_HEADLINE


def test_easy_pack_nodes_tags_easy_positive_negative_as_prompt_roles():
    api = {
        "4": {"class_type": "easy positive", "inputs": {"positive": "good"}},
        "5": {"class_type": "easy negative", "inputs": {"negative": "bad"}},
    }
    out = list(easy_pack_nodes(api, None))
    by_role = {next(iter(p.roles)): p for p in out}
    assert by_role["prompt"].value == "good"
    assert by_role["negative_prompt"].value == "bad"
    assert by_role["prompt"].tier == TIER_HEADLINE
    assert by_role["negative_prompt"].tier == TIER_HEADLINE


def test_easy_pack_nodes_ignores_unrelated_easy_classes():
    api = {"7": {"class_type": "easy showAnything", "inputs": {"any": "x"}}}
    assert list(easy_pack_nodes(api, None)) == []


# ── synthetic unit tests: discover() merges roles across predicates ───────────


def test_discover_merges_roles_from_class_type_and_polarity():
    api = {
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0], "seed": 42}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "hi"}},
    }
    params = discover(api)
    text = params_by_address(params, "6", "text")
    assert text is not None
    # Picked up by both class_type_roles (text_encode) and prompt_polarity (prompt)
    assert text.roles == {"text_encode", "prompt"}
    assert "class_type_roles" in text.source_predicates
    assert "prompt_polarity" in text.source_predicates
    assert "frontend_widget_pool" in text.source_predicates


def test_discover_lifts_tier_to_common_for_role_tagged_params():
    api = dict([_api("3", "KSampler", seed=42, steps=20, model=["1", 0])])
    params = discover(api)
    seed = params_by_address(params, "3", "seed")
    assert seed.tier == TIER_COMMON


def test_discover_keeps_advanced_tier_for_unknown_class_widgets():
    api = dict([_api("5", "WanVideoBlockSwap", blocks_to_swap=25, offload_img=True)])
    params = discover(api)
    for p in params:
        assert p.tier == TIER_ADVANCED


def test_params_by_role_finds_role_tagged_params():
    api = dict([_api("3", "KSampler", seed=42, steps=20, cfg=8.0, model=["1", 0])])
    params = discover(api)
    assert len(params_by_role(params, "seed")) == 1
    assert len(params_by_role(params, "steps")) == 1
    assert len(params_by_role(params, "cfg")) == 1


# ── parameterized real-workflow tests ─────────────────────────────────────────
#
# Add entries to ``WORKFLOW_FIXTURES`` to extend coverage. Each name maps to
# ``tests/data/workflows/<name>.json``. An optional sidecar
# ``tests/data/workflows/<name>.expected.json`` lets a fixture additionally
# assert ``min_total_params`` (int) and ``roles_present`` (list[str]); when
# absent, only the universal invariants run. Anything that doesn't fit those
# two keys belongs in a normal pytest function below — write the assertion
# directly rather than extending the sidecar schema.

_WORKFLOWS_DIR = Path(__file__).parent.parent / "data" / "workflows"

WORKFLOW_FIXTURES: list[str] = [
    "yt_bgswap_v01",
]


def _load_workflow(name: str) -> dict:
    """Load ``<name>.json``, booting the node system if UI-format.

    `convert_ui_to_api` needs `import_all_nodes_in_workspace` to populate
    the node registry. Unknown classes (e.g. WanVideoWrapper) survive via
    `preserve_unknown_nodes=True`.
    """
    workflow = json.loads((_WORKFLOWS_DIR / f"{name}.json").read_text())
    if "nodes" in workflow and "links" in workflow:
        from comfy.nodes.package import import_all_nodes_in_workspace
        from comfy.nodes_context import get_nodes
        if len(get_nodes()) == 0:
            import_all_nodes_in_workspace()
    return workflow


def _maybe_load_expected(name: str) -> dict | None:
    path = _WORKFLOWS_DIR / f"{name}.expected.json"
    return json.loads(path.read_text()) if path.exists() else None


@pytest.mark.parametrize("name", WORKFLOW_FIXTURES, ids=WORKFLOW_FIXTURES)
def test_real_workflow(name: str):
    workflow = _load_workflow(name)
    params = discover(workflow)

    # Universal invariants: every param resolves and none holds a link value.
    from comfy.entrypoints.workflow_params import _is_link, _to_api
    api, _ = _to_api(workflow)
    for p in params:
        assert p.node_id in api, (
            f"param node_id {p.node_id!r} missing from API workflow"
        )
        assert p.widget_name in (api[p.node_id].get("inputs") or {}), (
            f"param widget {p.widget_name!r} missing on node {p.node_id} ({p.class_type})"
        )
        assert not _is_link(p.value), (
            f"param {p.address} has a link value {p.value!r}"
        )

    expected = _maybe_load_expected(name)
    if expected is None:
        return
    if "min_total_params" in expected:
        assert len(params) >= expected["min_total_params"], (
            f"discover() returned {len(params)} params; expected at least "
            f"{expected['min_total_params']}"
        )
    for role in expected.get("roles_present", []):
        assert params_by_role(params, role), f"role {role!r} not present"


# Case-specific assertions that don't fit the sidecar schema go here as plain
# tests. Keep them thin and named after what they verify.

def test_yt_bgswap_v01_image_loader_is_loadimage_node_42():
    workflow = _load_workflow("yt_bgswap_v01")
    params = discover(workflow)
    image_inputs = params_by_role(params, "image_input")
    assert len(image_inputs) == 1
    assert image_inputs[0].node_id == "42"
    assert image_inputs[0].class_type == "LoadImage"


def test_yt_bgswap_v01_set_node_pairs_promote_named_variables_to_headline():
    """Set_Model, Set_Prompt, Set_VAE, Set_CLIP, Set_VideoIn each bless their
    upstream loader/encoder node, which lands them at headline tier with a
    set:<name> role. The exact source nodes are baked into the workflow:
    Set_Model←6 (WanVideoModelLoader), Set_VAE←8 (WanVideoVAELoader),
    Set_CLIP←7 (LoadWanVideoT5TextEncoder), Set_VideoIn←11 (VHS_LoadVideoFFmpeg),
    Set_Prompt←73 (WanVideoTextEncode).
    """
    workflow = _load_workflow("yt_bgswap_v01")
    params = discover(workflow)

    expected = {
        "set:model":    ("6",  "WanVideoModelLoader"),
        "set:vae":      ("8",  "WanVideoVAELoader"),
        "set:clip":     ("7",  "LoadWanVideoT5TextEncoder"),
        "set:videoin":  ("11", "VHS_LoadVideoFFmpeg"),
        "set:prompt":   ("73", "WanVideoTextEncode"),
    }
    for role, (expected_node_id, expected_class) in expected.items():
        matches = params_by_role(params, role)
        assert matches, f"role {role!r} has no params"
        assert all(p.tier == TIER_HEADLINE for p in matches), (
            f"role {role!r} should be headline tier"
        )
        assert all(p.node_id == expected_node_id for p in matches), (
            f"role {role!r} should target node {expected_node_id} ({expected_class})"
        )
        assert all(p.class_type == expected_class for p in matches)
