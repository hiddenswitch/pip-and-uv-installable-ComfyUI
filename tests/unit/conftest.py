"""Unit-test session fixtures.

Pre-loads the real node system exactly once per pytest session so the
expensive ``import_all_nodes_in_workspace`` call happens at a predictable
point (before any test executes) and before any test has a chance to
mutate ``sys.modules`` or per-module ``NODE_CLASS_MAPPINGS``. Later
fixtures that need the full node set should reuse this snapshot instead
of calling ``import_all_nodes_in_workspace`` themselves.
"""
import os

import pytest


# These tests exercise the *software* graph-visible weight-cast fallback: the
# prefetch scheduler and graph-visible weight resolution that simulate weight
# offload by setting ``module.weight = None`` and rely on shapes recorded in
# the materialization spec. That fallback is what runs when the Aimdo dynamic
# VRAM allocator is unavailable (Linux CI, most setups). When the real Aimdo
# allocator is active (Windows dynamic VRAM), weight residency is managed by
# the allocator and the live runtime expects a resident weight, so these
# null-the-weight simulations don't apply and are skipped.
_GRAPH_VISIBLE_FALLBACK_TESTS = frozenset({
    "test_comfy_weight_custom_ops_are_present_in_fx_graph",
    "test_comfy_weight_custom_ops_compile_with_eager_backend",
    "test_comfy_weight_custom_ops_track_overlapping_invocations",
    "test_comfy_weight_prefetch_token_is_consumed_by_prefetched_resolve",
    "test_compiled_manual_cast_uses_graph_visible_op_even_when_resident",
    "test_compiled_model_stabilizes_small_manual_cast_parameters_before_compile",
    "test_dynamic_vbar_prefetch_fallback_release_tracks_materialized_tensors",
    "test_graph_visible_runtime_uses_distinct_invocations_for_repeated_module",
    "test_graph_visible_runtime_uses_recorded_materialization_shape",
    "test_manual_cast_compile_tracks_replaced_parameters",
    "test_manual_cast_compile_uses_graph_visible_weight_resolution",
    "test_manual_cast_linear_preserves_eager_output",
    "test_model_patcher_dynamic_records_weight_materialization_spec",
    "test_non_vbar_offload_falls_back_when_shared_cast_buffer_is_unavailable",
    "test_weight_prefetch_scheduler_budgets_from_shapes_when_module_lookup_misses",
    "test_weight_prefetch_scheduler_can_cross_exemplar_dependency",
    "test_weight_prefetch_scheduler_keeps_existing_window_across_demand_resolve",
    "test_weight_prefetch_scheduler_keeps_live_patch_function_on_demand_path",
    "test_weight_prefetch_scheduler_lookahead_zero_leaves_demand_resolves",
    "test_weight_prefetch_scheduler_respects_byte_budget",
    "test_weight_prefetch_scheduler_respects_live_lookahead_window",
    "test_weight_prefetch_scheduler_respects_per_weight_prefetch_cap",
    "test_weight_prefetch_scheduler_rewrites_future_resolves_from_fx_graph",
    "test_weight_prefetch_scheduler_uses_materialization_spec_budget",
})


def pytest_collection_modifyitems(config, items):
    try:
        from comfy import memory_management
        aimdo_active = memory_management.aimdo_allocator is not None
    except Exception:
        aimdo_active = False
    if not aimdo_active:
        return
    skip = pytest.mark.skip(
        reason="graph-visible weight-cast fallback test; the live Aimdo dynamic VRAM "
               "allocator manages residency on this platform"
    )
    for item in items:
        if getattr(item, "originalname", None) in _GRAPH_VISIBLE_FALLBACK_TESTS or item.name in _GRAPH_VISIBLE_FALLBACK_TESTS:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def _preloaded_nodes():
    """Full snapshot of all importable nodes, frozen at session start.

    Copies NODE_CLASS_MAPPINGS into a standalone ExportedNodes so later
    ``import_all_nodes_in_workspace`` calls (which ``.clear()`` the shared
    ``_nodes_local.nodes`` instance) can't erase entries we already saw.
    Normally forces ``disable_all_custom_nodes=False`` in case an earlier
    test has left a Configuration on the current context that would skip
    custom node loading. Device-specific CI can set
    ``COMFYUI_UNIT_DISABLE_CUSTOM_NODES=1`` to isolate core unit tests from
    custom-node import side effects.
    """
    from comfy.cli_args import default_configuration
    from comfy.nodes.package import import_all_nodes_in_workspace
    from comfy.nodes.package_typing import ExportedNodes
    from comfy.execution_context import context_configuration

    cfg = default_configuration()
    disable_custom_nodes = os.environ.get("COMFYUI_UNIT_DISABLE_CUSTOM_NODES", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    cfg.disable_all_custom_nodes = disable_custom_nodes
    with context_configuration(cfg):
        live = import_all_nodes_in_workspace()

    snapshot = ExportedNodes()
    snapshot.NODE_CLASS_MAPPINGS.update(live.NODE_CLASS_MAPPINGS)
    snapshot.NODE_DISPLAY_NAME_MAPPINGS.update(live.NODE_DISPLAY_NAME_MAPPINGS)
    snapshot.EXTENSION_WEB_DIRS.update(live.EXTENSION_WEB_DIRS)
    return snapshot


@pytest.fixture(scope="session", autouse=True)
def _eagerly_preload_nodes(_preloaded_nodes):
    """Force ``_preloaded_nodes`` to be constructed before any test runs.

    Session fixtures are otherwise lazy; we need this one eager because a
    handful of earlier tests in the sweep mutate ``sys.modules`` in ways
    that cause subsequent ``import_all_nodes_in_workspace`` calls to drop
    several hundred custom nodes.
    """
    yield _preloaded_nodes
