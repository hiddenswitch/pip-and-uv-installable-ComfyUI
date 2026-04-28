"""Quick offline test: registry / filter logic."""
def test_resolve_host_filter_with_host_only():
    from comfy.component_model.workflow_hosts import resolve_host_filter, list_hosts
    all_hosts = list_hosts()
    assert {h.id for h in all_hosts} >= {"civitai", "civitai_red", "comfyui-org"}
    sel = resolve_host_filter(["civitai"], None)
    assert {h.id for h in sel} == {"civitai"}


def test_resolve_host_filter_csv():
    from comfy.component_model.workflow_hosts import resolve_host_filter
    sel = resolve_host_filter(["civitai,comfyui-org"], None)
    assert {h.id for h in sel} == {"civitai", "comfyui-org"}


def test_resolve_host_filter_without_host():
    from comfy.component_model.workflow_hosts import resolve_host_filter
    sel = resolve_host_filter(None, ["civitai_red"])
    assert "civitai_red" not in {h.id for h in sel}
    assert "civitai" in {h.id for h in sel}


def test_comfyui_org_top_returns_results_offline():
    from comfy.component_model.workflow_hosts import get_host
    h = get_host("comfyui-org")
    out = h.top(5)
    assert out and len(out) <= 5
    assert all(r.uri.startswith("comfyui-org://t/") for r in out)


def test_comfyui_org_search_offline():
    from comfy.component_model.workflow_hosts import get_host
    h = get_host("comfyui-org")
    out = h.search("kontext", limit=3)
    assert out
    assert all("kontext" in r.title.lower() for r in out)
