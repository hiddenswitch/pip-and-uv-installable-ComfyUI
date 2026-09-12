from comfy import model_management


def test_share_pinned_memory_budget_divides_the_host_budget(monkeypatch):
    monkeypatch.setattr(model_management, "MAX_PINNED_MEMORY", 100 * 1024 ** 2)
    monkeypatch.setattr(model_management, "_UNSHARED_MAX_PINNED_MEMORY", None)

    model_management.share_pinned_memory_budget(4)
    assert model_management.MAX_PINNED_MEMORY == 25 * 1024 ** 2

    # Re-sharing derives from the original host budget, not the shared one.
    model_management.share_pinned_memory_budget(2)
    assert model_management.MAX_PINNED_MEMORY == 50 * 1024 ** 2


def test_share_pinned_memory_budget_is_a_no_op_for_one_process_or_disabled_pinning(monkeypatch):
    monkeypatch.setattr(model_management, "MAX_PINNED_MEMORY", 100 * 1024 ** 2)
    monkeypatch.setattr(model_management, "_UNSHARED_MAX_PINNED_MEMORY", None)
    model_management.share_pinned_memory_budget(1)
    assert model_management.MAX_PINNED_MEMORY == 100 * 1024 ** 2

    monkeypatch.setattr(model_management, "MAX_PINNED_MEMORY", -1)
    model_management.share_pinned_memory_budget(3)
    assert model_management.MAX_PINNED_MEMORY == -1
