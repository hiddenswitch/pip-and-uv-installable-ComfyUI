from types import SimpleNamespace

from comfy.model_prefetch import _dynamic_vbar_modules


def test_dynamic_vbar_modules_excludes_unallocated_modules():
    unallocated = SimpleNamespace(_v=None)
    allocated = SimpleNamespace(_v=(object(), 0, 1))
    module = SimpleNamespace(modules=lambda: [unallocated, allocated, SimpleNamespace()])

    assert _dynamic_vbar_modules(module) == [allocated]
