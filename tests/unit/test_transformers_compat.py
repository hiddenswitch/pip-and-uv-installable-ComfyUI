from types import SimpleNamespace

from comfy.transformers_compat import patch_transformers_finegrained_fp8_import


def test_patch_transformers_finegrained_fp8_import_stubs_optional_integration(monkeypatch):
    module_name = "transformers.integrations.finegrained_fp8"
    monkeypatch.delitem(__import__("sys").modules, module_name, raising=False)
    torch_module = SimpleNamespace()

    patch_transformers_finegrained_fp8_import(torch_module)

    stub = __import__("sys").modules[module_name]
    assert stub.ALL_FP8_EXPERTS_FUNCTIONS == {}
    assert not hasattr(torch_module, "float8_e8m0fnu")


def test_patch_transformers_finegrained_fp8_import_preserves_supported_torch(monkeypatch):
    module_name = "transformers.integrations.finegrained_fp8"
    monkeypatch.delitem(__import__("sys").modules, module_name, raising=False)
    torch_module = SimpleNamespace(float8_e8m0fnu=object())

    patch_transformers_finegrained_fp8_import(torch_module)

    assert module_name not in __import__("sys").modules
