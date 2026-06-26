from __future__ import annotations

import torch

from comfy.weight_cast import GraphVisibleWeightCastRuntime
from comfy.weight_cast_ops import module_key_tensor, register_module_with_stable_key


def test_module_key_tensor_tracks_stable_registered_key() -> None:
    module = torch.nn.Linear(2, 2)

    key = register_module_with_stable_key(module, "tests.unit.weight_cast.stable")
    key_tensor = module_key_tensor(module)

    assert key_tensor.dtype == torch.int64
    assert key_tensor.ndim == 0
    assert int(key_tensor.item()) == key


def test_stable_registered_key_reuses_key_tensor() -> None:
    module = torch.nn.Linear(2, 2)

    key = register_module_with_stable_key(module, "tests.unit.weight_cast.stable")
    first_tensor = module_key_tensor(module)
    second_key = register_module_with_stable_key(module, "tests.unit.weight_cast.stable")
    second_tensor = module_key_tensor(module)

    assert second_key == key
    assert second_tensor is first_tensor


def test_graph_visible_resident_modules_reuse_invocation_id() -> None:
    module = torch.nn.Linear(2, 2)

    assert not GraphVisibleWeightCastRuntime._needs_unique_invocation(module)

    module.weight_function = [lambda weight: weight]
    assert GraphVisibleWeightCastRuntime._needs_unique_invocation(module)
