"""Regression tests for fused RMS normalization and rotary embedding paths."""

import pytest
import torch
from torch import nn

import comfy.ldm.joyimage.model as joyimage_model
import comfy.ldm.lumina.model as lumina_model
from comfy import ops


class _FixedProjection(nn.Module):
    def __init__(self, output_features: int):
        super().__init__()
        self.output_features = output_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = torch.arange(
            inputs.shape[0] * inputs.shape[1] * self.output_features,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        return values.reshape(inputs.shape[0], inputs.shape[1], self.output_features)


class _TrackingNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return inputs


def _fake_cast(calls):
    def cast(norm, inputs, *, offloadable):
        calls.append(("cast", norm, inputs.shape, offloadable))
        return norm.weight, None, f"stream-{len(calls)}"

    return cast


def _fake_uncast(calls):
    def uncast(norm, weight, bias, stream):
        calls.append(("uncast", norm, weight, bias, stream))

    return uncast


def _fake_lumina_attention(query, _key, _value, _heads, _mask, **_kwargs):
    return query.movedim(1, 2).flatten(2)


def _make_lumina_attention(*, heads: int, kv_heads: int) -> lumina_model.JointAttention:
    attention = lumina_model.JointAttention.__new__(lumina_model.JointAttention)
    nn.Module.__init__(attention)
    attention.n_local_heads = heads
    attention.n_local_kv_heads = kv_heads
    attention.n_kv_heads = kv_heads
    attention.n_rep = heads // kv_heads
    attention.head_dim = 2
    attention.qk_norm = True
    attention.qkv = _FixedProjection((heads + 2 * kv_heads) * attention.head_dim)
    attention.out = nn.Identity()
    attention.q_norm = _TrackingNorm(attention.head_dim)
    attention.k_norm = _TrackingNorm(attention.head_dim)
    return attention


def test_lumina_fused_rms_rope_equal_heads_casts_and_uncasts(monkeypatch):
    calls = []
    attention = _make_lumina_attention(heads=2, kv_heads=2)

    def rms_rope(query, key, rope, query_scale, key_scale, epsilon):
        calls.append(("rms_rope", query.shape, key.shape, rope, query_scale, key_scale, epsilon))
        return query, key

    monkeypatch.setattr(lumina_model.model_management, "in_training", False)
    monkeypatch.setattr(lumina_model.ops, "cast_bias_weight", _fake_cast(calls))
    monkeypatch.setattr(lumina_model.ops, "uncast_bias_weight", _fake_uncast(calls))
    monkeypatch.setattr(lumina_model.quant_ops.ck, "rms_rope", rms_rope)
    monkeypatch.setattr(
        lumina_model.quant_ops.ck,
        "rms_rope1",
        lambda *_args, **_kwargs: pytest.fail("GQA kernel must not run for equal head counts"),
    )
    monkeypatch.setattr(lumina_model, "optimized_attention_masked", _fake_lumina_attention)
    monkeypatch.setattr(
        lumina_model,
        "apply_rope",
        lambda *_args, **_kwargs: pytest.fail("fallback rope must not run during inference"),
    )

    result = attention(torch.zeros(1, 3, 4), None, torch.ones(1))

    assert result.shape == (1, 3, 4)
    assert [call[0] for call in calls] == ["cast", "cast", "rms_rope", "uncast", "uncast"]
    assert attention.q_norm.forward_calls == 0
    assert attention.k_norm.forward_calls == 0


def test_lumina_fused_rms_rope_gqa_uses_single_tensor_kernel(monkeypatch):
    calls = []
    attention = _make_lumina_attention(heads=2, kv_heads=1)

    def rms_rope1(tensor, rope, scale, epsilon):
        calls.append(("rms_rope1", tensor.shape, rope, scale, epsilon))
        return tensor

    monkeypatch.setattr(lumina_model.model_management, "in_training", False)
    monkeypatch.setattr(lumina_model.ops, "cast_bias_weight", _fake_cast(calls))
    monkeypatch.setattr(lumina_model.ops, "uncast_bias_weight", _fake_uncast(calls))
    monkeypatch.setattr(lumina_model.quant_ops.ck, "rms_rope1", rms_rope1)
    monkeypatch.setattr(
        lumina_model.quant_ops.ck,
        "rms_rope",
        lambda *_args, **_kwargs: pytest.fail("equal-head kernel must not run for GQA"),
    )
    monkeypatch.setattr(lumina_model, "optimized_attention_masked", _fake_lumina_attention)

    result = attention(torch.zeros(1, 3, 4), None, torch.ones(1))

    assert result.shape == (1, 3, 4)
    assert [call[0] for call in calls].count("rms_rope1") == 2


def test_lumina_training_uses_norm_and_rope_fallback(monkeypatch):
    attention = _make_lumina_attention(heads=2, kv_heads=2)
    rope_calls = []

    def apply_rope(query, key, rope):
        rope_calls.append(rope)
        return query, key

    monkeypatch.setattr(lumina_model.model_management, "in_training", True)
    monkeypatch.setattr(
        lumina_model.ops,
        "cast_bias_weight",
        lambda *_args, **_kwargs: pytest.fail("fused weight casting must not run while training"),
    )
    monkeypatch.setattr(lumina_model, "apply_rope", apply_rope)
    monkeypatch.setattr(lumina_model, "optimized_attention_masked", _fake_lumina_attention)

    attention(torch.zeros(1, 3, 4), None, torch.ones(1))

    assert attention.q_norm.forward_calls == 1
    assert attention.k_norm.forward_calls == 1
    assert len(rope_calls) == 1


def test_joyimage_fused_rms_rope_uses_relative_ops_import(monkeypatch):
    calls = []
    attention = joyimage_model.JoyImageAttention(
        dim=4,
        num_attention_heads=2,
        attention_head_dim=2,
        operations=ops.disable_weight_init,
    )

    def rms_rope(query, key, rope, query_scale, key_scale, epsilon):
        calls.append(("rms_rope", query.shape, key.shape, rope, query_scale, key_scale, epsilon))
        return query, key

    monkeypatch.setattr(joyimage_model.ops, "cast_bias_weight", _fake_cast(calls))
    monkeypatch.setattr(joyimage_model.ops, "uncast_bias_weight", _fake_uncast(calls))
    monkeypatch.setattr(joyimage_model.comfy_kitchen, "rms_rope", rms_rope)
    monkeypatch.setattr(
        joyimage_model,
        "optimized_attention",
        lambda query, _key, _value, **_kwargs: query,
    )

    image, text = attention(
        torch.zeros(1, 2, 4),
        torch.zeros(1, 3, 4),
        torch.ones(1),
    )

    assert image.shape == (1, 2, 4)
    assert text.shape == (1, 3, 4)
    assert [call[0] for call in calls] == ["cast", "cast", "rms_rope", "uncast", "uncast"]
