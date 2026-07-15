"""Qwen Image attention regression tests."""

import torch

import comfy.ldm.qwen_image.model as qwen_image_model
import comfy.ops as comfy_ops


def test_qwen_attention_disables_low_precision_attention(monkeypatch):
    captured_options = {}

    def fake_attention(query, key, value, heads, mask, **kwargs):
        del key, value, heads, mask
        captured_options.update(kwargs)
        batch, attention_heads, sequence, head_dim = query.shape
        return torch.zeros(batch, sequence, attention_heads * head_dim)

    monkeypatch.setattr(qwen_image_model, "optimized_attention_masked", fake_attention)
    monkeypatch.setattr(qwen_image_model, "apply_rope1", lambda tensor, _rope: tensor)

    attention = qwen_image_model.Attention(
        query_dim=8,
        dim_head=4,
        heads=2,
        operations=comfy_ops.disable_weight_init,
    )
    attention(
        hidden_states=torch.zeros(1, 2, 8),
        encoder_hidden_states=torch.zeros(1, 3, 8),
    )

    assert captured_options["low_precision_attention"] is False
