import torch

from comfy import memory_management


def test_aimdo_enabled_is_a_module_property(monkeypatch):
    monkeypatch.setattr(memory_management, "aimdo_allocator", None)
    assert memory_management.aimdo_enabled is False
    monkeypatch.setattr(memory_management, "aimdo_allocator", object())
    assert memory_management.aimdo_enabled is True


def test_stable_audio_3_vae_forces_full_load_without_dynamic_vram(monkeypatch):
    """The SA3 audio VAE branch reads ``memory_management.aimdo_enabled`` as an
    attribute; it must resolve to the property value, not a callable."""
    from comfy import sd

    class _StubVAE(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(sd, "SA3AudioVAE", _StubVAE)
    state = {"decoder.layers.3.transformers.0.pre_norm.alpha": torch.zeros(1)}

    monkeypatch.setattr(memory_management, "aimdo_allocator", None)
    assert sd.VAE(sd=dict(state), device=torch.device("cpu")).disable_offload is True

    monkeypatch.setattr(memory_management, "aimdo_allocator", object())
    assert sd.VAE(sd=dict(state), device=torch.device("cpu")).disable_offload is False
