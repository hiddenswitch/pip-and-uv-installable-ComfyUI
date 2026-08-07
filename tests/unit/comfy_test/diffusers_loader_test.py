import torch

from comfy.nodes.base_nodes import DiffusersLoader


def test_diffusers_loader_uses_root_sharded_checkpoint(monkeypatch, tmp_path):
    model_dir = tmp_path / "HiDream-O1-Image-Dev"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    index_path = model_dir / "model.safetensors.index.json"
    index_path.write_text('{"weight_map": {}}', encoding="utf-8")

    calls = {}

    def fake_load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        output_clipvision=False,
        embedding_directory=None,
        output_model=True,
        model_options=None,
        te_model_options=None,
        disable_dynamic=False,
    ):
        calls["ckpt_path"] = ckpt_path
        calls["output_vae"] = output_vae
        calls["output_clip"] = output_clip
        calls["model_options"] = model_options
        return ("model", "clip", "vae", None)

    monkeypatch.setattr("comfy.nodes.base_nodes.folder_paths.get_folder_paths", lambda folder: [str(tmp_path)] if folder == "diffusers" else [])
    monkeypatch.setattr("comfy.nodes.base_nodes.sd.load_checkpoint_guess_config", fake_load_checkpoint_guess_config)

    assert DiffusersLoader().load_checkpoint(model_dir.name, weight_dtype="fp8_e4m3fn") == ("model", "clip", "vae")
    assert calls["ckpt_path"] == str(index_path)
    assert calls["output_vae"] is True
    assert calls["output_clip"] is True
    assert calls["model_options"]["dtype"] is torch.float8_e4m3fn
