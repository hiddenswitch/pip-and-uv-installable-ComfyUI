"""`comfyui workflows submit` must share prompt-mutation plumbing with
`run-workflow` so every override (--add-lora, --compile, --image, …) is
applied before POSTing to the daemon. Guards against regressions where
the submit branch grows its own copy of the override pipeline.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
from pathlib import Path
from unittest import mock

import pytest

from comfy.cli_args_types import Configuration
from comfy.cmd.sub_workflows import _submit_workflows


def _sdxl_wf() -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
            "seed": 0, "steps": 20, "cfg": 7.0, "sampler_name": "euler",
            "scheduler": "normal", "denoise": 1.0,
        }},
        "6": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": "test"}},
    }


def _run_submit(**kwargs) -> dict:
    """Call _submit_workflows with every override off by default; capture the
    prompt dict that would have been POSTed and return it.

    Runs in a dedicated thread with its own event loop so tests work even
    when the pytest-asyncio session loop or Playwright's greenlet loop is
    active on the current thread.
    """
    captured: dict = {}

    async def fake_post_json(server, path, body=None):
        captured["body"] = body
        captured["server"] = server
        captured["path"] = path
        return {"prompt_id": "abc123"}

    workflows = kwargs.pop("workflows", [])
    server = kwargs.pop("server", None)
    # Tests historically used `set_overrides`; map to the Configuration field name `set`.
    if "set_overrides" in kwargs:
        kwargs["set"] = kwargs.pop("set_overrides")
    config = Configuration(**{k: v for k, v in kwargs.items() if v is not None})

    def _thread_target():
        with mock.patch("comfy.cmd.server_connection.post_json", new=fake_post_json):
            asyncio.run(_submit_workflows(workflows, server, config))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_thread_target).result()

    return captured


@pytest.fixture
def wf_path(tmp_path) -> Path:
    p = tmp_path / "wf.json"
    p.write_text(json.dumps(_sdxl_wf()))
    return p


class TestSubmitParity:
    def test_prompt_override(self, wf_path):
        captured = _run_submit(workflows=[str(wf_path)], prompt="cat on a sofa")
        # Both CLIPTextEncode nodes get rewritten in a single-encoder workflow;
        # at minimum the positive one must carry the new text.
        assert any(n["inputs"].get("text") == "cat on a sofa"
                   for n in captured["body"].values()
                   if n["class_type"] == "CLIPTextEncode")

    def test_seed_override(self, wf_path):
        captured = _run_submit(workflows=[str(wf_path)], seed=12345)
        assert captured["body"]["5"]["inputs"]["seed"] == 12345

    def test_dimensions_override(self, wf_path):
        captured = _run_submit(workflows=[str(wf_path)], width=1024, height=768)
        assert captured["body"]["4"]["inputs"]["width"] == 1024
        assert captured["body"]["4"]["inputs"]["height"] == 768

    def test_image_override_rewrites_loader(self, wf_path):
        captured = _run_submit(
            workflows=[str(wf_path)],
            image=["https://example.com/new.png"],
        )
        # LoadImage is converted to LoadImageFromURL on URL input.
        load_nodes = [n for n in captured["body"].values()
                      if n["class_type"] in ("LoadImage", "LoadImageFromURL")]
        assert len(load_nodes) == 1
        assert load_nodes[0]["class_type"] == "LoadImageFromURL"
        assert load_nodes[0]["inputs"]["value"] == "https://example.com/new.png"

    def test_add_lora_splices_lora(self, wf_path):
        captured = _run_submit(
            workflows=[str(wf_path)],
            add_lora=["mine.safetensors:0.6"],
        )
        loras = [n for n in captured["body"].values()
                 if n["class_type"] == "LoraLoader"
                 and n["inputs"].get("lora_name") == "mine.safetensors"]
        assert len(loras) == 1
        assert loras[0]["inputs"]["strength_model"] == 0.6
        # Splice lands between Checkpoint and KSampler: the sampler's model
        # ref now points at the new LoRA node.
        ksampler = captured["body"]["5"]
        new_id = next(nid for nid, n in captured["body"].items()
                      if n is loras[0])
        assert ksampler["inputs"]["model"] == [new_id, 0]

    def test_compile_wraps_model_chain(self, wf_path):
        captured = _run_submit(workflows=[str(wf_path)], compile=True)
        compiles = [(nid, n) for nid, n in captured["body"].items()
                    if n["class_type"] == "TorchCompileModel"]
        assert len(compiles) == 1
        new_id, _ = compiles[0]
        # KSampler's model input now points at the compile node.
        assert captured["body"]["5"]["inputs"]["model"] == [new_id, 0]

    def test_lora_then_compile_compose(self, wf_path):
        captured = _run_submit(
            workflows=[str(wf_path)],
            add_lora=["mine.safetensors"],
            compile=True,
        )
        compile_node = next(n for n in captured["body"].values()
                            if n["class_type"] == "TorchCompileModel")
        # Compile's input is the new LoraLoader (latest predecessor of the
        # sampler), so the traced graph covers the LoRA.
        ref_id, _ = compile_node["inputs"]["model"]
        ref = captured["body"][ref_id]
        assert ref["class_type"] == "LoraLoader"

    def test_compile_wraps_model_before_guider(self):
        from comfy.component_model.prompt_utils import enable_compile

        prompt = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "model.safetensors"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "prompt"},
            },
            "3": {
                "class_type": "CFGGuider",
                "inputs": {"cfg": 5, "model": ["1", 0], "positive": ["2", 0]},
            },
            "4": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {"guider": ["3", 0]},
            },
        }

        compiled = enable_compile(prompt)
        compile_id = next(
            nid for nid, node in compiled.items()
            if node["class_type"] == "TorchCompileModel"
        )

        assert compiled[compile_id]["inputs"]["model"] == ["1", 0]
        assert compiled["3"]["inputs"]["model"] == [compile_id, 0]
        assert compiled["4"]["inputs"]["guider"] == ["3", 0]

    def test_set_override(self, wf_path):
        captured = _run_submit(
            workflows=[str(wf_path)],
            set_overrides=["5.inputs.cfg=3.5"],
        )
        assert captured["body"]["5"]["inputs"]["cfg"] == 3.5

    def test_multiple_workflows_each_get_posted(self, wf_path, tmp_path):
        wf2 = tmp_path / "wf2.json"
        wf2.write_text(json.dumps(_sdxl_wf()))

        posts: list[dict] = []

        async def fake_post_json(server, path, body=None):
            posts.append(body)
            return {"prompt_id": f"id_{len(posts)}"}

        def _thread_target():
            with mock.patch("comfy.cmd.server_connection.post_json", new=fake_post_json):
                config = Configuration(prompt="cat", seed=1)
                asyncio.run(_submit_workflows(
                    [str(wf_path), str(wf2)], None, config,
                ))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_thread_target).result()

        assert len(posts) == 2
        for body in posts:
            assert body["5"]["inputs"]["seed"] == 1

    def test_quantity_posts_multiple_seeded_jobs(self, wf_path):
        posts: list[dict] = []

        async def fake_post_json(server, path, body=None):
            posts.append(body)
            return {"prompt_id": f"id_{len(posts)}"}

        def _thread_target():
            with mock.patch("comfy.cmd.server_connection.post_json", new=fake_post_json):
                config = Configuration(quantity=3, seed=50)
                asyncio.run(_submit_workflows([str(wf_path)], None, config))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_thread_target).result()

        assert [body["5"]["inputs"]["seed"] for body in posts] == [50, 51, 52]
