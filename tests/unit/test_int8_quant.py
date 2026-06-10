import json
import unittest

import torch


def has_gpu():
    return torch.cuda.is_available()


from comfy.cli_args import args  # noqa: E402

if not has_gpu():
    args.cpu = True

from comfy import ops  # noqa: E402
from comfy.quant_ops import QuantizedTensor, int8_quantization_available  # noqa: E402
import comfy.utils  # noqa: E402

if int8_quantization_available():
    from comfy.quant_ops_int8 import (
        Int8ConvRotLayout,
        Int8RowwiseLayout,
        regular_hadamard,
        rotate_groups,
    )

requires_int8 = unittest.skipUnless(int8_quantization_available(), "requires comfy_kitchen-backed quantized tensors")


def _comfy_quant_tensor(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _read_conf(t: torch.Tensor) -> dict:
    return json.loads(t.numpy().tobytes())


@requires_int8
class TestRegularHadamard(unittest.TestCase):
    def test_properties(self):
        for n in (4, 16, 64, 256):
            h = regular_hadamard(n)
            self.assertTrue(torch.allclose(h, h.t()), f"H{n} not symmetric")
            self.assertTrue(torch.allclose(h @ h, torch.eye(n), atol=1e-5), f"H{n} not self-inverse")
            # normalized rows have unit norm
            self.assertTrue(torch.allclose(h.norm(dim=1), torch.ones(n), atol=1e-5))

    def test_rejects_non_power_of_four(self):
        for n in (2, 8, 32, 100):
            with self.assertRaises(ValueError):
                regular_hadamard(n)

    def test_rotation_is_exact_involution(self):
        x = torch.randn(8, 512)
        rotated = rotate_groups(x, 256)
        self.assertFalse(torch.allclose(rotated, x))
        back = rotate_groups(rotated, 256)
        self.assertTrue(torch.allclose(back, x, atol=1e-4))

    def test_rotation_preserves_matmul(self):
        x = torch.randn(8, 512)
        w = torch.randn(32, 512)
        ref = x @ w.t()
        out = rotate_groups(x, 256) @ rotate_groups(w, 256).t()
        self.assertTrue(torch.allclose(out, ref, atol=1e-3))

    def test_rotation_rejects_indivisible(self):
        with self.assertRaises(ValueError):
            rotate_groups(torch.randn(4, 100), 256)


@requires_int8
class TestInt8Layouts(unittest.TestCase):
    def test_rowwise_roundtrip(self):
        w = torch.randn(64, 512, dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "Int8RowwiseLayout")
        self.assertEqual(qt._qdata.dtype, torch.int8)
        self.assertEqual(tuple(qt._params.scale.shape), (64, 1))
        dq = qt.dequantize()
        self.assertEqual(dq.dtype, torch.bfloat16)
        rel = (dq.float() - w.float()).abs().max() / w.float().abs().max()
        self.assertLess(rel.item(), 0.02)

    def test_convrot_roundtrip_returns_original_space(self):
        w = torch.randn(64, 512, dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "Int8ConvRotLayout")
        dq = qt.dequantize()
        rel = (dq.float() - w.float()).abs().max() / w.float().abs().max()
        self.assertLess(rel.item(), 0.03)

    def test_scalar_scale_legacy_tensorwise(self):
        w = torch.randn(16, 32)
        scale = (w.abs().max() / 127.0).reshape(())
        q = (w / scale).round().clamp(-128, 127).to(torch.int8)
        params = Int8RowwiseLayout.Params(scale=scale, orig_dtype=torch.float32, orig_shape=(16, 32))
        qt = QuantizedTensor(q, "Int8RowwiseLayout", params)
        rel = (qt.dequantize() - w).abs().max() / w.abs().max()
        self.assertLess(rel.item(), 0.02)

    def test_state_dict_tensors(self):
        w = torch.randn(16, 256, dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "Int8ConvRotLayout")
        sd = qt.state_dict("weight")
        self.assertEqual(sorted(sd.keys()), ["weight", "weight_scale"])
        self.assertEqual(sd["weight"].dtype, torch.int8)

    def test_extra_state_dict_conf(self):
        self.assertEqual(Int8RowwiseLayout.extra_state_dict_conf(), {})
        conf = Int8ConvRotLayout.extra_state_dict_conf()
        self.assertTrue(conf["convrot"])
        self.assertEqual(conf["convrot_groupsize"], 256)

    def test_should_quantize_input(self):
        self.assertFalse(Int8RowwiseLayout.should_quantize_input(torch.zeros(16, 64)))
        self.assertTrue(Int8RowwiseLayout.should_quantize_input(torch.zeros(17, 64)))

    def test_stochastic_rounding_seeded(self):
        w = torch.randn(8, 256)
        q1, p1 = Int8RowwiseLayout.quantize(w, stochastic_rounding=1234)
        q2, p2 = Int8RowwiseLayout.quantize(w, stochastic_rounding=1234)
        q3, _ = Int8RowwiseLayout.quantize(w, stochastic_rounding=99)
        self.assertTrue(torch.equal(q1, q2))
        self.assertFalse(torch.equal(q1, q3))
        dq = q1.float() * p1.scale
        self.assertLess(((dq - w).abs().max() / w.abs().max()).item(), 0.03)


@requires_int8
class TestInt8Normalization(unittest.TestCase):
    def _foreign_layer_sd(self, conf=None, prefix="lin."):
        w = torch.randn(32, 256, dtype=torch.bfloat16)
        qdata, params = Int8RowwiseLayout.quantize(w)
        sd = {
            f"{prefix}weight": qdata,
            f"{prefix}weight_scale": params.scale,
            f"{prefix}bias": torch.zeros(32, dtype=torch.bfloat16),
        }
        if conf is not None:
            sd[f"{prefix}comfy_quant"] = _comfy_quant_tensor(conf)
        return sd

    def test_formatless_convrot_json_normalized(self):
        sd = self._foreign_layer_sd({"convrot": True, "convrot_groupsize": 256, "per_row": True})
        out, _ = comfy.utils.convert_old_quants(sd, model_prefix="", metadata={})
        conf = _read_conf(out["lin.comfy_quant"])
        self.assertEqual(conf["format"], "int8_convrot")
        self.assertEqual(conf["convrot_groupsize"], 256)

    def test_bare_int8_weight_scale_normalized(self):
        sd = self._foreign_layer_sd(None)
        out, _ = comfy.utils.convert_old_quants(sd, model_prefix="", metadata={})
        conf = _read_conf(out["lin.comfy_quant"])
        self.assertEqual(conf["format"], "int8")

    def test_existing_format_untouched(self):
        sd = self._foreign_layer_sd({"format": "int8_convrot", "convrot_groupsize": 256})
        out, _ = comfy.utils.convert_old_quants(sd, model_prefix="", metadata={})
        self.assertEqual(_read_conf(out["lin.comfy_quant"])["format"], "int8_convrot")

    def test_float_weight_not_marked(self):
        sd = {"lin.weight": torch.randn(8, 8), "lin.weight_scale": torch.ones(8, 1)}
        out, _ = comfy.utils.convert_old_quants(sd, model_prefix="", metadata={})
        self.assertNotIn("lin.comfy_quant", out)

    def test_detection(self):
        sd = self._foreign_layer_sd(None)
        out, _ = comfy.utils.convert_old_quants(sd, model_prefix="", metadata={})
        self.assertEqual(comfy.utils.detect_layer_quantization(out, ""), {"mixed_ops": True})


@requires_int8
class TestInt8MixedPrecisionLoad(unittest.TestCase):
    def _make_linear(self, fmt="int8_convrot", in_features=256, out_features=32, quant_config=None):
        mpo = ops.mixed_precision_ops(quant_config if quant_config is not None else {"mixed_ops": True}, torch.bfloat16)
        lin = mpo.Linear(in_features, out_features, device="cpu")
        w = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        layout = "Int8ConvRotLayout" if fmt == "int8_convrot" else "Int8RowwiseLayout"
        qdata, params = ops.get_layout_class(layout).quantize(w)
        conf = {"format": fmt}
        if fmt == "int8_convrot":
            conf["convrot_groupsize"] = 256
        sd = {
            "weight": qdata,
            "weight_scale": params.scale,
            "bias": torch.zeros(out_features, dtype=torch.bfloat16),
            "comfy_quant": _comfy_quant_tensor(conf),
        }
        missing, unexpected = lin.load_state_dict(sd, strict=False)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        return lin, w

    def test_load_prequantized(self):
        for fmt in ("int8", "int8_convrot"):
            lin, w = self._make_linear(fmt)
            self.assertIsInstance(lin.weight, QuantizedTensor)
            self.assertEqual(lin.quant_format, fmt)
            x = torch.randn(24, 256, dtype=torch.bfloat16)
            out = lin(x)
            ref = torch.nn.functional.linear(x, w)
            rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
            self.assertLess(rel.item(), 0.05, fmt)

    def test_small_batch_fallback(self):
        lin, w = self._make_linear("int8_convrot")
        x = torch.randn(4, 256, dtype=torch.bfloat16)
        out = lin(x)
        ref = torch.nn.functional.linear(x, w)
        rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel.item(), 0.05)

    def test_lora_bake_dequant_patch_requant(self):
        lin, w = self._make_linear("int8_convrot")
        dq = lin.convert_weight(lin.weight)
        self.assertEqual(dq.dtype, torch.bfloat16)
        # convert_weight returns original (de-rotated) space
        rel = (dq.float() - w.float()).abs().max() / w.float().abs().max()
        self.assertLess(rel.item(), 0.03)
        delta = (torch.randn(32, 4) @ torch.randn(4, 256)).to(torch.bfloat16) * 0.05
        lin.set_weight(dq + delta, seed=4242)
        self.assertIsInstance(lin.weight, QuantizedTensor)
        x = torch.randn(24, 256, dtype=torch.bfloat16)
        out = lin(x)
        ref = torch.nn.functional.linear(x, w + delta)
        rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel.item(), 0.06)

    def test_state_dict_roundtrip(self):
        lin, w = self._make_linear("int8_convrot")
        sd = lin.state_dict(prefix="lin.")
        conf = _read_conf(sd["lin.comfy_quant"])
        self.assertEqual(conf["format"], "int8_convrot")
        self.assertTrue(conf["convrot"])
        self.assertEqual(sd["lin.weight"].dtype, torch.int8)

        lin2 = ops.mixed_precision_ops({"mixed_ops": True}, torch.bfloat16).Linear(256, 32, device="cpu")
        missing, unexpected = lin2.load_state_dict({k[len("lin."):]: v for k, v in sd.items()}, strict=False)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        x = torch.randn(24, 256, dtype=torch.bfloat16)
        self.assertTrue(torch.allclose(lin(x).float(), lin2(x).float(), atol=1e-3))


@requires_int8
class TestQuantizeOnLoad(unittest.TestCase):
    def _ops(self, fmt="int8_convrot", exclude=()):
        return ops.mixed_precision_ops(
            {"mixed_ops": True, "quantize_on_load": fmt, "exclude_layers": tuple(exclude)},
            torch.bfloat16,
        )

    def test_eligible_layer_quantizes(self):
        lin = self._ops().Linear(256, 32, device="cpu")
        w = torch.randn(32, 256, dtype=torch.bfloat16)
        lin.load_state_dict({"weight": w, "bias": torch.zeros(32, dtype=torch.bfloat16)}, strict=False)
        self.assertIsInstance(lin.weight, QuantizedTensor)
        self.assertEqual(lin.quant_format, "int8_convrot")
        x = torch.randn(24, 256, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w)
        rel = (lin(x).float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel.item(), 0.05)

    def test_excluded_layer_stays_float(self):
        mpo = self._ops(exclude=("img_in",))

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.img_in = mpo.Linear(256, 32, device="cpu")

        m = M()
        m.load_state_dict({"img_in.weight": torch.randn(32, 256, dtype=torch.bfloat16),
                           "img_in.bias": torch.zeros(32, dtype=torch.bfloat16)}, strict=False)
        self.assertNotIsInstance(m.img_in.weight, QuantizedTensor)
        self.assertEqual(m.img_in.weight.dtype, torch.bfloat16)

    def test_convrot_indivisible_stays_float(self):
        lin = self._ops().Linear(100, 32, device="cpu")
        lin.load_state_dict({"weight": torch.randn(32, 100, dtype=torch.bfloat16),
                             "bias": torch.zeros(32, dtype=torch.bfloat16)}, strict=False)
        self.assertNotIsInstance(lin.weight, QuantizedTensor)

    def test_plain_int8_quantizes_indivisible(self):
        lin = self._ops(fmt="int8").Linear(100, 32, device="cpu")
        lin.load_state_dict({"weight": torch.randn(32, 100, dtype=torch.bfloat16),
                             "bias": torch.zeros(32, dtype=torch.bfloat16)}, strict=False)
        self.assertIsInstance(lin.weight, QuantizedTensor)

    def test_prequantized_checkpoint_wins(self):
        mpo = self._ops(fmt="int8")
        lin = mpo.Linear(256, 32, device="cpu")
        w = torch.randn(32, 256, dtype=torch.bfloat16)
        qdata, params = ops.get_layout_class("Int8ConvRotLayout").quantize(w)
        lin.load_state_dict({
            "weight": qdata,
            "weight_scale": params.scale,
            "bias": torch.zeros(32, dtype=torch.bfloat16),
            "comfy_quant": _comfy_quant_tensor({"format": "int8_convrot", "convrot_groupsize": 256}),
        }, strict=False)
        self.assertEqual(lin.quant_format, "int8_convrot")


class TestApplyQuantizeOnLoad(unittest.TestCase):
    def _cfg(self):
        class Cfg:
            quant_config = None
            custom_operations = None
            int8_quant_exclude = ("img_in",)

        return Cfg()

    @requires_int8
    def test_sets_quant_config(self):
        from comfy.sd import _apply_quantize_on_load
        cfg = self._cfg()
        _apply_quantize_on_load(cfg, {"quantize_on_load": "int8_convrot"})
        self.assertEqual(cfg.quant_config, {
            "mixed_ops": True,
            "quantize_on_load": "int8_convrot",
            "exclude_layers": ("img_in",),
        })

    @requires_int8
    def test_already_quantized_wins(self):
        from comfy.sd import _apply_quantize_on_load
        cfg = self._cfg()
        cfg.quant_config = {"mixed_ops": True}
        _apply_quantize_on_load(cfg, {"quantize_on_load": "int8"})
        self.assertEqual(cfg.quant_config, {"mixed_ops": True})

    def test_dropdown_options(self):
        from comfy.nodes.base_nodes import get_model_options_for_dtype
        from comfy.ldm.flux.weight_dtypes import FLUX_WEIGHT_DTYPES
        self.assertIn("int8", FLUX_WEIGHT_DTYPES)
        self.assertIn("int8_convrot", FLUX_WEIGHT_DTYPES)
        opts = get_model_options_for_dtype("int8_convrot")
        self.assertEqual(opts, {"quantize_on_load": "int8_convrot"})
        self.assertNotIn("dtype", get_model_options_for_dtype("int8"))


if __name__ == "__main__":
    unittest.main()
