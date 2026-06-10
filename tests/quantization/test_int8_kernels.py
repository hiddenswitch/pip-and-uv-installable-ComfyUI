"""GPU tests for the INT8 w8a8 triton kernels and their compile behavior."""
import unittest

import torch


def has_gpu():
    return torch.cuda.is_available()


from comfy.cli_args import args  # noqa: E402

if not has_gpu():
    args.cpu = True

from comfy import int8_kernels, ops  # noqa: E402
from comfy.quant_ops import QuantizedTensor, int8_quantization_available  # noqa: E402

requires_cuda_int8 = unittest.skipUnless(
    has_gpu() and int8_quantization_available() and int8_kernels.int8_compute_capable(),
    "requires a CUDA device with int8 tensor cores",
)
requires_triton = unittest.skipUnless(int8_kernels.int8_fast_available(), "requires triton")


@requires_cuda_int8
class TestInt8Kernels(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(0)

    def test_quantize_rowwise_matches_eager(self):
        for k in (256, 3072, 40960):  # includes rows far wider than one block
            x = torch.randn(8, k, device=self.device, dtype=torch.bfloat16)
            q, s = torch.ops.comfy_int8.quantize_rowwise(x)
            qe, se = int8_kernels.quantize_rowwise_eager(x)
            self.assertTrue(torch.allclose(s, se, rtol=1e-3), f"k={k}")
            self.assertLessEqual((q.int() - qe.int()).abs().max().item(), 1, f"k={k}")

    def test_gemm_matches_reference(self):
        m, k, n = 128, 512, 384
        x = torch.randn(m, k, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(n, k, device=self.device, dtype=torch.bfloat16)
        bias = torch.randn(n, device=self.device, dtype=torch.bfloat16)
        x_q, x_s = torch.ops.comfy_int8.quantize_rowwise(x)
        w_q, w_s = torch.ops.comfy_int8.quantize_rowwise(w)
        out = torch.ops.comfy_int8.gemm(x_q, x_s, w_q, w_s, bias, torch.bfloat16)
        ref = torch.nn.functional.linear(x, w, bias)
        rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel.item(), 0.05)

    def test_gemm_scalar_weight_scale(self):
        m, k, n = 64, 256, 128
        x = torch.randn(m, k, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(n, k, device=self.device, dtype=torch.bfloat16)
        scale = (w.abs().max() / 127.0).reshape(()).float()
        w_q = (w.float() / scale).round().clamp(-128, 127).to(torch.int8)
        x_q, x_s = torch.ops.comfy_int8.quantize_rowwise(x)
        out = torch.ops.comfy_int8.gemm(x_q, x_s, w_q, scale, None, torch.bfloat16)
        ref = torch.nn.functional.linear(x, w)
        rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel.item(), 0.06)

    def test_gemm_int_mm_and_eager_fallbacks_agree(self):
        m, k, n = 128, 512, 384
        x = torch.randn(m, k, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(n, k, device=self.device, dtype=torch.bfloat16)
        x_q, x_s = int8_kernels.quantize_rowwise_eager(x)
        w_q, w_s = int8_kernels.quantize_rowwise_eager(w)
        # Exact reference: integer accumulation in fp64, then scales.
        exact = ((x_q.double() @ w_q.double().t()) * x_s.double() * w_s.double().t()).float()
        eager = int8_kernels._gemm_dequant_eager(x_q, x_s, w_q, w_s, None, torch.float32)
        # The eager fallback uses a float matmul, which may round through TF32.
        self.assertTrue(torch.allclose(eager, exact, rtol=2e-2, atol=2e-2))
        if int8_kernels._int_mm_supported(x_q, w_q):
            imm = int8_kernels._gemm_dequant_int_mm(x_q, x_s, w_q, w_s, None, torch.float32)
            self.assertTrue(torch.allclose(imm, exact, rtol=1e-4, atol=1e-4))

    def test_int8_available_independent_of_ck_triton(self):
        # Mirrors tests/quantization/test_kitchen_backends.py: on Ampere the
        # comfy_kitchen "triton" backend is force-disabled for fp8e4nv, but the
        # int8 kernels must remain available.
        cap = torch.cuda.get_device_capability()
        if int8_kernels.TRITON_AVAILABLE and cap >= (7, 5):
            self.assertTrue(int8_kernels.int8_fast_available())


@requires_cuda_int8
class TestInt8Compile(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch._dynamo.reset()

    def _block(self, dev):
        mpo = ops.mixed_precision_ops({"mixed_ops": True, "quantize_on_load": "int8_convrot"}, torch.bfloat16)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = mpo.Linear(512, 1024, device=dev)
                self.fc2 = mpo.Linear(1024, 512, device=dev)

            def forward(self, x):
                return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

        blk = Block()
        w1 = torch.randn(1024, 512, dtype=torch.bfloat16)
        w2 = torch.randn(512, 1024, dtype=torch.bfloat16)
        blk.load_state_dict({
            "fc1.weight": w1, "fc1.bias": torch.zeros(1024, dtype=torch.bfloat16),
            "fc2.weight": w2, "fc2.bias": torch.zeros(512, dtype=torch.bfloat16),
        }, strict=False)
        return blk.to(dev), w1.to(dev), w2.to(dev)

    @requires_triton
    def test_fullgraph_keeps_int8_gemm(self):
        dev = torch.device("cuda")
        blk, w1, w2 = self._block(dev)
        self.assertIsInstance(blk.fc1.weight, QuantizedTensor)
        x = torch.randn(128, 512, device=dev, dtype=torch.bfloat16)

        graphs = []

        def backend(gm, inputs):
            graphs.append(gm)
            return gm.forward

        compiled = torch.compile(blk, fullgraph=True, backend=backend)
        out = compiled(x)
        self.assertEqual(len(graphs), 1)  # no graph breaks
        code = "\n".join(g.code for g in graphs)
        self.assertIn("comfy_int8", code)
        self.assertIn("quantize_rowwise", code)
        self.assertIn("gemm", code)

        ref = torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(x, w1)), w2)
        rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel.item(), 0.05)

    @requires_triton
    def test_inductor_end_to_end(self):
        dev = torch.device("cuda")
        blk, w1, w2 = self._block(dev)
        x = torch.randn(128, 512, device=dev, dtype=torch.bfloat16)
        compiled = torch.compile(blk, fullgraph=True)
        out = compiled(x)
        out = compiled(x)
        ref = torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(x, w1)), w2)
        rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel.item(), 0.05)


if __name__ == "__main__":
    unittest.main()
