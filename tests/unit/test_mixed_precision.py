import json
import unittest
from unittest import mock

import torch

from comfy.cli_args_types import Configuration


def has_gpu():
    return torch.cuda.is_available()

from comfy.cli_args import args
if not has_gpu():
    args.cpu = True

from comfy import ops
from comfy import model_management
from comfy.model_base import _format_quantized_storage_summary
from comfy.quant_ops import QuantizedTensor
import comfy.utils


class SimpleModel(torch.nn.Module):
    def __init__(self, operations=ops.disable_weight_init):
        super().__init__()
        self.layer1 = operations.Linear(10, 20, device="cpu", dtype=torch.bfloat16)
        self.layer2 = operations.Linear(20, 30, device="cpu", dtype=torch.bfloat16)
        self.layer3 = operations.Linear(30, 40, device="cpu", dtype=torch.bfloat16)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        x = self.layer3(x)
        return x


class TestMixedPrecisionOps(unittest.TestCase):
    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_quant_layout_class_fallbacks_exist(self):
        self.assertIsNotNone(ops.get_layout_class("TensorCoreFP8E4M3Layout"))
        self.assertIsNotNone(ops.get_layout_class("TensorCoreFP8E5M2Layout"))
        self.assertIsNotNone(ops.get_layout_class("TensorCoreNVFP4Layout"))

    def test_all_layers_standard(self):
        """Test that model with no quantization works normally"""
        # Create model
        model = SimpleModel(operations=ops.mixed_precision_ops({}))

        # Initialize weights manually
        model.layer1.weight = torch.nn.Parameter(torch.randn(20, 10, dtype=torch.bfloat16))
        model.layer1.bias = torch.nn.Parameter(torch.randn(20, dtype=torch.bfloat16))
        model.layer2.weight = torch.nn.Parameter(torch.randn(30, 20, dtype=torch.bfloat16))
        model.layer2.bias = torch.nn.Parameter(torch.randn(30, dtype=torch.bfloat16))
        model.layer3.weight = torch.nn.Parameter(torch.randn(40, 30, dtype=torch.bfloat16))
        model.layer3.bias = torch.nn.Parameter(torch.randn(40, dtype=torch.bfloat16))

        # Initialize weight_function and bias_function
        for layer in [model.layer1, model.layer2, model.layer3]:
            layer.weight_function = []
            layer.bias_function = []

        # Forward pass
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))
        self.assertEqual(output.dtype, torch.bfloat16)

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_mixed_precision_load(self):
        """Test loading a mixed precision model from state dict"""
        # Configure mixed precision: layer1 is FP8, layer2 and layer3 are standard
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            },
            "layer3": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        # Create state dict with mixed precision
        fp8_weight1 = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        fp8_weight3 = torch.randn(40, 30, dtype=torch.float32).to(torch.float8_e4m3fn)

        state_dict = {
            # Layer 1: FP8 E4M3FN
            "layer1.weight": fp8_weight1,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),

            # Layer 2: Standard BF16
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),

            # Layer 3: FP8 E4M3FN
            "layer3.weight": fp8_weight3,
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
            "layer3.weight_scale": torch.tensor(1.5, dtype=torch.float32),
        }

        state_dict, _ = comfy.utils.convert_old_quants(state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})
        # Create model and load state dict (strict=False because custom loading pops keys)
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict, strict=False)

        # Verify weights are wrapped in QuantizedTensor
        self.assertIsInstance(model.layer1.weight, QuantizedTensor)
        self.assertEqual(model.layer1.weight._layout_cls, "TensorCoreFP8E4M3Layout")

        # Layer 2 should NOT be quantized
        self.assertNotIsInstance(model.layer2.weight, QuantizedTensor)

        # Layer 3 should be quantized
        self.assertIsInstance(model.layer3.weight, QuantizedTensor)
        self.assertEqual(model.layer3.weight._layout_cls, "TensorCoreFP8E4M3Layout")

        # Verify scales were loaded
        self.assertEqual(model.layer1.weight._params.scale.item(), 2.0)
        self.assertEqual(model.layer3.weight._params.scale.item(), 1.5)

        # Forward pass
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        with torch.inference_mode():
            output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_disabled_fp8_compute_preserves_scaled_quantized_weight(self):
        """Disabled fp8 kernels must still dequantize with the stored scale."""
        for quant_format, weight_dtype in (
            ("float8_e4m3fn", torch.float8_e4m3fn),
            ("float8_e5m2", torch.float8_e5m2),
        ):
            with self.subTest(quant_format=quant_format):
                layer_quant_config = {
                    "layer1": {
                        "format": quant_format,
                        "params": {}
                    }
                }
                fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(weight_dtype)
                state_dict = {
                    "layer1.weight": fp8_weight,
                    "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
                    "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),
                    "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
                    "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
                    "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
                    "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
                }
                state_dict, _ = comfy.utils.convert_old_quants(
                    state_dict,
                    metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
                )

                model = SimpleModel(operations=ops.mixed_precision_ops({}, disabled={quant_format}))
                model.load_state_dict(state_dict, strict=False)

                self.assertTrue(model.layer1._full_precision_mm)
                self.assertIsInstance(model.layer1.weight, QuantizedTensor)
                self.assertEqual(model.layer1.weight._params.scale.item(), 2.0)
                self.assertEqual(model.layer1.weight._qdata.dtype, weight_dtype)

                input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
                with torch.inference_mode():
                    output = model(input_tensor)

                self.assertEqual(output.shape, (5, 40))

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_quantized_layer_keeps_fp8_resident_storage_with_bf16_compute(self):
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {},
            }
        }
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(1.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }
        state_dict, _ = comfy.utils.convert_old_quants(
            state_dict,
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )

        model = SimpleModel(operations=ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16, disabled={"float8_e4m3fn"}))
        model.load_state_dict(state_dict, strict=False)

        self.assertTrue(model.layer1._full_precision_mm)
        self.assertIsInstance(model.layer1.weight, QuantizedTensor)
        self.assertEqual(model.layer1.weight.dtype, torch.bfloat16)
        self.assertEqual(model.layer1.weight._qdata.dtype, torch.float8_e4m3fn)
        self.assertEqual(
            _format_quantized_storage_summary(model),
            "torch.float8_e4m3fn=1",
        )

        with torch.inference_mode():
            output = model(torch.randn(5, 10, dtype=torch.bfloat16))

        self.assertEqual(output.shape, (5, 40))
        self.assertEqual(output.dtype, torch.bfloat16)

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_quantized_layer_can_disable_fp8_resident_storage_for_benchmarking(self):
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {},
            }
        }
        state_dict = {
            "layer1.weight": torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn),
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(1.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }
        state_dict, _ = comfy.utils.convert_old_quants(
            state_dict,
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )

        model = SimpleModel(
            operations=ops.mixed_precision_ops(
                {},
                compute_dtype=torch.bfloat16,
                disabled_storage={"float8_e4m3fn"},
            )
        )
        model.load_state_dict(state_dict, strict=False)

        self.assertNotIsInstance(model.layer1.weight, QuantizedTensor)
        self.assertEqual(model.layer1.weight.dtype, torch.bfloat16)
        self.assertEqual(_format_quantized_storage_summary(model), "")

        with torch.inference_mode():
            output = model(torch.randn(5, 10, dtype=torch.bfloat16))

        self.assertEqual(output.shape, (5, 40))
        self.assertEqual(output.dtype, torch.bfloat16)

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "requires torch fp8 dtype")
    def test_native_fp8_weight_storage_used_when_device_feature_test_passes(self):
        """Native fp8 checkpoints stay resident as fp8 when the device supports upcasted ops."""
        with (
            mock.patch.object(model_management, "args", Configuration(fp8_storage=True)),
            mock.patch.object(model_management, "supports_fp8_storage", return_value=True),
            mock.patch.object(model_management, "should_use_fp16", return_value=False),
            mock.patch.object(model_management, "should_use_bf16", return_value=True),
        ):
            selected_dtype = model_management.unet_dtype(
                device=torch.device("cpu"),
                model_params=10,
                supported_dtypes=(torch.bfloat16, torch.float32),
                weight_dtype=torch.float8_e4m3fn,
            )
            compute_dtype = model_management.unet_manual_cast(
                selected_dtype,
                torch.device("cpu"),
                supported_dtypes=(torch.bfloat16, torch.float32),
            )

        self.assertEqual(selected_dtype, torch.float8_e4m3fn)
        self.assertEqual(compute_dtype, torch.bfloat16)

        layer = ops.manual_cast.Linear(10, 20, device="cpu", dtype=selected_dtype)
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        fp8_bias = torch.randn(20, dtype=torch.float32).to(torch.float8_e4m3fn)
        layer.load_state_dict({"weight": fp8_weight, "bias": fp8_bias}, strict=False)

        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.bias.dtype, torch.float8_e4m3fn)

        with torch.inference_mode():
            output = layer(torch.randn(5, 10, dtype=compute_dtype))

        self.assertEqual(output.shape, (5, 20))
        self.assertEqual(output.dtype, compute_dtype)

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "requires torch fp8 dtype")
    def test_native_fp8_weight_storage_falls_back_when_device_feature_test_fails(self):
        with (
            mock.patch.object(model_management, "args", Configuration(fp8_storage=True)),
            mock.patch.object(model_management, "supports_fp8_storage", return_value=False),
            mock.patch.object(model_management, "should_use_fp16", return_value=False),
            mock.patch.object(model_management, "should_use_bf16", return_value=True),
        ):
            selected_dtype = model_management.unet_dtype(
                device=torch.device("cpu"),
                model_params=10,
                supported_dtypes=(torch.bfloat16, torch.float32),
                weight_dtype=torch.float8_e4m3fn,
            )

        self.assertEqual(selected_dtype, torch.bfloat16)

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "requires torch fp8 dtype")
    def test_native_fp8_weight_storage_flag_can_disable_feature_detected_storage(self):
        with (
            mock.patch.object(model_management, "args", Configuration(fp8_storage=False)),
            mock.patch.object(model_management, "supports_fp8_storage", return_value=True),
            mock.patch.object(model_management, "should_use_fp16", return_value=False),
            mock.patch.object(model_management, "should_use_bf16", return_value=True),
        ):
            selected_dtype = model_management.unet_dtype(
                device=torch.device("cpu"),
                model_params=10,
                supported_dtypes=(torch.bfloat16, torch.float32),
                weight_dtype=torch.float8_e4m3fn,
            )

        self.assertEqual(selected_dtype, torch.bfloat16)

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_state_dict_quantized_preserved(self):
        """Test that quantized weights are preserved in state_dict()"""
        # Configure mixed precision
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        # Create and load model
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict1 = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(3.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict1, _ = comfy.utils.convert_old_quants(state_dict1, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict1, strict=False)

        # Save state dict
        state_dict2 = model.state_dict()

        # Verify layer1.weight is a QuantizedTensor with scale preserved
        self.assertTrue(torch.equal(state_dict2["layer1.weight"].view(torch.uint8), fp8_weight.view(torch.uint8)))
        self.assertEqual(state_dict2["layer1.weight_scale"].item(), 3.0)
        self.assertEqual(model.layer1.weight._layout_cls, "TensorCoreFP8E4M3Layout")

        # Verify non-quantized layers are standard tensors
        self.assertNotIsInstance(state_dict2["layer2.weight"], QuantizedTensor)
        self.assertNotIsInstance(state_dict2["layer3.weight"], QuantizedTensor)

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_weight_function_compatibility(self):
        """Test that weight_function (LoRA) works with quantized layers"""
        # Configure FP8 quantization
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        # Create and load model
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict, _ = comfy.utils.convert_old_quants(state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict, strict=False)

        # Add a weight function (simulating LoRA)
        # This should trigger dequantization during forward pass
        def apply_lora(weight):
            lora_delta = torch.randn_like(weight) * 0.01
            return weight + lora_delta

        model.layer1.weight_function.append(apply_lora)

        # Forward pass should work with LoRA (triggers weight_function path)
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))

    def test_error_handling_unknown_format(self):
        """Test that unknown formats raise error"""
        # Configure with unknown format
        layer_quant_config = {
            "layer1": {
                "format": "unknown_format_xyz",
                "params": {}
            }
        }

        # Create state dict
        state_dict = {
            "layer1.weight": torch.randn(20, 10, dtype=torch.bfloat16),
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict, _ = comfy.utils.convert_old_quants(state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})

        # Load should raise KeyError for unknown format in QUANT_FORMAT_MIXINS
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        with self.assertRaises(KeyError):
            model.load_state_dict(state_dict, strict=False)

if __name__ == "__main__":
    unittest.main()
