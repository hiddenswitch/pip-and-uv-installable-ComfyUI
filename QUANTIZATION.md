# The Comfy guide to Quantization


## How does quantization work?

Quantization aims to map a high-precision value x_f to a lower precision format with minimal loss in accuracy. These smaller formats then serve to reduce the models memory footprint and increase throughput by using specialized hardware.

When simply converting a value from FP16 to FP8 using the round-nearest method we might hit two issues:
- The dynamic range of FP16 (-65,504, 65,504) far exceeds FP8 formats like E4M3 (-448, 448) or E5M2 (-57,344, 57,344), potentially resulting in clipped values
- The original values are concentrated in a small range (e.g. -1,1) leaving many FP8-bits "unused"

By using a scaling factor, we aim to map these values into the quantized-dtype range, making use of the full spectrum. One of the easiest approaches, and common, is using per-tensor absolute-maximum scaling.

```
absmax = max(abs(tensor))
scale = amax / max_dynamic_range_low_precision

# Quantization
tensor_q = (tensor / scale).to(low_precision_dtype)

# De-Quantization
tensor_dq = tensor_q.to(fp16) * scale

tensor_dq ~ tensor
```

Given that additional information (scaling factor) is needed to "interpret" the quantized values, we describe those as derived datatypes.


## Quantization in Comfy

```
QuantizedTensor (torch.Tensor subclass)
  ↓ __torch_dispatch__
Two-Level Registry (generic + layout handlers)
  ↓
MixedPrecisionOps + Metadata Detection
```

### Representation

To represent these derived datatypes, ComfyUI uses a subclass of torch.Tensor to implements these using the `QuantizedTensor` class found in `comfy/quant_ops.py`

A `Layout` class defines how a specific quantization format behaves:
- Required parameters
- Quantize method
- De-Quantize method

```python
from comfy.quant_ops import QuantizedLayout

class MyLayout(QuantizedLayout):
    @classmethod
    def quantize(cls, tensor, **kwargs):
        # Convert to quantized format
        qdata = ...
        params = {'scale': ..., 'orig_dtype': tensor.dtype}
        return qdata, params
    
    @staticmethod
    def dequantize(qdata, scale, orig_dtype, **kwargs):
        return qdata.to(orig_dtype) * scale
```

To then run operations using these QuantizedTensors we use two registry systems to define supported operations. 
The first is a **generic registry** that handles operations common to all quantized formats (e.g., `.to()`, `.clone()`, `.reshape()`).

The second registry is layout-specific and allows to implement fast-paths like nn.Linear.
```python
from comfy.quant_ops import register_layout_op

@register_layout_op(torch.ops.aten.linear.default, MyLayout)
def my_linear(func, args, kwargs):
    # Extract tensors, call optimized kernel
    ...
```
When `torch.nn.functional.linear()` is called with QuantizedTensor arguments, `__torch_dispatch__` automatically routes to the registered implementation.
For any unsupported operation, QuantizedTensor will fallback to call `dequantize` and dispatch using the high-precision implementation.


### Mixed Precision

The `MixedPrecisionOps` class (lines 542-648 in `comfy/ops.py`) enables per-layer quantization decisions, allowing different layers in a model to use different precisions. This is activated when a model config contains a `layer_quant_config` dictionary that specifies which layers should be quantized and how.

When `comfy_kitchen` is unavailable, ComfyUI disables FP8 and NVFP4 mixed-precision quantized layers and falls back to loading those weights in the normal compute dtype instead. This is the expected behavior on older torch builds such as the 2.2.x lane, where `comfy_kitchen` cannot provide its tensor subclass implementation.

**Architecture:**

```python
class MixedPrecisionOps(disable_weight_init):
    _layer_quant_config = {}  # Maps layer names to quantization configs
    _compute_dtype = torch.bfloat16  # Default compute / dequantize precision
```

**Key mechanism:**

The custom `Linear._load_from_state_dict()` method inspects each layer during model loading:
- If the layer name is **not** in `_layer_quant_config`: load weight as regular tensor in `_compute_dtype`
- If the layer name **is** in `_layer_quant_config`: 
  - Load weight as `QuantizedTensor` with the specified layout (e.g., `TensorCoreFP8Layout`)
  - Load associated quantization parameters (scales, block_size, etc.)

**Why it's needed:**

Not all layers tolerate quantization equally. Sensitive operations like final projections can be kept in higher precision, while compute-heavy matmuls are quantized. This provides most of the performance benefits while maintaining quality.

The system is selected in `pick_operations()` when `model_config.layer_quant_config` is present, making it the highest-priority operation mode.


## Checkpoint Format

Quantized checkpoints are stored as standard safetensors files with quantized weight tensors and associated scaling parameters, plus a `_quantization_metadata` JSON entry describing the quantization scheme.

The quantized checkpoint will contain the same layers as the original checkpoint but:
- The weights are stored as quantized values, sometimes using a different storage datatype. E.g. uint8 container for fp8.
- For each quantized weight a number of additional scaling parameters are stored alongside depending on the recipe.
- We store a metadata.json in the metadata of the final safetensor containing the `_quantization_metadata` describing which layers are quantized and what layout has been used.

### Scaling Parameters details
We define 4 possible scaling parameters that should cover most recipes in the near-future:
- **weight_scale**: quantization scalers for the weights
- **weight_scale_2**: global scalers in the context of double scaling
- **pre_quant_scale**: scalers used for smoothing salient weights
- **input_scale**: quantization scalers for the activations

| Format | Storage dtype | weight_scale | weight_scale_2 | pre_quant_scale | input_scale |
|--------|---------------|--------------|----------------|-----------------|-------------|
| float8_e4m3fn | float32 | float32 (scalar) | - | - | float32 (scalar) |
| int8 | int8 | float32 ([out, 1] per-row, scalar accepted) | - | - | accepted but ignored |
| int8_convrot | int8 | float32 ([out, 1] per-row) | - | - | accepted but ignored |

You can find the defined formats in `comfy/quant_ops.py` (QUANT_ALGOS).

### INT8 w8a8 (`int8`, `int8_convrot`)

Symmetric absmax quantization with no zero points: `q = round(x / s).clamp(-128, 127)` with `s = absmax / 127`, one fp32 scale per output row. Activations are quantized dynamically per token on every forward (`comfy/int8_kernels.py`, triton fused per-token quant + int8 GEMM with int32 accumulation and dequant epilogue), so any `input_scale` tensors in a checkpoint are consumed and re-emitted on save but never used at runtime. The triton kernels handle any batch size including single tokens (tile sizes, not M, satisfy the tensor-core minimum); only the no-triton `torch._int_mm` fallback needs M > 16 and falls back to a dequantized matmul below that.

`int8_convrot` additionally stores weights in a rotated space: each contiguous group of 256 input channels is right-multiplied by a normalized regular Hadamard matrix (block-diagonal rotation, `comfy/quant_ops_int8.py`). The matrix is symmetric and orthogonal, so the same rotation applied to the activations preserves the matmul result exactly in full precision while spreading activation outliers across channels, which reduces absmax quantization loss. `dequantize()` always de-rotates back to original weight space, which is what makes LoRA application (dequantize, patch, requantize) and every dequantization fallback correct without special cases. Layers whose `in_features` is not divisible by 256 are not eligible for convrot.

Per-layer `comfy_quant` JSON:

```json
{"format": "int8"}
{"format": "int8_convrot", "convrot_groupsize": 256, "convrot": true, "per_row": true}
```

The `convrot`/`per_row` keys are redundant with the format and kept for compatibility with checkpoints produced by ComfyUI-INT8-Fast. Foreign int8 checkpoints are normalized on load (`comfy.utils.convert_old_quants`): a per-layer `comfy_quant` JSON without a `"format"` key, or a bare int8 `.weight` with a sibling `.weight_scale` and no marker at all (ModelOpt int8 row-wise exports), both map onto these formats automatically, so `CheckpointLoaderSimple`/`UNETLoader` load them with no extra nodes.

Hardware: int8 tensor-core matmul needs NVIDIA sm_75 (Turing) or newer and notably includes all of Ampere (sm_80/sm_86), where fp8 compute is unavailable. The triton kernels are independent of the comfy_kitchen "triton" backend (which is disabled below sm_89 for fp8e4nv reasons). On unsupported devices int8 weights still load (half the memory) and dequantize per forward.

### Ad-hoc quantization (quantize-on-load)

The `weight_dtype` dropdown on UNETLoader and related loaders accepts `int8` and `int8_convrot`. Unlike the fp8 entries this is not a plain dtype cast: eligible Linear weights are absmax-quantized per row (with the Hadamard rotation for convrot) while the model loads, on the GPU when one is available. Sensitive layers (embedders, modulation/adaLN, final projections) are excluded per architecture via `int8_quant_exclude` on the model configs in `comfy/supported_models.py`. Saving the model afterwards writes a distributable int8 checkpoint through the regular state_dict path.

### Quantization Metadata

The metadata stored alongside the checkpoint contains:
- **format_version**: String to define a version of the standard
- **layers**: A dictionary mapping layer names to their quantization format. The format string maps to the definitions found in `QUANT_ALGOS`. 

Example:
```json
{
  "_quantization_metadata": {
    "format_version": "1.0",
    "layers": {
      "model.layers.0.mlp.up_proj": {"format": "float8_e4m3fn"},
      "model.layers.0.mlp.down_proj": {"format": "float8_e4m3fn"},
      "model.layers.1.mlp.up_proj": {"format": "float8_e4m3fn"}
    }
  }
}
```


## Creating Quantized Checkpoints

To create compatible checkpoints, use any quantization tool provided the output follows the checkpoint format described above and uses a layout defined in `QUANT_ALGOS`.

### Weight Quantization

Weight quantization is straightforward - compute the scaling factor directly from the weight tensor using the absolute maximum method described earlier. Each layer's weights are quantized independently and stored with their corresponding `weight_scale` parameter.

### Calibration (for Activation Quantization)

Activation quantization (e.g., for FP8 Tensor Core operations) requires `input_scale` parameters that cannot be determined from static weights alone. Since activation values depend on actual inputs, we use **post-training calibration (PTQ)**:

1. **Collect statistics**: Run inference on N representative samples
2. **Track activations**: Record the absolute maximum (`amax`) of inputs to each quantized layer
3. **Compute scales**: Derive `input_scale` from collected statistics
4. **Store in checkpoint**: Save `input_scale` parameters alongside weights

The calibration dataset should be representative of your target use case. For diffusion models, this typically means a diverse set of prompts and generation parameters.
