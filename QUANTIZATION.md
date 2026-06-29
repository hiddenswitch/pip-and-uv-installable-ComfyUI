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
| int8_tensorwise | int8 | float32 ([out, 1] per-row, scalar accepted) | - | - | - |
| svdquant_w4a4 | int8 (packed int4) | bf16/fp16 ([in/64, out] per-group) | - | - | - |
| awq_w4a16 | int8 (packed int4) | bf16/fp16 ([in/64, out] per-group) | - | - | - |

You can find the defined formats in `comfy/quant_ops.py` (QUANT_ALGOS).

### INT8 tensorwise (`int8_tensorwise`)

Native INT8 support is provided by comfy_kitchen's `TensorWiseINT8Layout`.
Checkpoints store int8 weights plus `weight_scale`; activation handling and
matmul dispatch belong to comfy_kitchen, not this fork. Optional ConvRot
metadata is preserved on the kitchen layout params when present.

Per-layer `comfy_quant` JSON:

```json
{"format": "int8_tensorwise"}
{"format": "int8_tensorwise", "convrot_groupsize": 256, "convrot": true, "per_row": true}
```

Legacy foreign INT8 checkpoints are normalized on load (`comfy.utils.convert_old_quants`): `format: "int8"`, `format: "int8_convrot"`, formatless ComfyUI-INT8-Fast JSON, and bare int8 `.weight` plus `.weight_scale` all map to `format: "int8_tensorwise"`. `convrot` and `convrot_groupsize` are retained for checkpoints that carry them.

Hardware: native INT8 acceleration depends on comfy_kitchen and the active
device/backend. On unsupported devices, INT8 weights can still load and
dequantize through the mixed-precision fallback path.

### SVDQuant W4A4 and AWQ W4A16 (`svdquant_w4a4`, `awq_w4a16`)

Offline-calibrated 4-bit formats implemented natively by comfy_kitchen
(`comfy_kitchen/tensor/svdquant_w4a4.py`, `awq_w4a16.py`) following nunchaku's
conventions. SVDQuant stores packed int4 weights plus a rank-R SVD low-rank
correction (`weight_proj_down`/`weight_proj_up`) and input smoothing
(`weight_smooth_factor`); the kernel fuses int4 activation quantization, the
low-rank branch, and the int4 GEMM. AWQ W4A16 keeps activations in bf16/fp16
(used for modulation linears) with per-group scales and zero points
(`weight_zeros`). Both are load-only: `quantize()` raises, calibration happens
offline (DeepCompressor). Per-layer JSON: `{"format": "svdquant_w4a4",
"act_unsigned": true}` marks nunchaku's post-GELU fc2 layers;
`{"format": "awq_w4a16", "group_size": 64}` for the AWQ layers. LoRA bake on
these layers falls back to a dense patched weight (no requantization path).

Caveats: the current comfy_kitchen CUDA kernels are tuned for Blackwell; on
Ampere (sm_86) the w4a4 path runs but measures slower than the int8 w8a8 path,
so on 30-series/A-series treat these formats as a 4x weight-memory feature
rather than a speedup. Checkpoints in raw nunchaku packing (key names
`qweight`/`wscales`/`proj_down`, nunchaku tile-swizzled int4 order, fused qkv
projections, int32-packed AWQ weights) are not yet converted automatically;
only kitchen-format serialization (the `weight*` suffixes above) loads today.

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
