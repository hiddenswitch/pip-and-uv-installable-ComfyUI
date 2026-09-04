# Configuration

The CLI and embedded API use the same `Configuration` object. Explicit values
win over defaults; launcher identity (`RANK`, `WORLD_SIZE`, and friends) is
resolved only when an external process group is being used.

## Starting ComfyUI

```console
comfyui                         # serve is the default command
comfyui serve --listen 0.0.0.0
comfyui --cuda-device 0,1 --tensor-parallel-size 2
comfyui run-workflow image_flux2_text_to_image --all
comfyui models list
comfyui workflows list
comfyui nodes list
comfyui env
```

In Python, pass a configuration directly; no setup call, subprocess, or
`PYTHONPATH` modification is required:

```python
from comfy import Comfy
from comfy.component_model.configuration import Configuration

configuration = Configuration(cuda_device="0,1", tensor_parallel_size=2)
app = Comfy(configuration=configuration)
```

## Memory and device selection

`guess_settings` is enabled by default. It selects a suitable device, dtype,
attention implementation, and DynamicVRAM policy from the hardware and the
requested model. An explicit CLI or `Configuration` value always wins.

Use `--reserve-vram` to leave a fixed amount available for the desktop or
other workloads. `--novram` is an explicit compatibility escape hatch, not the
normal recommendation; DynamicVRAM can eject dependencies when memory is
needed and works across pipeline stages.

## Distributed configuration

The model-parallel flags are ordinary configuration values:

```console
comfyui --cuda-device 0,1 --tensor-parallel-size 2
comfyui --cuda-device 0,1 --pipeline-parallel-size 2
comfyui --cuda-device 0,1 --ulysses-degree 2
comfyui --cuda-device 0,1 --ring-degree 2
```

For `torchrun` or another launcher, canonical `RANK`, `WORLD_SIZE`,
`LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` are read
alongside common MPI/PMI/Slurm aliases. Do not configure normal application
behavior by inventing additional environment variables.

## Tracing and benchmarking

ComfyUI emits OpenTelemetry spans for workflow execution and sampling. Set the
configured OTLP/JSONL exporter destination through the CLI configuration and
compare the sampler span after one warm-up run. The sampler span excludes
custom-node import and checkpoint-load time, which makes TP comparisons
meaningful. See [distributed inference](distributed.md#collected-tp-benchmark)
for the collected reference table.

## Embedded applications

The embedded entry point and the server use the same configuration surface:

```python
from comfy import Comfy
from comfy.component_model.configuration import Configuration

app = Comfy(configuration=Configuration(guess_settings=True))
```

Keep model paths, device choices, and feature flags in `Configuration` so an
embedded run and a CLI run have identical behavior.
