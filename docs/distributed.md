# Distributed, Multi-Process and Multi-GPU Comfy

This package supports multi-processing across machines using RabbitMQ. This means you can launch multiple ComfyUI backend workers and queue prompts against them from multiple frontends.

It also supports local pipeline and tensor parallel inference for selected diffusion transformers. These are separate features: RabbitMQ distributes whole prompts among workers, pipeline parallelism splits transformer blocks across devices, and tensor parallelism shards individual matrix multiplications across devices.

## Exact Pipeline Parallel Models

Pipeline parallelism is transparent. Keep the ordinary **Load Diffusion Model** (`UNETLoader`) in the workflow and select more than one local device when starting ComfyUI:

```shell
comfyui --cuda-device 0,1
```

Supported Qwen Image checkpoints (including Qwen Image Layered), MiniMax H3,
and Flux2 Dev then use one contiguous stage per selected device. Selecting one
device keeps the ordinary single-device DynamicVRAM path. Flux2 Dev's eight
double-stream and 48 single-stream blocks are treated as one ordered 56-block
sequence; other Flux2 geometries are rejected instead of being guessed.

The loader reads the safetensors header first and makes a provisional contiguous partition so it can materialize only each stage's owned tensors. In the single-process path it then measures those loaded modules through the same stored-versus-materialized weight geometry used by DynamicVRAM and replans if the real geometry changes a boundary. Entry layers live on the first stage, exit layers live on the last stage, and no complete checkpoint state dict is materialized. INT8 ConvRot and ordinary safetensors checkpoints use the same ownership path.

The selected device order is the pipeline order. Automatic partitioning compares each prospective stage's DynamicVRAM weight geometry against the memory DynamicVRAM can make available on that device. The capacity signal includes allocator-free memory plus the device's reclaimable Aimdo VBAR residency, so externally occupied or differently loaded GPUs can receive asymmetric block ranges. The optimizer minimizes the worst per-device pressure and then overflow; equal layer counts and equal checkpoint byte counts are not objectives. If a stage is larger than resident capacity, its normal DynamicVRAM patcher streams and evicts weights on that GPU.

At each boundary the executor asks an injected pipeline-operations provider to transfer the activation payload. The operations mux selects asynchronous CUDA peer copies when adjacent devices support them. Otherwise it can select process-peer execution: one rank and model stage per device, a Gloo control group, and `torch.distributed` tensor sends through an NCCL group. NCCL chooses its own NVLink, PCIe, or network transport. Both providers produce the same pending-intermediate object: it owns destination buffers, serialized metadata, source references, and the completion event until the consuming stage waits on it. CUDA peer and NCCL communication use side streams; the compute stream waits at consumption rather than synchronizing the host. This follows the deferred-receive pattern used by vLLM's V2 pipeline runner.

Both providers use destination-owned buffers and a fresh serialized metadata representation; the process-peer provider does not share Python tensor state between ranks. Quantized and mixed-precision model operations remain independently injected, so pipeline transport composes with INT8 ConvRot instead of replacing its linear operations.

Pipeline stages participate in ComfyUI memory management as a group. Their weights are patched and loaded independently on their assigned devices, LoRA weight patches are routed to the owning stage, and a partial grouped load is rolled back if a later stage fails. A process-peer stage is represented by a remote model-manageable object, so ordinary per-device pressure loads, partially unloads, and resets it through the same request as local stages. Models already occupying either GPU, including text encoders, are ejected only when that generalized pressure decision needs their memory. Rank processes and reusable activation buffers survive sampler cleanup and are released with the owning model executor. The final denoised result is returned to the first stage's device so the normal sampler contract is unchanged.

Current restrictions:

- automatic local selection currently requires CUDA devices; the process-peer provider additionally requires an available NCCL `torch.distributed` backend;
- diffusion-model wrappers and transformer block replacement patches are rejected because they may require cross-stage Python execution; ordinary weight LoRAs are supported;
- hybrid tensor plus pipeline parallelism is not yet implemented.

The inference regression workflow is `tests/inference/workflows/qwen-image-layered-pipeline-0.json`. For example, on a two-GPU host:

```shell
comfyui run-workflow tests/inference/workflows/qwen-image-layered-pipeline-0.json \
  --cuda-device 0,1 --image input/example.png --steps 1
```

## Local Tensor Parallel Models

Tensor parallelism is also transparent to the workflow. MiniMax H3 safetensors checkpoints support Megatron-style TP when an explicit tensor-parallel size and an ordered device list are selected:

```shell
comfyui run-workflow tests/inference/workflows/minimax-h3-fl2va-0.json \
  --cuda-device 0,1 --tensor-parallel-size 2 \
  --disable-all-custom-nodes
```

There is no tensor-parallel workflow node. The loader decorates the checkpoint's normal Comfy operations provider. MiniMax QKV and gate/up projections become section-aware column-parallel linears; attention output and MLP down projections become row-parallel linears whose partial hidden states are summed through an injected collective. This composes with mixed-precision INT8 ConvRot operations and DynamicVRAM instead of replacing either one. Each rank loads a copied 9.83 GiB shard of the pruned MiniMax INT8 ConvRot checkpoint, and each rank remains an ordinary model-manageable participant on its own device.

The internal multiprocessing executor uses a Gloo control group and an NCCL device group. CPU inputs are broadcast through Gloo, CUDA inputs and activations through NCCL, and row-parallel reductions are asynchronous, out-of-place NCCL all-reduces whose completion is attached to the current compute stream. NCCL selects NVLink, PCIe, or another available transport. MiniMax's token-refiner preprocessing is executed on every rank through the same model-call executor as denoising, because it contains the same TP-sharded attention and MLP projections.

Internal workers receive their accelerator assignment as a stable device UUID
rather than a parent-process logical CUDA index. They resolve that identity in
their own visibility namespace before creating the NCCL group. This keeps
ordered lists such as `--cuda-device 1,0` correct even when the parent imported
Torch before applying `CUDA_VISIBLE_DEVICES`. `LOCAL_RANK` remains the canonical
process rank on the host and is not overloaded as a device identifier. PP and
TP share this device-identity provider.

W3C OpenTelemetry context is injected into TP process commands and extracted by
each rank. Worker-command spans are therefore children of the sampler or model
operation that issued them instead of unrelated traces.

Current restrictions:

- MiniMax H3 is the first supported TP model family;
- TP currently requires CUDA, NCCL, a safetensors checkpoint, and the internal multiprocessing executor;
- tensor-parallel LoRA patch routing and hybrid TP+PP are not yet implemented.

On two 24 GB RTX A5000 GPUs, the unchanged shipped `video_minimax_h3_t2v`
template (0.4 MP, five seconds/124 frames, 20 steps, SageAttention, seed
556589502035082) completed the sampler in 121.917 seconds at 0.16405 it/s with
TP=2. The same template measured 218.726 seconds at 0.09144 it/s with TP=1:
TP=2 reduced sampler latency by 44.3% and increased sampling throughput by
79.4% in that run. GPU 0 had an unrelated graphical workload during the TP=2
measurement, so rerun on idle GPUs for controlled hardware comparisons. See
`llms.txt`, “Measuring performance with OpenTelemetry,” for exact commands and
measurement rules.

### Launcher topology

ComfyUI uses the process identity standardized by `torchrun` and TorchElastic. CLI values take precedence over the canonical environment, which takes precedence over compatible PMI, Open MPI, MVAPICH, and Slurm aliases.

| CLI | Canonical environment | Meaning |
|---|---|---|
| `--rank` | `RANK` | Global process and pipeline-stage rank. |
| `--world-size` | `WORLD_SIZE` | Total process count. |
| `--local-rank` | `LOCAL_RANK` | Rank on this host. `--local_rank` is accepted for older launchers. |
| `--local-world-size` | `LOCAL_WORLD_SIZE` | Process count on this host. |
| `--master-addr` | `MASTER_ADDR` | Rendezvous host. |
| `--master-port` | `MASTER_PORT` | Rendezvous port. |
| `--pipeline-parallel-size`, `-pp` | `COMFYUI_PIPELINE_PARALLEL_SIZE` | Pipeline-stage count; defaults to world size under an external launcher. |
| `--tensor-parallel-size`, `-tp` | `COMFYUI_TENSOR_PARALLEL_SIZE` | Tensor-parallel rank count; defaults to one. |
| `--distributed-executor-backend` | `COMFYUI_DISTRIBUTED_EXECUTOR_BACKEND` | `auto`, single-process `peer`, internal `mp`, or `external_launcher`. |

An external launch maps one process to one pipeline stage, so pipeline size must
equal world size. Rank zero is the ComfyUI driver and owns the first stage; the
last rank owns model output. Other ranks disable custom nodes and remain in the
rank service. Each process binds to `cuda:LOCAL_RANK`. Before partitioning,
rank zero asks every rank for the memory its local DynamicVRAM allocator can
make available, so planning sees all stages without rank zero querying a remote
host's CUDA device. Once assigned, each rank makes its ordinary local
DynamicVRAM pressure decision when its stage is loaded; existing text encoders
or other models are ejected only when that pressure requires it.

Use `torchrun` without changing the workflow:

```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  --no-python "$(command -v comfyui)" run-workflow \
  tests/inference/workflows/qwen-image-layered-pipeline-0.json \
  --image input/example.png --pipeline-parallel-size 2 \
  --disable-all-custom-nodes
```

`serve`, `worker`, and `run-workflow` share the same startup lifecycle and
topology resolver. `--daemon` is intentionally rejected under an external
launcher. Hybrid data/pipeline topology is not implemented yet.

For two hosts with one GPU each, run the same workflow command on both hosts.
The checkpoint, workflow, inputs, and ComfyUI code must be available at the
same paths. Replace the address with rank zero's routable address:

```shell
# Host 0
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=2 --nproc-per-node=1 \
  --node-rank=0 --master-addr=10.0.0.10 --master-port=29500 \
  --no-python "$(command -v comfyui)" run-workflow \
  tests/inference/workflows/qwen-image-layered-pipeline-0.json \
  --image input/example.png --pipeline-parallel-size 2

# Host 1
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=2 --nproc-per-node=1 \
  --node-rank=1 --master-addr=10.0.0.10 --master-port=29500 \
  --no-python "$(command -v comfyui)" run-workflow \
  tests/inference/workflows/qwen-image-layered-pipeline-0.json \
  --image input/example.png --pipeline-parallel-size 2
```

NCCL selects its network transport. On a 10 GbE network, set standard NCCL
interface variables only when automatic interface discovery selects the wrong
NIC. This exact pipeline mode transfers one activation payload at each stage
boundary per denoising step. It is the first network-capable MVP; sequence
parallel methods such as Ulysses and PipeFusion are separate future operations
providers, not changes to the native model forward.

## Getting Started

ComfyUI has two roles: `worker` and `frontend`. An unlimited number of workers can consume and execute workflows (prompts) in parallel; and an unlimited number of frontends can submit jobs. All of the frontends' API calls will operate transparently against your collection of workers, including progress notifications from the websocket.

To share work among multiple workers and frontends, ComfyUI uses RabbitMQ or any AMQP-compatible message queue like SQS or Kafka.

### Example with RabbitMQ and File Share

On a machine in your local network, install **Docker** and run RabbitMQ:

```shell
docker run -it --rm --name rabbitmq -p 5672:5672 rabbitmq:latest
```

Find the machine's main LAN IP address:

**Windows (PowerShell)**:

```pwsh
Get-NetIPConfiguration | Where-Object { $_.InterfaceAlias -like '*Ethernet*' -and $_.IPv4DefaultGateway -ne $null } | ForEach-Object { $_.IPv4Address.IPAddress }
```

**Linux**

```shell
ip -4 addr show $(ip route show default | awk '/default/ {print $5}') | grep -oP 'inet \K[\d.]+'
```

**macOS**

```shell
ifconfig $(route get default | grep interface | awk '{print $2}') | awk '/inet / {print $2; exit}'
```

On my machine, this prints `10.1.0.100`, which is a local LAN IP that other hosts on my network can reach.

On this machine, you can also set up a file share for models, outputs and inputs.

Once you have installed this Python package following the installation steps, you can start a worker using:

**Starting a Worker:**

```shell
# you must replace the IP address with the one you printed above
comfyui-worker --distributed-queue-connection-uri="amqp://guest:guest@10.1.0.100"
```

All the normal command line arguments are supported. This means you can use `--cwd` to point to a file share containing the `models/` directory:

```shell
comfyui-worker --cwd //10.1.0.100/shared/workspace --distributed-queue-connection-uri="amqp://guest:guest@10.1.0.100"
```

**Starting a Frontend:**

```shell
comfyui --listen --distributed-queue-connection-uri="amqp://guest:guest@10.1.0.100" --distributed-queue-frontend
```

However, the frontend will **not** be able to find the output images or models to show the client by default. You must specify a place where the frontend can find the **same** outputs and models that are available to the backends:

```shell
comfyui --cwd //10.1.0.100/shared/workspace --listen --distributed-queue-connection-uri="amqp://guest:guest@10.1.0.100" --distributed-queue-frontend
```

You can carefully mount network directories into `outputs/` and `inputs/` such that they are shared among workers and frontends; you can store the `models/` on each machine, or serve them over a file share too.

### Operating

The frontend expects to find the referenced output images in its `--output-directory` or in the default `outputs/` under `--cwd` (aka the "workspace").

This means that workers and frontends do **not** have to have the same argument to `--cwd`. The paths that are passed to the **frontend**, such as the `inputs/` and `outputs/` directories, must have the **same contents** as the paths passed as those directories to the workers.

Since reading models like large checkpoints over the network can be slow, you can use `--extra-model-paths-config` to specify additional model paths. Or, you can use `--cwd some/path`, where `some/path` is a local directory, and, and mount `some/path/outputs` to a network directory.

Known models listed in [**model_downloader.py**](../comfy/model_downloader.py) are downloaded using `huggingface_hub` with the default `cache_dir`. This means you can mount a read-write-many volume, like an SMB share, into the default cache directory. Read more about this [here](https://huggingface.co/docs/huggingface_hub/en/guides/download).
