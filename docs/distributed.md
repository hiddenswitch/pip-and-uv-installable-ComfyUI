# Distributed, Multi-Process and Multi-GPU Comfy

This package supports multi-processing across machines using RabbitMQ. This means you can launch multiple ComfyUI backend workers and queue prompts against them from multiple frontends.

It also supports local pipeline parallel inference for selected diffusion transformers. These are separate features: RabbitMQ distributes whole prompts among workers, while pipeline parallelism splits one model invocation across devices in one worker.

## Local Pipeline Parallel Models

Pipeline parallelism is transparent. Keep the ordinary **Load Diffusion Model** (`UNETLoader`) in the workflow and select more than one local device when starting ComfyUI:

```shell
comfyui --cuda-device 0,1
```

Supported Qwen Image checkpoints (including Qwen Image Layered) and MiniMax H3 then use one contiguous stage per selected device. Selecting one device keeps the ordinary single-device DynamicVRAM path. Tensor-parallel execution is not yet implemented.

The loader reads the safetensors header first and makes a provisional contiguous partition so it can materialize only each stage's owned tensors. In the single-process path it then measures those loaded modules through the same stored-versus-materialized weight geometry used by DynamicVRAM and replans if the real geometry changes a boundary. Entry layers live on the first stage, exit layers live on the last stage, and no complete checkpoint state dict is materialized. INT8 ConvRot and ordinary safetensors checkpoints use the same ownership path.

The selected device order is the pipeline order. Automatic partitioning compares each prospective stage's DynamicVRAM weight geometry against the memory DynamicVRAM can make available on that device. The capacity signal includes allocator-free memory plus the device's reclaimable Aimdo VBAR residency, so externally occupied or differently loaded GPUs can receive asymmetric block ranges. The optimizer minimizes the worst per-device pressure and then overflow; equal layer counts and equal checkpoint byte counts are not objectives. If a stage is larger than resident capacity, its normal DynamicVRAM patcher streams and evicts weights on that GPU.

At each boundary the executor asks an injected pipeline-operations provider to transfer the activation payload. The operations mux selects asynchronous CUDA peer copies when adjacent devices support them. Otherwise it can select process-peer execution: one rank and model stage per device, a Gloo control group, and `torch.distributed` tensor sends through an NCCL group. NCCL chooses its own NVLink, PCIe, or network transport. Both providers use destination-owned buffers and a fresh serialized metadata representation; the process-peer provider does not share Python tensor state between ranks. Quantized and mixed-precision model operations remain independently injected, so pipeline transport composes with INT8 ConvRot instead of replacing its linear operations.

Pipeline stages participate in ComfyUI memory management as a group. Their weights are patched and loaded independently on their assigned devices, LoRA weight patches are routed to the owning stage, and a partial grouped load is rolled back if a later stage fails. A process-peer stage is represented by a remote model-manageable object, so ordinary per-device pressure loads, partially unloads, and resets it through the same request as local stages. Models already occupying either GPU, including text encoders, are ejected only when that generalized pressure decision needs their memory. Rank processes and reusable activation buffers survive sampler cleanup and are released with the owning model executor. The final denoised result is returned to the first stage's device so the normal sampler contract is unchanged.

Current restrictions:

- automatic local selection currently requires CUDA devices; the process-peer provider additionally requires an available NCCL `torch.distributed` backend;
- diffusion-model wrappers and transformer block replacement patches are rejected because they may require cross-stage Python execution; ordinary weight LoRAs are supported;
- tensor parallelism and multi-host stage discovery are not yet implemented.

The inference regression workflow is `tests/inference/workflows/qwen-image-layered-pipeline-0.json`. For example, on a two-GPU host:

```shell
comfyui run-workflow tests/inference/workflows/qwen-image-layered-pipeline-0.json \
  --cuda-device 0,1 --image input/example.png --steps 1
```

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
