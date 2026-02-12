# ComfyUI LTS

A vanilla, up-to-date fork of [ComfyUI](https://github.com/comfyanonymous/comfyui) intended for long term support (LTS) from [AppMana](https://appmana.com) and [Hidden Switch](https://hiddenswitch.com).

## Used By

Used in production by **Scopely**, a game studio, **Livepeer** and **Nunchaku Tech**. Used by innovators at **Ferrero Group**, **Hyundai** and **Nike**.

If you need to develop an application or plugin around ComfyUI, this fork stays compatible and up-to-date with upstream, fixing numerous bugs and adding features. It also packages tacit knowledge about running diffusion models and art workflows, distributed inference, deployment on Kubernetes, and other production tasks that Claude and Gemini cannot do.

## Key Features and Differences

This LTS fork adds development, embedding, automated testing, LLM and distributed inference features to ComfyUI, but maintains compatibility with custom nodes from the ecosystem.

- **Pip and UV Installable:** Install via `pip` or `uv` directly from GitHub. No manual cloning required for users.
- **Automatic Model Downloading:** Missing models (e.g., Stable Diffusion, FLUX, LLMs) are downloaded on-demand from Hugging Face or CivitAI.
- **Docker and Containers:** First-class support for Docker and Kubernetes with optimized containers for NVIDIA and AMD.
- **Distributed Inference:** Run scalable inference clusters with multiple workers and frontends using RabbitMQ.
- **Embedded / Library:** Use ComfyUI as a Python library (`import comfy`) inside your own applications without the web server. Runs like `diffusers`.
- **Vanilla Custom Nodes:** Fully compatible with existing ComfyUI custom nodes (ComfyUI-Manager, WanVideoWrapper, KJNodes, etc.). Clone into `custom_nodes/` and install dependencies into your venv.
- **LTS Custom Nodes:** A curated set of "Installable" custom nodes (ControlNet, AnimateDiff, IPAdapter) optimized for this fork.
- **LLM Support:** Native support for Large Language Models (LLaMA, Phi-3, etc.) and multi-modal workflows.
- **API and Configuration:** Enhanced API endpoints and extensive configuration options via CLI args, env vars, and config files.
- **Tests:** Automated test suite ensuring stability for new features.

## Quickstart (Windows & Linux, One Line)

Install `uv`, then:

```shell
uvx --python 3.12 --torch-backend=auto --from "git+https://github.com/hiddenswitch/ComfyUI.git" comfyui post-workflow https://raw.githubusercontent.com/hiddenswitch/pip-and-uv-installable-ComfyUI/refs/heads/master/tests/inference/workflows/z_image-0.json --guess-settings --prompt "a girl with red hair" --steps 9
```

## Quickstart (Linux)

### UI Users

For users who want to run ComfyUI for generating images and videos.

1.  **Install `uv`**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a Workspace**:
    ```bash
    mkdir comfyui-workspace
    cd comfyui-workspace
    ```

3.  **Install and Run**:
    ```bash
    # Create a virtual environment
    uv venv --python 3.12
    
    # Install ComfyUI LTS
    # --torch-backend=auto installs the correct torch, torchvision and torchaudio for your platform.
    # Omit --torch-backend if you want to keep your currently installed PyTorch.
    uv pip install --torch-backend=auto "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
    
    # Run (--guess-settings auto-detects GPU type, VRAM mode, attention backend)
    uv run --no-sync comfyui --guess-settings
    ```

### Developers

For developers contributing to the codebase or building on top of it.

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/hiddenswitch/ComfyUI.git
    cd ComfyUI
    ```

2.  **Setup Environment**:
    ```bash
    # Create virtual environment
    uv venv --python 3.12
    source .venv/bin/activate

    # Install in editable mode with dev dependencies
    uv pip install -e .[dev]
    ```

3.  **Run**:
    ```bash
    uv run --no-sync comfyui --guess-settings
    ```

### Using ComfyUI as a Library

ComfyUI can run embedded inside your own Python application. No server is started, no subprocesses are used. Use the `Comfy` async context manager to execute workflows directly:

```python
from comfy.client.embedded_comfy_client import Comfy

async with Comfy() as client:
    outputs = await client.queue_prompt(workflow_dict)
    # All models unloaded and VRAM released on exit
```

Build workflows programmatically with `GraphBuilder`, or paste JSON from the web UI (both API and UI format workflows are accepted). Stream previews during inference with `queue_with_progress`.

See [Embedded / Library Usage](docs/embedded.md) for complete examples.

## Documentation

Full documentation is available in [docs/index.md](docs/index.md).

### Core
- [Installation & Getting Started](docs/installing.md)
- [Hardware Compatibility](docs/compatibility.md)
- [Configuration](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)

### Features & Workflows
- [Large Language Models](docs/llm.md)
- [Video Workflows](docs/video.md) (AnimateDiff, SageAttention, etc.)
- [Other Features](docs/other_features.md) (SVG, Ideogram)

### Extending ComfyUI
- [Custom Nodes](docs/custom_nodes.md) (Installing & Authoring)
- [Embedded / Library Usage](docs/embedded.md) (Python, GraphBuilder, Streaming)
- [Testing Workflows](docs/testing.md) (pytest, Image Snapshots)
- [API Usage](docs/api.md) (REST, WebSocket)

### Deployment
- [Distributed / Multi-GPU](docs/distributed.md)
- [Docker & Containers](docs/docker.md)

### Development
- [Linting](docs/linting.md)
- [Merging Upstream](docs/merging.md)
