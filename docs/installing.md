# Getting Started

## Installing

Pick the block for your platform and accelerator and paste it into a terminal. Models used by workflows are downloaded automatically.

- The **workspace directory** is where ComfyUI stores the `.venv`, downloaded models, outputs, and `custom_nodes/`. You can move it anywhere; the examples use `ComfyUI_Workspace`.
- The **virtual environment** (`.venv`) is an isolated Python install inside the workspace. Activate it before running `comfyui`.
- **`uv`** is the Python package manager used here. It creates the venv and installs ComfyUI plus the correct PyTorch wheels.
- **GPU settings** are chosen in two places: `uv pip install --torch-backend=...` selects the PyTorch wheel, and `comfyui serve --guess-settings` auto-detects runtime settings such as VRAM mode and attention backend. CUDA users should use `--torch-backend=auto`.

### Windows + CUDA

```powershell
irm https://astral.sh/uv/install.ps1 | iex
$env:Path = "$HOME\.local\bin;$env:Path"
New-Item -ItemType Directory -Force "$HOME\Documents\ComfyUI_Workspace"
cd $HOME\Documents\ComfyUI_Workspace
uv venv --python 3.12
uv pip install --torch-backend=auto --extra-index-url https://nodes.appmana.com/simple/ comfyui
uv pip install triton-windows
uv pip install --extra-index-url https://nodes.appmana.com/simple/ sageattention flash-attn
.\.venv\Scripts\Activate.ps1
comfyui --help
comfyui serve --guess-settings
```

### Windows + ROCm

```powershell
irm https://astral.sh/uv/install.ps1 | iex
$env:Path = "$HOME\.local\bin;$env:Path"
New-Item -ItemType Directory -Force "$HOME\Documents\ComfyUI_Workspace"
cd $HOME\Documents\ComfyUI_Workspace
uv venv --python 3.12
uv pip install --torch-backend=rocm7.2 --extra-index-url https://nodes.appmana.com/simple/ comfyui
.\.venv\Scripts\Activate.ps1
comfyui --help
comfyui serve --guess-settings --fp32-vae
```

### macOS

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/Documents/ComfyUI_Workspace"
cd "$HOME/Documents/ComfyUI_Workspace"
uv venv --python 3.12
uv pip install --torch-backend=auto --extra-index-url https://nodes.appmana.com/simple/ comfyui
source .venv/bin/activate
comfyui --help
comfyui serve --guess-settings
```

Apple Silicon does not natively support FP8 tensor operations. The `fp4-fp8-for-torch-mps` package is installed automatically and provides software emulation of FP8 compute on MPS via Metal kernels, enabling FP8-quantized models to run on macOS.

### Linux + CUDA

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/ComfyUI_Workspace"
cd "$HOME/ComfyUI_Workspace"
uv venv --python 3.12
uv pip install --torch-backend=auto --extra-index-url https://nodes.appmana.com/simple/ comfyui
uv pip install --extra-index-url https://nodes.appmana.com/simple/ sageattention flash-attn
source .venv/bin/activate
comfyui --help
comfyui serve --guess-settings
```

### Linux + ROCm

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/ComfyUI_Workspace"
cd "$HOME/ComfyUI_Workspace"
uv venv --python 3.12
uv pip install --torch-backend=rocm7.2 --extra-index-url https://nodes.appmana.com/simple/ comfyui
source .venv/bin/activate
comfyui --help
comfyui serve --guess-settings --fp32-vae
```

For bare Ubuntu with AMD GPUs, install the AMDGPU/ROCm stack before installing PyTorch. Verify `/dev/kfd` and `/dev/dri/renderD*` exist before debugging ComfyUI.

```shell
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.hip)"
```

### Linux + Lesser-Known ROCm Builds

Use this block when your AMD GPU needs one of AMD's architecture-specific nightly indexes, such as Strix Halo (`gfx1151`), MI300 (`gfx94X`), MI350 (`gfx950`), or consumer RDNA 3/4 indexes. This is the ROCm exception where you use `--index-url` instead of `--torch-backend`, because the wheel index is selected by GPU architecture. Set `ROCM_INDEX_URL` to the index for your GPU before running the rest.

Common ROCm nightly indexes:

| GPU family | gfx arch | `ROCM_INDEX_URL` |
| --- | --- | --- |
| RX 9000 / RDNA 4 | `gfx120X` | `https://rocm.nightlies.amd.com/v2/gfx120X-all/` |
| RX 7900 / RDNA 3 dGPU | `gfx1100`, `gfx1101` | `https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/` |
| RX 7600 / Framework 16 / 780M | `gfx1102`, `gfx1103` | `https://rocm.nightlies.amd.com/v2/gfx110X-all/` |
| Strix Halo | `gfx1151` | `https://rocm.nightlies.amd.com/v2/gfx1151/` |
| MI300A / MI300X | `gfx942` | `https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/` |
| MI350X / MI355X | `gfx950` | `https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/` |

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/ComfyUI_Workspace"
cd "$HOME/ComfyUI_Workspace"
uv venv --python 3.12
export ROCM_INDEX_URL="https://rocm.nightlies.amd.com/v2/gfx120X-all/"
uv pip install --index-url "$ROCM_INDEX_URL" --pre torch torchaudio torchvision triton
uv pip install --extra-index-url https://nodes.appmana.com/simple/ comfyui
source .venv/bin/activate
comfyui --help
comfyui serve --guess-settings --fp32-vae
```

`HSA_OVERRIDE_GFX_VERSION=11.0.0` can make gfx1102 advertise gfx1100 ISA so torch's gfx1100 kernels run on it. ComfyUI sets this automatically when it detects a local gfx1102/gfx1103 GPU on a torch wheel that does not include those arches; set the variable yourself only when you need to override the auto-detection.

### Intel Arc / Max / iGPU (XPU, Linux)

Use XPU when `torch.xpu.is_available()` is true. The host needs a recent Intel GPU kernel/firmware stack exposing `/dev/dri/renderD*` and the Intel userland compute stack. The recommended Linux path is Intel's XPU PyTorch container:

```shell
docker run --rm -it --device /dev/dri intel/intel-extension-for-pytorch:2.8.10-xpu bash
```

For bare Ubuntu, install the Intel GPU runtime first, then install ComfyUI with the XPU PyTorch backend:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/ComfyUI_Workspace"
cd "$HOME/ComfyUI_Workspace"
uv venv --python 3.11
uv pip install --torch-backend=xpu --extra-index-url https://nodes.appmana.com/simple/ comfyui
source .venv/bin/activate
python -c "import torch; print(torch.xpu.is_available())"
comfyui --help
comfyui serve --guess-settings
```

### Running Again Later

To start ComfyUI again after closing your terminal, `cd` into your workspace and run:

```shell
cd ~/ComfyUI_Workspace
source .venv/bin/activate
comfyui serve --guess-settings
```

On Windows:
```powershell
cd ~\Documents\ComfyUI_Workspace
.\.venv\Scripts\Activate.ps1
comfyui serve --guess-settings
```

### Upgrading

```shell
uv pip install --upgrade --extra-index-url https://nodes.appmana.com/simple/ comfyui
```

For NVIDIA users who want to ensure the correct CUDA version is maintained:
```shell
uv pip install --torch-backend=auto --upgrade --extra-index-url https://nodes.appmana.com/simple/ comfyui
```

## CUDA and PyTorch

### You Do Not Need the CUDA SDK

ComfyUI does not require the CUDA Toolkit (nvcc) to be installed on your system. PyTorch ships with its own CUDA runtime libraries bundled inside the pip package. The only requirement is an NVIDIA driver that supports the CUDA version used by your PyTorch build.

### Checking Your Driver's CUDA Version

To see the highest CUDA version supported by your installed driver:

```shell
nvidia-smi
```

Look for the "CUDA Version" in the top-right corner of the output. For example, `CUDA Version: 12.8` means your driver supports CUDA 12.8 and below.

### Understanding `--torch-backend`

The `--torch-backend` flag tells `uv` which PyTorch package index to use when resolving `torch` and its related packages (`torchvision`, `torchaudio`, etc.). Without it, `uv` would install CPU-only PyTorch.

- `--torch-backend=auto` — automatically detects your platform and selects the appropriate CUDA version
- `--torch-backend=cu128` — explicitly selects CUDA 12.8
- `--torch-backend=cu130` — explicitly selects CUDA 13.0
- `--torch-backend=rocm7.2` — selects ROCm 7.2 PyTorch wheels
- `--torch-backend=xpu` — selects Intel XPU PyTorch wheels
- `--torch-backend=cpu` — CPU-only (no GPU acceleration)

For AMD architecture-specific ROCm nightlies such as `gfx120X-all`, `gfx1151`, `gfx94X-dcgpu`, or `gfx950-dcgpu`, use the `--index-url` block above instead of `--torch-backend`.

This flag also works when installing prerelease (nightly) PyTorch builds:

```shell
# Install the latest prerelease torch with auto-detected CUDA
uv pip install --torch-backend=auto --prerelease=allow torch

# Install a specific prerelease version
uv pip install --torch-backend=auto --prerelease=allow "torch>=2.9.0.dev"
```

This is useful for testing new PyTorch features or getting early access to new CUDA version support.

### Installing a Specific CUDA Build

If you need to target a specific CUDA version (e.g., to match your driver or for compatibility):

```shell
# Install with CUDA 12.8
uv pip install --torch-backend=cu128 --extra-index-url https://nodes.appmana.com/simple/ comfyui

# Install with CUDA 13.0
uv pip install --torch-backend=cu130 --extra-index-url https://nodes.appmana.com/simple/ comfyui
```

See the [uv PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for more details.

## XPU and ROCm Runtime Notes

### Intel XPU

- `torch.xpu` is the Intel accelerator backend exposed by XPU-enabled PyTorch.
- ComfyUI uses it only if the installed torch build already supports XPU.
- The recommended Linux path is the Intel XPU PyTorch container used in CI: `intel/intel-extension-for-pytorch:2.8.10-xpu`.
- On bare Ubuntu, install the Intel GPU driver/runtime stack first, then verify `torch.xpu.is_available()` before installing ComfyUI.

### AMD ROCm

- On AMD, accelerated PyTorch typically reports through `torch.cuda` even though the backend is ROCm.
- `torch.version.hip` is the quickest way to confirm you are running a ROCm torch build.
- For consumer RDNA 3+ GPUs, use the architecture-specific nightly wheel index from the compatibility matrix instead of generic wheels.
- If ComfyUI starts but VAE decode is unstable on AMD, run with `--fp32-vae`.

## Running Workflows from Civitai, Hugging Face, or a URL

`comfyui run-workflow` accepts URIs in addition to template names and local paths:

```shell
# Civitai workflow by model id (resolves to the latest version's primary file)
comfyui run-workflow civitai://m/2304098 --help          # read author notes first
comfyui run-workflow civitai://m/2304098 --all           # then run

# Civitai workflow by version id, the NSFW mirror, or HTTPS URLs (auto-canonicalized)
comfyui run-workflow civitai://v/2521513 --all
comfyui run-workflow civitai-red://m/12345 --all
comfyui run-workflow https://civitai.com/models/2304098 --all

# Hugging Face direct file or repo
comfyui run-workflow hf://owner/repo/workflow.json --all
comfyui run-workflow https://huggingface.co/owner/repo/blob/main/wf.json --all

# Any URL that returns workflow JSON, a zip containing one, or a PNG with an embedded graph
comfyui run-workflow https://gist.githubusercontent.com/.../wf.json --all
```

**Always run `--help` first.** When a workflow URI is from Civitai, the help output renders every `Note` and `MarkdownNote` node from the graph as the first sections, with markdown headings, lists, and links rendered properly (each link prints as label + full URL on the next line). Workflow authors put download URLs and "place in" hints there; reading them is faster than discovering missing files at run time.

Set `CIVITAI_API_TOKEN=<your-token>` in the env before running so the `civitai://` fsspec backend can authenticate against early-access / NSFW gated downloads. Get one at https://civitai.com/user/account → API Keys.

### What `--all` does on a Civitai workflow URI

- **Hydrates the model index from the workflow's author.** All of that user's Civitai uploads (checkpoints, LoRAs, VAEs) become resolvable by filename.
- **Mines `Note` / `MarkdownNote` URLs.** Every markdown link `[label](url)` whose URL points at a model file (`.safetensors`, `.gguf`, `.ckpt`, etc.) is extracted, attributed to a folder via the nearest `Place in:` hint, and registered as a typed `Downloadable` (`HuggingFile` for `huggingface.co`, `FsspecFile("civitai://v/<id>")` for `civitai.com/api/download/models/`, `UrlFile` otherwise) so the right cache + auth path applies.
- **Builds custom-node facade wheels locally** for any package without a pre-built wheel on `nodes.appmana.com`. See [Custom Nodes — Local Facade Build for Long-Tail Packages](custom_nodes.md#local-facade-build-for-long-tail-packages).

### Foreign workflow formats

Some Civitai uploads are A1111 / Forge `.txt` parameter dumps or Fooocus JSON presets rather than ComfyUI graphs. Those are auto-translated to a generic ComfyUI checkpoint → LoRA → KSampler → VAEDecode → SaveImage workflow using the dump's prompt, sampler, steps, CFG, seed, dimensions, model, and LoRAs. SwarmUI / InvokeAI / Krita-AI shapes raise `UnsupportedWorkflowFormatError` (a subclass of `ApiValueError`).

### Discovering workflows

```shell
# Top 20 across every host (civitai + civitai-red + comfyui-org + huggingface)
comfyui workflows top --limit 20

# Filter by host and family
comfyui workflows top --with-host civitai --period 30d --family wan --limit 10
comfyui workflows top --without-host civitai-red,tensorart

# Substring search across hosts
comfyui workflows search "flux kontext" --limit 10
```

`--family` accepts `wan`, `flux`, `sdxl`, `ltxv`. `--period` accepts `Day` / `Week` / `Month` / `Year` / `AllTime`, plus the shorthands `30d` / `180d` / `360d`. URIs from the output paste straight into `comfyui run-workflow <URI> --help`.

## Model Downloading

ComfyUI LTS supports downloading models on demand.

Known models will be downloaded from Hugging Face or CivitAI.

To support licensed models like Flux, you will need to login to Hugging Face from the command line.

1. Activate your Python environment by `cd` followed by your workspace directory. For example, if your workspace is located in `~/Documents/ComfyUI_Workspace`, do:

```shell
cd ~/Documents/ComfyUI_Workspace
```

Then, on Windows: `& .venv/scripts/activate.ps1`; on macOS: `source .venv/bin/activate`.

2. Login with Huggingface:

```shell
uv pip install huggingface-cli
huggingface-cli login
```

3. Agree to the terms for a repository. For example, visit https://huggingface.co/black-forest-labs/FLUX.1-dev, login with your HuggingFace account, then choose **Agree**.

To disable model downloading, start with the command line argument `--disable-known-models`: `comfyui --disable-known-models`. However, this will generally only increase your toil for no return.

### Saving Space on Windows

To save space, you will need to enable **Developer Mode** in the Windows Settings, then reboot your computer. This way, Hugging Face can download models into a common place for all your apps, and place small "link" files that ComfyUI and others can read instead of whole copies of models.

## Using ComfyUI in Google Colab

Access an example Colab Notebook here: https://colab.research.google.com/drive/1Gd9F8iYRJW-LG8JLiwGTKLAcXLJ5eH78?usp=sharing

This demonstrates running a workflow inside colab and accessing the UI remotely.

## Using a "Python Embedded" "Portable" Style Distribution

This is a "ComfyUI" "Portable" style distribution with a "`python_embedded`" directory, carefully spelled correctly. It includes Python 3.12, `torch==2.7.1+cu128`, `sageattention` and the ComfyUI-Manager.

On **Windows**:

1. Download all the files in this the latest release: ([`comfyui_portable.exe`](https://github.com/hiddenswitch/ComfyUI/releases/download/latest/comfyui_portable.exe), [`comfyui_portable.7z.001`](https://github.com/hiddenswitch/ComfyUI/releases/download/latest/comfyui_portable.7z.001) and [`comfyui_portable.7z.002`](https://github.com/hiddenswitch/ComfyUI/releases/download/latest/comfyui_portable.7z.002)).
2. Run `comfyui_portable.exe` to extract a workspace containing an embedded Python 3.12.
3. Double-click on `comfyui.bat` inside `ComfyUI_Workspace` to start the server.

## LTS Custom Nodes

These packages have been adapted to be installable with `pip` and download models to the correct places:

- **ELLA T5 Text Conditioning for SD1.5**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-ella.git`
- **IP Adapter**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-ipadapter-plus`
- **ControlNet Auxiliary Preprocessors**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-controlnet-aux.git`.
- **LayerDiffuse Alpha Channel Diffusion**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-layerdiffuse.git`.
- **BRIA Background Removal**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-bria-bg-removal.git`
- **Video Frame Interpolation**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-video-frame-interpolation`
- **Video Helper Suite**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-video-helper-suite`
- **AnimateDiff Evolved**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-animatediff-evolved`
- **Impact Pack**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-impact-pack`
- **TensorRT**: `uv pip install git+https://github.com/AppMAna/appmana-comfyui-nodes-tensorrt`

Custom nodes are generally supported by this fork. Use these for a bug-free experience.

Request first-class, LTS support for more nodes by [creating a new issue](https://github.com/hiddenswitch/ComfyUI/issues/new). Remember, ordinary custom nodes from the ComfyUI ecosystem work in this fork. Create an issue if you experience a bug or if you think something needs more attention.

##### Running with TLS

To serve with `https://` on Windows easily, use [Caddy](https://github.com/caddyserver/caddy/releases/download/v2.7.6/caddy_2.7.6_windows_amd64.zip). Extract `caddy.exe` to a directory, then run it:

```shell
caddy reverse-proxy --from localhost:443 --to localhost:8188 --tls self_signed
```

## Performance

### Memory Offloading (`--novram`)

If you have 16GB of VRAM or less, start ComfyUI with `--novram`:

```shell
uv run comfyui --novram
```

Despite the name, `--novram` does not prevent GPU usage. It aggressively offloads model weights from VRAM when they are not actively needed. On modern systems with fast PCIe connections, this has minimal impact on inference speed while allowing you to run much larger models.

### Model Quantization

Inference speed is proportional to a model's size in memory. Quantized models run faster because they consume less memory bandwidth. However, not all quantization formats are equal:

- **FP8 quantizations** (e.g., `fp8_e4m3fn`) offer the best quality-to-speed tradeoff. They are smaller than full-precision models while maintaining high output quality.
- **GGUF quantizations** produce noticeably worse output quality. You are better off using a full-precision or FP8 model with `--novram` than using a GGUF model that fits in VRAM.

For diffusion models, the amount of the model resident in VRAM at any given time does not meaningfully affect inference speed, because the bottleneck is the sequential denoising steps, not weight loading. Use the highest quality quantization available and rely on `--novram` for memory management.

### Swap and Pinned Memory

If your system has swap enabled and you have less than 16GB of VRAM, you should disable pinned memory:

```shell
uv run comfyui --novram --disable-pinned-memory
```

Pinned (page-locked) memory cannot be swapped out by the OS. On memory-constrained systems with swap enabled, this can cause the remaining unpinned memory to thrash to disk, resulting in worse performance than not using pinned memory at all.

### NVIDIA Ampere and Newer (`--fast cublas_ops`) (CUDA only)

If you have an Ampere GPU (RTX 30 series, A100) or newer (RTX 40 series, RTX 50 series), enable cuBLAS optimizations:

```shell
uv run comfyui --fast cublas_ops
```

This uses optimized cuBLAS matrix multiplication kernels that are available on compute capability 8.0+ hardware.

## Triton

Triton is used by some custom nodes and performance features.

### CUDA

**Linux:**
```shell
uv pip install --torch-backend=auto triton
```

**Windows:**
```powershell
uv pip install triton-windows
```

See https://github.com/woct0rdho/triton-windows for details.

### ROCm

The ROCm nightly indexes include Triton under the package name `triton` (not `pytorch-triton-rocm`). This works on both Linux and Windows. Install it from the same index URL you used for PyTorch:

**RX 9000 (RDNA 4):**
```shell
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ --pre triton
```

**RX 7000 (RDNA 3):**
```shell
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ --pre triton
```

**Strix Halo (RDNA 3.5):**
```shell
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre triton
```

**Instinct MI300 (CDNA 3):**
```shell
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ --pre triton
```

**Instinct MI350 (CDNA 4):**
```shell
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/ --pre triton
```

If you followed the ROCm installation steps above, Triton was already installed alongside PyTorch.

## CUDA Extension Wheels

The Windows + CUDA and Linux + CUDA blocks install `sageattention` and `flash-attn` from the AppMana package facade. The default facade index serves CUDA 13.0 builds:

```shell
uv pip install --extra-index-url https://nodes.appmana.com/simple/ sageattention flash-attn
```

Use a CUDA segment when you need a specific build. For example:

```shell
uv pip install --extra-index-url https://nodes.appmana.com/simple/cu130/ sageattention flash-attn
uv pip install --extra-index-url https://nodes.appmana.com/simple/cu128/ sageattention flash-attn
uv pip install --extra-index-url https://nodes.appmana.com/simple/cu126/ flash-attn
```

`sageattention` is available for `cu128` and `cu130`. `flash-attn` has broader CUDA coverage through prebuilt wheels. These packages are CUDA-only and are not supported on macOS.

To start with SageAttention explicitly enabled:

```shell
comfyui serve --guess-settings --use-sage-attention
```
