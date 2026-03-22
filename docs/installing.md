# Getting Started

## Installing

These instructions will install an interactive ComfyUI using the command line. Find your platform and GPU below and copy-paste the complete sequence of commands.

When you run workflows that use well-known models, they will be downloaded automatically.

### Linux — NVIDIA (CUDA)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/ComfyUI_Workspace && cd ~/ComfyUI_Workspace
uv venv --python 3.12
uv pip install --torch-backend=auto "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
uv run --no-sync comfyui
```

### Linux — AMD RX 7000 (RDNA 3)

Requires the latest AMDGPU driver.

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/ComfyUI_Workspace && cd ~/ComfyUI_Workspace
uv venv --python 3.12
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ --pre torch torchaudio torchvision triton
uv pip install "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
uv run --no-sync comfyui --fp32-vae
```

### Linux — AMD RX 9000 (RDNA 4)

Requires the latest AMDGPU driver.

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/ComfyUI_Workspace && cd ~/ComfyUI_Workspace
uv venv --python 3.12
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ --pre torch torchaudio torchvision triton
uv pip install "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
uv run --no-sync comfyui --fp32-vae
```

### Linux — Intel Arc / Max / iGPU (XPU, Ubuntu)

Use XPU when `torch.xpu.is_available()` is true. On Ubuntu, the host needs both:

- a recent Intel GPU kernel/firmware stack that exposes `/dev/dri/renderD*`
- the Intel userland compute stack used by XPU workloads

In this repo, the CI lane itself runs inside Intel's XPU PyTorch container:

```shell
docker run --rm -it --device /dev/dri intel/intel-extension-for-pytorch:2.8.10-xpu bash
```

That container provides the XPU-enabled PyTorch userspace. The host still has to provide the kernel-side GPU support. Our worker provisioning currently does two Intel-specific things:

- uses Ubuntu 24.04 HWE kernel on workers
- patches DG2/Arc firmware by installing upstream `dg2_guc_70.bin`, rebuilding initramfs, and holding `linux-firmware`

The Kubernetes runner hook also requires the Intel device plugin resources:

- `gpu.intel.com/xe: 1`
- `gpu.intel.com/family: A_Series`

The Intel PPA / Intel graphics repo below is about the userland compute runtime. It does not replace the need for the host kernel driver and firmware.

For Ubuntu desktop/client GPUs, Intel's documented package flow is:

**Ubuntu 24.04 / newer desktop path**
```shell
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt-get update
sudo apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc
sudo apt-get install -y libze-dev intel-ocloc
```

**Ubuntu 22.04 path**
```shell
sudo apt-get update
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
sudo apt update
sudo apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo libze-dev intel-ocloc
```

Inside that environment:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/ComfyUI_Workspace && cd ~/ComfyUI_Workspace
uv venv --python 3.11
uv pip install "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
python -c "import torch; print(torch.xpu.is_available())"
uv run --no-sync comfyui
```

If you are installing directly on Ubuntu instead of using the container, verify that these work before installing ComfyUI:

```shell
python -c "import torch; print(torch.xpu.is_available())"
python -c "import torch; print(torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'no xpu')"
clinfo | grep "Device Name"
```

If `torch.xpu.is_available()` is false, fix the Intel driver/runtime stack first. ComfyUI will not make XPU appear if the base PyTorch/XPU environment is not healthy.

### Linux — AMD ROCm (Ubuntu)

Use ROCm when `torch.cuda.is_available()` is true on an AMD system with a ROCm build of PyTorch. On Ubuntu, the host needs:

- the AMDGPU kernel driver
- ROCm userland compatible with your GPU architecture
- a ROCm PyTorch build matching that architecture

Unlike the Intel XPU lane, our ROCm CI lane uses a ROCm PyTorch container but still depends on the host AMDGPU/ROCK kernel stack and `/dev/kfd` / `/dev/dri` being healthy before the container starts.

AMD's documented Ubuntu flow is to install the `amdgpu-install` package first, then install the ROCm stack from that package. For example:

```shell
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/noble/amdgpu-install_6.4.60401-1_all.deb
sudo apt install ./amdgpu-install_6.4.60401-1_all.deb
```

After that, install the ROCm components appropriate for your machine, then install the matching ROCm PyTorch wheels. For ComfyUI, the important part is that the host AMDGPU/ROCm stack must be working before you install torch or start a ROCm container.

For RDNA 3 and newer consumer GPUs, this repo uses AMD's architecture-specific nightly wheels. The pattern is:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/ComfyUI_Workspace && cd ~/ComfyUI_Workspace
uv venv --python 3.12
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ --pre torch torchaudio torchvision triton
uv pip install "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.hip)"
uv run --no-sync comfyui --fp32-vae
```

For Ubuntu host setup, the key rule is: install the ROCm stack that matches your GPU family first, then install the matching PyTorch wheels. Do not mix a generic CPU torch wheel with ROCm expectations. On bare metal, verify `/dev/kfd` and `/dev/dri/renderD*` exist before debugging ComfyUI itself.

Verify the ROCm path before debugging ComfyUI itself:

```shell
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.hip)"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no rocm device')"
```

If those checks fail, fix the ROCm driver/runtime installation first.

### Windows — NVIDIA (CUDA)

Open **Windows PowerShell**, then:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
choco install -y uv
mkdir ~\Documents\ComfyUI_Workspace
cd ~\Documents\ComfyUI_Workspace
uv venv --python 3.12
uv pip install --torch-backend=auto "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
uv run --no-sync comfyui
```

### Windows — AMD RX 7000 (RDNA 3)

Requires the latest Adrenaline driver. Open **Windows PowerShell**, then:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
choco install -y uv
mkdir ~\Documents\ComfyUI_Workspace
cd ~\Documents\ComfyUI_Workspace
uv venv --python 3.12
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ --pre torch torchaudio torchvision triton
uv pip install "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
uv run --no-sync comfyui --fp32-vae
```

### Windows — AMD RX 9000 (RDNA 4)

Requires the latest Adrenaline driver. Open **Windows PowerShell**, then:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
choco install -y uv
mkdir ~\Documents\ComfyUI_Workspace
cd ~\Documents\ComfyUI_Workspace
uv venv --python 3.12
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ --pre torch torchaudio torchvision triton
uv pip install "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
uv run --no-sync comfyui --fp32-vae
```

### macOS (Apple Silicon)

```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
HOMEBREW_NO_AUTO_UPDATE=1 brew install uv
mkdir -p ~/Documents/ComfyUI_Workspace && cd ~/Documents/ComfyUI_Workspace
uv venv --python 3.12
uv pip install --torch-backend=auto "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
uv run --no-sync comfyui
```

#### FP8 Support on MPS

Apple Silicon does not natively support FP8 tensor operations. The `fp4-fp8-for-torch-mps` package is installed automatically and provides software emulation of FP8 compute on MPS via Metal kernels, enabling FP8-quantized models (e.g., `ltx-2.3-22b-dev-fp8.safetensors`) to run on macOS.

### Running Again Later

To start ComfyUI again after closing your terminal, `cd` into your workspace and run:

```shell
cd ~/ComfyUI_Workspace
uv run --no-sync comfyui
```

On Windows:
```powershell
cd ~\Documents\ComfyUI_Workspace
uv run --no-sync comfyui
```

### Upgrading

```shell
uv pip install --upgrade "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
```

For NVIDIA users who want to ensure the correct CUDA version is maintained:
```shell
uv pip install --torch-backend=auto --upgrade "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
```

### Other AMD GPUs (ROCm)

The following architectures are also supported. Install PyTorch from the matching index, then install ComfyUI:

**Strix Halo iGPU (RDNA 3.5, `gfx1151`):**
```shell
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision triton
uv pip install "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
```

**Instinct MI300A / MI300X (CDNA 3, `gfx942`):**
```shell
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ --pre torch torchaudio torchvision triton
uv pip install "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
```

**Instinct MI350X / MI355X (CDNA 4, `gfx950`):**
```shell
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/ --pre torch torchaudio torchvision triton
uv pip install "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
```

**RX 6000 (RDNA 2) and RX 5000 (RDNA 1):** These architectures are no longer well supported by AMD. There are no architecture-specific builds available.

## Why `--no-sync`?

By default, `uv run` performs a project sync before running the command. This means it checks the lockfile, resolves dependencies, and potentially modifies your environment every time you run ComfyUI. This is undesirable because:

- It adds startup latency
- It can unexpectedly change your installed packages
- It can fail if network is unavailable

Using `--no-sync` skips this automatic sync. You already installed packages explicitly with `uv pip install`, so there is no need for `uv run` to re-resolve them. See the [uv documentation on automatic sync](https://docs.astral.sh/uv/concepts/projects/sync/#automatic-lock-and-sync) for more details.

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
- `--torch-backend=cpu` — CPU-only (no GPU acceleration)

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
uv pip install --torch-backend=cu128 "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"

# Install with CUDA 13.0
uv pip install --torch-backend=cu130 "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
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
uv run --no-sync comfyui --novram
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
uv run --no-sync comfyui --novram --disable-pinned-memory
```

Pinned (page-locked) memory cannot be swapped out by the OS. On memory-constrained systems with swap enabled, this can cause the remaining unpinned memory to thrash to disk, resulting in worse performance than not using pinned memory at all.

### NVIDIA Ampere and Newer (`--fast cublas_ops`) (CUDA only)

If you have an Ampere GPU (RTX 30 series, A100) or newer (RTX 40 series, RTX 50 series), enable cuBLAS optimizations:

```shell
uv run --no-sync comfyui --fast cublas_ops
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

## Stable ABI CUDA extensions

AppMana publishes stable-ABI wheels for SageAttention and Nunchaku. These wheels use
Python `abi3` and the PyTorch stable ABI, so they are the preferred install path for
CUDA builds on PyTorch `>= 2.9`.

Use `cu130` for new installations. That is the preferred choice for NVIDIA RTX 20xx
and newer GPUs, including RTX 30xx, 40xx, and 50xx cards.

Use `cu128` only when you specifically need CUDA 12.8 compatibility.

These packages are not supported on macOS.

### SageAttention (CUDA only)

SageAttention accelerates attention computation in diffusion models.

Install `triton-windows` first if you are on Windows, then install SageAttention from
the matching AppMana stable-ABI index:

**Recommended for new installs: CUDA 13.0 (`cu130`)**

```shell
uv pip install triton-windows
uv pip install --no-deps sageattention --index-url https://appmana.github.io/forks-sageattention-stable-abi/cu130
```

**Fallback: CUDA 12.8 (`cu128`)**

```shell
uv pip install triton-windows
uv pip install --no-deps sageattention --index-url https://appmana.github.io/forks-sageattention-stable-abi/cu128
```

Or build from source instead (requires CUDA Toolkit and build tools):

```shell
sudo apt-get install -y build-essential nvidia-cuda-toolkit
uv pip install --no-build-isolation "sageattention@git+https://github.com/woct0rdho/SageAttention.git"
```

#### Running with SageAttention

```shell
uv run --no-sync comfyui --use-sage-attention
```

### Nunchaku (CUDA only)

Nunchaku provides optimized 4-bit inference kernels. For PyTorch `>= 2.9`, install it
from the matching AppMana stable-ABI index:

**Recommended for new installs: CUDA 13.0 (`cu130`)**

```shell
uv pip install --no-deps nunchaku --index-url https://appmana.github.io/forks-nunchaku-stable-abi/cu130
```

**Fallback: CUDA 12.8 (`cu128`)**

```shell
uv pip install --no-deps nunchaku --index-url https://appmana.github.io/forks-nunchaku-stable-abi/cu128
```
