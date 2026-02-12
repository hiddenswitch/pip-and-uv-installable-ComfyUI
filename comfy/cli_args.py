"""
Stub parser for upstream merge compatibility.

All ``parser.add_argument`` calls are no-ops. Real CLI parsing is handled
by Typer in ``comfy.cmd.cli``. The module-level ``args`` attribute returns
the current execution context's ``Configuration`` via a module property.

When upstream adds new ``parser.add_argument(...)`` lines, they merge
cleanly because the stub silently accepts them. You must also:
  1. Add the corresponding field to ``cli_args_types.py`` Configuration.
  2. Add a ``typer.Option`` to the appropriate command in ``comfy/cmd/cli.py``.
"""
from __future__ import annotations

import enum
import sys

from .cli_args_types import Configuration, LatentPreviewMethod, PerformanceFeature
from .component_model.module_property import create_module_properties
class _StubGroup:
    def add_argument(self, *a, **kw):
        return self

    def add_mutually_exclusive_group(self, **kw):
        return _StubGroup()


class _StubParser(_StubGroup):
    def parse_args(self, args=None):
        return Configuration()

    def parse_known_args(self, args=None, **kw):
        return Configuration(), []

    def parse_known_args_with_config_files(self, args=None, **kw):
        from .cli_args_types import ParsedArgs
        return ParsedArgs(Configuration(), [], [])

parser = _StubParser()


# ===================================================================
# Upstream add_argument calls — all no-ops, kept for clean merges.
# ===================================================================

parser.add_argument('-w', "--cwd", type=str, default=None,
                    help="Specify the working directory. If not set, this is the current working directory. models/, input/, output/ and other directories will be located here by default.")
parser.add_argument("--base-paths", type=str, nargs='+', default=[],
                    help="Additional base paths for custom nodes, models and inputs.")
parser.add_argument('-H', "--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0,::",
                    help="Specify the IP address to listen on (default: 127.0.0.1). You can give a list of ip addresses by separating them with a comma like: 127.2.2.2,127.3.3.3 If --listen is provided without an argument, it defaults to 0.0.0.0,:: (listens on all ipv4 and ipv6)")
parser.add_argument("--port", type=int, default=8188, help="Set the listen port.")
parser.add_argument("--enable-cors-header", type=str, default=None, metavar="ORIGIN", nargs="?", const="*",
                    help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'.")
parser.add_argument("--max-upload-size", type=float, default=100, help="Set the maximum upload size in MB.")
parser.add_argument("--base-directory", type=str, default=None, help="Set the ComfyUI base directory for models, custom_nodes, input, output, temp, and user directories.")
parser.add_argument("--extra-model-paths-config", type=str, default=[], metavar="PATH", nargs='+',
                    help="Load one or more extra_model_paths.yaml files.")
parser.add_argument("--output-directory", type=str, default=None, help="Set the ComfyUI output directory.")
parser.add_argument("--temp-directory", type=str, default=None, help="Set the ComfyUI temp directory.")
parser.add_argument("--input-directory", type=str, default=None, help="Set the ComfyUI input directory.")
parser.add_argument("--auto-launch", action="store_true", help="Automatically launch ComfyUI in the default browser.")
parser.add_argument("--disable-auto-launch", action="store_true", help="Disable auto launching the browser.")
parser.add_argument("--cuda-device", type=int, default=None, metavar="DEVICE_ID",
                    help="Set the id of the cuda device this instance will use.")
parser.add_argument("--default-device", type=int, default=None, metavar="DEFAULT_DEVICE_ID", help="Set the id of the default device.")

cm_group = parser.add_mutually_exclusive_group()
cm_group.add_argument("--cuda-malloc", action="store_true", help="Enable cudaMallocAsync.")
cm_group.add_argument("--disable-cuda-malloc", action="store_true", default=True, help="Disable cudaMallocAsync.")

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--force-fp32", action="store_true", help="Force fp32.")
fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16.")
fp_group.add_argument("--force-bf16", action="store_true", help="Force bf16.")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument("--fp32-unet", action="store_true", help="Run the diffusion model in fp32.")
fpunet_group.add_argument("--fp64-unet", action="store_true", help="Run the diffusion model in fp64.")
fpunet_group.add_argument("--bf16-unet", action="store_true", help="Run the diffusion model in bf16.")
fpunet_group.add_argument("--fp16-unet", action="store_true", help="Run the diffusion model in fp16.")
fpunet_group.add_argument("--fp8_e4m3fn-unet", action="store_true", help="Store unet weights in fp8_e4m3fn.")
fpunet_group.add_argument("--fp8_e5m2-unet", action="store_true", help="Store unet weights in fp8_e5m2.")
fpunet_group.add_argument("--fp8_e8m0fnu-unet", action="store_true", help="Store unet weights in fp8_e8m0fnu.")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--fp16-vae", action="store_true", help="Run the VAE in fp16.")
fpvae_group.add_argument("--fp32-vae", action="store_true", help="Run the VAE in full precision fp32.")
fpvae_group.add_argument("--bf16-vae", action="store_true", help="Run the VAE in bf16.")

parser.add_argument("--cpu-vae", action="store_true", help="Run the VAE on the CPU.")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--fp8_e4m3fn-text-enc", action="store_true", help="Store text encoder weights in fp8 (e4m3fn).")
fpte_group.add_argument("--fp8_e5m2-text-enc", action="store_true", help="Store text encoder weights in fp8 (e5m2).")
fpte_group.add_argument("--fp16-text-enc", action="store_true", help="Store text encoder weights in fp16.")
fpte_group.add_argument("--fp32-text-enc", action="store_true", help="Store text encoder weights in fp32.")
fpte_group.add_argument("--bf16-text-enc", action="store_true", help="Store text encoder weights in bf16.")

parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1, help="Use torch-directml.")
parser.add_argument("--oneapi-device-selector", type=str, default=None, metavar="SELECTOR_STRING", help="Sets the oneAPI device(s).")
parser.add_argument("--disable-ipex-optimize", action="store_true", help="Disables ipex.optimize.")
parser.add_argument("--supports-fp8-compute", action="store_true", help="Act as if device supports fp8 compute.")

parser.add_argument("--preview-method", type=LatentPreviewMethod, default=LatentPreviewMethod.Auto, help="Default preview method.")
parser.add_argument("--preview-size", type=int, default=512, help="Sets the maximum preview size for sampler nodes.")

cache_group = parser.add_mutually_exclusive_group()
cache_group.add_argument("--cache-classic", action="store_true", help="Use old style caching.")
cache_group.add_argument("--cache-lru", type=int, default=0, help="Use LRU caching.")
cache_group.add_argument("--cache-none", action="store_true", help="Disable caching.")
cache_group.add_argument("--cache-ram", nargs='?', const=4.0, type=float, default=0, help="RAM pressure caching.")

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument("--use-split-cross-attention", action="store_true", help="Use split cross attention.")
attn_group.add_argument("--use-quad-cross-attention", action="store_true", help="Use sub-quadratic cross attention.")
attn_group.add_argument("--use-pytorch-cross-attention", action="store_true", help="Use PyTorch 2.0 cross attention.", default=True)
attn_group.add_argument("--use-sage-attention", action="store_true", help="Use sage attention.")
attn_group.add_argument("--use-flash-attention", action="store_true", help="Use FlashAttention.")

parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")

upcast = parser.add_mutually_exclusive_group()
upcast.add_argument("--force-upcast-attention", action="store_true", help="Force attention upcasting.")
upcast.add_argument("--dont-upcast-attention", action="store_true", help="Disable attention upcasting.")

parser.add_argument("--enable-manager", action="store_true", help="Enable the ComfyUI-Manager feature.")
manager_group = parser.add_mutually_exclusive_group()
manager_group.add_argument("--disable-manager-ui", action="store_true", help="Disables only the ComfyUI-Manager UI.")
manager_group.add_argument("--enable-manager-legacy-ui", action="store_true", help="Enables the legacy UI of ComfyUI-Manager.")

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--gpu-only", action="store_true", help="Store and run everything on the GPU.")
vram_group.add_argument("--highvram", action="store_true", help="Keep models in GPU memory.")
vram_group.add_argument("--normalvram", action="store_true", help="Force normal vram use.")
vram_group.add_argument("--lowvram", action="store_true", help="Split unet for less vram.")
vram_group.add_argument("--novram", action="store_true", help="Minimal vram usage.")
vram_group.add_argument("--cpu", action="store_true", help="Use CPU for everything.")

parser.add_argument("--reserve-vram", type=float, default=0, help="VRAM to reserve in GB.")
parser.add_argument("--async-offload", nargs='?', const=2, type=int, default=None, metavar="NUM_STREAMS", help="Use async weight offloading.")
parser.add_argument("--disable-async-offload", action="store_true", help="Disable async weight offloading.")
parser.add_argument("--force-non-blocking", action="store_true", help="Force non-blocking operations.")
parser.add_argument("--default-hashing-function", type=str, choices=['md5', 'sha1', 'sha256', 'sha512'], default='sha256', help="Hash function for comparisons.")
parser.add_argument("--disable-smart-memory", action="store_true", help="Disable smart memory management.")
parser.add_argument("--deterministic", action="store_true", help="Use deterministic algorithms.")

parser.add_argument("--fast", nargs="*", type=PerformanceFeature, default=set(), help="Enable performance optimizations.")
parser.add_argument("--disable-pinned-memory", action="store_true", help="Disable pinned memory use.")

parser.add_argument("--mmap-torch-files", action="store_true", help="Use mmap for ckpt/pt files.")
parser.add_argument("--disable-mmap", action="store_true", help="Don't use mmap for safetensors.")

parser.add_argument("--dont-print-server", action="store_true", help="Don't print server output.")
parser.add_argument("--quick-test-for-ci", action="store_true", help="Quick test for CI.")
parser.add_argument("--windows-standalone-build", default=hasattr(sys, 'frozen') and getattr(sys, 'frozen'),
                    action="store_true", help="Windows standalone build.")

parser.add_argument("--disable-metadata", action="store_true", help="Disable saving prompt metadata.")
parser.add_argument("--disable-all-custom-nodes", action="store_true", help="Disable all custom nodes.")
parser.add_argument("--whitelist-custom-nodes", type=str, nargs='+', default=[], help="Custom nodes to load.")
parser.add_argument("--blacklist-custom-nodes", type=str, nargs='+', default=[], help="Custom nodes to never load.")
parser.add_argument("--disable-api-nodes", action="store_true", help="Disable API nodes.")
parser.add_argument("--enable-eval", action="store_true", help="Enable eval nodes.")

parser.add_argument("--multi-user", action="store_true", help="Enable per-user storage.")
parser.add_argument("--create-directories", action="store_true", help="Create default directories then exit.")
parser.add_argument("--log-stdout", action="store_true", help="Send output to stdout.")

parser.add_argument("--plausible-analytics-base-url", required=False, help="Analytics base URL.")
parser.add_argument("--plausible-analytics-domain", required=False, help="Analytics domain.")
parser.add_argument("--analytics-use-identity-provider", action="store_true", help="Use identity for analytics.")
parser.add_argument("--distributed-queue-connection-uri", type=str, default=None, help="AMQP URL.")
parser.add_argument('--distributed-queue-worker', required=False, action="store_true", help='Run as worker.')
parser.add_argument('--distributed-queue-frontend', required=False, action="store_true", help='Run as frontend.')
parser.add_argument("--distributed-queue-name", type=str, default="comfyui", help="Queue name.")
parser.add_argument("--external-address", required=False, help="External address base URL.")
parser.add_argument("--logging-level", type=str, default='INFO', help='Logging level.')
parser.add_argument("--disable-known-models", action="store_true", help="Disable known model downloads.")
parser.add_argument("--max-queue-size", type=int, default=65536, help="Max queue size.")

parser.add_argument("--otel-service-name", type=str, default="comfyui", help="OTel service name.")
parser.add_argument("--otel-service-version", type=str, default="0.0.1", help="OTel service version.")
parser.add_argument("--otel-exporter-otlp-endpoint", type=str, default=None, help="OTLP endpoint.")
parser.add_argument("--force-channels-last", action="store_true", help="Force channels last format.")
parser.add_argument("--force-hf-local-dir-mode", action="store_true", help="HuggingFace local_dir mode.")
parser.add_argument("--enable-video-to-image-fallback", action="store_true", help="Enable video-to-image fallback.")

parser.add_argument("--front-end-version", type=str, default="comfyanonymous/ComfyUI@latest", help="Frontend version.")
parser.add_argument('--panic-when', nargs='+', type=str, default=[], help="Exception class names to panic on.")
parser.add_argument("--front-end-root", type=str, default=None, help="Local frontend directory.")
parser.add_argument("--executor-factory", type=str, default="ThreadPoolExecutor", help="Executor type.")
parser.add_argument("--openai-api-key", required=False, type=str, default=None, help="OpenAI API key.")
parser.add_argument("--ideogram-api-key", required=False, type=str, default=None, help="Ideogram API key.")
parser.add_argument("--anthropic-api-key", required=False, type=str, help="Anthropic API key.")
parser.add_argument("--user-directory", type=str, default=None, help="User directory.")
parser.add_argument("--enable-compress-response-body", action="store_true", help="Compress response body.")
parser.add_argument("--comfy-api-base", type=str, default="https://api.comfy.org", help="ComfyUI API base URL.")
parser.add_argument("--block-runtime-package-installation", action="store_true", help="Block runtime installs.")
parser.add_argument("--disable-assets-autoscan", action="store_true", help="Disable asset scanning.")
parser.add_argument("--database-url", type=str, default=None, help="Database URL.")
parser.add_argument("--workflows", type=str, nargs='+', default=[], help="Execute workflows and exit.")
parser.add_argument("--prompt", type=str, default=None, help="Override positive prompt.")
parser.add_argument("--negative-prompt", type=str, default=None, help="Override negative prompt.")
parser.add_argument("--steps", type=int, default=None, help="Override sampling steps.")
parser.add_argument("--seed", type=int, default=None, help="Override seed.")
parser.add_argument("--image", type=str, nargs='+', default=None, help="Override image inputs.")
parser.add_argument("--video", type=str, nargs='+', default=None, help="Override video inputs.")
parser.add_argument("--audio", type=str, nargs='+', default=None, help="Override audio inputs.")
parser.add_argument("-o", "--output", type=str, default=None, help="Override output directory.")
parser.add_argument("--guess-settings", action="store_true", help="Auto-detect best settings.")

parser.add_argument("--disable-requests-caching", action="store_true", help="Disable requests caching.")
parser.add_argument("--disable-manager-model-fallback", action="store_true", default=False, help="Disable manager model fallback.")
parser.add_argument("--refresh-manager-models", action="store_true", default=False, help="Fetch latest model list.")


class EnumAction:
    def __init__(self, **kw):
        pass

    def __call__(self, *a, **kw):
        pass


DEFAULT_VERSION_STRING = "comfyanonymous/ComfyUI@latest"


_module_properties = create_module_properties()


@_module_properties.getter
def _args() -> Configuration:
    from .execution_context import current_execution_context
    return current_execution_context().configuration


args: Configuration


def default_configuration() -> Configuration:
    """Return a Configuration with all defaults (no CLI parsing)."""
    return Configuration()


def cli_args_configuration() -> Configuration:
    """Return a Configuration with all defaults.

    In the Typer-based CLI, real parsing is in ``comfy.cmd.cli``. This
    function exists for backward compat and returns defaults.
    """
    return Configuration()


def enables_dynamic_vram():
    config = _args()
    return PerformanceFeature.DynamicVRAM in config.fast and not config.highvram and not config.gpu_only
