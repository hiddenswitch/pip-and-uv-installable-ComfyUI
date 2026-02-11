"""
Typer CLI application for ComfyUI.

Single entry point for all commands: serve, worker, post-workflow,
create-directories, list-workflow-templates.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from ..cli_args_types import Configuration, LatentPreviewMethod, PerformanceFeature

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="comfyui",
    no_args_is_help=False,
    context_settings={"auto_envvar_prefix": "COMFYUI"},
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DEFAULT_VERSION_STRING = "comfyanonymous/ComfyUI@latest"

_VRAM_MODES = ("gpu_only", "highvram", "normalvram", "lowvram", "novram", "cpu")
_PRECISION_MODES = ("force_fp32", "force_fp16", "force_bf16")
_UNET_MODES = ("fp32_unet", "fp64_unet", "bf16_unet", "fp16_unet", "fp8_e4m3fn_unet", "fp8_e5m2_unet", "fp8_e8m0fnu_unet")
_VAE_MODES = ("fp16_vae", "fp32_vae", "bf16_vae")
_TEXT_ENC_MODES = ("fp8_e4m3fn_text_enc", "fp8_e5m2_text_enc", "fp16_text_enc", "fp32_text_enc", "bf16_text_enc")
_ATTENTION_MODES = ("use_split_cross_attention", "use_quad_cross_attention", "use_pytorch_cross_attention", "use_sage_attention", "use_flash_attention")
_CACHE_MODES = ("cache_classic", "cache_lru", "cache_none", "cache_ram")
_UPCAST_MODES = ("force_upcast_attention", "dont_upcast_attention")


def _set_config_context(config: Configuration):
    """Set config into the execution context so main_pre reads the correct values.

    Must be called before any import that transitively triggers main_pre.py's
    import-time side effects (e.g. importing main.py, model_management, nodes.package).
    """
    from dataclasses import replace
    from ..execution_context import comfyui_execution_context, current_execution_context
    ctx = replace(current_execution_context(), configuration=config)
    comfyui_execution_context.set(ctx)


def _validate_mutex(params: dict, group_name: str, fields: tuple[str, ...]):
    """Raise if more than one field in a mutually exclusive group is set to a truthy non-default value."""
    set_modes = [f for f in fields if params.get(f)]
    if len(set_modes) > 1:
        flags = ", ".join(f"--{f.replace('_', '-')}" for f in set_modes)
        raise typer.BadParameter(f"Only one of {flags} can be set ({group_name})")


def _build_config(params: dict) -> Configuration:
    """Build Configuration from Typer-parsed parameters."""
    _validate_mutex(params, "VRAM mode", _VRAM_MODES)
    _validate_mutex(params, "precision", _PRECISION_MODES)
    _validate_mutex(params, "UNet precision", _UNET_MODES)
    _validate_mutex(params, "VAE precision", _VAE_MODES)
    _validate_mutex(params, "text encoder precision", _TEXT_ENC_MODES)
    _validate_mutex(params, "attention", _ATTENTION_MODES)
    _validate_mutex(params, "upcast attention", _UPCAST_MODES)

    filtered = {
        k: v for k, v in params.items()
        if v is not None and k not in ("ctx", "config") and not k.startswith("_")
    }

    if "fast" in filtered:
        raw = filtered["fast"]
        items = set()
        for v in raw:
            for piece in v.split(","):
                piece = piece.strip()
                if piece:
                    items.add(PerformanceFeature(piece))
        filtered["fast"] = items

    if "preview_method" in filtered and isinstance(filtered["preview_method"], str):
        filtered["preview_method"] = LatentPreviewMethod(filtered["preview_method"])

    for list_field in ("base_paths", "extra_model_paths_config", "panic_when",
                       "whitelist_custom_nodes", "blacklist_custom_nodes",
                       "image", "video", "audio", "workflows"):
        if list_field in filtered and isinstance(filtered[list_field], (list, tuple)):
            expanded = []
            for v in filtered[list_field]:
                for piece in str(v).split(","):
                    piece = piece.strip()
                    if piece:
                        expanded.append(piece)
            filtered[list_field] = expanded

    # windows_standalone_build enables auto_launch, but disable_auto_launch always wins
    if filtered.get("windows_standalone_build"):
        filtered["auto_launch"] = True
    if filtered.get("disable_auto_launch"):
        filtered["auto_launch"] = False
    if filtered.get("force_fp16"):
        filtered["fp16_unet"] = True

    config = Configuration(**filtered)
    return config


def _load_config_file(path: str) -> dict:
    """Load a YAML or JSON config file."""
    import yaml
    with open(path) as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f) or {}


def _find_default_config_file() -> Optional[str]:
    """Find the first default config file that exists."""
    for name in ("config.yaml", "config.json", "config.cfg", "config.ini"):
        if os.path.exists(name):
            return name
    return None


# ---------------------------------------------------------------------------
# Typer callback: config file loading, default-to-serve
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """ComfyUI - The most powerful and modular diffusion model GUI and backend."""
    if ctx.invoked_subcommand is None:
        # No subcommand specified - this shouldn't happen because
        # entrypoint() inserts "serve" when no subcommand is given.
        # But as a safety net, invoke serve.
        ctx.invoke(serve)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    # -- Paths --
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory (default: current directory)."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths for custom nodes, models and inputs."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory for models, custom_nodes, input, output, temp, user."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths YAML config files."),
    output_directory: Optional[str] = typer.Option(None, "--output-directory", help="Output directory. Overrides --base-directory."),
    temp_directory: Optional[str] = typer.Option(None, "--temp-directory", help="Temp directory. Overrides --base-directory."),
    input_directory: Optional[str] = typer.Option(None, "--input-directory", help="Input directory. Overrides --base-directory."),
    user_directory: Optional[str] = typer.Option(None, "--user-directory", help="User directory (absolute path). Overrides --base-directory."),

    # -- Server --
    listen: str = typer.Option("127.0.0.1", "-H", "--listen", help="IP address to listen on."),
    port: int = typer.Option(8188, help="Listen port."),
    enable_cors_header: Optional[str] = typer.Option(None, "--enable-cors-header", help="Enable CORS with optional origin (default '*')."),
    max_upload_size: float = typer.Option(100.0, "--max-upload-size", help="Max upload size in MB."),
    auto_launch: bool = typer.Option(False, "--auto-launch", help="Auto-launch browser on start."),
    disable_auto_launch: bool = typer.Option(False, "--disable-auto-launch", help="Disable auto launching the browser."),
    external_address: Optional[str] = typer.Option(None, "--external-address", help="Base URL for external addresses reported by the API."),
    multi_user: bool = typer.Option(False, "--multi-user", help="Enable per-user storage."),
    enable_compress_response_body: bool = typer.Option(False, "--enable-compress-response-body", help="Enable compressing response body."),

    # -- CUDA --
    cuda_device: Optional[int] = typer.Option(None, "--cuda-device", help="CUDA device ID."),
    default_device: Optional[int] = typer.Option(None, "--default-device", help="Default device ID (other devices stay visible)."),
    cuda_malloc: bool = typer.Option(False, "--cuda-malloc", help="Enable cudaMallocAsync."),
    disable_cuda_malloc: bool = typer.Option(True, "--disable-cuda-malloc", help="Disable cudaMallocAsync."),

    # -- Precision --
    force_fp32: bool = typer.Option(False, "--force-fp32", help="Force fp32."),
    force_fp16: bool = typer.Option(False, "--force-fp16", help="Force fp16."),
    force_bf16: bool = typer.Option(False, "--force-bf16", help="Force bf16."),

    # -- UNet precision --
    fp32_unet: bool = typer.Option(False, "--fp32-unet", help="Run diffusion model in fp32."),
    fp64_unet: bool = typer.Option(False, "--fp64-unet", help="Run diffusion model in fp64."),
    bf16_unet: bool = typer.Option(False, "--bf16-unet", help="Run diffusion model in bf16."),
    fp16_unet: bool = typer.Option(False, "--fp16-unet", help="Run diffusion model in fp16."),
    fp8_e4m3fn_unet: bool = typer.Option(False, "--fp8_e4m3fn-unet", help="Store unet weights in fp8_e4m3fn."),
    fp8_e5m2_unet: bool = typer.Option(False, "--fp8_e5m2-unet", help="Store unet weights in fp8_e5m2."),
    fp8_e8m0fnu_unet: bool = typer.Option(False, "--fp8_e8m0fnu-unet", help="Store unet weights in fp8_e8m0fnu."),

    # -- VAE precision --
    fp16_vae: bool = typer.Option(False, "--fp16-vae", help="Run VAE in fp16."),
    fp32_vae: bool = typer.Option(False, "--fp32-vae", help="Run VAE in fp32."),
    bf16_vae: bool = typer.Option(False, "--bf16-vae", help="Run VAE in bf16."),
    cpu_vae: bool = typer.Option(False, "--cpu-vae", help="Run VAE on CPU."),

    # -- Text encoder precision --
    fp8_e4m3fn_text_enc: bool = typer.Option(False, "--fp8_e4m3fn-text-enc", help="Store text encoder in fp8 (e4m3fn)."),
    fp8_e5m2_text_enc: bool = typer.Option(False, "--fp8_e5m2-text-enc", help="Store text encoder in fp8 (e5m2)."),
    fp16_text_enc: bool = typer.Option(False, "--fp16-text-enc", help="Store text encoder in fp16."),
    fp32_text_enc: bool = typer.Option(False, "--fp32-text-enc", help="Store text encoder in fp32."),
    bf16_text_enc: bool = typer.Option(False, "--bf16-text-enc", help="Store text encoder in bf16."),

    # -- DirectML / oneAPI --
    directml: Optional[int] = typer.Option(None, "--directml", help="Use torch-directml (-1 for auto)."),
    oneapi_device_selector: Optional[str] = typer.Option(None, "--oneapi-device-selector", help="oneAPI device selector."),
    disable_ipex_optimize: bool = typer.Option(False, "--disable-ipex-optimize", help="Disable ipex.optimize."),
    supports_fp8_compute: bool = typer.Option(False, "--supports-fp8-compute", help="Act as if device supports fp8 compute."),

    # -- Preview --
    preview_method: str = typer.Option("auto", "--preview-method", help="Preview method: none, auto, latent2rgb, taesd."),
    preview_size: int = typer.Option(512, "--preview-size", help="Max preview size for sampler nodes."),

    # -- Cache --
    cache_classic: bool = typer.Option(False, "--cache-classic", help="Use old style (aggressive) caching."),
    cache_lru: int = typer.Option(0, "--cache-lru", help="LRU cache size (0 = disabled)."),
    cache_none: bool = typer.Option(False, "--cache-none", help="Disable caching entirely."),
    cache_ram: float = typer.Option(0, "--cache-ram", help="RAM pressure cache headroom in GB (0 = disabled, default 4GB when flag used)."),

    # -- Attention --
    use_split_cross_attention: bool = typer.Option(False, "--use-split-cross-attention", help="Use split cross attention."),
    use_quad_cross_attention: bool = typer.Option(False, "--use-quad-cross-attention", help="Use sub-quadratic cross attention."),
    use_pytorch_cross_attention: bool = typer.Option(False, "--use-pytorch-cross-attention", help="Use PyTorch 2.0 cross attention."),
    use_sage_attention: bool = typer.Option(False, "--use-sage-attention", help="Use sage attention."),
    use_flash_attention: bool = typer.Option(False, "--use-flash-attention", help="Use FlashAttention."),
    disable_xformers: bool = typer.Option(False, "--disable-xformers", help="Disable xformers."),

    # -- Attention upcast --
    force_upcast_attention: bool = typer.Option(False, "--force-upcast-attention", help="Force attention upcasting."),
    dont_upcast_attention: bool = typer.Option(False, "--dont-upcast-attention", help="Disable attention upcasting."),

    # -- Manager --
    enable_manager: bool = typer.Option(False, "--enable-manager", help="Enable ComfyUI-Manager."),
    disable_manager_ui: bool = typer.Option(False, "--disable-manager-ui", help="Disable ComfyUI-Manager UI."),
    enable_manager_legacy_ui: bool = typer.Option(False, "--enable-manager-legacy-ui", help="Enable legacy Manager UI."),

    # -- VRAM --
    gpu_only: bool = typer.Option(False, "--gpu-only", help="Store and run everything on GPU."),
    highvram: bool = typer.Option(False, "--highvram", help="Keep models in GPU memory."),
    normalvram: bool = typer.Option(False, "--normalvram", help="Force normal VRAM use."),
    lowvram: bool = typer.Option(False, "--lowvram", help="Split unet for less VRAM."),
    novram: bool = typer.Option(False, "--novram", help="Minimal VRAM usage."),
    cpu: bool = typer.Option(False, "--cpu", help="Use CPU for everything."),

    # -- Memory --
    reserve_vram: float = typer.Option(0, "--reserve-vram", help="VRAM to reserve in GB."),
    async_offload: Optional[int] = typer.Option(None, "--async-offload", help="Async weight offloading (default streams: 2)."),
    disable_async_offload: bool = typer.Option(False, "--disable-async-offload", help="Disable async weight offloading."),
    force_non_blocking: bool = typer.Option(False, "--force-non-blocking", help="Force non-blocking operations."),
    disable_smart_memory: bool = typer.Option(False, "--disable-smart-memory", help="Disable smart memory management."),
    disable_pinned_memory: bool = typer.Option(False, "--disable-pinned-memory", help="Disable pinned memory."),

    # -- Performance --
    fast: Optional[list[str]] = typer.Option(None, "--fast", help="Enable optimizations: fp16_accumulation, fp8_matrix_mult, cublas_ops, autotune, dynamic_vram."),
    deterministic: bool = typer.Option(False, "--deterministic", help="Use deterministic algorithms."),
    default_hashing_function: str = typer.Option("sha256", "--default-hashing-function", help="Hash function: md5, sha1, sha256, sha512."),
    force_channels_last: bool = typer.Option(False, "--force-channels-last", help="Force channels last format."),
    force_hf_local_dir_mode: bool = typer.Option(False, "--force-hf-local-dir-mode", help="Use local_dir for HuggingFace downloads."),

    # -- Mmap --
    mmap_torch_files: bool = typer.Option(False, "--mmap-torch-files", help="Use mmap for ckpt/pt files."),
    disable_mmap: bool = typer.Option(False, "--disable-mmap", help="Don't use mmap for safetensors."),

    # -- Output / Logging --
    dont_print_server: bool = typer.Option(False, "--dont-print-server", help="Don't print server output."),
    log_stdout: bool = typer.Option(False, "--log-stdout", help="Send output to stdout instead of stderr."),
    logging_level: str = typer.Option("INFO", "--logging-level", help="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL."),

    # -- Testing --
    quick_test_for_ci: bool = typer.Option(False, "--quick-test-for-ci", help="Quick test for CI."),
    windows_standalone_build: bool = typer.Option(False, "--windows-standalone-build", help="Enable Windows standalone build features."),

    # -- Metadata / Nodes --
    disable_metadata: bool = typer.Option(False, "--disable-metadata", help="Disable saving prompt metadata."),
    disable_all_custom_nodes: bool = typer.Option(False, "--disable-all-custom-nodes", help="Disable all custom nodes."),
    whitelist_custom_nodes: Optional[list[str]] = typer.Option(None, "--whitelist-custom-nodes", help="Custom node folders to load when --disable-all-custom-nodes."),
    blacklist_custom_nodes: Optional[list[str]] = typer.Option(None, "--blacklist-custom-nodes", help="Custom node folders to never load."),
    disable_api_nodes: bool = typer.Option(False, "--disable-api-nodes", help="Disable API nodes."),
    enable_eval: bool = typer.Option(False, "--enable-eval", help="Enable Python eval nodes."),
    enable_video_to_image_fallback: bool = typer.Option(False, "--enable-video-to-image-fallback", help="Enable video-to-image fallback."),

    # -- Legacy (handled within serve) --
    create_directories: bool = typer.Option(False, "--create-directories", help="Create default directories then exit."),

    # -- Analytics --
    plausible_analytics_base_url: Optional[str] = typer.Option(None, "--plausible-analytics-base-url", help="Analytics base URL."),
    plausible_analytics_domain: Optional[str] = typer.Option(None, "--plausible-analytics-domain", help="Analytics domain."),
    analytics_use_identity_provider: bool = typer.Option(False, "--analytics-use-identity-provider", help="Use platform identifiers for analytics."),

    # -- Distributed --
    distributed_queue_connection_uri: Optional[str] = typer.Option(None, "--distributed-queue-connection-uri", help="AMQP URL for distributed queue."),
    distributed_queue_worker: bool = typer.Option(False, "--distributed-queue-worker", help="Run as distributed worker."),
    distributed_queue_frontend: bool = typer.Option(False, "--distributed-queue-frontend", help="Run as distributed frontend."),
    distributed_queue_name: str = typer.Option("comfyui", "--distributed-queue-name", help="Distributed queue name."),

    # -- Known models / Queue --
    disable_known_models: bool = typer.Option(False, "--disable-known-models", help="Disable automatic model downloads."),
    max_queue_size: int = typer.Option(65536, "--max-queue-size", help="Max prompt queue size."),

    # -- Tracing --
    otel_service_name: str = typer.Option("comfyui", "--otel-service-name", envvar="OTEL_SERVICE_NAME", help="OpenTelemetry service name."),
    otel_service_version: Optional[str] = typer.Option(None, "--otel-service-version", envvar="OTEL_SERVICE_VERSION", help="OpenTelemetry service version."),
    otel_exporter_otlp_endpoint: Optional[str] = typer.Option(None, "--otel-exporter-otlp-endpoint", envvar="OTEL_EXPORTER_OTLP_ENDPOINT", help="OTLP endpoint URL."),

    # -- Frontend --
    front_end_version: str = typer.Option(DEFAULT_VERSION_STRING, "--front-end-version", help="Frontend version: [owner]/[repo]@[version]."),
    front_end_root: Optional[str] = typer.Option(None, "--front-end-root", help="Local frontend directory. Overrides --front-end-version."),

    # -- Panic / Executor --
    panic_when: Optional[list[str]] = typer.Option(None, "--panic-when", help="Exception class names to panic on."),
    executor_factory: str = typer.Option("ThreadPoolExecutor", "--executor-factory", help="Executor type: ThreadPoolExecutor or ProcessPoolExecutor."),

    # -- API Keys --
    openai_api_key: Optional[str] = typer.Option(None, "--openai-api-key", envvar="OPENAI_API_KEY", help="OpenAI API key."),
    ideogram_api_key: Optional[str] = typer.Option(None, "--ideogram-api-key", envvar="IDEOGRAM_API_KEY", help="Ideogram API key."),
    anthropic_api_key: Optional[str] = typer.Option(None, "--anthropic-api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key."),

    # -- ComfyAPI --
    comfy_api_base: str = typer.Option("https://api.comfy.org", "--comfy-api-base", help="ComfyUI API base URL."),
    block_runtime_package_installation: bool = typer.Option(False, "--block-runtime-package-installation", help="Block runtime pip/uv installs."),
    disable_assets_autoscan: bool = typer.Option(False, "--disable-assets-autoscan", help="Disable asset scanning on startup."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Database URL (default: SQLite in user data dir)."),

    # -- Workflows (legacy) --
    workflows: Optional[list[str]] = typer.Option(None, "--workflows", help="Execute workflow(s) and exit."),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Override positive prompt."),
    negative_prompt: Optional[str] = typer.Option(None, "--negative-prompt", help="Override negative prompt."),
    steps: Optional[int] = typer.Option(None, "--steps", help="Override sampling steps."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Override seed."),
    image: Optional[list[str]] = typer.Option(None, "--image", help="Override image inputs."),
    video: Optional[list[str]] = typer.Option(None, "--video", help="Override video inputs."),
    audio: Optional[list[str]] = typer.Option(None, "--audio", help="Override audio inputs."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Override output directory."),

    # -- Misc --
    guess_settings: bool = typer.Option(False, "--guess-settings", help="Auto-detect best settings for this machine."),
    disable_requests_caching: bool = typer.Option(False, "--disable-requests-caching", help="Disable requests caching."),
    disable_manager_model_fallback: bool = typer.Option(False, "--disable-manager-model-fallback", help="Disable manager model database fallback."),
    refresh_manager_models: bool = typer.Option(False, "--refresh-manager-models", help="Fetch latest model list from GitHub."),
):
    """Start the ComfyUI server (default command)."""
    from ..component_model.setup import setup_pre_torch, setup_post_torch

    params = {k: v for k, v in locals().items() if k != "ctx"}
    for key in ("fast", "base_paths", "extra_model_paths_config", "panic_when",
                "whitelist_custom_nodes", "blacklist_custom_nodes", "workflows",
                "image", "video", "audio"):
        if params.get(key) is None:
            params[key] = []

    if params.get("otel_service_version") is None:
        from .. import __version__
        params["otel_service_version"] = __version__

    config = _build_config(params)

    setup_pre_torch(config)
    _set_config_context(config)
    setup_post_torch(config)

    from .main import _start_comfyui
    try:
        asyncio.run(_start_comfyui(configuration=config))
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

@app.command()
def worker(
    # Worker-specific
    distributed_queue_connection_uri: str = typer.Option(..., "--distributed-queue-connection-uri", help="AMQP URL for distributed queue."),
    distributed_queue_name: str = typer.Option("comfyui", "--distributed-queue-name", help="Queue name."),
    executor_factory: str = typer.Option("ThreadPoolExecutor", "--executor-factory", help="Executor type."),

    # Paths
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
    output_directory: Optional[str] = typer.Option(None, "--output-directory", help="Output directory."),
    temp_directory: Optional[str] = typer.Option(None, "--temp-directory", help="Temp directory."),
    input_directory: Optional[str] = typer.Option(None, "--input-directory", help="Input directory."),

    # GPU/VRAM
    cuda_device: Optional[int] = typer.Option(None, "--cuda-device", help="CUDA device ID."),
    default_device: Optional[int] = typer.Option(None, "--default-device", help="Default device ID."),
    gpu_only: bool = typer.Option(False, "--gpu-only", help="GPU only mode."),
    highvram: bool = typer.Option(False, "--highvram", help="Keep models in GPU memory."),
    normalvram: bool = typer.Option(False, "--normalvram", help="Normal VRAM mode."),
    lowvram: bool = typer.Option(False, "--lowvram", help="Low VRAM mode."),
    novram: bool = typer.Option(False, "--novram", help="Minimal VRAM."),
    cpu: bool = typer.Option(False, "--cpu", help="CPU only."),

    # Precision
    force_fp32: bool = typer.Option(False, "--force-fp32", help="Force fp32."),
    force_fp16: bool = typer.Option(False, "--force-fp16", help="Force fp16."),
    force_bf16: bool = typer.Option(False, "--force-bf16", help="Force bf16."),
    fp16_unet: bool = typer.Option(False, "--fp16-unet", help="fp16 unet."),
    bf16_unet: bool = typer.Option(False, "--bf16-unet", help="bf16 unet."),
    fp8_e4m3fn_unet: bool = typer.Option(False, "--fp8_e4m3fn-unet", help="fp8 unet."),

    # Memory
    reserve_vram: float = typer.Option(0, "--reserve-vram", help="VRAM to reserve in GB."),
    disable_smart_memory: bool = typer.Option(False, "--disable-smart-memory", help="Disable smart memory."),
    disable_pinned_memory: bool = typer.Option(False, "--disable-pinned-memory", help="Disable pinned memory."),

    # Performance
    fast: Optional[list[str]] = typer.Option(None, "--fast", help="Performance optimizations."),
    deterministic: bool = typer.Option(False, "--deterministic", help="Deterministic mode."),

    # Attention
    use_pytorch_cross_attention: bool = typer.Option(False, "--use-pytorch-cross-attention", help="PyTorch cross attention."),
    use_sage_attention: bool = typer.Option(False, "--use-sage-attention", help="Sage attention."),
    use_flash_attention: bool = typer.Option(False, "--use-flash-attention", help="Flash attention."),
    disable_xformers: bool = typer.Option(False, "--disable-xformers", help="Disable xformers."),

    # Custom nodes
    disable_all_custom_nodes: bool = typer.Option(False, "--disable-all-custom-nodes", help="Disable custom nodes."),
    blacklist_custom_nodes: Optional[list[str]] = typer.Option(None, "--blacklist-custom-nodes", help="Blacklist custom nodes."),

    # Misc
    logging_level: str = typer.Option("INFO", "--logging-level", help="Log level."),
    guess_settings: bool = typer.Option(False, "--guess-settings", help="Auto-detect settings."),
    block_runtime_package_installation: bool = typer.Option(True, "--block-runtime-package-installation", help="Block runtime installs (default True for workers)."),
    disable_known_models: bool = typer.Option(False, "--disable-known-models", help="Disable automatic model downloads."),

    # Tracing
    otel_service_name: str = typer.Option("comfyui", "--otel-service-name", envvar="OTEL_SERVICE_NAME", help="OTel service name."),
    otel_service_version: Optional[str] = typer.Option(None, "--otel-service-version", envvar="OTEL_SERVICE_VERSION", help="OTel service version."),
    otel_exporter_otlp_endpoint: Optional[str] = typer.Option(None, "--otel-exporter-otlp-endpoint", envvar="OTEL_EXPORTER_OTLP_ENDPOINT", help="OTLP endpoint."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Database URL."),
    panic_when: Optional[list[str]] = typer.Option(None, "--panic-when", help="Panic on exceptions."),
):
    """Run as a distributed queue worker."""
    from ..component_model.setup import setup_pre_torch, setup_post_torch

    params = {k: v for k, v in locals().items() if k != "ctx"}
    for key in ("fast", "base_paths", "extra_model_paths_config", "blacklist_custom_nodes", "panic_when"):
        if params.get(key) is None:
            params[key] = []

    if params.get("otel_service_version") is None:
        from .. import __version__
        params["otel_service_version"] = __version__

    config = _build_config(params)
    config.distributed_queue_worker = True
    config.distributed_queue_frontend = False

    setup_pre_torch(config)
    _set_config_context(config)
    setup_post_torch(config)

    from ..entrypoints.worker import run_worker
    try:
        asyncio.run(run_worker(config))
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# post-workflow
# ---------------------------------------------------------------------------

@app.command(name="post-workflow")
def post_workflow(
    workflows: list[str] = typer.Argument(..., help="Workflow files, URIs, '-' for stdin, or literal JSON."),

    prompt: Optional[str] = typer.Option(None, "--prompt", help="Override positive prompt."),
    negative_prompt: Optional[str] = typer.Option(None, "--negative-prompt", help="Override negative prompt."),
    steps: Optional[int] = typer.Option(None, "--steps", help="Override sampling steps."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Override seed."),
    image: Optional[list[str]] = typer.Option(None, "--image", help="Override image inputs."),
    video: Optional[list[str]] = typer.Option(None, "--video", help="Override video inputs."),
    audio: Optional[list[str]] = typer.Option(None, "--audio", help="Override audio inputs."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Override output directory."),

    # Minimal shared options for headless execution
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
    output_directory: Optional[str] = typer.Option(None, "--output-directory", help="Output directory."),
    input_directory: Optional[str] = typer.Option(None, "--input-directory", help="Input directory."),
    temp_directory: Optional[str] = typer.Option(None, "--temp-directory", help="Temp directory."),

    # GPU/VRAM
    cuda_device: Optional[int] = typer.Option(None, "--cuda-device", help="CUDA device ID."),
    gpu_only: bool = typer.Option(False, "--gpu-only", help="GPU only."),
    highvram: bool = typer.Option(False, "--highvram", help="High VRAM."),
    novram: bool = typer.Option(False, "--novram", help="Minimal VRAM."),
    cpu: bool = typer.Option(False, "--cpu", help="CPU only."),
    force_fp16: bool = typer.Option(False, "--force-fp16", help="Force fp16."),
    fast: Optional[list[str]] = typer.Option(None, "--fast", help="Performance optimizations."),
    guess_settings: bool = typer.Option(False, "--guess-settings", help="Auto-detect settings."),
    logging_level: str = typer.Option("INFO", "--logging-level", help="Log level."),
    disable_all_custom_nodes: bool = typer.Option(False, "--disable-all-custom-nodes", help="Disable custom nodes."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Database URL."),
):
    """Execute workflow(s) and exit."""
    from ..component_model.setup import setup_pre_torch, setup_post_torch

    params = {k: v for k, v in locals().items() if k != "ctx"}
    for key in ("fast", "base_paths", "extra_model_paths_config", "image", "video", "audio"):
        if params.get(key) is None:
            params[key] = []

    if params.get("output") is not None:
        params["output_directory"] = params["output"]

    config = _build_config(params)

    setup_pre_torch(config)
    _set_config_context(config)
    setup_post_torch(config)

    from ..component_model.entrypoints_common import configure_application_paths
    configure_application_paths(config)

    from ..entrypoints.workflow import run_workflows
    try:
        asyncio.run(run_workflows(config.workflows, configuration=config))
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# create-directories
# ---------------------------------------------------------------------------

@app.command(name="create-directories")
def create_directories_cmd(
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    output_directory: Optional[str] = typer.Option(None, "--output-directory", help="Output directory."),
    input_directory: Optional[str] = typer.Option(None, "--input-directory", help="Input directory."),
    temp_directory: Optional[str] = typer.Option(None, "--temp-directory", help="Temp directory."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
    logging_level: str = typer.Option("INFO", "--logging-level", help="Log level."),
):
    """Create default model/input/output/temp directories and exit."""
    from ..component_model.setup import setup_pre_torch

    params = {k: v for k, v in locals().items() if k != "ctx"}
    for key in ("base_paths", "extra_model_paths_config"):
        if params.get(key) is None:
            params[key] = []

    config = _build_config(params)

    setup_pre_torch(config)
    _set_config_context(config)

    from ..execution_context import context_configuration
    from .folder_paths import create_directories
    from ..nodes.package import import_all_nodes_in_workspace
    with context_configuration(config):
        import_all_nodes_in_workspace(raise_on_failure=False)
        create_directories()


# ---------------------------------------------------------------------------
# list-workflow-templates
# ---------------------------------------------------------------------------

@app.command(name="list-workflow-templates")
def list_workflow_templates(
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    template_dir: Optional[list[str]] = typer.Option(None, "--template-dir", help="Extra directories to scan."),
    convert_to_api: bool = typer.Option(False, "--convert-to-api", help="Convert UI workflows to API format (boots node system)."),
):
    """List available workflow templates."""
    from .workflow_templates import list_templates
    list_templates(
        format=format,
        extra_dirs=template_dir or [],
        convert=convert_to_api,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

_KNOWN_COMMANDS = frozenset({
    "serve", "worker", "post-workflow", "list-workflow-templates",
    "create-directories",
})


def entrypoint():
    """Main CLI entrypoint. Defaults to 'serve' when no subcommand is given."""
    if len(sys.argv) <= 1:
        sys.argv.insert(1, "serve")
    elif sys.argv[1] not in _KNOWN_COMMANDS and sys.argv[1] not in ("--help", "-h"):
        sys.argv.insert(1, "serve")

    app()
