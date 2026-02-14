"""
Typer CLI application for ComfyUI.

Single entry point for all commands: serve, worker, post-workflow,
create-directories, list-workflow-templates.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click
import typer

from ..cli_args_types import (
    Configuration, LatentPreviewMethod, PerformanceFeature,
    VRAM_MODES, PRECISION_MODES, UNET_MODES, VAE_MODES, TEXT_ENC_MODES,
    ATTENTION_MODES, CACHE_MODES, UPCAST_MODES,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="comfyui",
    no_args_is_help=False,
    add_completion=False,
)

_COMFYUI_ENV = {"auto_envvar_prefix": "COMFYUI"}


DEFAULT_VERSION_STRING = "comfyanonymous/ComfyUI@latest"



_DIRECTORY_OPTS: list[tuple] = [
    ("cwd", Optional[str], typer.Option(None, "-w", "--cwd", help="Specify the working directory. If not set, this is the current working directory. models/, input/, output/ and other directories will be located here by default.")),
    ("base_paths", Optional[list[str]], typer.Option(None, "--base-paths", help="Additional base paths for custom nodes, models and inputs.")),
    ("base_directory", Optional[str], typer.Option(None, "--base-directory", help="Set the ComfyUI base directory for models, custom_nodes, input, output, temp, and user directories.")),
    ("extra_model_paths_config", Optional[list[str]], typer.Option(None, "--extra-model-paths-config", help="Load one or more extra_model_paths.yaml files.")),
    ("output_directory", Optional[str], typer.Option(None, "--output-directory", help="Set the ComfyUI output directory.")),
    ("temp_directory", Optional[str], typer.Option(None, "--temp-directory", help="Set the ComfyUI temp directory.")),
    ("input_directory", Optional[str], typer.Option(None, "--input-directory", help="Set the ComfyUI input directory.")),
    ("user_directory", Optional[str], typer.Option(None, "--user-directory", help="Set the ComfyUI user directory with an absolute path.")),
]

_DEVICE_OPTS: list[tuple] = [
    ("cuda_device", Optional[int], typer.Option(None, "--cuda-device", help="Set the id of the cuda device this instance will use.")),
    ("default_device", Optional[int], typer.Option(None, "--default-device", help="Set the id of the default device, all other devices will stay visible.")),
    ("cuda_malloc", bool, typer.Option(False, "--cuda-malloc", help="Enable cudaMallocAsync.")),
    ("disable_cuda_malloc", bool, typer.Option(True, "--disable-cuda-malloc", help="Disable cudaMallocAsync.")),
    ("directml", Optional[int], typer.Option(None, "--directml", help="Use torch-directml. -1 for auto-selection.")),
    ("oneapi_device_selector", Optional[str], typer.Option(None, "--oneapi-device-selector", help="Sets the oneAPI device(s) this instance will use.")),
    ("disable_ipex_optimize", bool, typer.Option(False, "--disable-ipex-optimize", help="Disable IPEX optimization for Intel GPUs.")),
    ("supports_fp8_compute", bool, typer.Option(False, "--supports-fp8-compute", help="ComfyUI will act like if the device supports fp8 compute.")),
]

_VRAM_OPTS: list[tuple] = [
    ("gpu_only", bool, typer.Option(False, "--gpu-only", help="Store and run everything on the GPU.")),
    ("highvram", bool, typer.Option(False, "--highvram", help="Keep models in GPU memory.")),
    ("normalvram", bool, typer.Option(False, "--normalvram", help="Default VRAM usage setting.")),
    ("lowvram", bool, typer.Option(False, "--lowvram", help="Reduce UNet's VRAM usage.")),
    ("novram", bool, typer.Option(False, "--novram", help="Minimize VRAM usage.")),
    ("cpu", bool, typer.Option(False, "--cpu", help="Use CPU for processing.")),
    ("reserve_vram", float, typer.Option(0, "--reserve-vram", help="Set the amount of vram in GB you want to reserve for use by your OS/other software.")),
]

_PRECISION_OPTS: list[tuple] = [

    ("force_fp32", bool, typer.Option(False, "--force-fp32", help="Force using FP32 precision.")),
    ("force_fp16", bool, typer.Option(False, "--force-fp16", help="Force using FP16 precision.")),
    ("force_bf16", bool, typer.Option(False, "--force-bf16", help="Force using BF16 precision.")),

    ("fp32_unet", bool, typer.Option(False, "--fp32-unet", help="Run the diffusion model in fp32.")),
    ("fp64_unet", bool, typer.Option(False, "--fp64-unet", help="Run the diffusion model in fp64.")),
    ("bf16_unet", bool, typer.Option(False, "--bf16-unet", help="Run the diffusion model in bf16.")),
    ("fp16_unet", bool, typer.Option(False, "--fp16-unet", help="Run the diffusion model in fp16.")),
    ("fp8_e4m3fn_unet", bool, typer.Option(False, "--fp8_e4m3fn-unet", help="Store unet weights in fp8_e4m3fn.")),
    ("fp8_e5m2_unet", bool, typer.Option(False, "--fp8_e5m2-unet", help="Store unet weights in fp8_e5m2.")),
    ("fp8_e8m0fnu_unet", bool, typer.Option(False, "--fp8_e8m0fnu-unet", help="Store unet weights in fp8_e8m0fnu.")),

    ("fp16_vae", bool, typer.Option(False, "--fp16-vae", help="Run the VAE in FP16 precision.")),
    ("fp32_vae", bool, typer.Option(False, "--fp32-vae", help="Run the VAE in full precision fp32.")),
    ("bf16_vae", bool, typer.Option(False, "--bf16-vae", help="Run the VAE in BF16 precision.")),
    ("cpu_vae", bool, typer.Option(False, "--cpu-vae", help="Run the VAE on the CPU.")),

    ("fp8_e4m3fn_text_enc", bool, typer.Option(False, "--fp8_e4m3fn-text-enc", help="Store text encoder weights in fp8 (e4m3fn).")),
    ("fp8_e5m2_text_enc", bool, typer.Option(False, "--fp8_e5m2-text-enc", help="Store text encoder weights in fp8 (e5m2).")),
    ("fp16_text_enc", bool, typer.Option(False, "--fp16-text-enc", help="Store text encoder weights in fp16.")),
    ("fp32_text_enc", bool, typer.Option(False, "--fp32-text-enc", help="Store text encoder weights in fp32.")),
    ("bf16_text_enc", bool, typer.Option(False, "--bf16-text-enc", help="Store text encoder weights in bf16.")),
]

_ATTENTION_OPTS: list[tuple] = [
    ("use_split_cross_attention", bool, typer.Option(False, "--use-split-cross-attention", help="Use split cross-attention optimization.")),
    ("use_quad_cross_attention", bool, typer.Option(False, "--use-quad-cross-attention", help="Use sub-quadratic cross-attention optimization.")),
    ("use_pytorch_cross_attention", bool, typer.Option(False, "--use-pytorch-cross-attention", help="Use PyTorch's cross-attention function.")),
    ("use_sage_attention", bool, typer.Option(False, "--use-sage-attention", help="Use sage attention.")),
    ("use_flash_attention", bool, typer.Option(False, "--use-flash-attention", help="Use FlashAttention.")),
    ("disable_xformers", bool, typer.Option(False, "--disable-xformers", help="Disable xformers.")),
    ("force_upcast_attention", bool, typer.Option(False, "--force-upcast-attention", help="Force upcasting of attention.")),
    ("dont_upcast_attention", bool, typer.Option(False, "--dont-upcast-attention", help="Disable upcasting of attention.")),
]

_MEMORY_OPTS: list[tuple] = [
    ("async_offload", Optional[int], typer.Option(None, "--async-offload", help="Use async weight offloading. An optional argument controls the amount of offload streams.")),
    ("disable_async_offload", bool, typer.Option(False, "--disable-async-offload", help="Disable async weight offloading.")),
    ("force_non_blocking", bool, typer.Option(False, "--force-non-blocking", help="Force non-blocking operations for all applicable tensors.")),
    ("disable_smart_memory", bool, typer.Option(False, "--disable-smart-memory", help="Disable smart memory management.")),
    ("disable_pinned_memory", bool, typer.Option(False, "--disable-pinned-memory", help="Disable pinned memory use.")),
]

_CACHE_OPTS: list[tuple] = [
    ("cache_classic", bool, typer.Option(False, "--cache-classic", help="Use the old style (aggressive) caching.")),
    ("cache_lru", int, typer.Option(0, "--cache-lru", help="Use LRU caching with a maximum of N node results cached. May use more RAM/VRAM.")),
    ("cache_none", bool, typer.Option(False, "--cache-none", help="Reduced RAM/VRAM usage at the expense of executing every node for each run.")),
    ("cache_ram", float, typer.Option(0, "--cache-ram", help="Use RAM pressure caching with the specified headroom threshold in GB.")),
]

_PREVIEW_OPTS: list[tuple] = [
    ("preview_method", str, typer.Option("auto", "--preview-method", click_type=click.Choice(["none", "auto", "latent2rgb", "taesd"]), help="Method for generating previews.")),
    ("preview_size", int, typer.Option(512, "--preview-size", help="Sets the maximum preview size for sampler nodes.")),
]

_PERF_OPTS: list[tuple] = [
    ("fast", Optional[list[str]], typer.Option(None, "--fast", help="Enable some untested and potentially quality deteriorating optimizations. Valid optimizations: fp16_accumulation, fp8_matrix_mult, cublas_ops, autotune, dynamic_vram.")),
    ("deterministic", bool, typer.Option(False, "--deterministic", help="Use deterministic algorithms where possible.")),
    ("default_hashing_function", str, typer.Option("sha256", "--default-hashing-function", click_type=click.Choice(["md5", "sha1", "sha256", "sha512"]), help="Hash function for duplicate filename / contents comparison.")),
    ("force_channels_last", bool, typer.Option(False, "--force-channels-last", help="Force channels last format when inferencing the models.")),
    ("force_hf_local_dir_mode", bool, typer.Option(False, "--force-hf-local-dir-mode", help="Download HF repos with local_dir instead of cache_dir.")),
    ("mmap_torch_files", bool, typer.Option(False, "--mmap-torch-files", help="Use mmap when loading ckpt/pt files.")),
    ("disable_mmap", bool, typer.Option(False, "--disable-mmap", help="Don't use mmap when loading safetensors.")),
]

_NODE_OPTS: list[tuple] = [
    ("disable_metadata", bool, typer.Option(False, "--disable-metadata", help="Disable saving metadata with outputs.")),
    ("disable_all_custom_nodes", bool, typer.Option(False, "--disable-all-custom-nodes", help="Disable loading all custom nodes.")),
    ("whitelist_custom_nodes", Optional[list[str]], typer.Option(None, "--whitelist-custom-nodes", help="Specify custom node folders to load even when --disable-all-custom-nodes is enabled.")),
    ("blacklist_custom_nodes", Optional[list[str]], typer.Option(None, "--blacklist-custom-nodes", help="Specify custom node folders to never load. Accepts shell-style globs.")),
    ("disable_api_nodes", bool, typer.Option(False, "--disable-api-nodes", help="Disable loading all api nodes.")),
    ("enable_eval", bool, typer.Option(False, "--enable-eval", help="Enable nodes that can evaluate Python code in workflows.")),
    ("enable_video_to_image_fallback", bool, typer.Option(False, "--enable-video-to-image-fallback", help="Enable video-to-image fallback.")),
    ("disable_known_models", bool, typer.Option(False, "--disable-known-models", help="Disables automatic downloads of known models.")),
    ("disable_assets_autoscan", bool, typer.Option(False, "--disable-assets-autoscan", help="Disable asset scanning on startup for database synchronization.")),
]

_TELEMETRY_OPTS: list[tuple] = [
    ("otel_service_name", str, typer.Option("comfyui", "--otel-service-name", envvar="OTEL_SERVICE_NAME", help="The name of the service or application that is generating telemetry data.")),
    ("otel_service_version", Optional[str], typer.Option(None, "--otel-service-version", envvar="OTEL_SERVICE_VERSION", help="The version of the service or application that is generating telemetry data.")),
    ("otel_exporter_otlp_endpoint", Optional[str], typer.Option(None, "--otel-exporter-otlp-endpoint", envvar="OTEL_EXPORTER_OTLP_ENDPOINT", help="A base endpoint URL for any signal type, with an optionally-specified port number.")),
]

_LOGGING_OPTS: list[tuple] = [
    ("logging_level", str, typer.Option("INFO", "--logging-level", click_type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]), help="Specifies the logging level.")),
]

_MISC_OPTS: list[tuple] = [
    ("guess_settings", bool, typer.Option(False, "--guess-settings", help="Auto-detect best settings for this machine (GPU type, RAM, attention backend, etc.).")),
    ("database_url", Optional[str], typer.Option(None, "--database-url", help="Specify the database URL, e.g. 'sqlite:///:memory:'.")),
    ("panic_when", Optional[list[str]], typer.Option(None, "--panic-when", help="List of fully qualified exception class names to panic (sys.exit(1)) when a workflow raises it.")),
    ("executor_factory", str, typer.Option("ThreadPoolExecutor", "--executor-factory", help="Either ThreadPoolExecutor or ProcessPoolExecutor.")),
]

_WORKFLOW_OVERRIDE_OPTS: list[tuple] = [
    ("prompt", Optional[str], typer.Option(None, "--prompt", help="Override the positive prompt text in workflows.")),
    ("negative_prompt", Optional[str], typer.Option(None, "--negative-prompt", help="Override the negative prompt text in workflows.")),
    ("steps", Optional[int], typer.Option(None, "--steps", help="Override the number of sampling steps in workflows.")),
    ("seed", Optional[int], typer.Option(None, "--seed", help="Override the seed in sampler and noise nodes in workflows.")),
    ("image", Optional[list[str]], typer.Option(None, "--image", help="Override image inputs in workflows. Accepts file paths or URIs.")),
    ("video", Optional[list[str]], typer.Option(None, "--video", help="Override video inputs in workflows. Accepts file paths or URIs.")),
    ("audio", Optional[list[str]], typer.Option(None, "--audio", help="Override audio inputs in workflows. Accepts file paths or URIs.")),
    ("output", Optional[str], typer.Option(None, "-o", "--output", help="Override the output directory for workflows.")),
]

_COMPUTE_OPTS = (
    _DEVICE_OPTS + _VRAM_OPTS + _PRECISION_OPTS + _ATTENTION_OPTS +
    _MEMORY_OPTS + _CACHE_OPTS + _PREVIEW_OPTS + _PERF_OPTS
)

_ALL_SHARED_OPTS = (
    _DIRECTORY_OPTS + _COMPUTE_OPTS + _NODE_OPTS +
    _TELEMETRY_OPTS + _LOGGING_OPTS + _MISC_OPTS
)

_NULLABLE_LIST_FIELDS = frozenset({
    "fast", "base_paths", "extra_model_paths_config", "panic_when",
    "whitelist_custom_nodes", "blacklist_custom_nodes", "workflows",
    "image", "video", "audio",
})


def _with_options(*option_groups):
    """Add shared option groups to a Typer command function.

    The decorated function should accept **kwargs for the injected options.
    Options are appended to the function's signature so Typer discovers them.
    """
    combined = []
    for group in option_groups:
        combined.extend(group)

    def decorator(func):
        sig = inspect.signature(func)
        params = [p for p in sig.parameters.values()
                  if p.kind != inspect.Parameter.VAR_KEYWORD]
        for name, annotation, default in combined:
            params.append(inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default, annotation=annotation,
            ))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.__signature__ = sig.replace(parameters=params)
        return wrapper
    return decorator


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
    _validate_mutex(params, "VRAM mode", VRAM_MODES)
    _validate_mutex(params, "precision", PRECISION_MODES)
    _validate_mutex(params, "UNet precision", UNET_MODES)
    _validate_mutex(params, "VAE precision", VAE_MODES)
    _validate_mutex(params, "text encoder precision", TEXT_ENC_MODES)
    _validate_mutex(params, "attention", ATTENTION_MODES)
    _validate_mutex(params, "upcast attention", UPCAST_MODES)

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


def _parse_plugin_args(ctx: typer.Context, config: Configuration):
    """Load comfyui.custom_config entry points and parse their args from ctx.args."""
    from importlib.metadata import entry_points
    import configargparse

    parser = configargparse.ArgParser(add_help=False)
    for ep in entry_points(group='comfyui.custom_config'):
        plugin = ep.load()
        result = plugin(parser)
        if result is not None:
            parser = result

    if ctx.args:
        plugin_args, _ = parser.parse_known_args(ctx.args)
        for k, v in vars(plugin_args).items():
            setattr(config, k, v)


def _collect_params(local_vars: dict, kwargs: dict) -> dict:
    """Merge named locals and **kwargs into a single params dict for _build_config."""
    params = {k: v for k, v in local_vars.items() if k not in ("ctx", "kwargs")}
    params.update(kwargs)
    for key in _NULLABLE_LIST_FIELDS:
        if key in params and params[key] is None:
            params[key] = []
    return params


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """ComfyUI - The most powerful and modular diffusion model GUI and backend."""
    if ctx.invoked_subcommand is None:
        # No subcommand specified - this shouldn't happen because
        # entrypoint() inserts "serve" when no subcommand is given.
        # But as a safety net, invoke serve.
        ctx.invoke(serve)


@app.command(context_settings={**_COMFYUI_ENV, "allow_extra_args": True, "ignore_unknown_options": True})
@_with_options(_ALL_SHARED_OPTS, _WORKFLOW_OVERRIDE_OPTS)
def serve(
    ctx: typer.Context,

    listen: str = typer.Option("127.0.0.1", "-H", "--listen", help="Specify the IP address to listen on (default: 127.0.0.1). You can give a list of ip addresses by separating them with a comma like: 127.2.2.2,127.3.3.3 If --listen is provided without an argument, it defaults to 0.0.0.0,:: (listens on all ipv4 and ipv6)"),
    port: int = typer.Option(8188, help="Set the listen port."),
    enable_cors_header: Optional[str] = typer.Option(None, "--enable-cors-header", help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'."),
    max_upload_size: float = typer.Option(100.0, "--max-upload-size", help="Set the maximum upload size in MB."),
    auto_launch: bool = typer.Option(False, "--auto-launch", help="Automatically launch ComfyUI in the default browser."),
    disable_auto_launch: bool = typer.Option(False, "--disable-auto-launch", help="Disable auto launching the browser."),
    external_address: Optional[str] = typer.Option(None, "--external-address", help="Specifies a base URL for external addresses reported by the API, such as for image paths."),
    multi_user: bool = typer.Option(False, "--multi-user", help="Enable multi-user mode with per-user storage."),
    enable_compress_response_body: bool = typer.Option(False, "--enable-compress-response-body", help="Enable compressing response body."),
    enable_manager: bool = typer.Option(False, "--enable-manager", help="Enable the ComfyUI-Manager feature."),
    disable_manager_ui: bool = typer.Option(False, "--disable-manager-ui", help="Disables only the ComfyUI-Manager UI."),
    enable_manager_legacy_ui: bool = typer.Option(False, "--enable-manager-legacy-ui", help="Enables the legacy UI of ComfyUI-Manager."),
    dont_print_server: bool = typer.Option(False, "--dont-print-server", help="Don't print server output."),
    log_stdout: bool = typer.Option(False, "--log-stdout", help="Send normal process output to stdout instead of stderr (default)."),
    quick_test_for_ci: bool = typer.Option(False, "--quick-test-for-ci", help="Enable quick testing mode for CI."),
    windows_standalone_build: bool = typer.Option(False, "--windows-standalone-build", help="Enable features for standalone Windows build."),
    create_directories: bool = typer.Option(False, "--create-directories", help="Creates the default models/, input/, output/ and temp/ directories, then exits."),
    plausible_analytics_base_url: Optional[str] = typer.Option(None, "--plausible-analytics-base-url", help="Base URL for server-side analytics."),
    plausible_analytics_domain: Optional[str] = typer.Option(None, "--plausible-analytics-domain", help="Domain for analytics events."),
    analytics_use_identity_provider: bool = typer.Option(False, "--analytics-use-identity-provider", help="Use platform identifiers for analytics."),
    distributed_queue_connection_uri: Optional[str] = typer.Option(None, "--distributed-queue-connection-uri", help="Servers and clients will connect to this AMQP URL to form a distributed queue and exchange prompt execution requests and progress updates."),
    distributed_queue_worker: bool = typer.Option(False, "--distributed-queue-worker", help="Workers will pull requests off the AMQP URL."),
    distributed_queue_frontend: bool = typer.Option(False, "--distributed-queue-frontend", help="Frontends will start the web UI and connect to the provided AMQP URL to submit prompts."),
    distributed_queue_name: str = typer.Option("comfyui", "--distributed-queue-name", help="This name will be used by the frontends and workers to exchange prompt requests and replies."),
    max_queue_size: int = typer.Option(65536, "--max-queue-size", help="The API will reject prompt requests if the queue's size exceeds this value."),
    front_end_version: str = typer.Option(DEFAULT_VERSION_STRING, "--front-end-version", help="Specifies the version of the frontend to be used. Format: [owner]/[repo]@[version]."),
    front_end_root: Optional[str] = typer.Option(None, "--front-end-root", help="The local filesystem path to the directory where the frontend is located. Overrides --front-end-version."),
    openai_api_key: Optional[str] = typer.Option(None, "--openai-api-key", envvar="OPENAI_API_KEY", help="Configures the OpenAI API Key for the OpenAI nodes."),
    ideogram_api_key: Optional[str] = typer.Option(None, "--ideogram-api-key", envvar="IDEOGRAM_API_KEY", help="Configures the Ideogram API Key for the Ideogram nodes."),
    anthropic_api_key: Optional[str] = typer.Option(None, "--anthropic-api-key", envvar="ANTHROPIC_API_KEY", help="Configures the Anthropic API key for its nodes related to Claude functionality."),
    google_api_key: Optional[str] = typer.Option(None, "--google-api-key", envvar="GOOGLE_API_KEY", help="Google API key for Gemini models."),
    comfy_api_base: str = typer.Option("https://api.comfy.org", "--comfy-api-base", help="Set the base URL for the ComfyUI API."),
    block_runtime_package_installation: bool = typer.Option(False, "--block-runtime-package-installation", help="When set, custom nodes like ComfyUI Manager, Easy Use, Nunchaku and others will not be able to use pip or uv to install packages at runtime (experimental)."),
    workflows: Optional[list[str]] = typer.Option(None, "--workflows", help="Execute the API workflow(s) and exit. Each value can be a file path, a literal JSON string starting with '{', a URI (https://, s3://, hf://, etc.), or '-' for stdin."),
    disable_requests_caching: bool = typer.Option(False, "--disable-requests-caching", help="Disable requests caching."),
    disable_manager_model_fallback: bool = typer.Option(False, "--disable-manager-model-fallback", help="Disable manager model database fallback."),
    refresh_manager_models: bool = typer.Option(False, "--refresh-manager-models", help="Fetch latest model list from GitHub."),
    **kwargs,
):
    """Start the ComfyUI server (default command)."""
    from ..component_model.setup import setup_pre_torch, setup_post_torch

    params = _collect_params(locals(), kwargs)

    if params.get("otel_service_version") is None:
        from .. import __version__
        params["otel_service_version"] = __version__

    config = _build_config(params)
    _parse_plugin_args(ctx, config)

    setup_pre_torch(config)
    _set_config_context(config)
    setup_post_torch(config)

    from .main import _start_comfyui
    try:
        asyncio.run(_start_comfyui(configuration=config))
    except KeyboardInterrupt:
        pass



@app.command(context_settings={**_COMFYUI_ENV, "allow_extra_args": True, "ignore_unknown_options": True})
@_with_options(_ALL_SHARED_OPTS)
def worker(
    ctx: typer.Context,
    distributed_queue_connection_uri: str = typer.Option(..., "--distributed-queue-connection-uri", help="AMQP URL for distributed queue."),
    distributed_queue_name: str = typer.Option("comfyui", "--distributed-queue-name", help="Queue name."),
    block_runtime_package_installation: bool = typer.Option(True, "--block-runtime-package-installation", help="Block runtime installs (default True for workers)."),
    **kwargs,
):
    """Run as a distributed queue worker."""
    from ..component_model.setup import setup_pre_torch, setup_post_torch

    params = _collect_params(locals(), kwargs)

    if params.get("otel_service_version") is None:
        from .. import __version__
        params["otel_service_version"] = __version__

    config = _build_config(params)
    config.distributed_queue_worker = True
    config.distributed_queue_frontend = False
    _parse_plugin_args(ctx, config)

    setup_pre_torch(config)
    _set_config_context(config)
    setup_post_torch(config)

    from ..entrypoints.worker import run_worker
    try:
        asyncio.run(run_worker(config))
    except KeyboardInterrupt:
        pass



@app.command(name="post-workflow", context_settings=_COMFYUI_ENV)
@_with_options(_ALL_SHARED_OPTS, _WORKFLOW_OVERRIDE_OPTS)
def post_workflow(
    workflows: list[str] = typer.Argument(..., help="Workflow files, URIs, '-' for stdin, or literal JSON."),
    disable_progress: bool = typer.Option(False, "--disable-progress", help="Disable CLI progress bars."),
    block_runtime_package_installation: bool = typer.Option(False, "--block-runtime-package-installation", help="Block runtime package installations."),
    **kwargs,
):
    """Execute workflow(s) and exit."""
    from ..component_model.setup import setup_pre_torch, setup_post_torch

    params = _collect_params(locals(), kwargs)

    if params.get("output") is not None:
        params["output_directory"] = params["output"]

    if params.get("otel_service_version") is None:
        from .. import __version__
        params["otel_service_version"] = __version__

    config = _build_config(params)

    setup_pre_torch(config)
    _set_config_context(config)
    setup_post_torch(config)

    from ..component_model.entrypoints_common import configure_application_paths
    configure_application_paths(config)

    from ..execution_context import context_configuration
    from ..nodes.package import import_all_nodes_in_workspace
    with context_configuration(config):
        import_all_nodes_in_workspace(raise_on_failure=False)

    from ..entrypoints.workflow import run_workflows
    try:
        asyncio.run(run_workflows(config.workflows, configuration=config))
    except KeyboardInterrupt:
        pass



@app.command(name="create-directories", context_settings=_COMFYUI_ENV)
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
    from .folder_paths import create_directories  # pylint: disable=import-error
    from ..nodes.package import import_all_nodes_in_workspace
    with context_configuration(config):
        import_all_nodes_in_workspace(raise_on_failure=False)
        create_directories()



@app.command(name="list-workflow-templates", context_settings=_COMFYUI_ENV)
def list_workflow_templates(
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    template_dir: Optional[list[str]] = typer.Option(None, "--template-dir", help="Extra directories to scan."),
    convert_to_api: bool = typer.Option(False, "--convert-to-api", help="Convert UI workflows to API format (boots node system)."),
    all_templates: bool = typer.Option(False, "-a", "--all", help="Include API-key-requiring templates."),
):
    """List available workflow templates."""
    import sys
    from .workflow_templates import list_templates
    interactive = sys.stdout.isatty() and format == "table"
    list_templates(
        format=format,
        extra_dirs=template_dir or [],
        convert=convert_to_api,
        show_all=all_templates,
        interactive=interactive,
    )


@app.command(name="list-models", context_settings=_COMFYUI_ENV)
def list_models_cmd(
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    folder: Optional[str] = typer.Option(None, "--folder", help="Filter by model folder (checkpoints, loras, vae, etc)."),
    no_manager: bool = typer.Option(False, "--no-manager", help="Exclude comfyui_manager models."),
    check_exists: bool = typer.Option(False, "--check-exists", help="Check if models exist locally (requires path initialization)."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
    logging_level: str = typer.Option("INFO", "--logging-level", click_type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]), help="Logging level."),
):
    """List known downloadable models."""
    if check_exists:
        from ..component_model.setup import setup_pre_torch

        params = {}
        if cwd is not None:
            params["cwd"] = cwd
        if base_directory is not None:
            params["base_directory"] = base_directory
        if base_paths is not None:
            expanded = []
            for v in base_paths:
                for piece in str(v).split(","):
                    piece = piece.strip()
                    if piece:
                        expanded.append(piece)
            params["base_paths"] = expanded
        else:
            params["base_paths"] = []
        if extra_model_paths_config is not None:
            params["extra_model_paths_config"] = list(extra_model_paths_config)
        else:
            params["extra_model_paths_config"] = []
        params["logging_level"] = logging_level

        config = _build_config(params)
        setup_pre_torch(config)
        _set_config_context(config)

        from ..execution_context import context_configuration
        from ..nodes.package import import_all_nodes_in_workspace
        with context_configuration(config):
            import_all_nodes_in_workspace(raise_on_failure=False)

    from .list_models import list_models
    list_models(format=format, folder=folder, include_manager=not no_manager, check_exists=check_exists)


@app.command(name="integrity-check", context_settings=_COMFYUI_ENV)
def integrity_check(
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd"),
    base_directory: Optional[str] = typer.Option(None, "--base-directory"),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config"),
):
    """Print system diagnostics and verify installation integrity."""
    from ..component_model.setup import setup_pre_torch
    params = {k: v for k, v in locals().items() if k != "ctx"}
    params.setdefault("base_paths", [])
    params["extra_model_paths_config"] = params.get("extra_model_paths_config") or []
    config = _build_config(params)
    setup_pre_torch(config)
    _set_config_context(config)
    from .integrity_check import run_integrity_check
    run_integrity_check(config)


_KNOWN_COMMANDS = frozenset({
    "serve", "worker", "post-workflow", "list-workflow-templates",
    "list-models", "create-directories", "integrity-check",
})


def entrypoint():
    """Main CLI entrypoint. Defaults to 'serve' when no subcommand is given."""
    if len(sys.argv) <= 1:
        sys.argv.insert(1, "serve")
    elif sys.argv[1] not in _KNOWN_COMMANDS and sys.argv[1] not in ("--help", "-h"):
        sys.argv.insert(1, "serve")

    app()
