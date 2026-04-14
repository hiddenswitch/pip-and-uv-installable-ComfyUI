"""
Typer CLI application for ComfyUI.

Single entry point for all commands: serve, worker, stop, logs,
and sub-apps: models, workflows, nodes, jobs, env.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

# must be set before any transitive import of requests (e.g. via typer/click)
warnings.filterwarnings("ignore", message=".*doesn't match a supported version")

import click
import typer

from ..cli_args_types import (
    Configuration, LatentPreviewMethod, PerformanceFeature,
    VRAM_MODES, PRECISION_MODES, UNET_MODES, VAE_MODES, TEXT_ENC_MODES,
    ATTENTION_MODES, UPCAST_MODES,
)

logger = logging.getLogger(__name__)


class _ComfyGroup(typer.core.TyperGroup):
    """Custom group that renders compact help for ``-h`` and full help for ``--help``."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if _short_help_requested():
            self._format_short_help(ctx, formatter)
        else:
            super().format_help(ctx, formatter)

    def _format_short_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        from .. import __version__
        formatter.write(f"comfyui {__version__}\n\n")
        formatter.write("usage: comfyui <command> [options]\n\n")

        commands = []
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            commands.append((name, cmd.get_short_help_str(limit=60)))

        if commands:
            max_name = max(len(n) for n, _ in commands)
            for name, short_help in commands:
                formatter.write(f"  {name:<{max_name}}  {short_help}\n")

        formatter.write(f"\nRun 'comfyui <command> --help' for detailed usage.\n")


def _short_help_requested() -> bool:
    """True when the user passed ``-h`` (not ``--help``)."""
    return "-h" in sys.argv and "--help" not in sys.argv


app = typer.Typer(
    name="comfyui",
    no_args_is_help=False,
    add_completion=False,
    cls=_ComfyGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
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
    ("torch_device", Optional[str], typer.Option(None, "--torch-device", help="Set the torch device by name, e.g. cuda:1, cpu, mps. Overrides --cuda-device and --cpu.")),
    ("default_device", Optional[int], typer.Option(None, "--default-device", help="Set the id of the default device, all other devices will stay visible.")),
    ("cuda_malloc", bool, typer.Option(False, "--cuda-malloc", help="Enable cudaMallocAsync.")),
    ("disable_cuda_malloc", bool, typer.Option(True, "--disable-cuda-malloc", help="Disable cudaMallocAsync.")),
    ("directml", Optional[int], typer.Option(None, "--directml", help="Use torch-directml. -1 for auto-selection.")),
    ("oneapi_device_selector", Optional[str], typer.Option(None, "--oneapi-device-selector", help="Sets the oneAPI device(s) this instance will use.")),
    ("disable_ipex_optimize", bool, typer.Option(False, "--disable-ipex-optimize", help="Disable IPEX optimization for Intel GPUs.")),
    ("supports_fp8_compute", bool, typer.Option(False, "--supports-fp8-compute", help="ComfyUI will act like if the device supports fp8 compute.")),
    ("enable_comfy_kitchen_backends", Optional[list[str]], typer.Option(None, "--enable-comfy-kitchen-backends", help="Re-enable comfy_kitchen quantization backends previously disabled by guess-settings or another flag. Comma-separated. Valid: eager, cuda, triton.")),
    ("disable_comfy_kitchen_backends", Optional[list[str]], typer.Option(None, "--disable-comfy-kitchen-backends", help="Disable comfy_kitchen quantization backends. Use this to skip an op backend that crashes on your hardware (e.g. triton fp8e4nv on Ampere). Comma-separated. Valid: eager, cuda, triton.")),
]

_VRAM_OPTS: list[tuple] = [
    ("gpu_only", bool, typer.Option(False, "--gpu-only", help="Store and run everything on the GPU.")),
    ("highvram", bool, typer.Option(False, "--highvram", help="Keep models in GPU memory.")),
    ("normalvram", bool, typer.Option(False, "--normalvram", help="Default VRAM usage setting.")),
    ("lowvram", bool, typer.Option(False, "--lowvram", help="Reduce UNet's VRAM usage.")),
    ("novram", bool, typer.Option(False, "--novram", help="Minimize VRAM usage.")),
    ("cpu", bool, typer.Option(False, "--cpu", help="Use CPU for processing.")),
    ("reserve_vram", float, typer.Option(0, "--reserve-vram", help="Set the amount of vram in GB you want to reserve for use by your OS/other software.")),
    ("disable_dynamic_vram", bool, typer.Option(False, "--disable-dynamic-vram", help="Disable dynamic VRAM and use estimate based model loading.")),
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
    ("fp16_intermediates", bool, typer.Option(False, "--fp16-intermediates", help="Experimental: Use fp16 for intermediate tensors between nodes instead of fp32.")),
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
    ("cache_ram", float, typer.Option(0, "--cache-ram", help="Use RAM pressure caching with the specified headroom threshold. Default (when no value is provided): 25%% of system RAM (min 4GB, max 32GB).")),
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
    ("enable_assets", bool, typer.Option(False, "--enable-assets", help="Enable the assets API and asset management features.")),
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

_PIP_FACADE_OPTS: list[tuple] = [
    ("pip_facade_registry_base_url", str, typer.Option("https://api.comfy.org", "--pip-facade-registry-base-url", help="Base URL for the Comfy registry API used to resolve custom node versions.")),
    ("pip_facade_cache_prefix", Optional[str], typer.Option(None, "--pip-facade-cache-prefix", help="Writable fsspec URI prefix where generated facade wheels are cached, e.g. /var/cache/comfyui, file:///var/cache/comfyui, or s3://bucket/prefix.")),
    ("pip_facade_cache_revision", Optional[int], typer.Option(None, "--pip-facade-cache-revision", help="Override the wheel cache revision. Changing this invalidates all cached wheels.")),
    ("pip_facade_only_known_nodes", bool, typer.Option(False, "--pip-facade-only-known-nodes", help="Only expose nodes covered by the local custom node compatibility registry.")),
    ("pip_facade_snapshot_uri", Optional[str], typer.Option(None, "--pip-facade-snapshot-uri", help="Read facade registry metadata from this fsspec URI instead of querying the live registry API, e.g. file:///data/registry.sqlite.xz, s3://bucket/registry.sqlite.xz, or pkg://comfy.custom_nodes/pip_facade_registry_snapshot.sqlite.xz.")),
]

_PIP_FACADE_SNAPSHOT_OPTS: list[tuple] = [
    ("pip_facade_snapshot_output", str, typer.Argument(..., help="Output snapshot path. Use a .xz suffix or --pip-facade-snapshot-compression=xz for a compressed archive.")),
    ("pip_facade_snapshot_compression", str, typer.Option("auto", "--pip-facade-snapshot-compression", click_type=click.Choice(["auto", "none", "xz"]), help="Snapshot compression mode. 'auto' uses the output suffix.")),
    ("pip_facade_snapshot_overwrite", bool, typer.Option(False, "--pip-facade-snapshot-overwrite", help="Overwrite an existing snapshot output file.")),
]

_MISC_OPTS: list[tuple] = [
    ("guess_settings", bool, typer.Option(False, "-g", "--guess-settings", envvar="COMFYUI_GUESS_SETTINGS", help="Auto-detect best settings for this machine (GPU type, RAM, attention backend, etc.). Set env var COMFYUI_GUESS_SETTINGS=1 to enable by default.")),
    ("database_url", Optional[str], typer.Option(None, "--database-url", help="Specify the database URL, e.g. 'sqlite:///:memory:'.")),
    ("panic_when", Optional[list[str]], typer.Option(None, "--panic-when", help="List of fully qualified exception class names to panic (sys.exit(1)) when a workflow raises it.")),
    ("executor_factory", str, typer.Option("ThreadPoolExecutor", "--executor-factory", help="Either ThreadPoolExecutor or ProcessPoolExecutor.")),
]

_WORKFLOW_OVERRIDE_OPTS: list[tuple] = [
    ("prompt", Optional[str], typer.Option(None, "--prompt", help="Override the positive prompt text in workflows.")),
    ("negative_prompt", Optional[str], typer.Option(None, "--negative-prompt", help="Override the negative prompt text in workflows.")),
    ("steps", Optional[int], typer.Option(None, "--steps", help="Override the number of sampling steps in workflows.")),
    ("seed", Optional[int], typer.Option(None, "--seed", help="Override the seed in sampler and noise nodes in workflows.")),
    ("cfg", Optional[float], typer.Option(None, "--cfg", help="Override the CFG scale in sampler nodes.")),
    ("sampler", Optional[str], typer.Option(None, "--sampler", help="Override the sampler name in sampler nodes.")),
    ("scheduler", Optional[str], typer.Option(None, "--scheduler", help="Override the scheduler in sampler/scheduler nodes.")),
    ("denoise", Optional[float], typer.Option(None, "--denoise", help="Override the denoise strength in sampler nodes.")),
    ("width", Optional[int], typer.Option(None, "--width", help="Override the width in latent image nodes.")),
    ("height", Optional[int], typer.Option(None, "--height", help="Override the height in latent image nodes.")),
    ("batch_size", Optional[int], typer.Option(None, "--batch-size", help="Override the batch size in latent image nodes.")),
    ("checkpoint", Optional[str], typer.Option(None, "--checkpoint", help="Override the checkpoint model name.")),
    ("image", Optional[list[str]], typer.Option(None, "--image", help="Override image inputs in workflows. Accepts file paths or URIs.")),
    ("video", Optional[list[str]], typer.Option(None, "--video", help="Override video inputs in workflows. Accepts file paths or URIs.")),
    ("audio", Optional[list[str]], typer.Option(None, "--audio", help="Override audio inputs in workflows. Accepts file paths or URIs.")),
    ("output", Optional[str], typer.Option(None, "-o", "--output", help="Override the output directory for workflows.")),
    ("set", Optional[list[str]], typer.Option(None, "--set", help="Override arbitrary node inputs. Format: node_id.inputs.field=value")),
    ("add_lora", Optional[list[str]], typer.Option(None, "--add-lora", help="Inject a LoRA into the workflow. Format: name[:strength_model[:strength_clip]]. Bare name looks up in models/loras/; paths, hf:// URIs, and https:// URLs (including civitai.com) are resolved through the model downloader. Repeat to stack multiple LoRAs.")),
    ("compile", bool, typer.Option(False, "--compile", help="Wrap the workflow's MODEL chain tail with TorchCompileModel for a 2-4x step-time speedup after a first-run compile cost.")),
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
    "image", "video", "audio", "set", "add_lora",
    "enable_comfy_kitchen_backends", "disable_comfy_kitchen_backends",
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


def _parse_listen_address(value: str) -> tuple[str, int | None]:
    """Parse an optional port from a listen address.

    Supports ``host:port``, ``[ipv6]:port``, and bare addresses.
    Returns ``(host, port)`` where port is None if not specified.
    """
    # [ipv6]:port
    if value.startswith("[") and "]:" in value:
        bracket_end = value.index("]:")
        return value[1:bracket_end], int(value[bracket_end + 2:])
    # host:port (but not bare IPv6 like :: or ::1)
    if ":" in value and value.count(":") == 1:
        host, port_str = value.rsplit(":", 1)
        if port_str.isdigit():
            return host, int(port_str)
    return value, None


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
                       "image", "video", "audio", "workflows",
                       "enable_comfy_kitchen_backends", "disable_comfy_kitchen_backends"):
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
    if filtered.get("novram"):
        # --novram is the strictest VRAM mode: dynamic-VRAM (aimdo) actively
        # works against it by holding allocations to track peak usage, so
        # implicitly disable it.
        filtered["disable_dynamic_vram"] = True

    # Parse host:port from --listen (e.g. "0.0.0.0:8189" or "[::]:8189")
    if "listen" in filtered:
        listen_val = filtered["listen"]
        parsed_host, parsed_port = _parse_listen_address(listen_val)
        if parsed_port is not None:
            filtered["listen"] = parsed_host
            filtered["port"] = parsed_port

    config = Configuration(**filtered)
    if "database_url" in filtered:
        config._database_url_explicit = True
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


@app.command(context_settings={**_COMFYUI_ENV, "allow_extra_args": True, "ignore_unknown_options": True}, rich_help_panel="Server")
@_with_options(_ALL_SHARED_OPTS, _WORKFLOW_OVERRIDE_OPTS)
def serve(
    ctx: typer.Context,

    daemon: bool = typer.Option(False, "-d", "--daemon", help="Run as a background daemon."),
    pid_file: Optional[str] = typer.Option(None, "--pid-file", help="PID file path (default: ~/.comfyui/comfyui.pid)."),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Log file path (default: ~/.comfyui/comfyui.log)."),
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
    """Start the ComfyUI server (default command).

    When no subcommand is given, comfyui defaults to serve. Use --guess-settings
    to auto-detect optimal hardware configuration (recommended for most users).

    \b
    VRAM Management:
      --novram             Aggressively offload all model weights from GPU after
                           each operation. Use when you have 16 GB VRAM or less,
                           or when running large models. Despite the name, the GPU
                           is still used for computation.
      --lowvram            Keep some UNet layers on GPU but offload most. Less
                           aggressive than --novram, better throughput but uses
                           more VRAM.
      --normalvram         Default balanced mode. Models are loaded onto GPU as
                           needed and kept resident when memory allows.
      --highvram           Keep all models in GPU memory. Fastest, but requires
                           enough VRAM to hold all loaded models simultaneously.
      --disable-dynamic-vram
                           Disable runtime VRAM monitoring (PyTorch 2.8+). By
                           default, ComfyUI checks actual RAM and VRAM usage at
                           inference time to decide whether to offload. This flag
                           reverts to static size estimates.
      --reserve-vram N     Keep N GB of VRAM free for other applications. ComfyUI
                           subtracts this from available VRAM when deciding what
                           fits on GPU.

    \b
    Performance (--fast):
      cublas_ops           Use optimized cuBLAS matrix multiply on NVIDIA Ampere+
                           (RTX 30xx, 40xx, 50xx, A100). Auto-enabled by
                           --guess-settings on NVIDIA GPUs.
      fp16_accumulation    Use FP16 for accumulation. Faster but lower precision.
      fp8_matrix_mult      Use FP8 for matrix multiplication.
      autotune             Enable CUDA autotuning for kernel selection.
      dynamic_vram         Enable runtime VRAM monitoring (default on PyTorch 2.8+).
    \b
      Example: comfyui serve --fast cublas_ops,fp16_accumulation

    \b
    Precision:
      Model-specific precision flags (--fp16-vae, --fp32-vae, --bf16-unet, etc.)
      override the default auto-detection. Use --fp32-vae on AMD GPUs if you see
      VAE decode artifacts. Use --force-fp16 to halve memory usage globally.

    \b
    Examples:
      comfyui serve --guess-settings
      comfyui serve --novram --fast cublas_ops
      comfyui serve --listen 0.0.0.0 --port 8188
      comfyui serve --daemon --guess-settings
      comfyui serve --fp32-vae --use-sage-attention
      comfyui serve --workflows my_workflow.json --prompt "a sunset" --steps 20
    """
    from ..component_model.setup import setup_pre_torch, setup_post_torch

    _daemon = daemon
    _pid_file = pid_file
    _log_file = log_file

    params = _collect_params(locals(), kwargs)
    for _k in ("daemon", "pid_file", "log_file", "_daemon", "_pid_file", "_log_file"):
        params.pop(_k, None)

    if params.get("otel_service_version") is None:
        from .. import __version__
        params["otel_service_version"] = __version__

    config = _build_config(params)
    _parse_plugin_args(ctx, config)

    if _daemon:
        from .daemon import daemonize, default_pid_file, default_log_file
        daemonize(_pid_file or default_pid_file(), _log_file or default_log_file())

    setup_pre_torch(config)
    _set_config_context(config)
    setup_post_torch(config)

    from .main import _start_comfyui
    try:
        asyncio.run(_start_comfyui(configuration=config))
    except KeyboardInterrupt:
        pass



@app.command(context_settings={**_COMFYUI_ENV, "allow_extra_args": True, "ignore_unknown_options": True}, rich_help_panel="Server")
@_with_options(_ALL_SHARED_OPTS)
def worker(
    ctx: typer.Context,
    distributed_queue_connection_uri: str = typer.Option(..., "--distributed-queue-connection-uri", help="AMQP URL for distributed queue."),
    distributed_queue_name: str = typer.Option("comfyui", "--distributed-queue-name", help="Queue name."),
    block_runtime_package_installation: bool = typer.Option(True, "--block-runtime-package-installation", help="Block runtime installs (default True for workers)."),
    **kwargs,
):
    """Run as a distributed queue worker.

    Connects to a RabbitMQ broker and pulls workflow execution requests from a
    shared queue. Multiple workers can process jobs in parallel. Use with
    'comfyui serve --distributed-queue-frontend' on the frontend side.

    \b
    Example:
      comfyui worker --distributed-queue-connection-uri amqp://guest:guest@rabbitmq:5672/
      comfyui worker --distributed-queue-connection-uri amqp://... --guess-settings
    """
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



_NODES_INDEX_URL = "https://nodes.appmana.com/simple/"


def _install_workflow_requirements(workflow_sources: list[str]) -> None:
    """Install missing custom node packages for the given workflows."""
    import shutil
    import subprocess

    from ..component_model.asyncio_files import load_workflow_json
    from ..component_model.workflow_dependencies import resolve_workflow_packages_versioned

    uv = shutil.which("uv")
    if uv is None:
        logger.warning("uv not found, skipping custom node installation")
        return

    packages: set[str] = set()
    for source in workflow_sources:
        if source == "-":
            continue
        try:
            workflow = load_workflow_json(source)
        except Exception:
            continue
        for name, _ in resolve_workflow_packages_versioned(
            workflow, builtin_class_types=_load_core_class_types(),
        ):
            packages.add(name)

    if not packages:
        return

    # Filter out already-installed packages
    from importlib.metadata import distributions
    installed: set[str] = set()
    for d in distributions():
        name = (d.metadata or {}).get("Name")
        if name:
            installed.add(name.lower().replace("_", "-"))
    missing = sorted(p for p in packages if p not in installed)
    if not missing:
        return

    logger.info("Installing custom nodes: %s", ", ".join(missing))
    cmd = [uv, "pip", "install", "--python", sys.executable,
           "--extra-index-url", _NODES_INDEX_URL] + missing
    subprocess.run(cmd, check=True)


def _download_workflow_models(workflow_sources: list[str]) -> None:
    """Download missing models for the given workflows."""
    from ..component_model.asyncio_files import load_workflow_json
    from ..component_model.workflow_convert import is_ui_workflow, convert_ui_to_api
    from ..model_downloader import _known_models_db, get_or_download, canonicalize_path
    from . import folder_paths

    filename_index: dict[str, list[tuple[str, object]]] = {}
    for db in _known_models_db:
        for folder_name in db.folder_names:
            for item in db:
                for name in [str(item), item.filename, item.save_with_filename] + list(item.alternate_filenames):
                    key = canonicalize_path(name)
                    if key:
                        filename_index.setdefault(key, []).append((folder_name, item))

    seen: set[str] = set()
    for source in workflow_sources:
        if source == "-":
            continue
        try:
            workflow = load_workflow_json(source)
        except Exception:
            continue
        if is_ui_workflow(workflow):
            workflow = convert_ui_to_api(workflow)
        for node_data in workflow.values():
            if not isinstance(node_data, dict):
                continue
            for value in (node_data.get("inputs") or {}).values():
                if not isinstance(value, str) or not value:
                    continue
                key = canonicalize_path(value)
                if key in seen:
                    continue
                seen.add(key)
                matches = filename_index.get(key)
                if matches:
                    folder_name = matches[0][0]
                    if not folder_paths.get_full_path(folder_name, value):
                        logger.info("Downloading %s/%s", folder_name, value)
                        get_or_download(folder_name, value)


@app.command(name="run-workflow", context_settings=_COMFYUI_ENV, rich_help_panel="Workflows")
@_with_options(_ALL_SHARED_OPTS, _WORKFLOW_OVERRIDE_OPTS)
def run_workflow(
    workflows: list[str] = typer.Argument(..., help="Workflow files, URIs, '-' for stdin, or literal JSON."),
    all: bool = typer.Option(False, "--all", "-a", help="Install missing custom nodes and download missing models before running."),
    disable_progress: bool = typer.Option(False, "--disable-progress", help="Disable CLI progress bars."),
    block_runtime_package_installation: bool = typer.Option(False, "--block-runtime-package-installation", help="Block runtime package installations."),
    **kwargs,
):
    """Execute workflow(s) and exit.

    Run one or more workflows without starting the web server. Results are
    printed as JSON to stdout; logs go to stderr. Accepts file paths, URIs
    (https://, hf://, s3://), literal JSON strings, or '-' for stdin.

    \b
    With --all, automatically install missing custom nodes from
    nodes.appmana.com and download missing models before running:
      comfyui run-workflow workflow.json --all --guess-settings

    \b
    Override workflow parameters inline:
      --prompt "a cat"     Replace the positive text prompt
      --steps 20           Override sampling steps
      --seed 42            Set a fixed seed
      --set 5.inputs.denoise=0.8
                           Override any node input by ID

    \b
    Examples:
      comfyui run-workflow workflow.json --all --guess-settings
      comfyui run-workflow workflow.json --prompt "a sunset" --steps 20 --seed 42
      comfyui run-workflow https://example.com/workflow.json --all
      cat workflow.json | comfyui run-workflow -
      comfyui run-workflow workflow.json --novram --fast cublas_ops
    """
    from ..component_model.setup import setup_pre_torch, setup_post_torch

    _all = all
    params = _collect_params(locals(), kwargs)
    params.pop("all", None)
    params.pop("_all", None)

    if params.get("output") is not None:
        params["output_directory"] = params["output"]

    if params.get("otel_service_version") is None:
        from .. import __version__
        params["otel_service_version"] = __version__

    config = _build_config(params)

    # Publish the parsed config to the execution context BEFORE anything
    # (including --all's pip resolution) imports a comfy module that reads
    # args at module-import time — e.g. comfy.model_management latches its
    # VRAM state from args.{novram,lowvram,...} at import. Running
    # _install_workflow_requirements first made --novram/--lowvram silently
    # no-op because model_management saw the default Configuration.
    _set_config_context(config)

    if _all:
        _install_workflow_requirements(config.workflows)

    setup_pre_torch(config)
    setup_post_torch(config)

    from ..component_model.entrypoints_common import configure_application_paths
    configure_application_paths(config)

    if _all:
        _download_workflow_models(config.workflows)

    from ..execution_context import context_configuration
    from ..nodes.package import import_all_nodes_in_workspace
    with context_configuration(config):
        import_all_nodes_in_workspace(raise_on_failure=False)

    from ..entrypoints.workflow import run_workflows
    try:
        asyncio.run(run_workflows(config.workflows, configuration=config))
    except KeyboardInterrupt:
        pass



@app.command(name="create-directories", context_settings=_COMFYUI_ENV, rich_help_panel="Environment", hidden=True)
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



@app.command(name="list-workflow-templates", context_settings=_COMFYUI_ENV, hidden=True)
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


@app.command(name="list-models", context_settings=_COMFYUI_ENV, hidden=True)
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



@app.command(name="serve-pip", rich_help_panel="Package Index")
@_with_options(_LOGGING_OPTS, _PIP_FACADE_OPTS)
def serve_pip(
    listen: str = typer.Option("127.0.0.1", "-H", "--listen", help="Specify the IP address to listen on."),
    port: int = typer.Option(8190, help="Set the listen port."),
    **kwargs,
):
    """Serve a PEP 503 simple index for facade-packaged custom nodes.

    Builds pip-installable wheels on demand from ComfyUI ecosystem custom nodes.
    Combines ComfyUI-Manager's registry with the Comfy Node Registry (CNR) API,
    injects dependency metadata, and serves wheels through a standard pip index.

    The public instance is at https://nodes.appmana.com. Self-host for air-gapped
    environments or to control which nodes are available.

    \b
    Wheel cache: generated wheels are cached to avoid rebuilding. Use any writable
    fsspec URI (local path, s3://, etc.):
      --pip-facade-cache-prefix /var/cache/comfyui
      --pip-facade-cache-prefix s3://bucket/prefix

    \b
    Examples:
      comfyui serve-pip --listen 0.0.0.0 --port 8190
      comfyui serve-pip --pip-facade-only-known-nodes
      uv pip install --extra-index-url http://localhost:8190/simple/ comfyui-ltxvideo
    """
    from ..component_model.setup import setup_pre_torch, setup_post_torch

    params = _collect_params(locals(), kwargs)
    config = _build_config(params)
    setup_pre_torch(config)
    _set_config_context(config)
    setup_post_torch(config)

    from .serve_pip import run_serve_pip
    run_serve_pip(config)


@app.command(name="snapshot-pip-registry", rich_help_panel="Package Index")
@_with_options(_LOGGING_OPTS, _PIP_FACADE_OPTS, _PIP_FACADE_SNAPSHOT_OPTS)
def snapshot_pip_registry(**kwargs):
    """Snapshot the resolved facade registry into a compact SQLite artifact."""
    from ..component_model.setup import setup_pre_torch

    params = _collect_params(locals(), kwargs)
    config = _build_config(params)
    setup_pre_torch(config)
    _set_config_context(config)

    from .snapshot_pip_registry import run_snapshot_pip_registry
    run_snapshot_pip_registry(config)


def _load_core_class_types() -> frozenset[str]:
    from ..nodes.package import _import_and_enumerate_nodes_in_module
    from ..nodes.package_typing import ExportedNodes
    from functools import reduce
    from ..nodes import base_nodes
    from comfy_extras import nodes as comfy_extras_nodes
    import comfy_api_nodes
    core_nodes = reduce(
        lambda x, y: x.update(y),
        map(_import_and_enumerate_nodes_in_module, [base_nodes, comfy_extras_nodes, comfy_api_nodes]),
        ExportedNodes(),
    )
    return frozenset(core_nodes.NODE_CLASS_MAPPINGS.keys())


@app.command(name="workflow-requirements", rich_help_panel="Workflows", hidden=True)
def workflow_requirements(
    workflow_file: str = typer.Argument(..., help="Workflow file, URI, or literal JSON."),
    format: str = typer.Option("requirements_txt", "--format", "-f", help="Output format: requirements_txt, requirements_txt_versioned, requirements_txt_locked"),
    snapshot_uri: Optional[str] = typer.Option(None, "--pip-facade-snapshot-uri", help="Facade registry snapshot URI. Defaults to the bundled snapshot."),
):
    """Print custom node packages required by a workflow in pip requirements format."""
    from ..component_model.asyncio_files import load_workflow_json
    from ..component_model.workflow_dependencies import resolve_workflow_packages_versioned

    workflow = load_workflow_json(workflow_file)
    packages = resolve_workflow_packages_versioned(
        workflow,
        snapshot_uri=snapshot_uri,
        builtin_class_types=_load_core_class_types(),
    )

    for name, version in packages:
        if format == "requirements_txt_versioned" and version:
            typer.echo(f"{name}>={version}")
        elif format == "requirements_txt_locked" and version:
            typer.echo(f"{name}=={version}")
        else:
            typer.echo(name)


@app.command(name="start", rich_help_panel="Daemon")
def start(ctx: typer.Context):
    """Start ComfyUI as a background daemon with auto-detected settings.

    Equivalent to 'comfyui serve --daemon --guess-settings'. Detects your GPU,
    RAM, and available acceleration libraries, then starts ComfyUI in the
    background. Use 'comfyui stop' to shut it down and 'comfyui logs -f' to
    watch output.

    \b
    Examples:
      comfyui start
      comfyui logs -f
      comfyui stop
    """
    ctx.invoke(serve, daemon=True, guess_settings=True)


@app.command(name="stop", rich_help_panel="Daemon")
def stop(
    server: Optional[str] = typer.Option(None, "--server", envvar="COMFYUI_SERVER", help="Server URL for HTTP fallback."),
    pid_file: Optional[str] = typer.Option(None, "--pid-file", help="PID file path (default: ~/.comfyui/comfyui.pid)."),
):
    """Stop the ComfyUI daemon.

    Reads the PID from ~/.comfyui/comfyui.pid and sends SIGTERM. Falls back to
    the HTTP /interrupt endpoint if the PID file is missing.
    """
    from .daemon import stop_daemon, default_pid_file

    pf = pid_file or default_pid_file()
    if stop_daemon(pf):
        typer.echo("ComfyUI daemon stopped.")
        return
    try:
        from .server_connection import post_json
        asyncio.run(post_json(server, "/interrupt"))
        typer.echo("Sent interrupt to server.")
    except Exception as exc:
        typer.echo(f"Could not stop daemon: {exc}", err=True)
        raise typer.Exit(1)


@app.command(name="logs", rich_help_panel="Daemon")
def logs(
    follow: bool = typer.Option(False, "-f", "--follow", help="Follow log output."),
    server: Optional[str] = typer.Option(None, "--server", envvar="COMFYUI_SERVER", help="Server URL."),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Log file path (default: ~/.comfyui/comfyui.log)."),
):
    """Tail server logs.

    Without -f, prints the current log contents. With -f, follows the log file
    in real time (like tail -f). Tries the HTTP /internal/logs/raw endpoint
    first, falls back to reading ~/.comfyui/comfyui.log directly.

    \b
    Examples:
      comfyui logs
      comfyui logs -f
      comfyui logs --server http://remote:8188
    """
    from .daemon import default_log_file

    lf = log_file or default_log_file()
    log_path = Path(lf)

    if not follow:
        try:
            from .server_connection import fetch_text
            text = asyncio.run(fetch_text(server, "/internal/logs/raw"))
            typer.echo(text)
            return
        except Exception:
            pass
        if log_path.exists():
            typer.echo(log_path.read_text())
        else:
            typer.echo("No logs found.", err=True)
        return

    import time
    if not log_path.exists():
        typer.echo(f"Waiting for {lf}...", err=True)
        while not log_path.exists():
            time.sleep(0.5)
    with open(log_path) as f:
        f.seek(0, 2)
        try:
            while True:
                line = f.readline()
                if line:
                    typer.echo(line, nl=False)
                else:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass


def _register_sub_apps():
    from .sub_models import models_app
    from .sub_workflows import workflows_app
    from .sub_nodes import nodes_app
    from .sub_jobs import jobs_app
    from .sub_env import env_app

    app.add_typer(models_app, name="models", help="Manage models.", rich_help_panel="Workflows")
    app.add_typer(workflows_app, name="workflows", help="Run and manage workflows.", rich_help_panel="Workflows")
    app.add_typer(nodes_app, name="nodes", help="Inspect installed nodes.", rich_help_panel="Workflows")
    app.add_typer(jobs_app, name="jobs", help="List and cancel server jobs.", rich_help_panel="Daemon")
    app.add_typer(env_app, name="env", help="Environment and diagnostics.", rich_help_panel="Environment")


_KNOWN_COMMANDS = frozenset({
    "serve", "serve-pip", "worker", "start", "stop", "logs",
    "models", "workflows", "nodes", "jobs", "env",
    "run-workflow", "workflow-requirements", "list-workflow-templates",
    "list-models", "create-directories",
    "snapshot-pip-registry",
})


_BARE_FLAG_DEFAULTS: dict[str, str] = {
    "--listen": "0.0.0.0,::",
    "--enable-cors-header": "*",
    "--enable-cors": "*",
    "--cache-ram": "-1.0",
}
"""Flags that argparse accepted with ``nargs='?'``.

Typer/Click options always consume an argument, so a bare ``--listen``
(without a value) fails with *"Option '--listen' requires an argument"*.
We expand these in ``sys.argv`` before Click sees them.
"""


def _expand_bare_flags(argv: list[str]) -> list[str]:
    """Insert default values for flags listed in :data:`_BARE_FLAG_DEFAULTS`
    when they appear without a following value (end-of-args or next token
    starts with ``-``).
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in _BARE_FLAG_DEFAULTS:
            next_token = argv[i + 1] if i + 1 < len(argv) else None
            if next_token is None or next_token.startswith("-"):
                out.append(token)
                out.append(_BARE_FLAG_DEFAULTS[token])
                i += 1
                continue
        out.append(token)
        i += 1
    return out


def entrypoint():
    """Main CLI entrypoint. Defaults to 'serve' when no subcommand is given."""
    _register_sub_apps()

    if len(sys.argv) <= 1:
        sys.argv.insert(1, "serve")
    elif sys.argv[1].startswith("-") and sys.argv[1] not in ("--help", "-h"):
        sys.argv.insert(1, "serve")

    sys.argv[:] = _expand_bare_flags(sys.argv)

    app()
