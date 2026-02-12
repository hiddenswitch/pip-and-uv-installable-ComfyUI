# Configuration

This project supports configuration with command line arguments, the environment and a configuration file.

## Configuration File

Run `comfyui --help` for available commands, or `comfyui serve --help` for all server configuration options.

Args that start with `--` can also be set in a config file (`config.yaml`, `config.ini`, `config.conf` or `config.json` or specified via `-c`). Config file syntax allows: `key=value`, `flag=true`, `stuff=[a,b,c]` (for details, see syntax [here](https://goo.gl/R74nmi)). In general, command-line values override environment variables which override config file values which override defaults.

## Extra Model Paths

Copy [docs/examples/configuration/extra_model_paths.yaml](examples/configuration/extra_model_paths.yaml) to your working directory, and modify the folder paths to match your folder structure.

You can pass additional extra model path configurations with one or more copies of `--extra-model-paths-config=some_configuration.yaml`.

### Commands

```
comfyui --help
```

| Command | Description |
|---------|-------------|
| `serve` | Start the ComfyUI server (default command) |
| `post-workflow` | Execute workflow(s) and exit |
| `worker` | Run as a distributed queue worker |
| `create-directories` | Create default model/input/output/temp directories |
| `list-workflow-templates` | List available workflow templates |
| `list-models` | List known downloadable models |
| `integrity-check` | Print system diagnostics and verify installation integrity |

When no subcommand is given, `serve` is used by default (e.g. `comfyui --novram` is equivalent to `comfyui serve --novram`).

### `comfyui serve` Options

Run `comfyui serve --help` for the full list. All options can also be set via environment variables (prefixed with `COMFYUI_`).

### `Configuration` Object

All CLI options correspond to attributes on `comfy.cli_args_types.Configuration`. Use `default_configuration()` to create one programmatically:

```python
from comfy.cli_args import default_configuration
from comfy.cli_args_types import Configuration, PerformanceFeature

config: Configuration = default_configuration()
config.novram = True
config.fast = {PerformanceFeature.CublasOps}
config.guess_settings = True
```

`Configuration` is a `dict` subclass with attribute access and observer support — set attributes directly or call `config.update({...})`. See [Embedded / Library Usage](embedded.md) for usage with `Comfy()`.

### Auto-Detection (--guess-settings)

`--guess-settings` auto-detects the best settings for the current machine. It only touches settings still at their defaults — explicit flags always override guessed values.

```bash
comfyui serve --guess-settings
comfyui post-workflow my_workflow.json --guess-settings
```

What it detects:

| Condition | Action |
|-----------|--------|
| NVIDIA GPU present | Enables `--fast cublas_ops` |
| NVIDIA GPU with competing processes (e.g. Discord, games) | Enables `--novram` to avoid VRAM contention |
| AMD RDNA 4 GPU (gfx12xx) | Enables `--fp16-vae` |
| AMD GPU (older than RDNA 4) | Enables `--fp32-vae` |
| AMD GPU on Windows | Enables `--use-quad-cross-attention` |
| `sageattention` package installed | Enables `--use-sage-attention` |
| `xformers` package installed (no sageattention) | Keeps xformers enabled |
| No attention packages | Falls back to `--use-pytorch-cross-attention` |
| Less than 32 GB RAM | Enables `--disable-pinned-memory` |

For programmatic use, set `config.guess_settings = True` — detection runs automatically when `Comfy` starts:

```python
from comfy.cli_args import default_configuration

config = default_configuration()
config.guess_settings = True
```

### Performance Optimizations (--fast)

The `--fast` option accepts space-separated feature names (not comma-separated):

```bash
# Enable single optimization
comfyui --fast cublas_ops

# Enable multiple optimizations (space-separated)
comfyui --fast cublas_ops dynamic_vram

# Enable all available optimizations
comfyui --fast fp16_accumulation fp8_matrix_mult cublas_ops autotune dynamic_vram
```

Available optimizations:
- `fp16_accumulation` - Use fp16 for accumulation in matrix operations
- `fp8_matrix_mult` - Enable fp8 matrix multiplication
- `cublas_ops` - Use cuBLAS for linear operations
- `autotune` - Enable PyTorch autotuning
- `dynamic_vram` - Enable dynamic VRAM management
