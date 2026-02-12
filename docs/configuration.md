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
