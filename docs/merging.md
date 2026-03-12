# Merging Upstream Changes

This document covers synchronization tasks needed when merging upstream ComfyUI changes.

## Environment Setup

This project uses `uv` for dependency management. Before running any commands:

### Check if You're in a uv-Managed Environment

```bash
# Check for UV_VIRTUAL_ENV or if uv created the venv
echo $UV_VIRTUAL_ENV
# Or check the venv origin
cat $VIRTUAL_ENV/pyvenv.cfg | grep uv
```

### Package Installation

**In a uv-managed environment**: Always use `uv pip install` instead of `pip install`:

```bash
# Correct
uv pip install <package-name>
uv pip install -r requirements.txt

# Incorrect (do not use in uv environments)
pip install <package-name>
```

**Why**: Using `pip install` directly in a uv-managed environment can cause dependency resolution conflicts and inconsistent package states. The `uv` tool maintains its own lockfile and dependency graph.

### Quick Reference

| Task | Command |
|------|---------|
| Install package | `uv pip install <package>` |
| Install from requirements | `uv pip install -r requirements.txt` |
| Install editable | `uv pip install -e .` |
| Install with extras | `uv pip install "package[extra1,extra2]"` |
| Sync dependencies | `uv sync` |

## Linting

**IMPORTANT:** After fixing imports and other merge issues, run the linter on the **entire codebase**, not just the changed files. Upstream changes may introduce issues in unchanged files due to cross-module dependencies:

```bash
pylint -j 32 comfy/ comfy_extras/ comfy_api/ comfy_api_nodes/ comfy_compatibility/ comfy_execution/
```

The full lint must pass with no errors before the merge is complete.

See [Linting Guidelines](linting.md) for custom rules and common fixes.

## CLI Arguments

`comfy/cli_args.py` is a **stub parser** that accepts all upstream `parser.add_argument(...)` calls as no-ops. Real CLI parsing is handled by Typer in `comfy/cmd/cli.py`.

When upstream adds new CLI arguments to `comfy/cli_args.py`, you must also:

1. **`comfy/cli_args_types.py`** — Add the field to the `Configuration` class docstring and `__init__`
2. **`comfy/cmd/cli.py`** — Add a corresponding `typer.Option(...)` to the appropriate command(s)

### How the Stub Parser Works

Upstream's pattern is `parser.add_argument("--flag", ...)` at module level. Our `parser` is a `_StubParser` whose `add_argument` is a no-op (`return self`). This means:
- Upstream can add any `parser.add_argument` line and git merges cleanly
- The actual parsing happens in Typer via `comfy/cmd/cli.py`
- `cli_args.args` returns the current execution context's `Configuration` via a module property

### Example

If upstream adds:
```python
parser.add_argument("--disable-assets-autoscan", action="store_true", help="Disable asset scanning...")
```

Then:

**In `cli_args_types.py`** — add to the docstring and `__init__`:
```python
# Docstring:
disable_assets_autoscan (bool): Disable asset scanning on startup for database synchronization.

# __init__:
self.disable_assets_autoscan: bool = False
```

**In `comfy/cmd/cli.py`** — add to the `serve` command (and other commands if relevant):
```python
disable_assets_autoscan: bool = typer.Option(False, "--disable-assets-autoscan", help="Disable asset scanning."),
```

### Configuration Field Categories

When adding new CLI arguments, also check if they belong to special field categories in `comfy/component_model/configuration.py`:

**`AFFECTS_PATHS`** - Fields that affect folder paths. When these change, `folder_names_and_paths` is reinitialized:
- `cwd`, `base_directory`, `base_paths`
- `output_directory`, `input_directory`, `temp_directory`, `user_directory`
- `extra_model_paths_config`

**`MODEL_MANAGEMENT_ARGS`** - Fields that affect model management behavior (VRAM, precision, device selection). When these differ from defaults, `ProcessPoolExecutor` is required:
- VRAM modes: `lowvram`, `novram`, `highvram`, `gpu_only`, `cpu`
- Precision: `force_fp32`, `force_fp16`, `force_bf16`, `fp*_unet`, `fp*_vae`, `fp*_text_enc`
- Attention: `use_*_cross_attention`, `use_sage_attention`, `use_flash_attention`, `disable_xformers`
- Memory: `reserve_vram`, `disable_smart_memory`, `disable_pinned_memory`, `async_offload`
- Device: `directml`, `deterministic`, `force_channels_last`
- Performance: `fast` (includes `DynamicVRAM` feature)

If a new argument affects paths or model management, add it to the appropriate frozenset.

### Quick Check

After merging, diff the argument names across all three files:
```bash
# Stub parser (upstream adds here):
grep -oP '(?<=add_argument\(")[^"]+' comfy/cli_args.py | sed 's/^--//' | sed 's/-/_/g' | sort > /tmp/stub_args.txt

# Configuration fields:
grep -oP '(?<=self\.)[a-z_]+(?=:)' comfy/cli_args_types.py | sort > /tmp/types.txt

# Typer options (in serve command):
grep -oP '(?<=typer\.Option\()[^)]+' comfy/cmd/cli.py | sort > /tmp/typer_args.txt

diff /tmp/stub_args.txt /tmp/types.txt
```

## Git Merge Configuration

This fork frequently moves upstream top-level paths into `comfy/`, so plain Git defaults produce too much rename noise during upstream merges.

Configure Git once before merging upstream:

```bash
git config --global merge.renames true
git config --global merge.directoryRenames true
git config --global merge.renameLimit 999999
git config --global diff.renameLimit 999999
git config --global rerere.enabled true
git config --global rerere.autoUpdate true
git config --global merge.conflictStyle zdiff3
```

What these do:

- `merge.directoryRenames=true` tells Git to automatically follow directory moves like `app/assets -> comfy/app/assets` instead of leaving them as `CONFLICT (file location)`.
- `merge.renameLimit` and `diff.renameLimit` raise the rename detection ceiling so large upstream changes still get matched.
- `rerere.enabled=true` and `rerere.autoUpdate=true` record your conflict resolutions and replay them on later merges.
- `merge.conflictStyle=zdiff3` makes the remaining real conflicts much easier to inspect.

If a merge was already started with the wrong settings, abort and retry so Git recomputes the merge:

```bash
git merge --abort
git pull comfyui master
```

If upstream both moved and heavily edited files, you can retry with a lower rename similarity threshold:

```bash
git pull -s ort -Xfind-renames=30% comfyui master
```

Use the lower threshold only when needed; it can create false rename matches.

## Version String

Upstream uses `comfyui_version.py` at the repository root. We deleted this file and moved the version string to `comfy/__init__.py`.

When merging, accept our deletion of `comfyui_version.py` and update the `__version__` in `comfy/__init__.py` instead.

## Requirements

Upstream uses `requirements.txt` at the repository root. We deleted this file and moved all dependencies to `pyproject.toml`.

When merging, accept our deletion of `requirements.txt` and update version minimums in `pyproject.toml` instead. Key packages to watch:

- `comfyui-frontend-package`
- `comfyui-workflow-templates`
- `comfyui-embedded-docs`
- `comfy_kitchen`
- `comfy-aimdo`
- `comfyui_manager`

## Package Init Files

Upstream sometimes adds new Python packages without `__init__.py` files. After merging, check for missing init files and add empty ones where needed to make them proper Python packages.

## Directory Structure

Our fork moves some top-level directories into `comfy/`. When upstream adds or modifies files in these directories:

1. First, merge the upstream changes to the top-level location
2. In a separate commit, `git mv` the files to the correct location

Example: `app/assets` → `comfy/app/assets`

Also keep track of top-level modules we have relocated into `comfy/`. In particular, upstream `node_helpers.py` maps to `comfy/node_helpers.py` in this fork. Preserve that mapping explicitly during merges so upstream edits to `node_helpers.py` are surfaced and re-applied instead of showing up only as `Deleted by us`.

You will also have to move upstream root-level `comfy_extras/nodes_*.py` files into `comfy_extras/nodes/`, where they are scanned automatically. Use `git mv` and do not leave these files at the `comfy_extras/` package root.

Example:

```bash
git mv comfy_extras/nodes_math.py comfy_extras/nodes/nodes_math.py
git mv comfy_extras/nodes_sdpose.py comfy_extras/nodes/nodes_sdpose.py
```

After moving them, fix their imports so they are valid from the `comfy_extras.nodes` package (see [Import Fixes](#import-fixes) below).

This two-step approach keeps git history cleaner and makes conflicts easier to resolve.

Upstream may also still add tests under `tests-unit/`. Move those into `tests/unit/` in this fork, preserving the same relative structure.

Example: `tests-unit/seeder_test/test_seeder.py` → `tests/unit/seeder_test/test_seeder.py`

## Import Fixes

After moving directories, fix absolute imports to use relative imports whenever the target is inside the repo. Upstream uses absolute imports like:

- `import folder_paths` -> `from comfy.cmd import folder_paths`
- `import nodes` in moved node modules -> `from comfy.nodes import base_nodes as nodes`
- internal package imports like `from comfy.cmd.server import PromptServer` inside `comfy_api/latest/__init__.py` should become relative imports such as `from ...cmd.server import PromptServer`

```python
import app.assets.manager as manager
from app.database.db import create_session
from app.assets.helpers import some_function
```

Convert these to relative imports based on file location:

```python
from .. import manager
from ..database.db import create_session
from .helpers import some_function
```

For optional dependencies like `torchaudio`, keep the established local-import pattern after the move:

```python
def some_audio_method(...):
    try:
        import torchaudio  # pylint: disable=import-error
    except ImportError as exc:
        raise TorchAudioNotFoundError("torchaudio is required") from exc
```

Do not introduce new top-level `import torchaudio` lines in moved modules.

To reduce the impact of absolute to relative imports, sometimes it may make more sense to keep the module name:

#### Before:

**comfy/some_file.py**
```python
import comfy.model_management
```

#### After (Okay):
**comfy/some_file.py**
```python
from . model_management import get_torch_device, some_other_import, etc_etc
```

#### After (Okay):
**comfy/some_file.py**
```python
from . import model_management
```

Many import changes will cause name collisions (such as `from . import sd`, where `sd` is used as a variable name). Rename variables if a newly imported symbol would be shadowed by a variable.

## Alembic Migrations

Upstream keeps `alembic_db/versions/` at the repository root. We move it to `comfy/alembic_db/versions/`.

After merging new migrations:
```bash
git mv alembic_db/versions/* comfy/alembic_db/versions/
rmdir alembic_db/versions alembic_db
```

## Type Hints

This project has extensive typing annotations that typically look like `comfy/*_types.py`, `comfy/*_typing.py` and `comfy/component_model/*.py` for internal APIs.

Read them, and wherever you encounter code that has been added or interacts with types, align the type definitions to the merged code, and adopt the `NamedTuples` to replace untyped tuples where possible.

`NotRequired` should be imported from `typing_extensions` for better Python compatibility.

If it is illuminating to do so, label the iteratee with a type hint:

```python
some_list: List[NamedTupleType] = ...
# add this: type the iteratee, leaving it blank (no RHS value)
some_tuple: NamedTupleType
for some_tuple in some_list:
    ...
```

This fork also has numerous protocols that are used for functionality like populating `ModelPatcher` objects and adding functionality to them. Use them correctly.

## Testing New Functionality

New functionality will have content added in a variety of places by the upstream authors:

 - Sometimes in top level directories like `tests-unit/`: this should be moved to `tests/unit`
 - `tests/execution/test_execution.py`: this can stay

Move The files as needed. Remember to add `__init__.py` files.

The `conftest.py` needs to be updated in most upstream test code. Upstream test code starts `comfyui` in a subprocess which is usually unnecessary.

If you are testing RESTful API methods, you should adopt the creation of a ComfyUI subprocess in a loop using the code in the top level [conftest.py](../tests/conftest.py).

If you are testing functionality of ComfyUI generally, create a new `Comfy` instance and use an async wrapper correctly to use it. See [testing](./testing.md) for examples.

Observe that the configuration object is created with `default_configuration()` and will be the primary way you configure embedded or RESTful API server ComfyUI objects. Use it instead of passing raw command line args. When the upstream test parameterizes configurations with CLI args in the form of `--blah` passed via `pytestargs` (or some other similar approach), just parameterize the test normally using `pytest` features, making the appropriate change for fixtures versus test methods (i.e., fixtures will generally be parameterized with `request.params`), and you will just modify a `config = default_configuration()` object to implement the parameterization instead of raw args.

### Test Conftest Patterns

When adapting upstream test `conftest.py` files:

1. **Use top-level fixtures** - Import and use `comfy_background_server_from_config` from `tests/conftest.py` instead of duplicating server startup logic
2. **No environment variables** - Don't use `os.environ.get()` for test configuration. Use pytest parameterization if multiple configurations are needed
3. **No `pytest.addoption`** - Don't add custom CLI options. Parameterize fixtures with `request.param` instead
4. **Use `default_configuration()`** - Create configuration objects programmatically, not from CLI args

Example fixture pattern:
```python
from comfy.cli_args import default_configuration
from tests.conftest import comfy_background_server_from_config

@pytest.fixture(scope="session")
def my_server_config(tmp_path_factory) -> Configuration:
    config = default_configuration()
    config.base_directory = str(tmp_path_factory.mktemp("test"))
    config.cpu = True
    config.port = 0  # Let system assign port
    return config

@pytest.fixture(scope="session")
def server_url(my_server_config):
    for config, proc in comfy_background_server_from_config(my_server_config):
        yield f"http://{config.listen}:{config.port}"
```

### Test Assertion Updates

Upstream may change error types, message formats, or validation behavior. When tests fail after merge:

1. **Check error types** - Error `type` fields may change (e.g., `"invalid_prompt"` → `"missing_node_type"`)
2. **Check message content** - Error messages may be reworded
3. **Verify behavior is correct** - Ensure the test is checking for the right behavior, then update assertions to match

## Module-Level Properties

This fork uses module-level properties from `comfy/component_model/module_property.py` for configuration-dependent exports. This pattern allows module attributes to be evaluated at access time rather than import time.

### Why Use Module Properties

Some exports depend on runtime configuration (e.g., whether dynamic VRAM is enabled). The traditional approach of assigning at module level:

```python
# Bad: evaluated at import time, before configuration is known
CoreModelPatcher = ModelPatcher  # or ModelPatcherDynamic?
```

This leads to "radioactive" patterns where modules mutate each other's attributes after import.

### The Module Property Pattern

Instead, use a module property that evaluates at access time:

```python
from .component_model.module_property import create_module_properties

_module_properties = create_module_properties()

@_module_properties.getter
def _CoreModelPatcher() -> type[ModelPatcher]:
    """Module property - the underscore prefix is stripped."""
    return get_model_patcher_class()
```

Now `CoreModelPatcher` is a module attribute that calls `get_model_patcher_class()` each time it's accessed, returning the correct class based on current configuration.

### When to Use

Use module properties when:
- An export depends on runtime configuration
- You want to avoid import-time side effects
- The value might change during program execution

For `CoreModelPatcher` specifically:
- **Deprecated**: Use `get_model_patcher_class()` in new code
- **Module property**: Provides backwards compatibility for existing code that imports `CoreModelPatcher`

## Protocol Alignment

This fork uses protocols in `comfy/model_management_types.py` to define interfaces for model management. When upstream adds parameters to `ModelPatcher` or related classes:

1. **Update the protocol** - Add new attributes to `ModelManageable` protocol if they should be universally available
2. **Update the stub** - Add default implementations to `ModelManageableStub`
3. **Update dynamic variants** - Ensure `ModelPatcherDynamic` and similar classes accept and pass through new parameters

Example: When `ckpt_name` was added to track checkpoint paths:
```python
# In ModelManageable protocol
ckpt_name: Optional[str]

# In ModelManageableStub
ckpt_name: Optional[str] = None

# In ModelPatcherDynamic.__init__
def __init__(self, model, load_device, offload_device, size=0,
             weight_inplace_update=False, ckpt_name: Optional[str] = None):
    super().__init__(model, load_device, offload_device, size,
                     weight_inplace_update, ckpt_name=ckpt_name)
```

## Common Linting Issues

After merging, pylint often catches these issues:

### Undefined Variables

Upstream code sometimes uses variables before assignment in conditional branches:
```python
# Before (pylint error: possibly used before assignment)
if condition:
    output_ui = some_function()
return output_ui

# After
output_ui = []
if condition:
    output_ui = some_function()
return output_ui
```

### Variable Shadowing

When converting imports, watch for variable names that shadow imported modules or functions:
```python
# Before (pylint error: redefining name 'post_cast' from outer scope)
from . import post_cast
for post_cast in some_list:  # shadows the function!
    ...

# After
from . import post_cast
for tensor in some_list:
    ...
```

### Missing Imports

When code is moved, some imports may be lost:
```python
# Check for missing standard library imports
import logging  # often missing after refactoring

# Check for module-level function references
from .cli_args import args  # might need _args() function instead
```

### Avoid `__all__`

Never use `__all__` in this codebase. It's brittle and causes maintenance issues:

```python
# Don't do this
__all__ = ["function1", "function2", "ClassName"]

# Instead, just export what you need via normal imports
# and use explicit imports at the call site
```

If you need to re-export symbols from a module, use explicit imports with `# noqa: F401` to silence unused import warnings:

```python
from .helpers import some_function  # noqa: F401
```

### Undefined Module References

After converting `import comfy.module` to `from . import module`, update all usages:
```python
# Before
import comfy.model_management
x = comfy.model_management.get_torch_device()

# After
from . import model_management
x = model_management.get_torch_device()
```

## Adding New Models

When upstream adds new workflow files that reference new models, add those models to `comfy/model_downloader.py`.

### Step 1: Identify Models in Workflows

Check `git status` for new workflow files in `tests/inference/workflows/`. Read the workflow JSON files to find model references:

- `UNETLoader` → `unet_name` field → add to `KNOWN_UNET_MODELS`
- `CLIPLoader` → `clip_name` field → add to `KNOWN_CLIP_MODELS`
- `VAELoader` → `vae_name` field → add to `KNOWN_VAES`
- `CheckpointLoader` → `ckpt_name` field → add to `KNOWN_CHECKPOINTS`

### Step 2: Find HuggingFace Repository

Search for the model filename on HuggingFace to find the correct repository and path. For example:
- `flux-2-klein-base-4b.safetensors` → `black-forest-labs/FLUX.2-klein-base-4B`

### Step 3: Add to Model Downloader

Add a `HuggingFile` entry to the appropriate list in `comfy/model_downloader.py`:

```python
HuggingFile("repo-owner/repo-name", "path/to/model.safetensors"),
```

Group related models together with comments (e.g., `# Flux 2`).

### Step 4: Run Inference Tests

Run the inference tests for the new workflows to verify the models work correctly.

First, list available tests to find the workflow names:
```bash
pytest tests/inference --collect-only 2>&1 | grep -i "workflow-name"
```

Then run tests for specific workflows using `-k` to filter by workflow filename. Use `and` to combine multiple filters:
```bash
pytest -v tests/inference -k "workflow-name and normalvram"
```

Example for flux2-klein workflows:
```bash
pytest -v tests/inference -k "flux2-klein-0 and normalvram"
```

The `-k` flag matches test names containing the specified substrings. Common filters:
- Workflow name: `flux2-klein-0` matches `flux2-klein-0.json`
- VRAM mode: `normalvram` or `novram`
- Attention: `use_pytorch` or `sage_attention`

Avoid running all test variations by being specific with filters.

## Custom Nodes

Custom node compatibility is tested via `tests/custom_nodes/test_custom_node_execution.py`. The test clones each registered custom node, installs its dependencies, boots ComfyUI with the nodes loaded, and runs their bundled example workflows.

### Node Registry

All tested custom nodes are declared in `comfy/component_model/node_registry.py` as `CustomNodeSpec` entries in `CUSTOM_NODE_REGISTRY`:

```python
CustomNodeSpec(
    node_id="ComfyUI-Example-Node",
    repo_url="https://github.com/author/ComfyUI-Example-Node",
    display_name="ComfyUI-Example-Node",
    depends_on=("ComfyUI-VideoHelperSuite",),  # installed first
    priority="Mid",                              # "High" (default) or "Mid"
    needs_submodules=True,                       # if repo uses git submodules
    xfail=True,                                  # expected to fail
    xfail_reason="requires API keys at runtime",
)
```

Key fields:
- **`depends_on`**: Other node IDs that must be installed before this one
- **`xfail` / `xfail_reason`**: Mark nodes that cannot pass in CI (e.g., require API keys or auto-download large models at runtime)
- **`skip_requirements`**: Package names to exclude from `pip install` (e.g., packages already in the main venv)
- **`extra_requirements`**: Additional pip requirements not in the node's `requirements.txt`

### Adding Models for Custom Node Workflows

Custom node example workflows reference models by filename, often with subfolder prefixes (e.g., `sd1.5/dreamshaper_8.safetensors`). These must be present in `KNOWN_CHECKPOINTS` (or the appropriate known model list) in `comfy/model_downloader.py` so the download system can find and fetch them during test execution.

#### Using `alternate_filenames`

Community workflows may reference the same model under different paths depending on how the user organized their models folder. Use `alternate_filenames` on `HuggingFile` or `CivitFile` entries to register all known path variants:

```python
HuggingFile(
    "Lykon/DreamShaper",
    "DreamShaper_8_pruned.safetensors",
    save_with_filename="dreamshaper_8.safetensors",
    alternate_filenames=(
        "DreamShaper_8_pruned.safetensors",
        "sd1.5/dreamshaper_8.safetensors",   # subfolder-prefixed variant
    ),
),
```

How `alternate_filenames` works:
- During **validation**, `DownloadableFileList.view_for_validation()` adds all `alternate_filenames` to the set of accepted values — so `sd1.5/dreamshaper_8.safetensors` passes the combo-box check
- During **download**, `get_or_download()` matches the requested filename against `str(candidate)`, `candidate.filename`, `candidate.save_with_filename`, and all `candidate.alternate_filenames` — so any registered variant triggers the correct HuggingFace/CivitAI download

When adding a new custom node whose workflows reference models not yet in the known lists:

1. Run the test for that node and look for `value_not_in_list` errors — these show the exact filename the workflow expects
2. Check if the model already exists in `KNOWN_CHECKPOINTS` (or `KNOWN_LORAS`, `KNOWN_CONTROLNETS`, etc.) under a different name
3. If it exists, add the workflow's filename as an `alternate_filenames` entry
4. If it doesn't exist, add a new `HuggingFile` or `CivitFile` entry with the correct repo/version info

### Running Custom Node Tests

Tests are marked with `slow` and `git_clone`. Run a single node:

```bash
pytest tests/custom_nodes/test_custom_node_execution.py::TestCustomNodeExecution::test_execute_example_workflows[ComfyUI-Prompt-Combinator] \
  -v -s --log-cli-level=INFO --tb=short -m "slow and git_clone"
```

Run all custom node tests (slow, ~20 minutes):

```bash
pytest tests/custom_nodes/test_custom_node_execution.py -v -s --log-cli-level=INFO --tb=short -m "slow and git_clone"
```

The first run clones all nodes and installs dependencies into `~/.cache/comfy-test/custom_nodes/`. Subsequent runs reuse the cache. To force a fresh install:

```bash
rm -rf ~/.cache/comfy-test/custom_nodes/
```

### Test Infrastructure

- **`tests/custom_nodes/conftest.py`** — `build_config()`, `install_all_nodes()`, `make_base_dirs()`
- **`comfy/app/custom_node_manager.py`** — `CustomNodeManager.install_custom_node()` handles cloning and dependency installation
- **`comfy/component_model/site_packages.py`** — `add_node_site()` adds the `node_site/` directory to `sys.path`
- **`tests/custom_nodes/test_data/`** — Test media assets (image, audio, video) substituted into workflows during testing
