# Custom Nodes

Custom Nodes can be added to ComfyUI by copying and pasting Python files into your `./custom_nodes` directory.

## Installing Custom Nodes

There are two kinds of custom nodes: vanilla custom nodes, which generally expect to be dropped into the `custom_nodes` directory and managed by a tool called the ComfyUI Extension manager ("vanilla" custom nodes) and this repository's opinionated, installable custom nodes ("installable").

### Installing ComfyUI Manager

ComfyUI-Manager is a popular extension to help you install and manage other custom nodes. To install it, you will need `git` on your system.

#### Manual Install

The installation process for ComfyUI-Manager requires two steps: installing its Python dependencies, and then cloning its code into the `custom_nodes` directory.

1.  **Install dependencies.**
    First, ensure you have installed `comfyui` from this repository as described in the Installing section. Then, run the following command from your ComfyUI workspace directory (the one containing your `.venv` folder) to install the extra dependencies for ComfyUI-Manager:

    ```shell
    uv pip install --torch-backend=auto --upgrade "comfyui[comfyui_manager]@git+https://github.com/hiddenswitch/ComfyUI.git"
    ```

2.  **Clone the repository.**
    Next, you need to clone the ComfyUI-Manager repository into the `custom_nodes` directory within your ComfyUI workspace. Your workspace is the directory you created during the initial setup where you ran `uv venv` (e.g., `~/Documents/ComfyUI_Workspace`).

    If the `custom_nodes` directory does not exist in your workspace, create it first (e.g., `mkdir custom_nodes`). Then, from your workspace directory, run the following command:

    ```shell
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git ./custom_nodes/ComfyUI-Manager
    ```
    This command will place the manager's code into `custom_nodes/ComfyUI-Manager/`.

3.  **Restart ComfyUI.**
    After the cloning is complete, restart ComfyUI. You should now see a "Manager" button in the menu.

### PyPi Install

[ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager/tree/manager-v4)

**ComfyUI-Manager** is an extension that allows you to easily install, update, and manage custom nodes for ComfyUI.

### Setup

1. Install the manager dependencies:
   ```bash
   pip install -r manager_requirements.txt
   ```

2. Enable the manager with the `--enable-manager` flag when running ComfyUI:
   ```bash
   python main.py --enable-manager
   ```

### Command Line Options

| Flag | Description |
|------|-------------|
| `--enable-manager` | Enable ComfyUI-Manager |
| `--enable-manager-legacy-ui` | Use the legacy manager UI instead of the new UI (requires `--enable-manager`) |
| `--disable-manager-ui` | Disable the manager UI and endpoints while keeping background features like security checks and scheduled installation completion (requires `--enable-manager`) |



### Vanilla Custom Nodes

This fork is fully compatible with ordinary ComfyUI custom nodes from the ecosystem. As long as you install a node's dependencies into your virtual environment and clone it into the `custom_nodes/` directory that ComfyUI is scanning, everything will work.

#### Step 1: Open a Terminal in Your Workspace

Your workspace is the directory where you ran `uv venv` during installation (the one containing your `.venv` folder).

**Windows (PowerShell):**
```powershell
cd ~\Documents\ComfyUI_Workspace
.\.venv\Scripts\Activate.ps1
```

**macOS:**
```shell
cd ~/Documents/ComfyUI_Workspace
source .venv/bin/activate
```

**Linux:**
```shell
cd ~/Documents/ComfyUI_Workspace
source .venv/bin/activate
```

#### Step 2: Create the `custom_nodes` Directory (if it doesn't exist)

```shell
mkdir -p custom_nodes
```

On Windows PowerShell, use:
```powershell
if (!(Test-Path custom_nodes)) { mkdir custom_nodes }
```

#### Step 3: Clone the Custom Node and Install Dependencies

Clone the repository into `custom_nodes/` and install its Python dependencies:

```shell
git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git custom_nodes/ComfyUI-KJNodes
uv pip install -r custom_nodes/ComfyUI-KJNodes/requirements.txt
```

Some nodes may not have a `requirements.txt`. In that case, skip the `uv pip install` step.

#### Step 4: Restart ComfyUI

After cloning and installing dependencies, restart ComfyUI. The new nodes will be available in the node menu.

#### More Examples

```shell
# WAN Video Wrapper
git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git custom_nodes/ComfyUI-WanVideoWrapper
uv pip install -r custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt

# ComfyUI-Manager (also available as an installable, see above)
git clone --depth 1 https://github.com/Comfy-Org/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager
uv pip install -r custom_nodes/ComfyUI-Manager/requirements.txt
```

### Custom Nodes Authored for this Fork

Run `uv pip install "git+https://github.com/owner/repository"`, replacing the `git` repository with the installable custom nodes URL. This is just the GitHub URL.

---

## Programmatic Custom Node Management

This section documents how custom nodes are managed programmatically, covering the ComfyUI-Manager APIs, the node-to-package mapping system, dependency management, and the compatibility mitigations this fork applies.

### Architecture Overview

```
comfy-cli              comfyui_manager REST API       comfyui_manager Python API
(thin wrapper)    -->  POST /v2/manager/queue/task --> UnifiedManager (singleton)
                       GET  /v2/customnode/installed      |
                       GET  /v2/customnode/getmappings    +--> cnr_utils.install_node()  --> CNR API (api.comfy.org)
                                                          +--> git.Repo.clone_from()     --> GitHub
                                                          +--> manager_util.PIPFixer      --> pip/uv
```

`comfy-cli` (`comfy node install <name>`) does not implement installation logic. It shells out to `cm-cli.py` inside the cloned ComfyUI-Manager directory, which calls the same `UnifiedManager` methods documented below.

### Node Identification

Custom nodes are identified by three install types:

| Type | ID Format | Version | Example |
|------|-----------|---------|---------|
| **CNR** (ComfyUI Node Registry) | `node-name` | Semantic version (`1.2.3`) | `comfyui-impact-pack@1.0.0` |
| **Git clone** ("unknown") | `repo-name` | Git commit hash | `ComfyUI-KJNodes@unknown` |
| **Nightly** | `node-name` | `nightly` + commit hash | `comfyui-impact-pack@nightly` |

The `InstalledNodePackage` dataclass (`comfyui_manager.common.node_package`) stores this:

```python
@dataclass
class InstalledNodePackage:
    id: str           # node ID
    fullpath: str     # installation directory
    disabled: bool    # moved to .disabled/
    version: str      # semantic version, commit hash, "unknown", or "nightly"
```

### UnifiedManager Python API

The `UnifiedManager` singleton at `comfyui_manager.glob.manager_core.unified_manager` is the core programmatic interface. It requires ComfyUI's folder paths to be initialized.

#### Installation

```python
from comfyui_manager.glob.manager_core import unified_manager

# Install from ComfyUI Node Registry by name + version
result = unified_manager.cnr_install("comfyui-impact-pack", version_spec="1.0.0")
# result.result == True on success
# result.to_path == "/path/to/custom_nodes/comfyui-impact-pack"

# Install from git URL
result = unified_manager.repo_install(
    url="https://github.com/user/ComfyUI-SomeNodes",
    repo_path="/path/to/custom_nodes/ComfyUI-SomeNodes"
)

# Smart install (async) - resolves CNR vs git, handles enable/disable/version switching
result = await unified_manager.install_by_id(
    node_id="comfyui-impact-pack",
    version_spec="1.0.0",    # or "latest", "nightly", "unknown", None
    channel="default",
    mode="remote"
)
```

`cnr_install` calls `cnr_utils.install_node(node_id, version)` which hits the CNR API at `https://api.comfy.org/nodes/{node_id}/install?version={version}` to get a download URL, downloads the zip, extracts it into `custom_nodes/{node_id}/`, creates a `.tracking` file, then runs `execute_install_script` for pip dependencies and `install.py`.

`repo_install` runs `git.Repo.clone_from(url, repo_path, recursive=True)` then runs post-install scripts.

`install_by_id` is the highest-level method. When `version_spec` is `None`, it auto-resolves: checks if the node is already enabled (skip), disabled (enable it), or absent (install via CNR or git). It handles version switching between CNR/nightly transparently.

#### Version specification

The `resolve_node_spec` method parses version strings:

```python
# Accepts "name@version" format
spec = unified_manager.resolve_node_spec("comfyui-impact-pack@latest")
# Returns: ("comfyui-impact-pack", "1.2.3", True)  -- resolved semantic version

spec = unified_manager.resolve_node_spec("comfyui-impact-pack@nightly")
# Returns: ("comfyui-impact-pack", "nightly", True)

spec = unified_manager.resolve_node_spec("comfyui-impact-pack")
# Returns: ("comfyui-impact-pack", <auto-resolved>, False)
```

When `version_spec` is `"latest"`, it resolves to the concrete semantic version from the CNR map.

#### Uninstallation

```python
result = unified_manager.unified_uninstall("comfyui-impact-pack", is_unknown=False)
# Removes from active_nodes, nightly_inactive_nodes, and cnr_inactive_nodes
# Deletes the installation directory
```

This removes the node from all registries (active, inactive CNR, inactive nightly) and deletes the directory. It refuses to uninstall `comfyui-manager` itself.

#### Enable / Disable / Update

```python
unified_manager.unified_enable("node-id", version_spec="1.0.0")
unified_manager.unified_disable("node-id", is_unknown=False)
unified_manager.unified_update("node-id", version_spec="1.0.0")
unified_manager.cnr_switch_version("node-id", version_spec="2.0.0")
```

Disabling moves the node directory under `custom_nodes/.disabled/`. Enabling moves it back.

#### Return type

All methods return `ManagedResult`:

```python
class ManagedResult:
    action: str       # "install-cnr", "install-git", "uninstall", "enable", "skip", etc.
    result: bool      # True if successful
    msg: str          # error message on failure
    to_path: str      # installation path (for installs)
    target: str       # version spec
```

### REST API

When ComfyUI is running with `--enable-manager`, the manager exposes REST endpoints. All install/uninstall operations go through an async task queue.

#### Install a node

```
POST /v2/manager/queue/task
Content-Type: application/json

{
    "ui_id": "task-123",
    "client_id": "test",
    "kind": "install",
    "params": {
        "id": "comfyui-impact-pack",
        "version": "1.0.0",
        "selected_version": "1.0.0",
        "mode": "remote",
        "channel": "default"
    }
}
```

The `params` field is validated as `InstallPackParams` (Pydantic model in `comfyui_manager.data_models.generated_models`):

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Node ID or `publisher/node-name` |
| `version` | `str` | Semantic version or git commit hash |
| `selected_version` | `str` | `"latest"`, `"nightly"`, or specific version |
| `mode` | `"remote" \| "local" \| "cache"` | Database source |
| `channel` | `"default" \| "recent" \| "legacy" \| ...` | Channel |
| `skip_post_install` | `bool` | Skip post-install scripts |

#### Uninstall a node

```
POST /v2/manager/queue/task
Content-Type: application/json

{
    "ui_id": "task-124",
    "client_id": "test",
    "kind": "uninstall",
    "params": {
        "node_name": "comfyui-impact-pack",
        "is_unknown": false
    }
}
```

#### Other operations

| Kind | Params model | Description |
|------|-------------|-------------|
| `"enable"` | `EnablePackParams(cnr_id=...)` | Enable a disabled node |
| `"disable"` | `DisablePackParams(node_name=..., is_unknown=...)` | Disable a node |
| `"update"` | `UpdatePackParams(node_name=..., node_ver=...)` | Update to latest |
| `"fix"` | `FixPackParams(node_name=..., node_ver=...)` | Re-run dependency install |

#### Query endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v2/customnode/installed` | List all installed custom nodes |
| `GET /v2/manager/queue/status` | Task queue status |
| `GET /v2/manager/version` | Manager version |
| `GET /v2/customnode/getmappings?mode=remote` | Node class_type to package mapping |

### Mapping class_types to Custom Node Packages

ComfyUI-Manager maintains a bundled `extension-node-map.json` that maps git repository URLs to the list of `NODE_CLASS_MAPPINGS` keys (class_types) each package provides:

```json
{
    "https://github.com/user/ComfyUI-SomeNodes": [
        ["NodeClassA", "NodeClassB"],
        {
            "title_aux": "Some Nodes Pack",
            "nodename_pattern": "SomePrefix.*"
        }
    ]
}
```

The first element is the list of class_type strings. The second element contains metadata, including an optional `nodename_pattern` regex for nodes that follow a naming convention.

#### How the mapping works

The `GET /v2/customnode/getmappings` endpoint (`manager_server.py:1356`):

1. Loads `extension-node-map.json` via `get_data_by_mode(mode, 'extension-node-map.json')` (fetched from the channel URL remotely, or read from local cache/bundled file)
2. Calls `map_to_unified_keys()` which converts git-URL keys to CNR node IDs using `unified_manager.get_cnr_by_repo(url)`
3. Builds a set of all known class_types across all packages
4. Compares against `nodes.NODE_CLASS_MAPPINGS.keys()` (currently loaded class_types) to find unaccounted nodes
5. Applies `nodename_pattern` regex matching for missing nodes

#### Workflow dependency resolution

The function at `manager_core.py:2785` resolves which custom nodes a workflow requires:

1. Parses the workflow JSON, extracts all `class_type` values from nodes (skipping `Reroute`, `Note`, and `workflow/` prefixed names)
2. Loads `extension-node-map.json` and builds a **reverse map**: `class_type -> [repo_url]`
3. Applies a `preemption_map` for class_types claimed by core ComfyUI
4. Falls back to `nodename_pattern` regex for unresolved class_types
5. Returns `(used_exts, unknown_nodes)` -- the set of required extensions and any class_types that couldn't be mapped

To build a `class_type -> node_id` lookup for programmatic use:

```python
# After map_to_unified_keys(), the data is: cnr_node_id -> [class_type_list, metadata]
# Invert it:
reverse_map = {}
for node_id, (class_types, metadata) in mapping.items():
    for ct in class_types:
        reverse_map[ct] = node_id
```

### Dependency Management and pip Interception

Custom nodes manage their Python dependencies in several problematic ways. ComfyUI-Manager and this fork both have mitigations.

#### How custom nodes install dependencies

1. **`requirements.txt`** -- Most nodes ship a `requirements.txt`. ComfyUI-Manager reads this during `execute_install_script` and calls `pip install` for each line.
2. **`install.py`** -- Some nodes include an `install.py` script that runs arbitrary code (often calling pip directly).
3. **Runtime pip calls** -- Many nodes call `subprocess.run([sys.executable, "-m", "pip", "install", ...])` or `subprocess.Popen(...)` at import time or at execution time to install missing packages.
4. **`importlib` checks** -- Nodes often use `importlib.util.find_spec()` or try/except imports to check if packages are installed before attempting installation.

#### ComfyUI-Manager's pip safety system

ComfyUI-Manager applies several layers of protection during dependency installation (`manager_core.py:837`, `manager_util.py:407`):

**pip blacklist** -- Packages that must never be installed. Configured at startup in `prestartup_script.py`:
```python
cm_global.pip_blacklist = {'torch', 'torchaudio', 'torchsde', 'torchvision'}
```
Additional entries can be added via `pip_blacklist.list` in the manager's data directory.

**pip downgrade blacklist** -- Packages that must never be downgraded:
```python
cm_global.pip_downgrade_blacklist = [
    'torch', 'torchaudio', 'torchsde', 'torchvision',
    'transformers', 'safetensors', 'kornia'
]
```
If a requirements.txt specifies `transformers<=4.30.0` but a newer version is already installed, the install is skipped.

**pip overrides** -- Package name remapping via `pip_overrides.json`:
```python
# If pkg is in pip_overrides, replace it
def remap_pip_package(pkg):
    if pkg in cm_global.pip_overrides:
        return cm_global.pip_overrides[pkg]
    return pkg
```

**PIPFixer** -- Post-install fixup class (`manager_util.py:407`) that runs after every install. It:
- **Removes the `comfy` PyPI package** if installed (it conflicts with ComfyUI)
- **Rolls back torch changes** -- if a node's dependencies change the torch/torchvision/torchaudio versions, PIPFixer reinstalls the original versions
- **Fixes opencv conflicts** -- multiple opencv variants (`opencv-python`, `opencv-python-headless`, `opencv-contrib-python`, `opencv-contrib-python-headless`) cannot coexist at different versions. PIPFixer detects this and upgrades all installed variants to the highest version present.

**Requirements processing** -- `execute_install_script` reads `requirements.txt` with `robust_readlines` (handles encoding issues via `chardet`), applies `remap_pip_package` to each line, checks `is_blacklisted` (covers both blacklist and downgrade-blacklist), then calls `pip install` via `make_pip_cmd` (which uses `uv` when available).

#### This fork's pip interception

This fork adds additional protection in `comfy_compatibility/vanilla.py` to prevent custom nodes from installing packages at import time and execution time.

**`patch_pip_install_subprocess_run`** -- Patches `subprocess.run` to intercept calls matching the pattern `[sys.executable, '-s', '-m', 'pip', 'install', <package>]` (used by ComfyUI-Easy-Use and similar nodes). Returns a mock result with `returncode=0`.

**`patch_pip_install_popen`** -- Patches `subprocess.Popen` to intercept calls matching `[sys.executable, '-m', 'pip', 'install', ...]`. Returns a mock Popen instance with empty stdout/stderr. Has a special exception for `nunchaku` which is allowed through.

These patches are applied in two places:
- **At import time** (`vanilla_node_importing.py:164`): When `block_runtime_package_installation` is set in the configuration, the patches are active during `_exec_mitigations` for specific problematic nodes (comfyui-manager, comfyui_ryanonyheinside, comfyui-easy-use, comfyui_custom_nodes_alekpet).
- **At execution time** (`vanilla.py:255`): `vanilla_environment_node_execution_hooks` applies the patches during prompt execution when `block_runtime_package_installation` is enabled.

### Vanilla Environment Compatibility Layer

This fork restructures ComfyUI as an installable Python package (`comfy.*`), but vanilla custom nodes expect top-level modules like `nodes`, `folder_paths`, `execution`, `server`, etc. The compatibility layer in `comfy_compatibility/vanilla.py` bridges this gap.

`prepare_vanilla_environment()` (called once during startup) injects shims into `sys.modules`:

| Expected by custom nodes | Actual location in this fork |
|---|---|
| `import nodes` | `comfy.nodes.base_nodes` (via `_NodeShim`) |
| `import folder_paths` | `comfy.cmd.folder_paths` |
| `import execution` | `comfy.cmd.execution` |
| `import server` | `comfy.cmd.server` |
| `import model_patcher` | `comfy.model_patcher` |
| `import cuda_malloc` | `comfy.cmd.cuda_malloc` |
| `import latent_preview` | `comfy.cmd.latent_preview` |
| `import comfyui_version` | Synthetic module with `__version__` |
| `comfy_extras.*` | Re-exported under shortened names |

The `_NodeClassMappingsShim` provides a lazy, reference-counted view of `NODE_CLASS_MAPPINGS` that returns all currently loaded nodes when activated (during import/execution) or just base nodes when deactivated.

The `_PromptServerStub` (`vanilla_node_importing.py:53`) provides a stub `PromptServer.instance` so nodes that call `server.PromptServer.instance.send_sync()` during import don't crash.

### Manager Integration

`comfy/manager_integration.py` provides the bridge between this fork's startup sequence and comfyui_manager:

| Function | Purpose |
|---|---|
| `init_manager(args)` | Import and initialize comfyui_manager if `--enable-manager` is set |
| `prestartup()` | Run manager's prestartup script |
| `start()` | Start manager UI endpoints |
| `get_middleware()` | Get manager's aiohttp middleware |
| `should_be_disabled(module_path)` | Check if manager policy blocks a specific node |

`should_be_disabled` is called during node loading (`vanilla_node_importing.py:94,262`) to respect manager's enable/disable state for each custom node directory.

### Import Process for Vanilla Custom Nodes

The full loading sequence is in `comfy/nodes/vanilla_node_importing.py`:

1. **`mitigated_import_of_vanilla_custom_nodes()`** -- Entry point. Calls `prepare_vanilla_environment()`, collects `custom_nodes` paths, then runs prestartup and import phases.

2. **Prestartup phase** (`_vanilla_load_importing_execute_prestartup_script`):
   - Iterates over every directory in `custom_nodes/`
   - Skips `.disabled` directories and nodes blocked by manager policy
   - For ComfyUI-Manager specifically: patches its `security_check` to fail gracefully, suppresses its logging handler, and sets `COMFYUI_PATH`/`COMFYUI_FOLDERS_BASE_PATH` env vars
   - Executes each node's `prestartup_script.py` if present

3. **Import phase** (`_vanilla_load_custom_nodes_2`):
   - Iterates over every directory/file in `custom_nodes/`
   - Skips disabled nodes, blacklisted nodes, and nodes blocked by manager policy
   - For each module, calls `_vanilla_load_custom_nodes_1` which:
     - Imports the module via `importlib`
     - Applies `_exec_mitigations` for known problematic nodes (patches `folder_paths.__file__`, optionally blocks pip installs)
     - Extracts `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`, and `WEB_DIRECTORY`
   - Records import times for diagnostics

### Workspace Compatibility

`comfy_compatibility/workspace.py` handles the case where ComfyUI is run from a cloned upstream workspace directory (where `nodes.py`, `comfy/`, `comfy_extras/` etc. exist as bare directories without `__init__.py`). It:

1. Detects if the workspace has a `nodes.py` (indicating an upstream-style layout)
2. Creates `__init__.py` files in all directories containing `.py` files under `comfy/`, `comfy_extras/`, `comfy_execution/`, `comfy_api/`, and `comfy_config/`
3. Adds these files to `.git/info/exclude` to avoid polluting git status
4. Restarts the process so the new packages are importable

### Import Order Control

`comfy_compatibility/imports.py` provides `ImportContext`, a context manager that temporarily overrides Python's import resolution order for specific modules. It inserts a custom `PathFinder` into `sys.meta_path` that controls whether modules are resolved from the main script directory, the current working directory, or site-packages first. This is used to ensure the correct version of ambiguous modules (like `comfy`) is imported when both a workspace copy and a pip-installed copy exist.

### Directory Layout

```
custom_nodes/
    some-node/                    # active CNR or git-clone node
        __init__.py
        .tracking                 # present for CNR nodes (lists extracted files)
        pyproject.toml            # CNR nodes have version info here
        requirements.txt
        install.py                # optional post-install script
    .disabled/
        another-node/             # disabled node (moved here by manager)
```

### Testing Custom Node Compatibility

The existing test infrastructure at `tests/unit/manager_test/` shows the pattern:

1. Create a temporary `base_directory` with standard subdirectories (`models/`, `custom_nodes/`, `input/`, `output/`, `temp/`, `user/`)
2. Build a `Configuration` object with `enable_manager=True`, `cpu=True`, and a free port
3. Boot a ComfyUI server via `comfy_background_server_from_config(config)`
4. Use the REST API to install nodes, verify endpoints, and check behavior
5. The server runs in a separate process with full isolation

---

## Authoring Custom Nodes

These instructions will allow you to quickly author installable custom nodes.

#### Using `pyproject.toml` for projects with existing `requirements.txt`

Suppose your custom nodes called `my_comfyui_nodes` has a folder layout that looks like this:

```
__init__.py
some_python_file.py
requirements.txt
LICENSE.txt
some_directory/some_code.py
```

First, add an `__init__.py` to `some_directory`, so that it is a Python package:

```
__init__.py
some_python_file.py
requirements.txt
LICENSE.txt
some_directory/__init__.py
some_directory/some_code.py
```

Then, if your `NODE_CLASS_MAPPINGS` are declared in `__init__.py`, use the following as a `pyproject.toml`, substituting your actual project name:

**pyproject.toml**

```toml
[project]
name = "my_comfyui_nodes"
description = "My nodes description."
version = "1.0.0"
license = { file = "LICENSE.txt" }
dynamic = ["dependencies"]

[project.urls]
Repository = "https://github.com/your-github-username/my-comfyui-nodes"
#  Used by Comfy Registry https://comfyregistry.org

[tool.comfy]
PublisherId = "your-github-username"
DisplayName = "my_comfyui_nodes"
Icon = ""

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["my_comfyui_nodes", "my_comfyui_nodes.some_directory"]
package-dir = { "my_comfyui_nodes" = ".", "my_comfyui_nodes.some_directory" = "some_directory" }

[tool.setuptools.dynamic]
dependencies = { file = ["requirements.txt"] }

[project.entry-points."comfyui.custom_nodes"]
my_comfyui_nodes = "my_comfyui_nodes"
```

Observe that the directory should now be listed as a package in the `packages` and `package-dir` statement.

#### Using `setup.py`

Create a `requirements.txt`:

```
comfyui
```

Observe `comfyui` is now a requirement for using your custom nodes. This will ensure you will be able to access `comfyui` as a library. For example, your code will now be able to import the folder paths using `from comfyui.cmd import folder_paths`. Because you will be using my fork, use this:

```
comfyui @ git+https://github.com/hiddenswitch/ComfyUI.git
```

Additionally, create a `pyproject.toml`:

```
[build-system]
requires = ["setuptools", "wheel", "pip"]
build-backend = "setuptools.build_meta"
```

This ensures you will be compatible with later versions of Python.

Finally, move your nodes to a directory with an empty `__init__.py`, i.e., a package. You should have a file structure like this:

```
# the root of your git repository
/.git
/pyproject.toml
/requirements.txt
/mypackage_custom_nodes/__init__.py
/mypackage_custom_nodes/some_nodes.py
```

Finally, create a `setup.py` at the root of your custom nodes package / repository. Here is an example:

**setup.py**

```python
from setuptools import setup, find_packages
import os.path

setup(
    name="mypackage",
    version="0.0.1",
    packages=find_packages(),
    install_requires=open(os.path.join(os.path.dirname(__file__), "requirements.txt")).readlines(),
    author='',
    author_email='',
    description='',
    entry_points={
        'comfyui.custom_nodes': [
            'mypackage = mypackage_custom_nodes',
        ],
    },
)
```

All `.py` files located in the package specified by the entrypoint with your package's name will be scanned for node class mappings declared like this:

**some_nodes.py**:

```py
from comfy.nodes.package_typing import CustomNode


class Binary_Preprocessor(CustomNode):
    ...


NODE_CLASS_MAPPINGS = {
    "BinaryPreprocessor": Binary_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BinaryPreprocessor": "Binary Lines"
}
```

These packages will be scanned recursively.

Extending the `comfy.nodes.package_typing.CustomNode` provides type hints for authoring nodes.

## Adding Custom Configuration

Declare an entry point for configuration hooks in your **setup.py** that defines a function that takes and returns an
`configargparser.ArgParser` object:

**setup.py**

```python
setup(
    name="mypackage",
    ...
entry_points = {
    'comfyui.custom_nodes': [
        'mypackage = mypackage_custom_nodes',
    ],
    'comfyui.custom_config': [
        'mypackage = mypackage_custom_config:add_configuration',
    ]
},
)
```

**mypackage_custom_config.py**:

```python
import configargparse


def add_configuration(parser: configargparse.ArgParser) -> configargparse.ArgParser:
    parser.add_argument("--openai-api-key",
                        required=False,
                        type=str,
                        help="Configures the OpenAI API Key for the OpenAI nodes", env_var="OPENAI_API_KEY")
    return parser

```

You can now see your configuration option at the bottom of the `--help` command along with hints for how to use it:

```shell
$ comfyui --help
usage: comfyui.exe [-h] [-c CONFIG_FILE] [--write-out-config-file CONFIG_OUTPUT_PATH] [-w CWD] [-H [IP]] [--port PORT]
                   [--enable-cors-header [ORIGIN]] [--max-upload-size MAX_UPLOAD_SIZE] [--extra-model-paths-config PATH [PATH ...]]
...
                   [--openai-api-key OPENAI_API_KEY]

options:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config CONFIG_FILE
                        config file path
  --write-out-config-file CONFIG_OUTPUT_PATH
                        takes the current command line args and writes them out to a config file at the given path, then exits
  -w CWD, --cwd CWD     Specify the working directory. If not set, this is the current working directory. models/, input/, output/ and other
                        directories will be located here by default. [env var: COMFYUI_CWD]
  -H [IP], --listen [IP]
                        Specify the IP address to listen on (default: 127.0.0.1). If --listen is provided without an argument, it defaults to
                        0.0.0.0. (listens on all) [env var: COMFYUI_LISTEN]
  --port PORT           Set the listen port. [env var: COMFYUI_PORT]
...
  --distributed-queue-name DISTRIBUTED_QUEUE_NAME
                        This name will be used by the frontends and workers to exchange prompt requests and replies. Progress updates will be
                        prefixed by the queue name, followed by a '.', then the user ID [env var: COMFYUI_DISTRIBUTED_QUEUE_NAME]
  --external-address EXTERNAL_ADDRESS
                        Specifies a base URL for external addresses reported by the API, such as for image paths. [env var:
                        COMFYUI_EXTERNAL_ADDRESS]
  --openai-api-key OPENAI_API_KEY
                        Configures the OpenAI API Key for the OpenAI nodes [env var: OPENAI_API_KEY]
```

You can now start `comfyui` with:

```shell
uv run --no-sync comfyui --openai-api-key=abcdefg12345
```

or set the environment variable you specified:

```shell
export OPENAI_API_KEY=abcdefg12345
uv run --no-sync comfyui
```

or add it to your config file:

**config.yaml**:

```txt
openapi-api-key: abcdefg12345
```

```shell
comfyui --config config.yaml
```

Since `comfyui` looks for a `config.yaml` in your current working directory by default, you can omit the argument if
`config.yaml` is located in your current working directory:

```shell
uv run --no-sync comfyui
```

Your entry point for adding configuration options should **not** import your nodes. This gives you the opportunity to
use the configuration you added in your nodes; otherwise, if you imported your nodes in your configuration entry point,
the nodes will potentially be initialized without any configuration.

Access your configuration from `cli_args`:

```python
from comfy.cli_args import args
from comfy.cli_args_types import Configuration
from typing import Optional


# Add type hints when accessing args
class CustomConfiguration(Configuration):
    def __init__(self):
        super().__init__()
        self.openai_api_key: Optional[str] = None


args: CustomConfiguration


class OpenAINode(CustomNode):
    ...

    def execute(self):
        openai_api_key = args.open_api_key
```
