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

**IMPORTANT:** After fixing imports and other merge issues, run **both linters** on the **entire codebase** at least once, not just the changed files. Upstream changes may introduce issues in unchanged files due to cross-module dependencies, and CI checks exit codes — pre-existing warnings that don't matter for the score still fail the job.

```bash
ruff check comfy/ comfy_extras/ comfy_api/ comfy_api_nodes/ comfy_compatibility/ comfy_execution/
pylint -j 0 comfy/ comfy_extras/ comfy_api/ comfy_api_nodes/
```

Ruff handles standard Python lint rules; pylint runs only the custom checkers in `tests/*_checker.py`, including merge hygiene checks for version sync, direct Transformers imports, text encoder config forwarding, model inference coverage, packaged blueprints, CUDA allocator defaults, and workflow conversion cache format. Run them **raw** — never pipe through `head`, `tail`, or `grep`. Filtering hides the warnings CI will fail on. Both must exit 0 before the merge is complete.

See [Linting Guidelines](linting.md) for custom rules, the ruff/pylint split, and pragma-comment conventions.

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

## Entrypoint / `main.py` startup side effects

Upstream's real entrypoint is the root `main.py`: it runs module-level startup
code (device init, allocator setup, monkeypatches) as a side effect of import,
before the server starts. **Our root `main.py` is a thin shim** — it only calls
`comfy.cmd.main._start_comfyui`. The fork's actual startup runs through
`comfy/cmd/main_pre.py` and `comfy/component_model/setup.py::setup_post_torch`,
invoked by the Typer CLI.

This means: **any startup side effect upstream adds to or changes in root
`main.py` must be mirrored into `setup_post_torch` (or `main_pre.py`).** A git
merge of `main.py` will look clean — the lines apply or our shim absorbs them —
but the behavior is silently lost because our process never executes upstream's
`main.py` body.

This already bit us once, expensively: upstream PR #14116 (`e154da83`,
"Threaded Loader performance fixes (+ Aimdo 0.4.6)") reworked how `main.py`
activates dynamic VRAM (comfy-aimdo). The activation moved into
`comfy/aimdo_integration.py`, which only takes effect when something imports it.
Upstream's `main.py` imported it; our `setup_post_torch` did not. So from
**v0.23.0 onward dynamic VRAM was silently dormant** — `aimdo_allocator` stayed
`None`, `get_model_patcher_class()` always returned the legacy `ModelPatcher`,
and every release ran without streaming/offload. It was active through v0.22.x
(when `main.py` still called `comfy_aimdo.control.init()` directly) and went dark
at the merge, undetected because nothing tests `setup_post_torch`'s effects.

When merging `main.py`, check for module-level startup behavior and mirror it:

```bash
# What runs at import in upstream main.py (side-effecting calls before server start)
git show <upstream>:main.py | grep -nE "^\s*(import comfy_aimdo|comfy_aimdo\.|.*\.init\(|from .* import .*integration|set_per_process|cuda\.init)"

# What our startup path actually runs
grep -nE "aimdo_integration|comfy_aimdo|\.init\(|cuda\.init" comfy/cmd/main_pre.py comfy/component_model/setup.py
```

Specifically, dynamic VRAM must be activated by importing `comfy.aimdo_integration`
from `setup_post_torch` (the module self-gates on torch >= 2.8 and
`--disable-dynamic-vram`). If that import is missing, dynamic VRAM is off no
matter what `main.py` says.

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

If a merge was already started with the wrong settings, abort and retry so Git recomputes the merge. Do not continue a merge that was started without directory rename detection:

```bash
git merge --abort
git pull --no-commit --no-ff comfyui master
```

If upstream both moved and heavily edited files, you can retry with a lower rename similarity threshold:

```bash
git pull --no-commit --no-ff -s ort -Xfind-renames=30% comfyui master
```

Use the lower threshold only when needed; it can create false rename matches.

## Merge Workflow

Start from the local fork branch that should receive upstream, then recreate `develop` from it:

```bash
git branch -D develop
git checkout -b develop
git fetch comfyui master
```

Use a no-commit merge so conflict resolution and validation happen before the merge commit is created:

```bash
git pull --no-commit --no-ff -s ort -Xfind-renames=30% comfyui master
```

Before resolving conflicts, read the previous upstream merge sequences closely. Do this even when the conflict hunk looks obvious: the follow-up commits show the kinds of fixes that are easy to miss during the merge commit itself.

For the `0.16 -> 0.18` era merges, read these older examples too. They are especially useful for asset routes, workflow conversion, package-layout tests, custom-node compatibility, and context propagation:

- `5c6c95ea` - merge commit, `Merge branch 'master' of github.com:Comfy-Org/ComfyUI into develop`
- `de279e6e` - first follow-up, `Fix upstream merge follow-up issues`
- `bd4c2fd3` - package/test adaptation, `fix tests`
- `42a4902e` - file lock compatibility, `fix file lock`
- `fca287e2` - workflow conversion/model fix, `Merged`
- `7fe1d305` - package-layout/import/model/test cleanup, `fix merge issues`
- `154174a9` - lint cleanup, `fix linting issues`
- `dc09f472` - API import correction, `fix comfy_api import`
- `306f7790` - frontend workflow conversion parity, `fix workflow parity`
- `7d3e6ed2` - workflow conversion rewrite and Playwright cache refresh, `update workflow conversion code`
- `1f83d46c` - merge commit, `Merge branch 'master' of github.com:Comfy-Org/ComfyUI into develop`
- `49a0557f` - test import migration, `Fix test imports: from app. -> from comfy.app.`
- `219f5225` - test patch-target migration, `Fix mock patch paths in tests: app. -> comfy.app., folder_paths -> comfy.cmd.folder_paths`
- `f8731ae1` - folder path/context test migration, `Fix test mocks: use FolderNames/context, fix patch paths, remove missing utils.install_util`
- `1cb6ef78` - asset test restructure, `Restructure asset tests: use comfy_background_server_from_config, fix migration import`
- `8af843ce` - relocated Alembic path fix, `Fix alembic paths in migration test`
- `fec19d27` - thread context fix, `Fix seeder thread context: copy contextvars so folder_paths resolve correctly`
- `acff3997` - first context thread abstraction, `Add ContextThread helper for contextvars-aware threading`
- `7075f8da` - prompt-worker context propagation, `DRY context propagation: use ContextThread in main.py and seeder`
- `fe6ffadc` - executor-based context propagation, `Use ContextVarExecutor in seeder, fix asset hash collision in missing_sync test`
- `f62e979f` - final context propagation pattern, `Use ContextVarExecutor everywhere, remove unused ContextThread`
- `8923e655` - merge commit, `Merge branch 'master' of github.com:Comfy-Org/ComfyUI into develop`
- `298be37f` - merge-garbled code fix, `Fix merge-garbled indentation in utils.py, demote comfy_kitchen MXFP8 warning to debug`
- `d53b6414` - merge commit, `Merge branch 'master' of github.com:Comfy-Org/ComfyUI into develop`

For the `0.19 -> 0.20` style merge, use these commits as the baseline example:

- `ebdc4945` - merge commit, `Merge ComfyUI upstream master`
- `d58a7150` - follow-up move commit, `Move upstream extra nodes into package layout`
- `46ce90fd` - docs follow-up, `Document upstream merge workflow`
- `cddc27e9` - remaining follow-up cleanup, `Complete upstream merge follow-ups`

For the later upstream merge around `0.21`, also read:

- `921070e3` - merge commit, `Merge branch 'master' of github.com:Comfy-Org/ComfyUI into develop`
- `b9da815a` - move commit, `Move upstream extra nodes into package layout`
- `02420c71` - package-layout/import/model-management cleanup, `Complete upstream merge follow-ups`
- `9af07f57` - missed package import fix for PiD, `Fix PiD node helper import`
- `b5eb2276` - workflow CLI and converter regression fix, `Fix workflow run path and bypass type matching`
- `95a7276a` - test/inference/custom-node/model-support fixes, `Complete upstream merge test fixes`
- `d4f6d4bc` - GPU setting heuristic fix, `Ignore small GPU helper processes for novram`

Also read these additional post-`0.20` follow-up commits; they capture fixes that only showed up after deeper testing:

- `cc67e0b2` - model detection and known-repo coverage, `Add HiDream O1 known repos and detection test`
- `4bec5e9d` - sharded Diffusers loading plus inference workflow, `Add HiDream O1 inference workflow test`
- `6651d186` - conditioning wrapper bug from name shadowing, `Fix HiDream O1 conditioning wrappers`
- `c762a0dd` - shadowing lint enablement and cleanup, `Enable shadowing lint checks`
- `516af487` - unit coverage for shadowing cleanup touchpoints, `Add coverage for shadowing cleanup touchpoints`
- `7064da2a` - quantization fallback bug and Flux2 workflow, `Preserve fp8 scales when disabling fp8 kernels`
- `e0813036` - coverage gate for supported model classes, `Track supported model inference coverage`
- `48eecadf` - linked workflow parameter role resolution, `Resolve linked workflow parameter roles`
- `2c8f609a` - workflow quantity and seed expansion, `Add workflow quantity seed expansion`
- `9f53c9f0` - explicit disable flag precedence, `Honor disabled dynamic VRAM flag`

Read the whole diff of the merge commit and the few commits after it, not only the conflict hunks. For model-heavy conflicts, also read the whole upstream file and the whole fork file when the surrounding context matters. This is especially important for `model_management.py`, `model_patcher.py`, `model_base.py`, `sd.py`, `supported_models.py`, `ops.py`, `lora.py`, sampler files, and anything touching dynamic VRAM or model loading. The typical pattern is:

1. The merge commit contains upstream code plus direct conflict resolutions.
2. The next commit moves upstream root files into the fork package layout.
3. Later commits adapt imports, tests, docs, and fork-specific cleanup.

Do not guess from one conflict hunk when the file contains model-loading or memory-management behavior. Read enough of both sides to understand what upstream added, what the fork already changed, and which behavior must be preserved in the merge commit versus a follow-up commit.

Useful review commands:

```bash
git show --stat --summary --find-renames 921070e3 b9da815a 02420c71 9af07f57 b5eb2276 95a7276a d4f6d4bc
git show --name-status --find-renames 921070e3 b9da815a 02420c71 9af07f57 b5eb2276 95a7276a d4f6d4bc
git show --cc --patch 921070e3 -- comfy/model_management.py comfy/model_patcher.py comfy/model_base.py comfy/sd.py comfy/supported_models.py comfy/ops.py comfy/lora.py
git show --patch --find-renames 02420c71 95a7276a b5eb2276 9af07f57 d4f6d4bc
git show --patch --find-renames cc67e0b2 4bec5e9d 6651d186 c762a0dd 516af487 7064da2a e0813036 48eecadf 2c8f609a 9f53c9f0
```

Do the same for the older sequences:

```bash
git show --stat --summary --find-renames 5c6c95ea de279e6e bd4c2fd3 7fe1d305 306f7790 7d3e6ed2 1f83d46c 49a0557f 219f5225 f8731ae1 1cb6ef78 fec19d27 fe6ffadc f62e979f 8923e655 298be37f d53b6414
git show --cc --patch 5c6c95ea 1f83d46c 8923e655 d53b6414 -- comfy/model_management.py comfy/model_patcher.py comfy/model_base.py comfy/sd.py comfy/supported_models.py comfy/ops.py comfy/sample.py comfy/samplers.py comfy/app/assets comfy/component_model/workflow_convert.py
git show --patch --find-renames de279e6e bd4c2fd3 7fe1d305 306f7790 7d3e6ed2 49a0557f 219f5225 f8731ae1 1cb6ef78 fec19d27 fe6ffadc f62e979f 298be37f
git show --stat --summary --find-renames ebdc4945 d58a7150 46ce90fd cddc27e9
git show --cc --patch ebdc4945 -- comfy/model_management.py comfy/model_patcher.py comfy/model_base.py comfy/sd.py comfy/supported_models.py comfy/ops.py comfy/lora.py
git show --patch --find-renames cddc27e9 d58a7150
```

If the output is large, page through it instead of sampling only the first screen:

```bash
git show --patch --find-renames 95a7276a -- tests/inference tests/custom_nodes comfy/model_downloader.py | less
```

Resolve conflicts in the merge first. Keep directory rename detection enabled and let Git place files under renamed directories where it can. For delete/modify conflicts where this fork intentionally deleted upstream files, confirm the deletion is still intentional and use `git rm`.

Before committing the merge, run all of these checks:

```bash
git diff --name-only --diff-filter=U
git ls-files -u
rg -n '^(<<<<<<<|=======|>>>>>>>)' --glob '!**/tokenizer.json' --glob '!**/merges.txt' --glob '!**/*.model'
git diff --check
python3 -m py_compile <conflict-touched-python-files>
```

Also run focused semantic checks for common merge mistakes:

```bash
rg -n 'comfy\.' <files converted from upstream absolute imports>
rg -n 'intel_extension_for_pytorch|ipex|disable_ipex' comfy pyproject.toml
rg -n 'database-url|disable-assets-autoscan' comfy/cli_args.py comfy/cmd/cli.py
```

Then commit in this order:

1. Commit the upstream merge and conflict resolutions only.
2. In a second commit, move files that Git could not relocate automatically with `git mv`.
3. In later commits, make import fixes, tests, docs, or fork-specific cleanup.

Do not combine the merge and file moves into one commit. Keeping the move commit separate preserves useful history and makes later upstream merges less painful.

### Packaged Blueprints

This fork packages global blueprint subgraphs under `comfy.blueprints`, not the upstream root `blueprints/` directory. When upstream adds or updates root blueprints during a merge:

1. Move new or changed blueprint assets into `comfy/blueprints/`.
2. Preserve Git rename history for existing files by using `git mv` where possible.
3. Leave blueprint loading code pointed at `importlib.resources.files("comfy.blueprints")`, not a repository-relative path.
4. Verify the move commit appears as renames before committing or pushing.

Example commands:

```bash
git mv blueprints/* comfy/blueprints/
git mv blueprints/.glsl/* comfy/blueprints/.glsl/
rmdir blueprints/.glsl blueprints

git status --short
git diff --stat --find-renames
git diff --summary --find-renames
```

If shell globbing misses dotfiles or the upstream directory layout changes, inspect both directories manually before removing the root directory:

```bash
find blueprints comfy/blueprints -maxdepth 2 -type f | sort
```

The expected diff shape for existing blueprint files is `R100 blueprints/<name> comfy/blueprints/<name>`. New upstream blueprint files may appear as additions under `comfy/blueprints/`, but they should not remain under root `blueprints/`.

After moving blueprints, run the package-data and subgraph-manager tests:

```bash
uv run python -m pytest -q tests/unit/test_blueprints_package_data.py
```

## Common Follow-Up Fixes From Recent Merges

The recent merge history shows recurring work that should be expected, not treated as surprising one-off cleanup.

### Detailed Diff Observations

The examples below are from the reviewed merge and follow-up commits. Use them as models for the kind of whole-diff reading that should happen before closing an upstream merge.

- Upstream model additions usually arrive as several disconnected-looking file edits, but this fork needs them turned into a complete packaged workflow path. HiDream O1 is a good example: upstream support touched model detection, Diffusers loading, text/conditioning code, and latent/decode nodes. The fork follow-ups added known Hugging Face repos for `HiDream-ai/HiDream-O1-Image` and `HiDream-ai/HiDream-O1-Image-Dev`, added a minimal state-dict detection test for keys such as `t_embedder1.mlp.0.weight`, `x_embedder.proj1.weight`, and `visual.deepstack_merger_list.0.weight`, verified that extra visual keys are stripped during UNet state-dict processing, and added a one-step inference workflow using `DiffusersLoader`, `EmptyHiDreamO1LatentImage`, `VAEDecode`, and `SaveImage`.
- The HiDream O1 loader path also showed why reading the full loader diff matters. The fix was not just an import adjustment: `DiffusersLoader.load_checkpoint()` had to detect repos containing a root `model.safetensors.index.json` and route them through `sd.load_checkpoint_guess_config(...)[:3]` instead of `diffusers_load.load_diffusers(...)`. That preserves upstream sharded checkpoint behavior while still returning the tuple shape expected by this fork's node path. The unit test proves both the index-file routing and dtype propagation through `model_options`.
- Several bugs were caused by name shadowing introduced while adapting upstream code into this package layout. In HiDream O1 conditioning, a local variable named `conds` shadowed the imported `conds` module, so later references to `conds.CONDConstant` and `conds.CONDRegular` failed at runtime. The follow-up renamed the local value to `extra_cond_values`. This is the pattern that justified enabling ruff `A` and `PLR1704`, plus adding the AST-based import-shadowing test that looks for local bindings which later get used as attribute bases.
- The shadowing cleanup was broad because upstream-style variable names collided with this fork's package imports after relocation. Examples included `dir` becoming `extension_dir`, `tuple` becoming `address_tuple`, `sentencepiece_model_pb2 as model` becoming `sentencepiece_model`, `model_sampling` becoming `model_sampling_module`, `rope` becoming `rope_fn`, `filter` becoming `filter_kernel` or `should_quantize`, and `model_patcher` / `clip_vision` becoming explicit module aliases. These are not cosmetic changes when import names are later dereferenced.
- Quantization fixes require reading the whole operator and model-downloader context. One follow-up moved Flux2 klein FP8 filenames such as `flux-2-klein-4b-fp8.safetensors`, `flux-2-klein-9b-fp8.safetensors`, and `flux-2-klein-9b-kv-fp8.safetensors` into `KNOWN_UNET_MODELS` from the wrong model-family bucket. The same fix removed fallback logic that popped FP8 qconfig metadata like `weight_scale` when FP8 kernels were disabled. Disabled FP8 compute still needs the quantized tensor and scale metadata preserved; otherwise the fallback full-precision matmul path silently loses the calibrated weights.
- Supported-model inference coverage was added because upstream can register a new model class without any local end-to-end proof that this fork can download, load, condition, sample, and serialize output for it. `tests/inference/test_supported_model_coverage.py` maps each `supported_models.models` class to representative workflows and fails when a supported class is missing coverage, when coverage references an unknown class, or when a listed workflow file does not exist.
- Workflow parameter handling changed in ways that are easy to miss if only prompt execution is tested. The fork added linked-input role resolution so CLI parameters can follow primitive nodes through switch nodes. For example, a `--steps` override must update the active linked primitive behind a `ComfySwitchNode`, not blindly update every possible primitive or skip linked widgets entirely. The fix introduced helpers to resolve literal value sources and linked input sources, then tests verified that the active `full_steps` value changes while inactive `fast_steps` remains untouched.
- Workflow quantity support affected UI workflows, API workflows, submit, convert, and seed behavior. The fork added `Configuration.quantity`, Typer `--quantity`, validation that quantity is at least 1, `apply_ui_seed_quantity()` for frontend graphs including subgraphs, definitions, and legacy `extra.groupNodes`, and `expand_workflow_quantity()` for API and UI prompts. It respects frontend `control_after_generate` modes such as `fixed`, `randomize`, `increment`, and `decrement`, wraps at `0xffffffffffffffff`, and makes `workflows submit` and `workflows convert` produce multiple seeded prompts when requested.
- GPU defaults are another place where upstream behavior and fork heuristics diverge. A follow-up made `enables_dynamic_vram()` honor `disable_dynamic_vram` even when `enable_dynamic_vram=True`, because explicit disable flags must win over convenience flags. Another fixed local workstation behavior so tiny desktop GPU clients such as `gnome-remote-desktop-daemon` and `steamwebhelper` do not trigger `--novram`; only material non-Comfy GPU memory use should be treated as a competing process. The `main_pre` startup path also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` only when the variable is missing or blank and the run is not an XPU/oneAPI run; injecting that CUDA allocator setting into Intel XPU causes torch XPU `linear`/`conv2d` failures such as `RuntimeError: could not create a memory`.
- CLI workflow execution had a configuration-order regression that was invisible from the merge conflict alone. Shared setup moved into `_run_workflow_cli()` so both `comfyui run-workflow` and `comfyui workflows run` establish the execution context before importing code that reads `comfy.cli_args.args`. Without that ordering, flags such as `--novram`, `--lowvram`, `--fast`, model paths, and workflow options can silently fall back to defaults.
- The `0.16 -> 0.18` asset merge showed that upstream service code usually cannot be pasted into this fork without route gating and package relocation. The merge brought in a much larger asset stack: upload parsing, `AssetReference` services, tag queries, seeder/scanner services, and `/api/assets` routes. The fork adaptation gated routes behind an assets-enabled flag, made disabled routes return `503 SERVICE_DISABLED`, converted route handlers to call service-layer functions such as `list_assets_page()`, `get_asset_detail()`, `upload_from_temp_path()`, and `resolve_asset_for_download()`, and sanitized dangerous download MIME types with `X-Content-Type-Options: nosniff`.
- The same asset merge exposed compatibility shims that must be preserved for frontend and API callers. Older code still expected names such as `asset_info_id`, `get_asset_tags()`, `add_tags_to_asset_info()`, `remove_tags_from_asset_info()`, `ingest_fs_asset()`, and `set_asset_info_preview()`, so the follow-up added wrappers around the newer `AssetReference` service and query functions. During merges, do not remove these shims just because upstream renamed the internal concept; first check whether fork REST APIs, tests, or frontend routes still use the old public name.
- The asset tests are a good example of how upstream tests have to be rebuilt around this fork's execution context. Follow-ups changed `from app...` imports to `from comfy.app...`, rewrote mock targets from `folder_paths...` to `comfy.cmd.folder_paths...`, replaced raw folder-path monkeypatches with `FolderNames` plus `context_folder_names_and_paths`, moved upstream `tests-unit/seeder_test/test_seeder.py` into `tests/unit/seeder_test/test_seeder.py`, and used `comfy_background_server_from_config()` for REST-facing asset tests instead of duplicating subprocess setup.
- Background threads were a recurring hidden failure in the older merges. The asset seeder and prompt worker initially used plain `threading.Thread`, which lost `contextvars` and made folder paths resolve to defaults. The final follow-up replaced those with `ContextVarExecutor`, so work submitted from server or CLI contexts preserves configuration, folder paths, and execution state. Any future upstream `threading.Thread(...)` addition should be treated as suspicious until context propagation is verified.
- The 0.18 cache merge added pluggable cache-provider behavior into execution caching. The merge had to thread `enable_providers=True` through output caches such as `HierarchicalCache`, `LRUCache`, and `RAMPressureCache`, while leaving object caches alone. This kind of change crosses `comfy/cmd/execution.py`, `comfy_execution/caching.py`, and `comfy_execution/cache_provider.py`; test it with both unit cache-provider tests and a real workflow, because a broken cache can look like nondeterministic node execution.
- Several older model/memory fixes were dtype and device preservation issues, not just import fixes. VAE decode/encode paths switched from unconditional `.float()` to `vae_output_dtype()` and preallocated decode output buffers for chunked IO. Sampler paths forced noise and latent inputs to `torch.float32` at sampling time, then moved results back to the configured intermediate dtype/device. A later multigpu fix updated `LoadedModel.device` when `_switch_parent()` swaps a clone back to its parent patcher. These are examples where whole-file review is required because a one-line upstream change can violate this fork's memory and dtype invariants.
- Older workflow-conversion work relied on Playwright parity fixtures, not just hand-written unit cases. The follow-ups refreshed `tests/unit/playwright_cache/<frontend+templates>/...` and fixed conversion behavior for unknown nodes, subgraph output ordering, bypassed nodes, optional widget defaults, and frontend serialization differences. When the frontend or template package version changes, refresh parity fixtures deliberately and inspect whether the converter changed because of frontend behavior or because the fork's conversion code regressed.
- Custom-node compatibility with relocated `comfy_extras` required more than moving files. After root-level `comfy_extras/nodes_*.py` files moved into `comfy_extras/nodes/`, vanilla custom nodes still imported the old flat namespace. The compatibility fix added a meta path redirect from `comfy_extras.<name>` to `comfy_extras.nodes.<name>` and re-exported expected symbols such as `LatentUpscaleModelLoader` for ComfyUI-LTXVideo. Keep those compatibility probes in custom-node tests when upstream adds new node modules or when the package layout changes.
- The `8923e655` merge showed why merge output must be scanned for syntactically valid but semantically garbled code. One follow-up fixed indentation in `utils.py` after the merge and lowered a noisy `comfy_kitchen` MXFP8 warning to debug. A clean merge and passing import do not prove the merge is correct; inspect dense utility and quantization files for duplicated branches, misplaced indentation, and logging changes that would spam users or CI.
- The `d53b6414` merge added context-window slicing for several WAN conditioning paths. The fork had to preserve relative imports and call `slice_cond()` for keys like `vace_context`, `audio_embed`, `face_pixel_values`, and `pose_latents` with the right temporal dimension, scale, and offset. This is a model-code example where reading only the class registration is insufficient; the conditioning resize hooks are what make long-context video workflows behave correctly.

### Package-Layout Imports

Upstream often lands files with root-style imports. After moving files under this fork's package layout, check the whole moved file for imports, not just the lines Git marked as conflicts.

Examples from recent follow-ups:

- `comfy_extras/nodes_mediapipe.py` moved to `comfy_extras/nodes/nodes_mediapipe.py`; `import folder_paths` became `from comfy.cmd import folder_paths`, and later `from comfy_extras.mediapipe...` became relative imports from `..mediapipe...`.
- `comfy_extras/nodes_pid.py` moved cleanly but still needed `import node_helpers` changed to `from comfy import node_helpers`.
- newly added text encoders and model files such as `gpt_oss.py`, `pixeldit.py`, `sa3.py`, and `vae_sa3.py` needed upstream `import comfy.foo` references converted to local package imports.
- variables named like imported modules must be renamed, e.g. `import string` became `import string as string_module` in `nodes_string.py`.
- after enabling shadowing checks, module aliases should make intent explicit: use names like `model_sampling_module`, `model_patcher_module`, `clip_vision_module`, `sentencepiece_model`, `rope_fn`, and `filter_kernel` when upstream local variables would otherwise collide with imported modules.
- runtime helper tests should accompany risky renames. The shadowing cleanup added targeted coverage for vocoder alias-free filter helpers and OmitThink text filtering, because those are places where a rename can look mechanical but still break behavior.

Run import scans after the move commit:

```bash
rg -n '^(import|from) (folder_paths|server|nodes|app|node_helpers)\b' comfy comfy_extras comfy_api comfy_api_nodes tests
rg -n 'from comfy_extras\.' comfy_extras/nodes
```

Do not mechanically convert every absolute `comfy.*` import. Cross-package imports, public extension APIs, and tests sometimes need absolute paths. The goal is to remove broken root imports and avoid circular imports, not to make every import relative.

### Model Management And Dynamic VRAM

Model-management conflicts require whole-file review because this fork carries dynamic VRAM, pinned-memory, multigpu, mmap residency, and quantization changes that upstream does not have. Recent follow-ups fixed issues that were not obvious from one hunk:

- `LoadedModel.is_dead()` had to guard `real_model` before calling the weakref.
- mmap residency calls had to go through `loaded_model.model.model_mmap_residency()` because the wrapper owns the method.
- dynamic pin initialization in `ModelPatcherDynamic` had to tolerate missing `comfy_aimdo.host_buffer.lib` by using empty host-buffer stubs.
- multigpu clone selection needed to use the patcher's actual load device, not the process current CUDA device.
- `ops.pick_operations()` had to preserve this fork's inference-vs-training behavior: `disable_weight_init` in inference mode, `skip_init` otherwise.
- cuDNN attention selection needed runtime fallback when cuDNN/NVRTC compatibility fails.

When upstream changes these areas, read the full upstream and fork versions:

```bash
git show comfyui/master:comfy/model_management.py | less
git show HEAD:comfy/model_management.py | less
git diff --cc -- comfy/model_management.py comfy/model_patcher.py comfy/ops.py
```

Check for protocol drift too. If upstream adds required model-patcher attributes or methods, update `comfy/model_management_types.py`, `ModelManageableStub`, dynamic patchers, and tests.

#### Dynamic VRAM and mixed precision

Two things commonly look like a "dynamic VRAM regression" on mixed/quantized
checkpoints that stream. Only one is a bug.

1. **Load dtype — NOT a bug; do not "fix" it.** The dynamic patcher loads with
   `load_state_dict(assign=True)` so the vbar can own the tensors. `assign=True`
   *replaces* each param with the incoming tensor instead of `copy_`-casting it
   into the model's fp8 placeholder. For a mixed checkpoint (e.g.
   `flux2_dev_fp8mixed`, whose `_quantization_metadata` lists only the fp8
   layers) this **preserves** the higher-precision (bf16) layers the author
   stored for quality. The non-dynamic path coerces those layers to fp8 via
   `param.copy_()` (the legacy fp8-unet behaviour), so dynamic VRAM and legacy
   produce *different* output — but dynamic is the more faithful one, and that
   asymmetry must not be removed. Do **not** add a cast in
   `BaseModel.load_model_weights` to make them match: that silently down-converts
   the bf16 layers and defeats mixed precision. Pinned by
   `tests/unit/test_aimdo_dynamic_load.py::TestDynamicLoadPreservesMixedPrecision`.
   (Making *legacy* also preserve mixed precision would mean loading the model
   per-layer from `_quantization_metadata` instead of as a blanket fp8 unet — a
   larger, separate feature.)

2. **Streaming geometry — a real bug, fixed.** A scale-carrying weight
   (comfy_kitchen `QuantizedTensor`, or plain scaled-fp8) must cross the vbar in
   its **native** low-precision layout, not be densely cast to the compute dtype
   mid-stream. The dense-materialize transfer (`cast_to_gathered` with
   `target_geometries`) drops the per-tensor scale and diverges the streamed
   result from the resident one. `ops._streams_in_native_dtype()` gates this in
   `cast_modules_with_vbar.target_geometry_for`, the `direct_materialize`
   override, and the streaming/accounting geometries
   (`model_patcher.lowvram_materialization_geometry`).

The correctness target for streaming is therefore **streamed == resident under
the same patcher**, not dynamic == legacy.

To reproduce, force a fitting fp8 model to stream and compare to the resident
result (this is what `tests/unit/test_aimdo_dynamic_load.py` does via the
`limit_free_vram` helper, which patches `model_management.get_free_memory`):

```bash
COMFY_TEST_FP8_MODEL=models/diffusion_models/flux1-dev-fp8.safetensors \
  python -m pytest tests/unit/test_aimdo_dynamic_load.py -q
```

### New Model Family Support

When upstream adds a model family, the merge is not complete until the fork can load it through the packaged workflow path. Do all of the following:

1. Verify model-class registration in `comfy/model_base.py`, `comfy/supported_models.py`, `comfy/sd.py`, and related text encoder or latent-format files.
2. Add import fixes for new model files under `comfy/ldm/*` and `comfy/text_encoders/*`.
3. For every new class inheriting from `comfy.sd1_clip.SDClipModel`, make its `__init__` explicitly accept `textmodel_json_config=None` and pass a dict-compatible value to `super().__init__(..., textmodel_json_config=...)`. This fork's `SD1ClipModel` wrapper forwards `textmodel_json_config` to `clip_model(...)`; upstream classes often omit it because upstream does not use the same constructor plumbing. Missing this causes real workflow failures such as `TypeError: ... got an unexpected keyword argument 'textmodel_json_config'` during `CLIPLoader` execution.
4. Run pylint after text-encoder merges and treat `sd-clip-model-missing-config` from `tests/sd_clip_model_init_checker.py` as a merge blocker. Do not silence it unless the class does not actually subclass `SDClipModel`.
5. Register required downloadable filenames in `comfy/model_downloader.py`, including alternate filenames seen in workflow templates or custom-node examples.
6. Add a small one-step workflow under `tests/inference/workflows/`, or a dedicated template-based inference test when the upstream workflow is packaged as a template/subgraph.
7. Add coverage in `tests/inference/test_supported_model_coverage.py`.
8. Run the targeted workflow on a GPU and verify an output artifact, not just successful import.

Recent examples:

- PiD and PixelDiT support required `KNOWN_UNET_MODELS` entries for `pixeldit_1300m_1024px_mxfp8.safetensors` and `pid_flux1_512_to_2048_4step_mxfp8.safetensors`, plus a PixelDiT text encoder entry for `gemma_2_2b_it_elm_fp8_scaled.safetensors`.
- `tests/inference/workflows/pid-0.json` and `tests/inference/workflows/pixeldit-0.json` use one sampling step and `SaveImage` so the test can assert an `abs_path`.
- SAM3 coverage must include the SAM model classes and a workflow such as `sam3-segment-0.json`.
- StableAudio3 and Lens were added to supported-model coverage using existing or small representative workflows.
- HiDream O1 required both model detection coverage and a one-step inference workflow. The detection test uses a small synthetic state dict to prove `HiDreamO1` is selected and visual-only keys are stripped; the workflow proves the sharded `model.safetensors.index.json` path, latent node, decode, and save output work together.
- Flux2 FP8 work required both filename registration and operator behavior tests. The workflow exercises the template path, while the unit test proves disabled FP8 kernels still preserve `QuantizedTensor` weights and their scale metadata.
- Ideogram 4 showed the `textmodel_json_config` failure mode directly: upstream's `Qwen3VL8BModel(sd1_clip.SDClipModel)` needed the fork constructor argument added and merged with the model-specific Qwen3-VL config before the `image_ideogram4_t2i` template could run through `CLIPLoader`.

Use small workflows with one sampling step when possible. The purpose is smoke coverage that validates loading, conditioning, sampling, and output serialization. It is not image-quality benchmarking.

### Workflow Conversion And CLI Paths

Workflow runner regressions often appear only in end-to-end workflow tests. Recent fixes included:

- moving shared `run-workflow` setup into `_run_workflow_cli()` so both `comfyui run-workflow` and `comfyui workflows run` set the execution context before imports that read `args`;
- using all available node class types, not only core nodes, when resolving workflow requirements;
- treating comma-separated socket types such as `IMAGE,MASK` as matching when bypassing nodes;
- preserving frontend serialized widget positions for `forceInput` widgets so later widget values do not shift;
- resolving legacy `extra.groupNodes` outputs when old workflows lack modern subgraph boundary nodes;
- skipping optional frontend-injected widgets such as `CustomCombo` when the frontend did not serialize them;
- resolving direct parameter roles through linked primitive nodes and `ComfySwitchNode` so CLI overrides update the active source value instead of missing linked widgets or mutating inactive branches;
- expanding workflow quantity consistently across UI workflows, API prompts, `workflows submit`, and `workflows convert`, including frontend seed modes such as `fixed`, `randomize`, `increment`, and `decrement`.

Whenever upstream changes frontend workflow serialization, template workflows, group nodes, bypass handling, or CLI workflow execution, run the workflow-conversion unit tests and at least one real workflow through the CLI path:

```bash
uv run python -m pytest -q tests/unit/test_workflow_convert.py tests/unit/test_stream_json_objects.py
CUDA_VISIBLE_DEVICES=1 uv run python -m pytest -q tests/inference/test_workflows.py -k 'pid-0 or pixeldit-0 or sam3-segment-0' -s
CUDA_VISIBLE_DEVICES=1 uv run comfyui run-workflow --all tests/inference/workflows/pid-0.json
```

The inference test asserts `SaveImage` output paths. If you run the CLI manually, inspect the JSON output and confirm a real image file was produced.

#### Frontend conversion parity (`graphToPrompt`) and its cache

`tests/unit/test_workflow_convert_playwright.py::TestFrontendParity` is the authoritative cross-check: it runs the **real** compiled frontend in headless Chromium, calls `app.graphToPrompt()` for each template, and asserts `comfy/component_model/workflow_convert.py::convert_ui_to_api` produces the same API graph. `convert_ui_to_api` is a line-for-line translation of the frontend; when it drifts, port the TS rather than guessing. Key correspondences (frontend `~/Documents/ComfyUI_frontend`, checked out at the **installed** `comfyui-frontend-package` version — `git checkout v$(python -c 'import importlib.metadata as m;print(m.version("comfyui-frontend-package"))')`):

| Behaviour | Frontend (TS) | Fork (Python) |
| --- | --- | --- |
| Top-level conversion loop, widget/link serialization, drop muted (`NEVER`/`BYPASS`) and virtual nodes, prune links to removed nodes | `src/utils/executionUtil.ts` `graphToPrompt` | `convert_ui_to_api` |
| Subgraph flattening / execution-id assignment | `src/lib/litegraph/src/subgraph/SubgraphNode.ts` `getInnerNodes` (recurses `subgraphInstanceIdPath = [...path, this.id]`) | `_expand_subgraph` |
| Flattened execution id | `ExecutableNodeDTO.ts`: `this._id = [...this.subgraphNodePath, this.node.id].join(':')` | `workflow_convert.py:866`: `self.exec_id = ':'.join(str(x) for x in [*subgraph_node_path, nid])` |
| Bypass/Reroute input resolution | `ExecutableNodeDTO.resolveInput` / `resolveOutput` | `_resolve_source` / `_get_bypass_slot_index` |
| Promoted subgraph-widget lookup | `resolveConcretePromotedWidget.ts` | `_get_inner_widget_value` |

So a top-level subgraph instance node `267` emits its inner nodes as `267:<inner-id>`, and a subgraph nested one level deeper emits `267:<mid>:<leaf>` — exactly matching the frontend's colon-joined `subgraphNodePath`.

**Cache staleness gotcha.** Frontend outputs are cached under `tests/unit/playwright_cache/<frontend-version>+t<templates-versions>/<template_id>.json` and only regenerated via Playwright when the file is **missing**. The key encodes package *versions*, not template *content*, and `invalidate_stale_cache()` only deletes caches containing `class_type: null` (i.e. newly-added node types). So when a packaged template is **restructured** without that signal (e.g. `video_ltx2_3_t2v` changed from an image-to-video+MoGe-depth graph to a 51-node text-to-video subgraph), the old cache survives and the parity test fails with a structural diff (different node ids/prefixes) even though `convert_ui_to_api` is correct. The tell is that the cached frontend output references nodes that do not exist in the current template asset. Fix by regenerating the stale entries (requires `pip install playwright && python -m playwright install chromium`):

```bash
# delete the stale entries, then the test regenerates them from the real frontend
rm tests/unit/playwright_cache/<version-dir>/<template_id>.json
CUDA_VISIBLE_DEVICES=1 uv run python -m pytest \
  "tests/unit/test_workflow_convert_playwright.py::TestFrontendParity" -k "<template_id>"
```

Only treat a parity failure as a converter bug after confirming the regenerated cache still disagrees with Python.

### Asset Routes, Database, And Seeder

Upstream asset work often touches route handlers, service names, Alembic migrations, background scanning, and tests at the same time. This fork keeps asset code under `comfy/app/assets`, stores Alembic under `comfy/alembic_db`, and runs most request-facing tests through the fork's configuration object.

Recent older merge fixes included:

- gating `/api/assets` routes so the API returns a controlled disabled-service response when assets are unavailable instead of crashing after partial database initialization;
- preserving compatibility names around `AssetReference` such as `asset_info_id` wrappers until all fork callers are migrated;
- moving upstream Alembic paths from root `alembic_db` to `comfy/alembic_db` in both code and migration tests;
- copying uploaded files into reserved destinations with `os.O_EXCL` so duplicate uploads do not clobber an existing file;
- using `ContextVarExecutor` for the asset seeder so background scans inherit the current folder paths and configuration;
- adding per-test cleanup fixtures and unique test bytes so asset upload/list/sync tests do not collide by hash across tests.

When assets change, run the focused asset and migration tests:

```bash
uv run python -m pytest -q tests/unit/app_test/test_migrations.py tests/unit/assets_test --tb=short
```

Also check for old upstream package paths:

```bash
rg -n 'from app\.|import app\.|patch\("app\.|patch\("folder_paths\.|alembic_db' comfy tests docs
```

### Custom-Node Compatibility Shims

Moving upstream files into this fork's package layout can break third-party custom nodes that import upstream's flat namespaces. Do not judge compatibility only by core node imports.

The older ComfyUI-LTXVideo fix added lazy redirects from `comfy_extras.<name>` to `comfy_extras.nodes.<name>` and re-exported a symbol expected by the custom node. Keep this pattern in mind when upstream adds or moves root-level `comfy_extras/nodes_*.py` files.

Run at least one custom-node loading probe when package layout changes:

```bash
uv run python -m pytest -q tests/custom_nodes/test_pip_facade_installation.py -k 'ltxvideo or custom_node' --tb=short
```

### GPU Defaults And Guess Settings

Configuration heuristics affect local runs and CI. Recent fixes set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `main_pre` when the variable is missing or blank, while preserving an explicit user value, but skip that CUDA allocator default for XPU/oneAPI runs. On Intel Arc XPU, the CUDA allocator variable can make otherwise valid torch XPU `linear`/`conv2d` operations fail with `RuntimeError: could not create a memory`. Another fix changed `guess_settings` so tiny desktop GPU users such as `gnome-remote-desktop-daemon` and `steamwebhelper` do not force `--novram`; only material non-Comfy GPU memory use should do that.

When changing startup or GPU heuristics:

```bash
uv run python -m pytest -q tests/unit/test_cuda_env.py tests/unit/test_guess_settings.py
```

Also verify the logs on a workstation with benign GPU helper processes. The merge is not done if small helper processes still cause:

```text
competing GPU processes detected (...), enabling novram
```

## Version String

Upstream uses `comfyui_version.py` at the repository root. We deleted this file and moved the package version into `pyproject.toml`; the runtime CLI/server version is also mirrored in `comfy/__init__.py`.

When merging, accept our deletion of `comfyui_version.py` and update both version locations to the upstream ComfyUI version:

```bash
rg -n '^version = |__version__ = ' pyproject.toml comfy/__init__.py
```

Both values must match. `pyproject.toml` controls package metadata; `comfy/__init__.py` is what `comfyui --version`, server metadata, and workflow telemetry read.

The version field tracks **upstream** ComfyUI; keeping it aligned lets users reason about feature parity at a glance. Fork releases use a four-part **`v<UPSTREAM>.<FORK_PATCH>`** tag scheme — e.g. `v0.21.0.1`, `v0.21.0.2`, … — with the fork-patch counter bumped per fork-only release. Never invent upstream patch versions locally; those numbers belong to upstream. The Docker tag-match patterns already accept four-part versions (`type=match,pattern=v?(\d+\.\d+\.\d+\.\d+)` in `.github/workflows/docker-build*.yml`).

Place release tags on the finished, tested fork commit for that version, not merely on the upstream commit or the commit whose subject says `ComfyUI v...`. The correct tag target is the state after merge resolution, package-layout moves, model support additions, lint fixes, inference checks, and CI/test follow-ups are complete.

## Requirements

Upstream uses `requirements.txt` at the repository root. We deleted this file and moved all dependencies to `pyproject.toml`.

When merging, accept our deletion of `requirements.txt` and update version minimums in `pyproject.toml` instead. Do not only preserve the existing fork ranges. Read upstream's current `requirements.txt` and reconcile every pinned or minimum version into the package metadata:

```bash
git show comfyui/master:requirements.txt
rg -n 'dependencies = |comfyui-frontend-package|comfyui-workflow-templates|comfyui-embedded-docs|comfy[_-]kitchen|comfy-aimdo|numpy|aiohttp|yarl|kornia|filelock|SQLAlchemy' pyproject.toml
```

For upstream pins such as `pkg==X.Y.Z`, use an equivalent fork minimum when this package intentionally carries a range, e.g. `pkg>=X.Y.Z,<next-compatible-bound>`. For ordinary lower bounds, make sure `pyproject.toml` is at least as high as upstream. If upstream adds a base dependency that this fork does not have, add it to `dependencies` unless it is deliberately excluded with a documented reason. Recent examples include `comfyui-frontend-package`, `comfyui-workflow-templates`, `comfy-aimdo`, `numpy`, `kornia`, and `filelock`.

Key packages to watch:

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

You will also have to move upstream root-level `comfy_extras/nodes_*.py` files into `comfy_extras/nodes/`, where they are scanned automatically. Use `git mv` in the separate move commit and do not leave these files at the `comfy_extras/` package root.

Example:

```bash
git mv comfy_extras/nodes_math.py comfy_extras/nodes/nodes_math.py
git mv comfy_extras/nodes_sdpose.py comfy_extras/nodes/nodes_sdpose.py
```

After moving them, fix their imports so they are valid from the `comfy_extras.nodes` package (see [Import Fixes](#import-fixes) below).

Do not copy upstream's root `nodes.py` custom-node loader block into `comfy/nodes/base_nodes.py` when Git presents it as a directory-rename conflict. This fork keeps node loading under the package loader/facade paths; `base_nodes.py` should remain focused on core node definitions and mappings. For new upstream extra-node files, keep the merge commit faithful, then move `comfy_extras/nodes_*.py` into `comfy_extras/nodes/` in the separate move commit so the existing package scanner can discover them.

This two-step approach keeps git history cleaner and makes conflicts easier to resolve.

Upstream may also still add tests under `tests-unit/`. Move those into `tests/unit/` in this fork, preserving the same relative structure.

Example: `tests-unit/seeder_test/test_seeder.py` → `tests/unit/seeder_test/test_seeder.py`

## Import Fixes

After moving directories, fix absolute imports to use relative imports whenever the target is inside the repo. Upstream uses absolute imports like:

- `import folder_paths` -> `from comfy.cmd import folder_paths`
- `import nodes` in moved node modules -> `from comfy.nodes import base_nodes as nodes`
- prefer relative imports for moves within the same package tree
- exception: cross-package imports must stay absolute. For example, `comfy_api/latest/__init__.py` must import `PromptServer` as `from comfy.cmd.server import PromptServer`, because `comfy_api` and `comfy` are sibling top-level packages.

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

### Required Post-Merge Test Pass

Run the test layers in this order. Do not stop at unit tests when upstream added or modified models, nodes, workflow templates, media loaders, custom-node compatibility, or GPU settings.

```bash
# Fast correctness and conversion checks.
uv run python -m pytest -q tests/unit --tb=short
uv run python -m pytest -q tests/execution --tb=short

# Whole-codebase lint. Run raw, no grep/head/tail.
uv run ruff check comfy/ comfy_extras/ comfy_api/ comfy_api_nodes/ comfy_compatibility/ comfy_execution/
uv run pylint -j 0 comfy/ comfy_extras/ comfy_api/ comfy_api_nodes/

# Representative GPU inference. Use the idle GPU explicitly.
CUDA_VISIBLE_DEVICES=1 uv run python -m pytest -q tests/inference/test_supported_model_coverage.py
CUDA_VISIBLE_DEVICES=1 uv run python -m pytest -q tests/inference/test_workflows.py -k 'pid-0 or pixeldit-0 or sam3-segment-0' -s
```

If upstream added a new model family or a new loader path, add or update a workflow under `tests/inference/workflows/` and run it specifically. The workflow should have a `SaveImage`, `SaveAudio`, `SaveAnimatedWEBP`, or `PreviewString` terminal when possible so `tests/inference/test_workflows.py` can assert a concrete output. For image models, inspect the generated file when debugging; a test that only imports the node is not enough.

For custom-node compatibility, use GPU 1 and run targeted tests first:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m pytest -q tests/custom_nodes/test_custom_node_execution.py -k 'ComfyUI-WanVideoWrapper or SAM3 or PiD' -s --tb=short
```

Then run the broader custom-node suite when time permits:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m pytest -q tests/custom_nodes --tb=short
```

Custom-node failures should be classified. Fix compatibility regressions in the fork when they are caused by our package layout, import shims, workflow conversion, model discovery, or execution wrapper. Mark workflows as `xfail` only when the workflow is not actionable for the fork: removed/unpublished models, stale upstream workflow options, missing third-party custom nodes outside the test set, upstream bugs with local stub media, or workload size that consistently exceeds the local 24GB GPU timeout. Update `docs/custom_nodes.md` with every new xfail and the concrete reason.

When running coverage, use it to find missing tests rather than treating the percentage as the only goal:

```bash
uv run coverage run -m pytest tests/unit tests/execution
uv run coverage report
```

After coverage, propose or add focused tests for newly touched merge-sensitive code: workflow conversion edge cases, model downloader aliases, GPU heuristic parsing, config propagation, and package-layout imports.

### Inference Workflow Requirements

Inference workflows added during a merge should be cheap, deterministic enough for smoke testing, and representative of the new code path:

- use one sampling step for large diffusion models unless the model cannot exercise its path with one step;
- use local `pkg://tests.custom_nodes.test_data/...` media instead of remote URLs;
- use known filenames from `comfy/model_downloader.py`;
- include `SaveImage` or another terminal output node so the test asserts a produced artifact;
- add the model class to `tests/inference/test_supported_model_coverage.py`;
- keep model-specific workflow names obvious, e.g. `pid-0.json`, `pixeldit-0.json`, `sam3-segment-0.json`.

If a new workflow needs additional model files, register them before adding the workflow. Missing models discovered only during `--all` or inference tests usually mean `comfy/model_downloader.py` needs a new `HuggingFile`, `CivitFile`, `alternate_filenames`, or folder mapping.

### CLI Workflow Checks

Run at least one workflow through the CLI when merge changes touch `comfy/cmd/cli.py`, `comfy/cmd/sub_workflows.py`, setup order, model downloading, custom-node installation, or workflow conversion:

```bash
CUDA_VISIBLE_DEVICES=1 uv run comfyui run-workflow --all tests/inference/workflows/pid-0.json
CUDA_VISIBLE_DEVICES=1 uv run comfyui workflows run --all tests/inference/workflows/pixeldit-0.json
```

Watch for configuration-order bugs. The execution context must be set before imports that read `comfy.cli_args.args`; otherwise `--novram`, `--lowvram`, `--fast`, and model paths silently fall back to defaults.

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

### Threading and contextvars

This fork stores runtime state (folder paths, configuration, execution context) in `contextvars`. Plain `threading.Thread` does **not** propagate `contextvars` — child threads get an empty context, so values like `folder_names_and_paths` resolve to defaults instead of the configured paths.

When upstream adds new `threading.Thread(...)` calls, replace them with `ContextVarExecutor` from `comfy/distributed/executors.py`:

```python
# Bad: thread gets empty context, folder_paths uses defaults
import threading
threading.Thread(target=do_work, daemon=True, args=(x, y)).start()

# Good: executor propagates the caller's contextvars
from comfy.distributed.executors import ContextVarExecutor
executor = ContextVarExecutor(max_workers=1, thread_name_prefix="MyWorker")
executor.submit(do_work, x, y)
```

`ContextVarExecutor` is a `ThreadPoolExecutor` that captures `contextvars.copy_context()` on each `submit()` and runs the callable inside that context. Using a named executor pool (vs naked threads) makes it easier to track and debug active threads.

### Mock Patching folder_paths

Upstream tests often use `unittest.mock.patch` to override `folder_paths` attributes:

```python
# Upstream pattern (WRONG in this fork)
with patch("folder_paths.folder_names_and_paths", {"custom_nodes": (["/tmp"], None)}):
    ...
```

This doesn't work because `folder_paths.folder_names_and_paths` is a context-dependent property that reads from the execution context. Instead, use `FolderNames` and `context_folder_names_and_paths`:

```python
from comfy.component_model.folder_path_types import FolderNames
from comfy.execution_context import context_folder_names_and_paths

fn = FolderNames()
fn["custom_nodes"] = ([str(custom_nodes_dir)], set())
with context_folder_names_and_paths(fn):
    ...
```

Key differences from the upstream pattern:

- **Use `set()` not `None`** for supported_extensions — `FolderNames.__setitem__` wraps the value into a `ModelPaths` and `set(None)` raises `TypeError`
- **Create objects inside the context** — classes like `CustomNodeManager` capture `folder_paths.folder_names_and_paths` at `__init__` time, so they must be instantiated inside the `context_folder_names_and_paths` block
- **All `patch("folder_paths.xxx")` targets** must be rewritten to `patch("comfy.cmd.folder_paths.xxx")` if patching is still needed for non-dict attributes
- **All `patch("app.xxx")` targets** must be rewritten to `patch("comfy.app.xxx")`

### Mock Patching General Rules

When converting upstream test `patch()` targets:

| Upstream Target | This Fork Target |
|---|---|
| `"folder_paths.X"` | `"comfy.cmd.folder_paths.X"` or use `FolderNames`/`context_folder_names_and_paths` |
| `"app.X"` | `"comfy.app.X"` |
| `"server.X"` | `"comfy.cmd.server.X"` |
| `"nodes.X"` | `"comfy.nodes.X"` |
| `"execution.X"` | `"comfy.cmd.execution.X"` |

Watch for multi-line `patch()` calls where the target string is on a different line from `patch(` — simple find-and-replace may miss these.

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

### Logging Convention

Use `logger = logging.getLogger(__name__)` and `logger.info(...)` — never bare `logging.info(...)`. Upstream frequently uses bare `logging.xxx()` calls; convert them during merge.

```python
# Bad (upstream pattern)
import logging
logging.warning("something happened")

# Good
import logging
logger = logging.getLogger(__name__)
logger.warning("something happened")
```

Place `logger = logging.getLogger(__name__)` at module level, after imports.

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

## Workflow Conversion (Frontend Parity)

### Overview

The Python module `comfy/component_model/workflow_convert.py` implements the same logic as the frontend's `graphToPrompt` function. It converts UI-format (LiteGraph) workflows — which have `nodes`, `links`, and `widgets_values` — into API format (`{node_id: {"class_type": ..., "inputs": {...}}}`).

The Playwright parity test (`tests/unit/test_workflow_convert_playwright.py`) validates that the Python conversion produces identical output to the real frontend JavaScript for every template workflow shipped with `comfyui-workflow-templates`.

### Hard Rule: Port the Frontend Exactly

To fix any workflow conversion bug, you **must translate exactly what the frontend does in TypeScript into Python** by cloning the matching tag of the frontend repo and reading the source:

```bash
git clone --depth 1 --branch v<VERSION> https://github.com/Comfy-Org/ComfyUI_frontend.git /tmp/comfyui-frontend
```

Do not guess from cached output, infer from a single failing test, or invent heuristics. Every divergence between Python and the frontend is fixed by finding the corresponding TypeScript code path and replicating its behavior 1:1. If you cannot find the relevant frontend code, keep searching — do not patch around it.

### How the Test Works

1. A headless Chromium browser loads the compiled ComfyUI frontend
2. For each template workflow, the frontend's `app.loadGraphData()` and `app.graphToPrompt()` produce the authoritative API output
3. The Python `convert_ui_to_api()` converts the same workflow
4. Both outputs are normalized and compared; any differences fail the test

Frontend outputs are cached on disk under `tests/unit/playwright_cache/<version>/` keyed by the `comfyui-frontend-package` version. Playwright only runs when the frontend version changes.

### When to Run

Run the Playwright tests after:
- Upgrading `comfyui-frontend-package` (new cache needed)
- Modifying `comfy/component_model/workflow_convert.py`
- Adding new node types that affect template workflows

```bash
pytest tests/unit/test_workflow_convert_playwright.py -x --tb=short
```

The first run after a frontend upgrade takes ~8 minutes (generates cache for all templates). Subsequent runs use the cache and are fast.

### Fixing Mismatches

When tests fail, the mismatch output shows exactly which nodes and inputs differ between the Python and frontend outputs. To fix:

1. **Clone the frontend source** at the matching tag:
   ```bash
   git clone --depth 1 --branch v<VERSION> https://github.com/Comfy-Org/ComfyUI_frontend.git /tmp/comfyui-frontend
   ```

2. **Read the authoritative implementation** — the key files are:
   - `src/utils/executionUtil.ts` — the `graphToPrompt` function: widget serialization loop, link resolution, dangling link cleanup
   - `src/lib/litegraph/src/subgraph/ExecutableNodeDTO.ts` — `resolveInput`, `resolveOutput`, `_getBypassSlotIndex`, `_resolveSubgraphOutput`
   - `src/utils/executableGroupNodeDto.ts` — legacy group node DTO
   - `src/utils/executableGroupNodeChildDTO.ts` — child nodes inside group nodes
   - `src/utils/litegraphUtil.ts` — `compressWidgetInputSlots`, `matchesLegacyApi`

3. **Faithfully port the TypeScript logic to Python**. Do not innovate or approximate. The Python code must match the frontend behavior exactly. Search for `graphToPrompt` in the frontend codebase to find all relevant code paths.

4. **Delete the stale cache** for the affected version so Playwright regenerates it:
   ```bash
   rm -rf tests/unit/playwright_cache/<VERSION>/
   ```

5. **Re-run the tests** to verify all templates pass.

### Architecture Mapping

| Frontend (TypeScript) | Python (`workflow_convert.py`) |
|---|---|
| `graphToPrompt()` in `executionUtil.ts` | `convert_ui_to_api()` |
| `ExecutableNodeDTO` | `_NodeDTO` class + `_resolve_dto_input/output` |
| `ExecutableGroupNodeDTO` | Legacy group node handling in `_convert_legacy_group_node` |
| `compressWidgetInputSlots` | `_compress_widget_input_slots()` |
| `matchesLegacyApi` | `_matches_legacy_api_input()` |
| Widget serialization loop (lines 98-116) | `_map_widgets()`, `_map_widgets_dict()` |
| `{ __value__: array }` wrapping | `_wrap_value()` |
| `resolveInput` / `resolveOutput` | `_resolve_dto_input()` / `_resolve_dto_output()` |
| `_getBypassSlotIndex` | `_get_bypass_slot_index()` |
| `_resolveSubgraphOutput` | `_resolve_sg_output()` |
| Subgraph ID deduplication | `_ensure_global_id_uniqueness()` |
| Proxy widget overrides | `_compute_proxy_overrides()` |

### Excluded Templates

Templates are excluded in `_EXCLUDED_TEMPLATE_REASONS` when they depend on nodes unavailable to the test fixture. The exclusion tuple contains the missing node titles (or `<unknown>` when the title is not available). Common exclusion reasons:

- **`<unknown>`** — the template uses custom nodes not included in the base ComfyUI installation
- **`<frontend invalid link>`** — the workflow has broken links that the frontend also fails to resolve
- **`<frontend promoted widgets>`** — the template uses frontend-only promoted widget features
- **`<frontend group node outputs>`** — the template uses legacy group node output features

Never mark a workflow as impossible to convert due to a bug — all bugs are real and fixable. The only legitimate exclusion reason is missing nodes that don't exist in the test environment.

### Cache Management

To invalidate stale cache entries (e.g., after adding new node implementations):

```python
from tests.unit.test_workflow_convert_playwright import invalidate_stale_cache
deleted = invalidate_stale_cache()
print(f"Invalidated {len(deleted)} stale cache entries: {deleted}")
```

This removes cached entries where `class_type: null` (indicating missing node types at cache time that may now be available).

### CI Notes

The Playwright tests are **excluded from CI** (`--ignore=tests/unit/test_workflow_convert_playwright.py` in all CI test commands) because:
- They require Playwright browsers installed (`playwright install chromium`)
- They take ~8 minutes for a fresh run
- On Windows, they previously caused process deadlocks

Run them locally when upgrading the frontend package or modifying the conversion code.
