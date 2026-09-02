# Merging ComfyUI upstream

This fork carries a moving upstream frontend, workflow-template packages, and
runtime integrations. Treat an upstream merge as a parity exercise, not just
a conflict-resolution exercise.

## Before the merge

```console
git fetch upstream --prune
git status --short --branch
git log --oneline --decorate -8
git diff --stat upstream/master...develop
```

Work from a clean branch/worktree. Confirm the upstream commit and frontend
package versions before changing files. Enable `merge.directoryRenames=true`,
use a high rename limit, keep `rerere`, and prefer the `zdiff3` conflict style.
Never delete and recreate `develop`, reset away local commits, or resolve a
whole directory with an “ours” strategy.

## Rename rule

Every upstream `git mv` must be committed as its own move-only commit before
the moved file is edited:

```console
git mv old/path.py new/path.py
git commit -m "Move path.py to its upstream location"
# Only now modify new/path.py and commit the functional change separately.
```

Verify that Git records `R100`/rename detection. This preserves upstream’s
future updates and makes later merges reviewable.

## Merge and parity checklist

1. Merge the selected upstream commit without broad deletion or generated-file
   churn.
2. Inspect root entrypoints (`main.py`, packaging metadata, and setup helpers)
   for side effects that the fork’s shim may not inherit.
3. Compare the installed frontend package’s source and version with upstream;
   regenerate graph-to-prompt/cache artifacts only after verifying the source.
4. Reconcile `Configuration` and Typer help, package layout, model detection,
   known-model tables, workflow templates, and DynamicVRAM defaults.
5. Confirm `ContextVarExecutor` and process-pool boundaries still propagate
   configuration, progress, cancellation, and OpenTelemetry context.
6. Run frontend parity, unit, quantization, distributed, model-registration,
   and custom-node conversion tests. Use package resources rather than
   absolute workstation paths or `PYTHONPATH`.
7. Inspect the final diff for accidental dependency upgrades, especially Torch,
   CUDA/ROCm stacks, and custom-node requirements.

## Informative integration commits

| Commit | Lesson |
|---|---|
| `762012669` | Keep upstream moves separate from edits so rename detection survives future merges. |
| `cfdb6f513` | Re-check setup hooks after an upstream `main.py` merge; DynamicVRAM can disappear through a side effect, not a textual conflict. |
| `d1ee24e2a` | Frontend parity requires checking the shipped JavaScript source and regenerating the conversion cache with the correct cache key/version. |
| `f62e979f4` | Use `ContextVarExecutor` consistently across worker/process boundaries. |
| `81f4bd005` | Tests must resolve package resources with `importlib`; hardcoded workspace paths and `PYTHONPATH` hide portability bugs. |
| `c281de8ab` | Facade snapshots must reload safely between generations. |
| `7ad9433e8` / `ed0da0242` | An upstream release merge and its integration follow-up are separate validation events. |
| `1718d664d` | Release retagging must wait for cold image builds and their health checks. |

## Release gate

Do not tag until the merge commit has passed the same checks used for a normal
develop push, including frontend parity and the device-specific CI jobs. After
the tag, monitor image builds and health checks before publishing or retagging
any mutable image name.
