# Windows CI on Kubernetes (containerMode: kubernetes)

This documents how the Windows GitHub Actions CI runner works with actions-runner-controller (ARC) legacy mode (`actions.summerwind.dev/v1alpha1`) and `containerMode: kubernetes`.

## Architecture

The **runner agent** runs on a **Linux** pod. It registers with GitHub and listens for jobs. When a job arrives, the runner's **container hooks** (`/runner/k8s/index.js`) create a separate **Windows workflow pod** via the Kubernetes API. All workflow steps execute inside the Windows pod via `kubectl exec`.

```
Linux agent pod (summerwind/actions-runner)
  └─ container hooks (Node.js)
       └─ creates Windows workflow pod
            ├─ init container: fs-init (copies shell tools + node to /__e/)
            └─ job container: buildervscuda (runs workflow steps)
```

The hook extension ConfigMap (`hook-extension-windows-cuda-3090`) defines the Windows pod spec: node selector, GPU resources, init containers, volumes, and environment.

## Key files

- `appmana-cluster/src/clusters/appmana-cluster-03/action-runners/comfyui-action-runners/action-runners.yaml` — RunnerDeployment and autoscaler
- `appmana-cluster/src/clusters/appmana-cluster-03/action-runners/comfyui-action-runners/hook-configmaps.yaml` — Hook extension ConfigMaps (pod templates)

## Windows-specific workarounds

### 1. `rm "$0"` fails on Windows (file locking)

The container hooks generate shell scripts that start with `rm "$0"` (self-delete). On Linux this works because inodes allow deletion of open files. On Windows, the executing script file is locked and `rm` returns "Permission denied". With `set -e`, this aborts the entire script before any real work runs.

**Fix**: An init container on the Linux agent pod patches `/runner/k8s/index.js` before the runner starts:

```yaml
initContainers:
  - name: patch-hooks
    image: "summerwind/actions-runner:v2.332.0-ubuntu-22.04"
    command: ["sh", "-c"]
    args:
      - |
        cp /runnertmp/k8s/index.js /patched-hooks/index.js
        sed -i 's,rm \\"$0\\",rm \\"$0\\" 2>/dev/null || true,g' /patched-hooks/index.js
```

The patched file is mounted at `/runnertmp/k8s/` via an emptyDir volume, shadowing the original. When `startup.sh` copies `/runnertmp/*` to `/runner/`, it picks up the patched version.

### 2. Busybox applet re-execution needs `.exe` extension

The hooks call `/__e/sh -e <script>` and `/__e/tail` inside the Windows pod. These are busybox-w32 copies. When busybox runs an applet (like `ls`, `rm`, `find`), it re-executes itself via Windows `CreateProcess`. If the binary is named `sh` (no `.exe`), CreateProcess can't find it. Windows resolves `sh` → `sh.exe` automatically, so having both `sh` and `sh.exe` present fixes the issue.

**Fix**: The `fs-init` init container copies busybox as both `sh` and `sh.exe` (and `tail`/`tail.exe`, `env`/`env.exe`):

```
cp C:/busybox64.exe /mnt/externals/sh
cp C:/busybox64.exe /mnt/externals/sh.exe
cp C:/busybox64.exe /mnt/externals/tail
cp C:/busybox64.exe /mnt/externals/tail.exe
cp C:/busybox64.exe /mnt/externals/env
cp C:/busybox64.exe /mnt/externals/env.exe
```

### 3. Node.js required for JavaScript actions

Actions like `actions/checkout@v4` are JavaScript and need Node.js at `/__e/node20/bin/node`. On Linux, the hooks' default `fs-init` copies `/home/runner/externals/` from the actions-runner image. On Windows, we need Windows `node.exe`.

**Fix**: The `fs-init` init container uses the `buildervscuda` image (which has Node.js installed) and copies the full Node payload, not just `node.exe`, because the current runner/container path can also need `corepack`, `npm`, `npx`, and `node_modules` under `/__e/node20/bin`:

```
mkdir -p /mnt/externals/node20/bin
cp -r 'C:/Program Files/nodejs/'* /mnt/externals/node20/bin/
cp 'C:/Program Files/nodejs/node.exe' /mnt/externals/node20/bin/node
```

The init container must be named `fs-init` to override the hooks' default Linux-only init container.

### 4. `actions/checkout@v4` path handling is broken

The checkout action emits `::add-matcher::` with Windows paths that the Linux runner agent mangles (prepends the workspace path to an absolute Windows path). This causes a step failure even though git operations succeed.

Also, ARC now injects the job work volume itself. The Windows hook extension must not add its own `work` volume or `/__w` mount anymore, or pod creation fails with duplicate volume / duplicate mount-path validation errors.

**Fix**: Use a manual `git clone` instead:

```yaml
- name: Checkout git repo
  run: |
    git config --global credential.helper store
    echo "https://x-access-token:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
    git clone --depth=1 --branch "${GITHUB_REF_NAME}" "https://github.com/${GITHUB_REPOSITORY}.git" .
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### 5. Busybox `bash` doesn't support `--noprofile`

GitHub translates `shell: bash` to `bash --noprofile --norc -e -o pipefail {0}`. Busybox's bash applet doesn't support `--noprofile`.

**Fix**: Use `shell: sh -e {0}` instead of `shell: bash`.

### 6. `Initialize containers` step is slow (~15-20 minutes)

The hooks run `find . -exec stat -c '%b %n' {} \;` (via `listDirAllCommand`) to hash directory contents before and after each `cpToPod`. This is extremely slow on Windows containers due to `find` + `stat` per-file overhead. There is no workaround — this is intrinsic to the hooks' verification logic.

## Dependencies

- `uvloop` does not support Windows. It was pulled in via `flax` → `orbax-checkpoint` → `uvloop`. Since `flax` is unused, it was removed from dependencies.
- Any future dependency that unconditionally requires a Unix-only package will need platform-conditional handling in `pyproject.toml` or exclusion via `uv pip install --excludes`.

## Upstream status

- `containerMode: kubernetes` on Windows is **not officially supported** by GitHub or actions-runner-controller. The runner C# code blocks container operations on non-Linux (`ContainerOperationProvider.cs`). However, since the agent runs on Linux and the hooks use `kubectl exec` into the Windows pod, it works with the patches above.
- There are no env vars to control `rm "$0"`, shell paths, or externals behavior in the hooks. All fixes require patching `index.js` or overriding init containers.
- Relevant issues: [actions/runner#904](https://github.com/actions/runner/issues/904), [actions/runner-container-hooks#206](https://github.com/actions/runner-container-hooks/issues/206)
