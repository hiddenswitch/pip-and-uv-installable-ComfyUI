# Kandinsky 6 / HunyuanImage3 release candidate

This candidate is **not ready to tag or publish**. It started at `ed3459b68`
(0.38.0.1) and now includes master `088ef4eb5` (0.39.0.1), merged at
`6a8f02e38`, plus develop `0f373939d` at `aa46b9b0c`. The next patch release is 0.39.0.2. No release tag or image
promotion has been performed.

## Implemented

- Snapshot metadata stops serving cached generations when its source disappears,
  becomes unreadable, has invalid metadata, or exceeds the configured maximum
  age. Readiness checks the current snapshot and recovers after replacement.
  `--pip-facade-snapshot-max-age-seconds=0` keeps age enforcement optional.
- CUDA image promotion restores the validated `master-<sha>-<timestamp>` tag
  consumed by Flux. Cluster changes set `/readyz` and `/livez` probes, a 2100 s
  snapshot age limit, and zero stale grace for package indexes. Apply those
  changes only with an image supporting the new flag.
- Compatibility registry, bundled snapshot and shared custom-node requirements:
  `kandinsky6==1.0.2`, `kandinsky6-sr==0.1.2`, and
  `comfyui-hunyuanimage3==0.1.0`. The Hunyuan facade uses upstream commit
  `84ad3a3e2a54472e69e195269729f774b242d120`; there is no fork.
- Kandinsky loader adapter partitions attention heads and FFN channels, loads
  checkpoint slices per rank, imports the custom node in workers, and routes
  PiFlow calls through the distributed executor. Output projection bias is
  applied on rank zero before summation.
- Hunyuan adapter partitions output rows of transformer linears and expert
  banks, preserving the full W4A8 input quantization domain. Outputs are gathered
  before the next upstream operation. Prompt rewriting routes transformer calls
  through the executor; its cache remains local to the rewriting call and is
  transported for each step. This path needs a GPU performance check.
- CPU quantized-model capability checks no longer query CUDA properties for an
  explicitly non-CUDA device.

## Validation so far

- Facade regression rerun with network/device access restored: **31 passed**.
  This includes the final schema-validation and class-map changes.
- Combined CPU TP suite: **35 passed** (including TP2/TP4 projection parity,
  rank-local checkpoint ownership and non-CUDA capability cases).
- Installed Hunyuan transformer parity: **4 passed**, covering TP2/TP4 and
  one-token decode versus longer sequences. These use small float32 weights;
  they do not replace real W4A8 CUDA validation.
- Installed Kandinsky asymmetric-attention parity: **2 passed**, at TP2/TP4.
  Together with the shared suite, there are **41 passing CPU TP checks**.
- All three locally assembled facade wheels installed successfully; **21 nodes
  and 21 frontend schemas** registered through the compatibility loader.
- Upstream Hunyuan tests: **108 passed, 17 skipped, 1 failed**. The remaining
  failure is `test_generator_wraps_the_model_in_a_patcher`: its legacy generator
  test expects `generator.model.device` to be assigned by ModelPatcher. This
  fork tracks placement on the patcher. Do not hide the failure by adding an
  otherwise unused model attribute; verify the node loader's real offload path.
- Targeted Ruff checks and `git diff --check` pass.

## Benchmarks and remaining gates

`scripts/benchmark_custom_node_tp.py` runs a fresh server for TP1 and TP2, then
one cold generation and three warm generations with matched seeds 42–45.
It saves commands, NCCL logs, traces, prompts, histories, JSON/CSV results,
sampled device memory, aggregate process PSS and warm median speedup. It refuses
a TP2 run without a distributed-rank load report. Device-memory figures include
other GPU users; run on idle GPUs and retain the topology and package metadata.
Sampler timings are nullable when the custom sampler does not emit that span.

`release-artifacts/workflows/matrix.json` lists five upstream Hunyuan W4A8
samples, four Kandinsky Pro cases (standard/distilled × text/image conditioning),
and the SR smoke workflow. Primary Kandinsky copies bypass MagCache and disable
prompt beautification; standard Pro uses 50 steps, CFG 5 and audio scale 0.5302.
Converted API workflows and pinned model revision manifests are saved under
`release-artifacts/`. The benchmark can be invoked with:

```sh
comfyui workflows convert sample.json -o sample.api.json
python scripts/benchmark_custom_node_tp.py sample.api.json \
  --comfyui /path/to/release/venv/bin/comfyui \
  --output benchmark-results/sample --revisions model-revisions.json \
  --server-arg=--add-model-folder-path --server-arg=diffusion_models=/path/to/models
```

Full workflow validation began after GPU and network access were restored on
October 10. TP1 timing results appear below; TP2 results remain pending. The remaining model downloads are running; the full unit results appear below. The earlier asynchronous facade timeout no longer reproduces.

The current-base CUDA/NCCL W4A8 test passes for linears and expert banks,
including the installed Hunyuan sparse decode path and patched dense fallback.
It exposed and fixed a core expert-bank cast that dequantized a flat bank before
indexing it as if it still had an expert axis. These checks use synthetic weights.

Spectrum and MagCache transport now reconstruct call-local cache objects on each
rank and return the root update to the original controller. Worker inputs are
released before the next command; MagCache residuals are cleared after sampling,
including cancellation. The installed-cache tests cover skips and state updates;
real-model cache smoke checks remain pending. The implementation broadcasts cache
tensors per call, so its optional-cache timings must account for that overhead.

A fresh server exposed all 39 node types used by the ten converted API samples.
The primary K6 API workflows inline the disabled beautifier captions, avoiding an
unnecessary Qwen3.5 loader dependency. Both image-editing families use the bundled
K6 portrait; `release-artifacts/workflows/inputs.json` records its source and hash.

Before release:

1. Run real-model smoke checks for Spectrum and MagCache state transport.
2. Validate real checkpoint loading, W4A8 numerical parity, PiFlow, rewriting,
   cancellation, clone/offload/cleanup, and SR. Run the full workflow benchmark
   matrix; report regressions as well as improvements, with output comparisons.
3. Verify HTTP node schemas and source assets in a fresh server and clean wheel
   installation. Regenerate all CI locks without upgrading Torch/backends.
4. Reconcile master fixes and the Windows failures described in the release plan;
   complete frontend parity, quantization and normal develop/device CI gates.
5. Build and health-validate the facade image, apply the staged cluster changes,
   verify latest-version refresh and public installation, and confirm Flux moves.
6. Only then tag/publish 0.39.0.2 and run post-publication package/image checks.

The earlier operational recovery replaced the disconnected facade pod and banned
the Varnish `/simple` entries. The public SR index then served 0.1.2. The public SR index was rechecked on October 10 and still serves 0.1.2.
HunyuanImage3 is served by the local candidate facade but remains 404 publicly.

All seven CI lock targets resolved successfully through the candidate facade;
the core-only lock was unchanged. No previously locked package version changed.
The six custom-node locks add the three nodes and their missing dependencies.
Hunyuan wheel URLs target the public facade, so CI installation awaits deployment
of the validated facade candidate. The local index was used for resolution only.

Full local unit run on the 0.39 base: 8,385 passed, 2,165 skipped, one failure
(out of 10,551). The failure was a CUDA capability probe with no visible GPU.
After fixing that probe, all 39 targeted quantization/TP regressions passed.
The full run includes frontend conversion tests; the exact final release still
requires normal CI and device gates. Ruff passed across all core package trees,
and the changed TP modules passed Pylint.

The published distilled Hunyuan checkpoint now passes its upstream CUDA expert
bank test (core versus sliced decode and dequantized reference) and fingerprint
detection: two real-checkpoint tests passed on GPU 1. The first TP1 workflow exposed an upstream lookahead bug: it treated `_v=None`
as an allocated weight buffer. The adapter now uses ComfyUI prefetch queues,
with 25 targeted tests passing, including success and cancellation cleanup.
The restarted TP1 benchmark waits for unrelated compute jobs to release GPU 1;
TP2 likewise waits for both devices. No workflow timing is claimed from the failed run.

The Kandinsky generation samples include their upstream SR postprocessing branch.
End-to-end timings will include SR; its model uses the existing single-device
fallback because the new TP adapter targets the Pro generator. Report generator
and end-to-end timings separately when trace spans are available.

After the latest develop merge, allocator logging and CLI configuration checks
passed (7 passed, 1 skipped).

October 10 follow-up: the live known-node snapshot refresh contains 50 projects
and 951 versions, including Kandinsky 1.0.2. Its 36 Python source files match
the registry archive byte for byte; Pro model math is unchanged from 1.0.1.
All seven locks regenerated, changing only the Kandinsky version.

An incremental refresh of a copy of the production snapshot succeeded with
8,269 projects and 29,030 versions, adding HunyuanImage3. Public Kandinsky wheel
requests exposed disconnected cache mounts on two replicas. Both replicas were
replaced one at a time and their new cache mounts verified. Readiness now probes
local cache usability as well as snapshot freshness; liveness remains independent.
The cache/snapshot suite passed 32 tests, followed by all 16 server tests after
adding an actual filesystem recovery check. The new image still needs deployment.

The distilled Hunyuan text sample completed its TP1 runs: cold 104.01 seconds;
warm 45.42, 45.12, and 46.70 seconds (median 45.42). The first image was visually
inspected and is coherent. TP2 is pending an idle GPU 0; no speedup is claimed yet.
