# CI dependency locks

The CI environments install from the platform-specific `pylock.*.toml` files in
this directory. Backend locks omit ComfyUI and Torch: ComfyUI is installed from
the checkout, while the CUDA, XPU, and Windows jobs retain the Torch build
provided or explicitly selected by that backend. CUDA image builds freeze the
Python environment supplied by NGC, except for named compatibility exceptions,
as resolver overrides before adding ComfyUI. Hardware jobs test that immutable
candidate image without installing anything. All locks use the single
headless OpenCV distribution in `headless-requirements.txt` instead of
installing multiple distributions that provide the same `cv2` module.

Regenerate every lock from the repository root after changing project or custom
node dependencies:

```sh
ci/generate-locks.sh
```

Lock changes are reviewed and committed with the dependency change that caused
them. CI installs the locks without resolving dependency versions. Lock
generation intentionally uses `--no-config`: the files listed by the script are
the complete inputs, so an ambient project or user uv configuration cannot
silently change a lock. The AppMana node index is combined with the public
indexes only while generating the trusted lock; installs use the exact URLs and
hashes recorded in the resulting pylock files.

Pre-releases are selected only when a project explicitly requests one or no
stable distribution supports the target. This keeps Python 3.14 workable
without turning every dependency into a moving nightly-build target.

The macOS lock targets macOS 14 and Python 3.12. It includes the native macOS
PyTorch 2.13 wheels; Python 3.15 is not one of the CI targets.

`pylock.linux-py312-numpy1.toml` is the compatibility lock for the CUDA 12.8
NGC image. Its Torch and bundled extensions use the NumPy 1 ABI, so the whole
NumPy/OpenCV/SciPy set is resolved together under `numpy<2`; newer backends use
the regular Python 3.12 lock. `numpy1-overrides.txt` replaces the normal exact
OpenCV pin with the last NumPy-1-compatible wheel for that resolution.

`pylock.linux-py314-core.toml` intentionally omits the development extra and
custom nodes. Master uses it for a short wheel/install/import smoke; the larger
Python 3.14 CPU lock remains the single broad develop validation environment.

## Accelerator image promotion

CUDA and ROCm builds first publish immutable candidate tags such as
`<full-commit>-cuda133-<run>-<attempt>.dev` and
`<full-commit>-rocm-<run>-<attempt>.dev`. Hardware jobs execute
the application and tests already baked into those images; they do not mutate
the environment. Only after every required check succeeds is the exact manifest
retagged with the short commit and branch channel (`develop-*` or `latest-*`).

OCI defines annotations for source, revision, version, and base-image identity,
but it does not define a prerelease flag. The `.dev` tag suffix is therefore the
human-visible candidate channel; content digests provide the immutable identity.

The CUDA build has explicit, narrow exceptions to NGC's Python package set.
Its OpenCV provider is replaced by the single headless contrib provider required
by ComfyUI; `huggingface-hub` and `tokenizers` resolve with the supported
Transformers line; `protobuf` and `requests` resolve with ComfyUI's API and
object-store clients; and the official CPU TorchAudio wheel avoids TorchAudio's
exact CUDA-minor check while retaining Torch-backed tensor transforms. The
build compares every other NGC distribution before and after installation and
fails if any version changes.
