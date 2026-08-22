# CI dependency locks

The CI environments install from the platform-specific `pylock.*.toml` files in
this directory. Backend locks omit ComfyUI and Torch: ComfyUI is installed from
the checkout, while the CUDA, XPU, and Windows jobs retain the Torch build
provided or explicitly selected by that backend. All locks use the single
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

`pylock.linux-py314-core.toml` intentionally omits the development extra and
custom nodes. Master uses it for a short wheel/install/import smoke; the larger
Python 3.14 CPU lock remains the single broad develop validation environment.
