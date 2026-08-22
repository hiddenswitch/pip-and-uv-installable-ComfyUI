#!/usr/bin/env sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: $0 LOCK_FILE PYTHON" >&2
  exit 2
fi

lock_file=$1
python=$2

# Backend jobs deliberately provide their own Torch build. Remember it before
# installing the cross-platform dependency lock and fail if anything replaces
# it. CPU/macOS locks contain Torch themselves and therefore start without it.
torch_before=$(
  "$python" -c 'import torch; print(torch.__version__)' 2>/dev/null || true
)

uv pip install \
  --python "$python" \
  --no-build-isolation \
  -r "$lock_file"

# ComfyUI itself was omitted from every lock so the tested package is always
# this checkout. Its dependencies are already exact, so do not resolve again.
uv pip install --python "$python" --no-deps .

if [ -n "$torch_before" ]; then
  torch_after=$("$python" -c 'import torch; print(torch.__version__)')
  if [ "$torch_after" != "$torch_before" ]; then
    echo "Torch changed during locked install: $torch_before -> $torch_after" >&2
    exit 1
  fi
fi
