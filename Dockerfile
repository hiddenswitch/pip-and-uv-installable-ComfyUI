FROM nvcr.io/nvidia/pytorch:25.12-py3

ENV TZ="Etc/UTC"

ENV PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:True"
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_OVERRIDE=/workspace/overrides.txt

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# mitigates
# RuntimeError: Failed to import transformers.generation.utils because of the following error (look up to see its traceback):
# numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
RUN echo "onnxruntime-gpu==1.22.0" >> /workspace/overrides.txt; pip freeze | grep nvidia >> /workspace/overrides.txt; python -c "import torch; print(f'torch=={torch.__version__}')" >> /workspace/overrides.txt; pip freeze | grep numpy >> /workspace/overrides.txt; echo "opencv-python; python_version < '0'" >> /workspace/overrides.txt; echo "opencv-contrib-python; python_version < '0'" >> /workspace/overrides.txt; echo "opencv-python-headless; python_version < '0'" >> /workspace/overrides.txt; echo "opencv-contrib-python-headless!=4.11.0.86" >> /workspace/overrides.txt; echo "sentry-sdk; python_version < '0'" >> /workspace/overrides.txt

# mitigates https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
# mitigates AttributeError: module 'cv2.dnn' has no attribute 'DictValue' \
# see https://github.com/facebookresearch/nougat/issues/40
RUN pip install uv && uv --version && \
    apt-get update && apt-get install --no-install-recommends ffmpeg libsm6 libxext6 libcairo2-dev libxcb1 -y && \
    opencv_pkgs="$(pip list --format=freeze | grep opencv || true)" && \
    if [ -n "$opencv_pkgs" ]; then uv pip uninstall --system $opencv_pkgs; fi && \
    rm -rf /usr/local/lib/python3.12/dist-packages/cv2/ && \
    uv pip install wheel && \
    uv pip install --no-build-isolation "opencv-contrib-python-headless>=4.12.0.88" && \
    rm -rf /var/lib/apt/lists/*

# install sageattention
ADD pkg/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl /workspace/pkg/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
RUN uv pip install -U --no-deps --no-build-isolation spandrel timm tensorboard poetry "flash-attn<=2.8.0" "xformers==0.0.31.post1" "file:./pkg/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
# Some NGC images ship newer CUDA tags before public torchaudio wheels exist; in that case keep
# the image-bundled torchaudio instead of failing the build.
RUN torch_spec="$(python -c 'import torch, re; m = re.match(r\"(\\d+\\.\\d+\\.\\d+)\", torch.__version__); print(f\"{m.group(1)}+cu{torch.version.cuda.replace(\\\".\\\", \\\"\\\")}\")')" && \
    uv pip install --no-deps "torchaudio==${torch_spec}" --extra-index-url "https://download.pytorch.org/whl/cu$(python -c 'import torch; print(torch.version.cuda.replace(\".\", \"\"))')" || \
    echo "No published torchaudio wheel for ${torch_spec}; keeping image-bundled torchaudio."

# sources for building this dockerfile
# use these lines to build from the local fs
ADD . /workspace/src
ARG SOURCES="comfyui[attention,comfyui_manager]@./src"
# this builds from github
# useful if you are copying and pasted in order to customize this
# ARG SOURCES="comfyui[attention,comfyui_manager]@git+https://github.com/hiddenswitch/ComfyUI.git"
ENV SOURCES=$SOURCES

RUN uv pip install $SOURCES

WORKDIR /workspace
# addresses https://github.com/pytorch/pytorch/issues/104801
# and issues reported by importing nodes_canny
# smoke test
RUN python - <<'PY' \
&& comfyui --quick-test-for-ci --cpu --cwd /workspace
import importlib
import torch
import xformers
import cv2
import diffusers.hooks

if importlib.util.find_spec("sageattention") is not None:
    try:
        import sageattention
    except Exception as exc:  # pragma: no cover - image smoke test
        print(f"Skipping sageattention smoke import: {exc}")
PY

EXPOSE 8188
CMD ["python", "-m", "comfy.cmd.main", "--listen", "--use-sage-attention", "--reserve-vram=0", "--logging-level=INFO", "--enable-cors"]
