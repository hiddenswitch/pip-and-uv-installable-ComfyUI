FROM nvcr.io/nvidia/pytorch:25.12-py3

# Use uv without replacing the package manager/runtime selected by the NGC
# image.  The NGC image's CUDA/Torch stack is the ABI contract for this image.
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

ENV TZ="Etc/UTC"

ENV PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:True"
ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_OVERRIDE=/workspace/overrides.txt
ENV UV_CACHE_DIR=/root/.cache/uv

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# mitigates
# RuntimeError: Failed to import transformers.generation.utils because of the following error (look up to see its traceback):
# numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
RUN mkdir -p /workspace /root/.cache/uv && \
    uv pip freeze --system | awk '/^(nvidia-|torch==|torchvision==|torchaudio==|triton==|flash-attn==)/' > /workspace/overrides.txt; \
    echo "onnxruntime-gpu==1.22.0" >> /workspace/overrides.txt; \
    uv pip freeze --system | awk '/^numpy==/' >> /workspace/overrides.txt; \
    printf "%s\\n" "opencv-python; python_version < '0'" >> /workspace/overrides.txt; \
    printf "%s\\n" "opencv-contrib-python; python_version < '0'" >> /workspace/overrides.txt; \
    printf "%s\\n" "opencv-python-headless; python_version < '0'" >> /workspace/overrides.txt; \
    echo "opencv-contrib-python-headless!=4.11.0.86" >> /workspace/overrides.txt; \
    printf "%s\\n" "sentry-sdk; python_version < '0'" >> /workspace/overrides.txt

# mitigates https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
# mitigates AttributeError: module 'cv2.dnn' has no attribute 'DictValue' \
# see https://github.com/facebookresearch/nougat/issues/40
RUN uv --version && \
    apt-get update && apt-get install --no-install-recommends ffmpeg libsm6 libxext6 libcairo2-dev libxcb1 zip unzip -y && \
    opencv_pkgs="$(uv pip freeze --system | grep -i '^opencv' | sed 's/==.*//' || true)" && \
    if [ -n "$opencv_pkgs" ]; then uv pip uninstall --system $opencv_pkgs; fi && \
    rm -rf /usr/local/lib/python3.12/dist-packages/cv2/ && \
    uv pip install wheel && \
    uv pip install --no-build-isolation "opencv-contrib-python-headless>=4.12.0.88" && \
    rm -rf /var/lib/apt/lists/*

# install CUDA acceleration extras from the AppMana stable-ABI indexes
RUN uv pip install -U --no-deps --no-build-isolation spandrel timm tensorboard poetry && \
    uv pip install --no-deps sageattention --index-url https://appmana.github.io/forks-sageattention-stable-abi/cu130 && \
    uv pip install --no-deps nunchaku --index-url https://appmana.github.io/forks-nunchaku-stable-abi/cu130

# sources for building this dockerfile
# use these lines to build from the local fs
ADD . /workspace/src
RUN rm -rf /workspace/src/comfy/cmd/web/extensions/pysssss/CustomScripts /workspace/src/comfy/cmd/web/extensions/pysssss/WD14Tagger
ARG SOURCES="comfyui@./src"
# this builds from github
# useful if you are copying and pasted in order to customize this
# ARG SOURCES="comfyui[attention,comfyui_manager]@git+https://github.com/hiddenswitch/ComfyUI.git"
ENV SOURCES=$SOURCES

RUN uv pip install $SOURCES

WORKDIR /workspace

EXPOSE 8188
CMD ["python", "-m", "comfy.cmd.main", "--listen", "--use-sage-attention", "--reserve-vram=0", "--logging-level=INFO", "--enable-cors"]
