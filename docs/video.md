# Video Workflows

ComfyUI LTS supports video workflows with AnimateDiff Evolved.

First, install this package using the [Installation Instructions](#installing).

Then, install the custom nodes packages that support video creation workflows:

```shell
uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-video-frame-interpolation
uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-video-helper-suite
uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-animatediff-evolved
uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-controlnet-aux.git
```

Start creating an AnimateDiff workflow. When using these packages, the appropriate models will download automatically.

## SageAttention

Improve the performance of your Mochi model video generation using **Sage Attention**:

| Device | PyTorch 2.5.1 | SageAttention | S.A. + TorchCompileModel |
|--------|---------------|---------------|--------------------------|
| A5000  | 7.52s/it      | 5.81s/it      | 5.00s/it (but corrupted) |

[Use the default Mochi Workflow.](https://github.com/comfyanonymous/ComfyUI_examples/raw/refs/heads/master/mochi/mochi_text_to_video_example.webp) This does not require any custom nodes or any change to your workflow.

**Installation**

See the [SageAttention installation instructions](installing.md#sageattention) for platform-specific setup.

To start ComfyUI with it:

```shell
uv run comfyui --use-sage-attention
```

![with_sage_attention.webp](./docs/assets/with_sage_attention.webp)
**With SageAttention**

![with_pytorch_attention](./docs/assets/with_pytorch_attention.webp)
**With PyTorch Attention**

## Flash Attention

Flash Attention 2 is supported on Linux only.

```shell
uv pip install --no-build-isolation flash_attn
```

To start ComfyUI with it:

```shell
uv run comfyui --use-flash-attention
```

![with_sage_attention.webp](./docs/assets/with_sage_attention.webp)
**With SageAttention**

![with_pytorch_attention](./docs/assets/with_pytorch_attention.webp)
**With PyTorch Attention**

## Cosmos Prompt Upsampling

The Cosmos prompt "upsampler," a fine tune of Mistral-Nemo-12b, correctly rewrites Cosmos prompts in the narrative style that NVIDIA's captioner used for the training data of Cosmos, improving generation results significantly.

Here is a comparison between a simple and "upsampled" prompt.

![prompt_upsampling_01.webp](assets/prompt_upsampling_01.webp)
**A dog is playing with a ball.**

![prompt_upsampling_02.webp](assets/prompt_upsampling_02.webp)
**In a sun-drenched park, a playful golden retriever bounds joyfully across the lush green grass, its tail wagging with excitement. The dog, adorned with a vibrant red collar, is captivated by a bright yellow ball, which it cradles gently in its mouth. The camera captures the dog's animated expressions, from eager anticipation to sheer delight, as it trots and leaps, showcasing its agility and enthusiasm. The scene is bathed in warm, golden-hour light, enhancing the vibrant colors of the dog's fur and the ball. The background features a serene tree line, framing the playful interaction and creating a tranquil atmosphere. The static camera angle allows for an intimate focus on the dog's joyful antics, inviting viewers to share in this heartwarming moment of pure canine happiness.**

To use the Cosmos upsampler, install the prerequisites:

```shell
uv pip install loguru pynvml
uv pip install --no-deps git+https://github.com/NVIDIA/Cosmos.git
```

Then, use the workflow embedded in the upsampled prompt by dragging and dropping the upsampled animation into your workspace.

The Cosmos upsampler ought to improve any text-to-image video generation pipeline. Use the `Video2World` upsampler nodes to download Pixtral-12b and upsample for an image to video workflow using NVIDIA's default prompt. Since Pixtral is not fine tuned, the improvement may not be significant over using another LLM.
