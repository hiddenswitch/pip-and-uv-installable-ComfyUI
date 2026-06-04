# NewBie Image Exp0.1 model card

Vendored snapshot from: https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1

Fetched: 2026-06-03

---

---
license: other
license_name: newbie-nc-1.0
license_link: LICENSE.md
language:
- en
pipeline_tag: text-to-image
library_name: diffusers
tags:
- next-dit
- text-to-image
- transformer
- image-generation
- Anime
---


<h1 align="center">NewBie image Exp0.1<br><sub><sup>Efficient Image Generation Base Model Based on Next-DiT</sup></sub></h1>

<div align="center">

[![GitHub-NewBie](https://img.shields.io/badge/GitHub-NewBie%20image%20Exp0.1-181717?logo=github&logoColor=white)](https://github.com/NewBieAI-Lab/NewBie-image-Exp0.1)&#160;
[![GitHub - LoRa Trainer](https://img.shields.io/badge/GitHub-LoRa%20Trainer-181717?logo=github&logoColor=white)](https://github.com/NewBieAI-Lab/NewbieLoraTrainer)&#160;
[![GitHub - ComfyUI-NewBie](https://img.shields.io/badge/GitHub-ComfyUI--NewBie-181717?logo=github&logoColor=white)](https://github.com/E-Anlia/ComfyUI-NewBie)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-NewBie%20image%20Exp0.1-yellow)](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1)&#160;
[![MS](https://img.shields.io/badge/🤖%20Checkpoint-NewBie%20image%20Exp0%2E1-624aff)](https://www.modelscope.cn/models/NewBieAi-lab/NewBie-image-Exp0.1)
![C5BDBA2F0B1D85D81D3A9DCADF6DED1F](https://cdn-uploads.huggingface.co/production/uploads/67fdc3911c5d7301352a0507/qB2wyVrTuYBtg_ToRb2wP.jpeg)
</div>

## 🧱 Exp0.1 Base
**NewBie image Exp0.1** is a **3.5B** parameter DiT model developed through research on the Lumina architecture.
Building on these insights, it adopts Next-DiT as the foundation to design a new NewBie architecture tailored for text-to-image generation.
The *NewBie image Exp0.1* model is trained within this newly constructed system, representing the first experimental release of the NewBie text-to-image generation framework.
#### Text Encoder
We use Gemma3-4B-it as the primary text encoder, conditioning on its penultimate-layer token hidden states. We also extract pooled text features from Jina CLIP v2, project them, and fuse them into the time/AdaLN conditioning pathway.
Together, Gemma3-4B-it and Jina CLIP v2 provide strong prompt understanding and improved instruction adherence.
#### VAE
Use the FLUX.1-dev 16channel VAE to encode images into latents, delivering richer, smoother color rendering and finer texture detail helping safeguard the stunning visual quality of NewBie image Exp0.1.

## 🖼️ Task type
<div align="center">
  
**NewBie image Exp0.1** is pretrain on a large corpus of high-quality anime data, enabling the model to generate remarkably detailed and visually striking anime style images.
![NewBie image preview](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/resolve/main/image/newbie_image.png)
We reformatted the dataset text into an **XML structured format** for our experiments. Empirically, this improved attention binding and attribute/element disentanglement, and also led to faster convergence.

Besides that, It also supports natural language and tags inputs.

**In multi character scenes, using XML structured prompt typically leads to more accurate image generation results.**
  </div>

<div style="display:flex; gap:16px; align-items:flex-start;">

  <div style="flex:1; min-width:0;">
    <details open style="box-sizing:border-box; border:1px solid #e5e7eb; border-radius:10px; padding:12px; height:260px; overflow:auto;">
      <summary><b>XML structured prompt</b></summary>

```prompt
  <character_1>
  <n>$character_1$</n>
  <gender>1girl</gender>
  <appearance>chibi, red_eyes, blue_hair, long_hair, hair_between_eyes, head_tilt, tareme, closed_mouth</appearance>
  <clothing>school_uniform, serafuku, white_sailor_collar, white_shirt, short_sleeves, red_neckerchief, bow, blue_skirt, miniskirt, pleated_skirt, blue_hat, mini_hat, thighhighs, grey_thighhighs, black_shoes, mary_janes</clothing>
  <expression>happy, smile</expression>
  <action>standing, holding, holding_briefcase</action>
  <position>center_left</position>
  </character_1>

  <character_2>
  <n>$character_2$</n>
  <gender>1girl</gender>
  <appearance>chibi, red_eyes, pink_hair, long_hair, very_long_hair, multi-tied_hair, open_mouth</appearance>
  <clothing>school_uniform, serafuku, white_sailor_collar, white_shirt, short_sleeves, red_neckerchief, bow, red_skirt, miniskirt, pleated_skirt, hair_bow, multiple_hair_bows, white_bow, ribbon_trim, ribbon-trimmed_bow, white_thighhighs, black_shoes, mary_janes, bow_legwear, bare_arms</clothing>
  <expression>happy, smile</expression>
  <action>standing, holding, holding_briefcase, waving</action>
  <position>center_right</position>
  </character_2>

  <general_tags>
  <count>2girls, multiple_girls</count>
  <style>anime_style, digital_art</style>
  <background>white_background, simple_background</background>
  <atmosphere>cheerful</atmosphere>
  <quality>high_resolution, detailed</quality>
  <objects>briefcase</objects>
  <other>alternate_costume</other>
  </general_tags>
```
</details> 

</div>
<div style="box-sizing:border-box; width:260px; height:260px; flex:0 0 260px; border:1px solid #e5e7eb; border-radius:10px; padding:12px; display:flex; align-items:center; justify-content:center;"> 
  <img src="https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/resolve/main/image/XML_prompt_image.png" alt="XML prompt image" style="max-width:100%; max-height:100%; object-fit:contain; display:block;" /> 
</div>
</div>
<h1 align="center"><br><sub><sup>XML structured prompt and attribute/element disentanglement showcase</sup></sub></h1>
</div>

## 🧰 Model Zoo
| Model | Hugging Face | ModelScope |
| :--- | :--- | :--- |
| **NewBie image Exp0.1** | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-NewBie%20image%20Exp0%2E1-yellow)](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1) | [![MS](https://img.shields.io/badge/🤖%20Checkpoint-NewBie%20image%20Exp0%2E1-624aff)](https://www.modelscope.cn/models/NewBieAi-lab/NewBie-image-Exp0.1) |
| **Gemma3-4B-it** | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-Gemma3--4B--it-yellow)](https://huggingface.co/google/gemma-3-4b-it) | [![MS](https://img.shields.io/badge/🤖%20Checkpoint-Gemma3--4B--it-624aff)](https://www.modelscope.cn/models/google/gemma-3-4b-it) |
| **Jina CLIP v2** | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-Jina%20CLIP%20v2-yellow)](https://huggingface.co/jinaai/jina-clip-v2) | [![MS](https://img.shields.io/badge/🤖%20Checkpoint-Jina%20CLIP%20v2-624aff)](https://www.modelscope.cn/models/jinaai/jina-clip-v2) |
| **FLUX.1-dev VAE** | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-FLUX%2E1--dev%20VAE-yellow)](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/vae/diffusion_pytorch_model.safetensors) | [![MS](https://img.shields.io/badge/🤖%20Checkpoint-FLUX%2E1--dev%20VAE-624aff)](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev/tree/master/vae) |

## 🚀 Quickstart
- **Diffusers**
```bash
pip install diffusers transformers accelerate safetensors torch --upgrade
# Recommended: install FlashAttention and Triton according to your operating system.
```
```python
import torch
from diffusers import NewbiePipeline

def main():
    model_id = "NewBie-AI/NewBie-image-Exp0.1"

    # Load pipeline
    pipe = NewbiePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
  # use float16 if your GPU does not support bfloat16

    prompt = "1girl"

    image = pipe(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=28,
    ).images[0]

    image.save("newbie_sample.png")
    print("Saved to newbie_sample.png")

if __name__ == "__main__":
    main()
```
- **ComfyUI**


## 💪 Training procedure
![NewBie image preview](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/resolve/main/image/NewBie_image_Exp0.1_Training.png)

## 🔬 Participate
#### *Core*
- **[Anlia](https://huggingface.co/E-Anlia) | [CreeperMZ](https://huggingface.co/CreeperMZ) | [L_A_X](https://huggingface.co/LAXMAYDAY) | [maikaaomi](https://huggingface.co/maikaaomi) | [waw1w1](https://huggingface.co/xuefei123456) | [LakiCat](https://huggingface.co/LakiCat) | [chenkin](https://huggingface.co/windsingai) | [aplxaplx](https://huggingface.co/aplx) | [NULL](https://huggingface.co/GuChen)** 
#### *Members*
- **[niangao233](https://huggingface.co/niangao233) | [ginkgowm](https://huggingface.co/ginkgowm) | [leafmoone](https://huggingface.co/leafmoone) | [NaviVoid](https://huggingface.co/NaviVoid) | [Emita](https://huggingface.co/Emita) | [TLFZ](https://huggingface.co/TLFZ) | [3HOOO](https://huggingface.co/3HOOO)**

## ✨ Acknowledgments
- Thanks to the [Alpha-VLLM Org](https://huggingface.co/Alpha-VLLM) for open sourcing the advanced [Lumina](https://huggingface.co/collections/Alpha-VLLM/lumina-family) family.
  which has been invaluable for our research.
- Thanks to [Google](https://huggingface.co/google) for open sourcing the powerful [Gemma3](https://huggingface.co/google/gemma-3-4b-it) LLM family
- Thanks to the [Jina AI Org](https://huggingface.co/jinaai) for open sourcing the [Jina](https://huggingface.co/jinaai/jina-clip-v2) family, enabling further research.
- Thanks to [Black Forest Labs](https://huggingface.co/black-forest-labs) for open sourcing the [FLUX VAE](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/vae) family.
  powerful 16channel VAE is one of the key components behind improved image quality.
- Thanks to [Neta.art](https://huggingface.co/neta-art) for fine-tuning and open sourcing the [Lumina-image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) base model.
  [Neta-Lumina](https://huggingface.co/neta-art/Neta-Lumina) gives us the opportunity to study the performance of Next-DiT on Anime Types.
- Thanks to [DeepGHS](https://huggingface.co/deepghs)/[narugo1992](https://huggingface.co/narugo1992)/[SumomoLee](https://huggingface.co/SumomoLee) for providing high-quality Anime Datasets.
- Thanks to [Nyanko](https://huggingface.co/nyanko7) for the early help and support.
- Thanks to [woctordho](https://huggingface.co/woctordho) for helping improve NewBie compatibility with community tools.
  
## 📖 Contribute
- *Neko, 衡鲍, XiaoLxl, xChenNing, Hapless, Lius*
- *WindySea, 秋麒麟热茶, 古柯, Rnglg2, Ly, GHOSTLXH*
- *Sarara, Seina, KKT机器人, NoirAlmondL, 天满, 暂时*
- *Wenaka喵, ZhiHu, BounDless, DetaDT, 紫影のソナーニル*
- *花火流光, R3DeK, 圣人A, 王王玉, 乾坤君Sennke, 砚青*
- *Heathcliff01, 无音, MonitaChan, WhyPing, TangRenLan*
- *HomemDesgraca, EPIC, ARKBIRD, Talan, 448, Hugs288*

## 🧭 Community Guide
#### *Getting Started Guide*
- [English](https://ai.feishu.cn/wiki/NZl9wm7V1iuNzmkRKCUcb1USnsh)
- [中文](https://ai.feishu.cn/wiki/P3sgwUUjWih8ZWkpr0WcwXSMnTb)
#### *LoRa Trainer*
- [English](https://www.notion.so/Newbie-AI-lora-training-tutorial-English-2c2e4ae984ab8177b312e318827657e6?source=copy_link)
- [中文](https://www.notion.so/Newbie-AI-lora-2b84f7496d81803db524f5fc4a9c94b9?source=copy_link)

## 💬 Communication
- [Discord](https://discord.gg/bDJjy7rBGm)
- [解构原典](https://pd.qq.com/s/a79to55q6)
- [ChatGroup](https://qm.qq.com/q/qnHFwN9fSE)

## 📜 License
**Model Weights:** Newbie Non-Commercial Community License (Newbie-NC-1.0).  
- Applies to: model weights/parameters/configs and derivatives (fine-tunes, LoRA, merges, quantized variants, etc.)
- For Non Commercial use only, and must be shared under the same license.
- See [LICENSE.md](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/blob/main/LICENSE.md)

**Code:** Apache License 2.0.  
- Applies to: training/inference scripts and related source code in this project.
- See: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)

## ⚠️ Disclaimer
**This model may produce unexpected or harmful outputs. Users are solely responsible for any risks and potential consequences arising from its use.**
