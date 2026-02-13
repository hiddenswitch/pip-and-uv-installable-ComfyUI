import functools
import re
from importlib.resources import as_file, files
from typing import TypedDict, NamedTuple

import PIL.Image
import numpy as np
import torch
from jaxtyping import Float

from comfy.component_model.tensor_types import RGBImageBatch, MaskBatch
from comfy.nodes.package_typing import CustomNode, InputTypes

_MODEL_PATH = 'vae-oid.npz'

_SEGMENT_DETECT_RE = re.compile(
    r'(.*?)' +
    r'<loc(\d{4})>' * 4 + r'\s*' +
    '(?:%s)?' % (r'<seg(\d{3})>' * 16) +
    r'\s*([^;<>]+)? ?(?:; )?',
)

PALIGEMMA_OUTPUT_NAME = "PALIGEMMA_OUTPUT"


class BoundingBox(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int


PaligemmaMask = Float[np.ndarray, "height width"]


class ExtractedPaligemmaSegmented(TypedDict):
    content: str
    xyxy: BoundingBox
    mask: PaligemmaMask | None
    name: str


class ExtractedPaligemmaContentOnly(TypedDict):
    content: str


ExtractedPaligemmaObject = ExtractedPaligemmaSegmented | ExtractedPaligemmaContentOnly
PostProcessResult = list[ExtractedPaligemmaObject]


class _ResBlock(torch.nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(features, features, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(features, features, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(features, features, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class _Decoder(torch.nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(embedding_dim, 128, 1),
            torch.nn.ReLU(),
            _ResBlock(128),
            _ResBlock(128),
            torch.nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        return self.decoder(x)


@functools.cache
def _get_reconstruct_masks():
    """Reconstructs masks from codebook indices.
    Returns:
      A function that expects indices shaped `[B, 16]` of dtype int32, each
      ranging from 0 to 127 (inclusive), and that returns decoded masks sized
      `[B, 64, 64, 1]`, of dtype float32, in range [-1, 1].
    """
    with as_file(files("comfy_extras.paligemma") / _MODEL_PATH) as f:
        checkpoint = dict(np.load(f))

    embeddings = torch.from_numpy(checkpoint['_vq_vae._embedding']).float()
    embedding_dim = embeddings.shape[1]

    model = _Decoder(embedding_dim)
    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('_vq_vae.') or key.startswith('encoder.'):
            continue
        state_dict[key] = torch.from_numpy(value).float()
    model.load_state_dict(state_dict)
    model.eval()

    @torch.inference_mode()
    def reconstruct_masks(codebook_indices):
        indices = torch.from_numpy(np.asarray(codebook_indices)).long()
        batch_size, num_tokens = indices.shape
        assert num_tokens == 16, indices.shape

        encodings = embeddings[indices.reshape(-1)]
        encodings = encodings.reshape(batch_size, 4, 4, embedding_dim)
        encodings = encodings.permute(0, 3, 1, 2)

        output = model(encodings)
        return output.permute(0, 2, 3, 1).numpy()

    return reconstruct_masks


def extract_objs(text, width, height, unique_labels=False) -> PostProcessResult:
    """Returns objs for a string with "<loc>" and "<seg>" tokens."""
    objs: list[ExtractedPaligemmaObject] = []
    seen = set()
    while text:
        m = _SEGMENT_DETECT_RE.match(text)
        if not m:
            break
        gs = list(m.groups())
        before = gs.pop(0)
        name = gs.pop()
        y1, x1, y2, x2 = [int(x) / 1024 for x in gs[:4]]

        y1, x1, y2, x2 = map(round, (y1 * height, x1 * width, y2 * height, x2 * width))
        seg_indices = gs[4:20]
        if seg_indices[0] is None:
            mask = None
        else:
            seg_indices = np.array([int(x) for x in seg_indices], dtype=np.int32)
            m64, = _get_reconstruct_masks()(seg_indices[None])[..., 0]
            m64 = np.clip(np.array(m64) * 0.5 + 0.5, 0, 1)
            m64 = PIL.Image.fromarray((m64 * 255).astype('uint8'))
            mask = np.zeros([height, width])
            if y2 > y1 and x2 > x1:
                mask[y1:y2, x1:x2] = np.array(m64.resize([x2 - x1, y2 - y1])) / 255.0

        content = m.group()
        if before:
            objs.append(dict(content=before))
            content = content[len(before):]
        while unique_labels and name in seen:
            name = (name or '') + "'"
        seen.add(name)
        paligemma_output_obj: ExtractedPaligemmaObject = {'content': content, 'xyxy': BoundingBox(x1, y1, x2, y2), 'mask': mask, 'name': name}
        objs.append(paligemma_output_obj)
        text = text[len(before) + len(content):]

    if text:
        objs.append(dict(content=text))

    return [obj for obj in objs if obj["content"] != '<eos>']


class PaligemmaPostProcess(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "generated_text": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "images": ("IMAGE", {}),
            }
        }

    CATEGORY = "language"
    RETURN_TYPES = (PALIGEMMA_OUTPUT_NAME,)
    RETURN_NAMES = ("paligemma output",)
    FUNCTION = "execute"

    def execute(self, generated_text: str = "", task: str = "", images: RGBImageBatch = None) -> tuple[PostProcessResult]:
        return extract_objs(generated_text, images.shape[-2], images.shape[-3]),


class PaligemmaOutputToMask(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "paligemma_output": (PALIGEMMA_OUTPUT_NAME, {"forceInput": True}),
            },
        }

    CATEGORY = "language"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("paligemma output",)
    FUNCTION = "execute"

    def execute(self, paligemma_output: PostProcessResult) -> tuple[MaskBatch]:
        masks = [torch.from_numpy(p["mask"]) for p in paligemma_output if "mask" in p]
        if len(masks) == 0:
            return torch.zeros((0, 0, 0)),
        return torch.stack(masks, dim=0),


NODE_CLASS_MAPPINGS = {}
for cls in (
        PaligemmaOutputToMask,
        PaligemmaPostProcess,
):
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
