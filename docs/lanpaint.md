# LanPaint workflow guidance

LanPaint edits are mask-constrained diffusion edits. The prompt describes
the whole intended image; the mask determines which pixels the workflow is
allowed to change.

## ComfyUI mask meaning

ComfyUI masks are grayscale tensors where `1.0` means masked/selected and
`0.0` means unmasked/unselected. The confusing part is that each node decides
what "masked" means for its operation: for one node it may mean "paste here",
for another it may mean "preserve this", "hide this", or "exclude this".
Always verify the polarity against the exact node path being used.

For the LanPaint Ideogram workflow used here, the editable bbox must be the
active mask region. In practice:

- `1.0` inside the target bbox means LanPaint may edit that region.
- `0.0` outside the target bbox means LanPaint should leave that region out of
  the edit/blend region.
- A mask with `0.0` inside the bbox and `1.0` outside is inverted for this
  workflow and will preserve the requested edit region instead of changing it.

To build a rectangular LanPaint edit mask with ComfyUI nodes, create a
full-zero mask for the whole image, create a one-valued rectangle for the
target bbox, then composite the rectangle onto the full mask with `or`. The
result is `1.0` in the bbox and `0.0` everywhere else.

## Prompt discipline

Treat structured prompts as image descriptions, not as instruction
documents. For a localized edit, reuse the original whole-image prompt and
change only the exact requested field.

Do not add:

- `constraints`
- "masked edit"
- "inpainting"
- replacement/process prose
- preservation/process prose
- changes to unrelated text regions

The mask carries the edit locality. The prompt should only describe the
target image.

## JSON prompt edits

When editing an Ideogram-style JSON prompt:

- Preserve the original whole-image JSON.
- Change only the requested element field.
- Use the bbox of that same changed element to build the LanPaint mask.
- Keep unrelated title, type-line, footer, flavor text, object descriptions,
  palettes, and background unchanged.

Ideogram JSON bboxes use `[y_min, x_min, y_max, x_max]` on a 0-1000 grid.
For pixel-space mask nodes, convert with:

```text
x = round(x_norm * image_width / 1000)
y = round(y_norm * image_height / 1000)
```

Use half-open pixel rectangles: `[left, top, right, bottom)`.

## Build the edit inside the workflow

Prefer workflow nodes over sidecar files:

- Use `PrimitiveStringMultiline` for the original JSON prompt.
- Use `EvalPython_*` or a custom node to change exactly one field.
- Use mask nodes such as `SolidMask` and `MaskComposite` to create the bbox
  mask in the graph.
- Wire the generated prompt into the text encoder.
- Wire the generated mask into the LanPaint latent mask and final blend.

Do not pre-bake sidecar prompt JSON, alpha PNGs, or mask PNGs unless the
user explicitly asks for files.
