"""Compatibility shims for third-party node types used in workflow templates.

These stubs allow workflow conversion (UI→API) to succeed for templates
that reference nodes from popular custom-node packs (ComfyMath, KJNodes, etc.)
even when those packs are not installed.
"""
from comfy.node_helpers import export_custom_nodes
from comfy.nodes.package_typing import CustomNode, InputTypes


class CM_IntToFloat(CustomNode):
    """ComfyMath CM_IntToFloat: INT input 'a' -> FLOAT output."""
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "a": ("INT", {"default": 0}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, a: int = 0):
        return (float(a),)


class GetImageRangeFromBatch(CustomNode):
    """KJNodes GetImageRangeFromBatch: extract a range from image/mask batches."""
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "start_index": ("INT", {"default": 0, "min": -1, "max": 4096, "step": 1}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
            }
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"

    def execute(self, start_index, num_frames, images=None, masks=None):
        chosen_images = None
        chosen_masks = None
        if images is not None:
            if start_index == -1:
                start_index = max(0, len(images) - num_frames)
            end_index = min(start_index + num_frames, len(images))
            chosen_images = images[start_index:end_index]
        if masks is not None:
            if start_index == -1:
                start_index = max(0, len(masks) - num_frames)
            end_index = min(start_index + num_frames, len(masks))
            chosen_masks = masks[start_index:end_index]
        return (chosen_images, chosen_masks)


export_custom_nodes()
