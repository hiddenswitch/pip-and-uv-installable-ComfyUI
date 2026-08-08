from __future__ import annotations

from functools import lru_cache

from ..model_patcher import get_model_patcher_class


class PipelineModelPatcherMixin:
    pipeline_additional_models_key = "pipeline_parallel"

    def _pipeline_stage_patchers(self):
        return self.get_additional_models_with_key(self.pipeline_additional_models_key)

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        applied = set(super().add_patches(patches, strength_patch, strength_model))
        for patcher in self._pipeline_stage_patchers():
            applied.update(patcher.add_patches(patches, strength_patch, strength_model))
        executor = self.get_attachment("pipeline_parallel_executor")
        if executor is not None and hasattr(executor, "add_patches"):
            applied.update(executor.add_patches(patches, strength_patch, strength_model))
        return list(applied)

    def model_state_dict(self, filter_prefix=None):
        state_dict = super().model_state_dict(filter_prefix=filter_prefix)
        for patcher in self._pipeline_stage_patchers():
            for key, value in patcher.model_state_dict(filter_prefix=filter_prefix).items():
                state_dict.setdefault(key, value)
        return state_dict

    def get_key_patches(self, filter_prefix=None):
        patches = super().get_key_patches(filter_prefix=filter_prefix)
        for patcher in self._pipeline_stage_patchers():
            for key, value in patcher.get_key_patches(filter_prefix=filter_prefix).items():
                patches.setdefault(key, value)
        return patches

@lru_cache(maxsize=2)
def get_pipeline_model_patcher_class(disable_dynamic=False):
    base = get_model_patcher_class(disable_dynamic=disable_dynamic)
    return type("PipelineModelPatcher", (PipelineModelPatcherMixin, base), {})
