"""
Workaround for a torch+xpu kernel crash on Intel Arc DG2 (Alchemist) GPUs.

Arc A310/380/580/750/770 with the xe driver crash with
UR_RESULT_ERROR_UNKNOWN -> DEVICE_LOST when torch.triu / torch.tril (or
their method / in-place variants) execute on an XPU tensor after a
CLIP-shaped allocation pattern. CPU detour avoids the bug.

Reproduces on torch 2.9.0+xpu, 2.10.0+xpu, 2.11.0+xpu (every currently
published xpu wheel). Self-contained Docker reproducer + upstream issue
link live in the repo's CI history.

This module monkey-patches torch.{triu,tril} and Tensor.{triu,tril,
triu_,tril_}. For non-XPU tensors the wrappers tail-call the original,
so CUDA / ROCm / MPS / CPU / non-DG2 XPU users see no behavior change.
The wrappers cost one extra Python frame on the unaffected path, which
is negligible because triu/tril are rarely-called and operate on small
tensors.

Activation: imported once from model_management.py when the
INTEL_XPU_DG2_TRIU_WORKAROUND flag fires. On other hardware this module
is never imported and incurs zero cost.

Cleanup: when upstream torch+xpu fixes the kernel, gate the activation
in model_management.py on the broken torch versions only, or remove the
flag and this module entirely.
"""
import torch


_orig_triu_func = torch.triu
_orig_tril_func = torch.tril
_orig_triu_method = torch.Tensor.triu
_orig_tril_method = torch.Tensor.tril
_orig_triu_inplace = torch.Tensor.triu_
_orig_tril_inplace = torch.Tensor.tril_


def _make_safe(orig):
    # Wraps torch.{triu,tril} (free fns) and Tensor.{triu,tril} (methods).
    # Both call shapes have the tensor as the first positional argument.
    def safe(*args, **kwargs):
        tensor = args[0] if args else None
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "xpu":
            out = kwargs.pop("out", None)
            cpu_result = orig(tensor.cpu(), *args[1:], **kwargs)
            if out is not None:
                return out.copy_(cpu_result.to(out.device))
            return cpu_result.to(tensor.device)
        return orig(*args, **kwargs)
    return safe


def _make_safe_inplace(orig_inplace):
    def safe_inplace(self, diagonal=0):
        if self.device.type == "xpu":
            cpu_self = self.cpu()
            orig_inplace(cpu_self, diagonal)
            self.copy_(cpu_self.to(self.device))
            return self
        return orig_inplace(self, diagonal)
    return safe_inplace


torch.triu = _make_safe(_orig_triu_func)
torch.tril = _make_safe(_orig_tril_func)
torch.Tensor.triu = _make_safe(_orig_triu_method)
torch.Tensor.tril = _make_safe(_orig_tril_method)
torch.Tensor.triu_ = _make_safe_inplace(_orig_triu_inplace)
torch.Tensor.tril_ = _make_safe_inplace(_orig_tril_inplace)
