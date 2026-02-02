import comfy.memory_management
import comfy.model_patcher

import comfy_aimdo.control
import comfy_aimdo.torch

from comfy.cli_args import enables_dynamic_vram

if enables_dynamic_vram():
    if comfy_aimdo.control.init_device(comfy.model_management.get_torch_device().index):
        if args.verbose == 'DEBUG':
            comfy_aimdo.control.set_log_debug()
        elif args.verbose == 'CRITICAL':
            comfy_aimdo.control.set_log_critical()
        elif args.verbose == 'ERROR':
            comfy_aimdo.control.set_log_error()
        elif args.verbose == 'WARNING':
            comfy_aimdo.control.set_log_warning()
        else: #INFO
            comfy_aimdo.control.set_log_info()

        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_allocator = comfy_aimdo.torch.get_torch_allocator()
        logging.info("DynamicVRAM support detected and enabled")
    else:
        logging.info("No working comfy-aimdo install detected. DynamicVRAM support disabled. Falling back to legacy ModelPatcher. VRAM estimates may be unreliable especially on Windows")
        comfy.memory_management.aimdo_allocator = None
