from __future__ import annotations

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def patch_hf_hub_download():
    try:
        import huggingface_hub
    except ImportError:
        yield
        return

    from .. import model_downloader
    from ..model_downloader_types import HuggingFile

    original = huggingface_hub.hf_hub_download

    def _intercepted(*args, **kwargs):
        repo_id = kwargs.get("repo_id") or (args[0] if len(args) > 0 else None)
        filename = kwargs.get("filename") or (args[1] if len(args) > 1 else None)
        subfolder = kwargs.get("subfolder")

        if repo_id is None or filename is None:
            return original(*args, **kwargs)

        if subfolder:
            full_filename = f"{subfolder}/{filename}"
        else:
            full_filename = filename

        hf_file = HuggingFile(repo_id=repo_id, filename=full_filename)
        logger.info("Intercepted hf_hub_download: repo_id=%s filename=%s", repo_id, full_filename)

        result = model_downloader.get_or_download("huggingface", str(hf_file), known_files=[hf_file])
        if result is not None:
            return result

        return original(*args, **kwargs)

    huggingface_hub.hf_hub_download = _intercepted
    try:
        yield
    finally:
        huggingface_hub.hf_hub_download = original


@contextmanager
def patch_snapshot_download():
    try:
        import huggingface_hub
    except ImportError:
        yield
        return

    from .. import model_downloader

    original = huggingface_hub.snapshot_download

    def _intercepted(*args, **kwargs):
        repo_id = kwargs.get("repo_id") or (args[0] if len(args) > 0 else None)
        if repo_id is None:
            return original(*args, **kwargs)

        allow_patterns = kwargs.get("allow_patterns")
        ignore_patterns = kwargs.get("ignore_patterns")

        logger.info("Intercepted snapshot_download: repo_id=%s allow_patterns=%s ignore_patterns=%s",
                     repo_id, allow_patterns, ignore_patterns)

        result = model_downloader.get_or_download_huggingface_repo(
            repo_id,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        if result is not None:
            return result

        return original(*args, **kwargs)

    huggingface_hub.snapshot_download = _intercepted
    try:
        yield
    finally:
        huggingface_hub.snapshot_download = original


@contextmanager
def patch_folder_paths_functions():
    from ..cmd import folder_paths
    from .. import model_downloader

    # Reentrant: if already patched, just yield without double-patching
    # (double-patching would set _original_get_filename_list to the patched
    # version, causing infinite recursion)
    if folder_paths.get_filename_list is model_downloader.get_filename_list:
        yield
        return

    original_get_full_path = folder_paths.get_full_path
    original_get_full_path_or_raise = folder_paths.get_full_path_or_raise
    original_get_filename_list = folder_paths.get_filename_list

    # Set originals on model_downloader so it can call them without infinite recursion
    # (model_downloader.get_or_download calls folder_paths.get_full_path internally)
    prev_orig_full_path = getattr(model_downloader, '_original_get_full_path', None)
    prev_orig_filename_list = getattr(model_downloader, '_original_get_filename_list', None)
    model_downloader._original_get_full_path = original_get_full_path
    model_downloader._original_get_filename_list = original_get_filename_list

    folder_paths.get_full_path = model_downloader.get_full_path
    folder_paths.get_full_path_or_raise = model_downloader.get_full_path_or_raise
    folder_paths.get_filename_list = model_downloader.get_filename_list
    try:
        yield
    finally:
        folder_paths.get_full_path = original_get_full_path
        folder_paths.get_full_path_or_raise = original_get_full_path_or_raise
        folder_paths.get_filename_list = original_get_filename_list
        model_downloader._original_get_full_path = prev_orig_full_path
        model_downloader._original_get_filename_list = prev_orig_filename_list


@contextmanager
def patch_folder_names_dict():
    from ..component_model.folder_path_types import FolderNames

    original_setitem = FolderNames.__setitem__

    def _intercepted_setitem(self, key, value):
        from ..cmd.folder_paths import add_model_folder_path
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            paths, extensions = value[0], value[1]
            ext_set = set(extensions) if extensions else None
            if isinstance(paths, (list, tuple)):
                for path in paths:
                    logger.info("Intercepted folder_names_and_paths[%r] write: path=%s extensions=%s", key, path, extensions)
                    add_model_folder_path(key, str(path), extensions=ext_set)
            else:
                logger.info("Intercepted folder_names_and_paths[%r] write: path=%s extensions=%s", key, paths, extensions)
                add_model_folder_path(key, str(paths), extensions=ext_set)
        else:
            logger.info("Intercepted folder_names_and_paths[%r] write (passthrough)", key)
            original_setitem(self, key, value)

    FolderNames.__setitem__ = _intercepted_setitem
    try:
        yield
    finally:
        FolderNames.__setitem__ = original_setitem


@contextmanager
def patch_torch_downloads():
    import torch.hub
    from .. import model_downloader
    from ..model_downloader_types import UrlFile

    original_download_url_to_file = torch.hub.download_url_to_file

    def _intercepted_download_url_to_file(url, dst, hash_prefix=None, progress=True):
        logger.info("Intercepted torch.hub.download_url_to_file: url=%s", url)
        url_file = UrlFile(url)
        result = model_downloader.get_or_download("checkpoints", str(url_file), known_files=[url_file])
        if result is not None:
            import shutil
            shutil.copy2(result, dst)
            return
        return original_download_url_to_file(url, dst, hash_prefix=hash_prefix, progress=progress)

    torch.hub.download_url_to_file = _intercepted_download_url_to_file

    original_load_url = None
    try:
        import torch.utils.model_zoo
        original_load_url = torch.utils.model_zoo.load_url

        def _intercepted_load_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None, weights_only=False):
            logger.info("Intercepted torch.utils.model_zoo.load_url: url=%s", url)
            url_file = UrlFile(url)
            result = model_downloader.get_or_download("checkpoints", str(url_file), known_files=[url_file])
            if result is not None:
                import torch
                return torch.load(result, map_location=map_location, weights_only=weights_only)
            return original_load_url(url, model_dir=model_dir, map_location=map_location,
                                     progress=progress, check_hash=check_hash, file_name=file_name,
                                     weights_only=weights_only)

        torch.utils.model_zoo.load_url = _intercepted_load_url
    except (ImportError, AttributeError):
        pass

    original_tv_download_url = None
    tv_utils = None
    try:
        import torchvision.datasets.utils as tv_utils_mod
        tv_utils = tv_utils_mod
        original_tv_download_url = tv_utils.download_url

        def _intercepted_tv_download_url(url, root, filename=None, md5=None):
            logger.info("Intercepted torchvision.datasets.utils.download_url: url=%s", url)
            url_file = UrlFile(url)
            result = model_downloader.get_or_download("checkpoints", str(url_file), known_files=[url_file])
            if result is not None:
                import shutil
                import os
                dst = os.path.join(root, filename or os.path.basename(url))
                shutil.copy2(result, dst)
                return
            return original_tv_download_url(url, root, filename=filename, md5=md5)

        tv_utils.download_url = _intercepted_tv_download_url
    except (ImportError, AttributeError):
        pass

    try:
        yield
    finally:
        torch.hub.download_url_to_file = original_download_url_to_file
        if original_load_url is not None:
            try:
                import torch.utils.model_zoo
                torch.utils.model_zoo.load_url = original_load_url
            except (ImportError, AttributeError):
                pass
        if original_tv_download_url is not None and tv_utils is not None:
            tv_utils.download_url = original_tv_download_url
