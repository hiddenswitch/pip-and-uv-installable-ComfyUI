from __future__ import annotations

import logging
from pathlib import Path

from comfy.app.custom_node_manager import CustomNodeManager
from comfy.component_model.site_packages import add_node_site, add_site_dir
from comfy.component_model.node_registry import CustomNodeSpec, CUSTOM_NODE_REGISTRY

logger = logging.getLogger(__name__)


def add_node_site_to_path(base_dir: Path) -> None:
    """Add the ``node_site`` directory to ``sys.path`` and ``PYTHONPATH``."""
    add_node_site(base_dir)
    # Ensure the test source root is on PYTHONPATH so that subprocess workers
    # (ProcessPoolExecutor with spawn context) can resolve pkg:// URIs for
    # test data packages like tests.custom_nodes.test_data.
    src_root = str(Path(__file__).resolve().parents[2])
    add_site_dir(src_root)


def install_custom_node_from_spec(spec: CustomNodeSpec, base_dir: Path) -> Path:
    return CustomNodeManager.install_custom_node(
        repo_url=spec.repo_url,
        target_dir=base_dir / "custom_nodes",
        git_ref=spec.git_ref,
        needs_submodules=spec.needs_submodules,
        skip_requirements=spec.skip_requirements,
        extra_requirements=list(spec.extra_requirements) if spec.extra_requirements else None,
    )


def install_all_nodes(base_dir: Path) -> dict[str, Path]:
    installed: dict[str, Path] = {}
    for spec in CUSTOM_NODE_REGISTRY:
        if spec.node_id in installed:
            continue
        try:
            path = install_custom_node_from_spec(spec, base_dir)
            installed[spec.node_id] = path
        except Exception:
            logger.warning("Failed to install %s", spec.node_id, exc_info=True)
    return installed


def make_base_dirs(base_dir: Path) -> None:
    for subdir in ("custom_nodes", "models", "input", "output", "temp", "user"):
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)


def build_config(base_dir: Path, port: int = 0, **overrides) -> "Configuration":
    from comfy.cli_args import default_configuration
    config = default_configuration()
    config.base_directory = str(base_dir)
    # Don't set novram/torch_device — they trigger ProcessPoolExecutor which
    # can't pickle prompt data containing torch.compile module refs.
    config.block_runtime_package_installation = True
    config.enable_manager = False
    config.listen = "127.0.0.1"
    config.port = port
    config.database_url = "sqlite:///:memory:"
    for k, v in overrides.items():
        setattr(config, k, v)
    return config
