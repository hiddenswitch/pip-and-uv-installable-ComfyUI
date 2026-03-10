from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..cmd import folder_paths
import glob
from aiohttp import web
import json
import logging
from functools import lru_cache

from ..json_util import merge_json_recursive
logger = logging.getLogger(__name__)


def _uv_pip_install_cmd() -> list[str]:
    """Return the base command for ``uv pip install``.

    Prefers the ``uv`` binary on ``PATH``; falls back to
    ``sys.executable -m uv`` if the ``uv`` package is installed.
    Raises :class:`RuntimeError` if neither is available.
    """
    import shutil
    uv_bin = shutil.which("uv")
    if uv_bin:
        return [uv_bin, "pip", "install"]
    if importlib.util.find_spec("uv") is not None:
        return [sys.executable, "-m", "uv", "pip", "install"]
    raise RuntimeError(
        "uv is required for custom node dependency installation but was not found. "
        "Install it with: pip install uv"
    )
# Extra locale files to load into main.json
EXTRA_LOCALE_FILES = [
    "nodeDefs.json",
    "commands.json",
    "settings.json",
]


def safe_load_json_file(file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error loading {file_path}")
        return {}


from ..execution_context import context_folder_names_and_paths


class CustomNodeManager:
    EXAMPLE_WORKFLOW_FOLDER_NAMES = ["example_workflows", "example", "examples", "workflow", "workflows"]

    @staticmethod
    def scan_example_workflows(custom_nodes_roots: list[str]) -> list[tuple[str, str, str]]:
        """Return ``(node_name, workflow_name, filepath)`` for all example workflows."""
        results = []
        for root in custom_nodes_roots:
            for folder_name in CustomNodeManager.EXAMPLE_WORKFLOW_FOLDER_NAMES:
                for filepath in glob.glob(os.path.join(root, f'*/{folder_name}/*.json')):
                    node_name = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
                    workflow_name = os.path.splitext(os.path.basename(filepath))[0]
                    results.append((node_name, workflow_name, filepath))
        return results

    # Default packages that must never be installed/overridden by custom node deps.
    DEFAULT_SKIP: frozenset[str] = frozenset({
        "torch", "torchvision", "torchaudio", "torchsde",
        "comfy", "comfyui",
    })

    @staticmethod
    def _extract_pkg_name(req: str) -> str:
        """Extract the bare package name from a pip requirement string."""
        for ch in (">=", "<=", "==", "!=", "~=", ">", "<", "[", ";", " "):
            req = req.split(ch)[0]
        return req.strip().lower().replace("-", "_")

    @staticmethod
    def install_custom_node(
        repo_url: str,
        target_dir: Path,
        *,
        git_ref: Optional[str] = None,
        needs_submodules: bool = False,
        skip_requirements: frozenset[str] = frozenset(),
        extra_requirements: Optional[list[str]] = None,
        git_timeout: Optional[int] = None,
    ) -> Path:
        """Clone a custom node repo and install its dependencies.

        Args:
            repo_url: Git URL of the custom node repository.
            target_dir: Directory under which the repo will be cloned
                        (e.g. ``base_dir / "custom_nodes"``).
            git_ref: Optional branch or tag to check out.
            needs_submodules: Whether to recurse submodules on clone.
            skip_requirements: Package names to exclude from installation.
            extra_requirements: Additional pip requirements to install.
            git_timeout: Timeout in seconds for git clone (default: 600, override with COMFY_GIT_CLONE_TIMEOUT env var).

        Returns:
            Path to the cloned repository.
        """
        node_id = repo_url.rstrip("/").rsplit("/", 1)[-1]
        repo_path = target_dir / node_id

        # --- clone ---
        if git_timeout is None:
            git_timeout = int(os.environ.get("COMFY_GIT_CLONE_TIMEOUT", "600"))

        cmd = ["git", "clone", "--depth=1"]
        if git_ref:
            cmd += ["--branch", git_ref]
        if needs_submodules:
            cmd.append("--recurse-submodules")
        cmd += [repo_url, str(repo_path)]
        logger.info("Cloning %s from %s (timeout=%ds)", node_id, repo_url, git_timeout)
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=git_timeout)

        # --- install deps ---
        import tempfile

        install_target = target_dir.parent / "node_site"
        install_target.mkdir(parents=True, exist_ok=True)

        req_file = repo_path / "requirements.txt"
        has_req_file = req_file.exists()

        # Build a constraints file that pins skip-packages to their installed
        # versions so uv won't reinstall them.
        all_skip = CustomNodeManager.DEFAULT_SKIP | skip_requirements
        constraints_lines: list[str] = []
        for pkg_name in sorted(all_skip):
            try:
                from importlib.metadata import version as pkg_version
                ver = pkg_version(pkg_name)
                constraints_lines.append(f"{pkg_name}=={ver}")
            except Exception:
                # Package not installed — override with an impossible constraint
                # so uv skips it.
                constraints_lines.append(f"{pkg_name}==99999.0.0")

        if has_req_file or extra_requirements:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="constraints_", delete=False
            ) as cf:
                cf.write("\n".join(constraints_lines) + "\n")
                constraints_path = cf.name

            try:
                pip_cmd = _uv_pip_install_cmd() + [
                    "--target", str(install_target),
                    "--constraint", constraints_path,
                ]

                if has_req_file:
                    pip_cmd += ["-r", str(req_file)]

                if extra_requirements:
                    pip_cmd += extra_requirements

                logger.info("Installing deps for %s via uv", node_id)
                result = subprocess.run(pip_cmd, capture_output=True, text=True, timeout=600)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"{node_id} dep install failed (rc={result.returncode}):\n{result.stderr[:2000]}"
                    )
            finally:
                os.unlink(constraints_path)

        # Clean up badly-packaged top-level directories that shadow real
        # packages when node_site is on sys.path.  For example, color_matcher
        # installs its test suite as a bare ``tests/`` package.
        _BANNED_TOPLEVEL = ("tests", "test", "examples", "example", "docs", "doc")
        for name in _BANNED_TOPLEVEL:
            bad_dir = install_target / name
            if bad_dir.is_dir():
                import shutil
                logger.info("Removing badly-packaged top-level directory from node_site: %s", bad_dir)
                shutil.rmtree(bad_dir, ignore_errors=True)

        return repo_path

    def __init__(self):
        # binds to context at init time
        self.folder_paths = folder_paths.folder_names_and_paths

    def build_translations(self):
        with context_folder_names_and_paths(self.folder_paths):
            return self._build_translations()

    @lru_cache(maxsize=1)
    def _build_translations(self):
        """Load all custom nodes translations during initialization. Translations are
        expected to be loaded from `locales/` folder.

        The folder structure is expected to be the following:
        - custom_nodes/
            - custom_node_1/
                - locales/
                    - en/
                        - main.json
                        - commands.json
                        - settings.json

        returned translations are expected to be in the following format:
        {
            "en": {
                "nodeDefs": {...},
                "commands": {...},
                "settings": {...},
                ...{other main.json keys}
            }
        }
        """

        translations = {}

        for folder in folder_paths.get_folder_paths("custom_nodes"):
            # Sort glob results for deterministic ordering
            for custom_node_dir in sorted(glob.glob(os.path.join(folder, "*/"))):
                locales_dir = os.path.join(custom_node_dir, "locales")
                if not os.path.exists(locales_dir):
                    continue

                for lang_dir in glob.glob(os.path.join(locales_dir, "*/")):
                    lang_code = os.path.basename(os.path.dirname(lang_dir))

                    if lang_code not in translations:
                        translations[lang_code] = {}

                    # Load main.json
                    main_file = os.path.join(lang_dir, "main.json")
                    node_translations = safe_load_json_file(main_file)

                    # Load extra locale files
                    for extra_file in EXTRA_LOCALE_FILES:
                        extra_file_path = os.path.join(lang_dir, extra_file)
                        key = extra_file.split(".")[0]
                        json_data = safe_load_json_file(extra_file_path)
                        if json_data:
                            node_translations[key] = json_data

                    if node_translations:
                        translations[lang_code] = merge_json_recursive(
                            translations[lang_code], node_translations
                        )

        return translations

    def add_routes(self, routes, webapp, loadedModules):

        @routes.get("/workflow_templates")
        async def get_workflow_templates(request):
            """Returns a web response that contains the map of custom_nodes names and their associated workflow templates. The ones without templates are omitted."""
            with context_folder_names_and_paths(self.folder_paths):
                entries = self.scan_example_workflows(
                    folder_paths.get_folder_paths("custom_nodes"))
            result = {}
            for node_name, workflow_name, _ in entries:
                result.setdefault(node_name, []).append(workflow_name)
            return web.json_response(result)

        for module_name, module_dir in loadedModules:
            for folder_name in self.EXAMPLE_WORKFLOW_FOLDER_NAMES:
                workflows_dir = os.path.join(module_dir, folder_name)

                if os.path.exists(workflows_dir):
                    if folder_name != "example_workflows":
                        logger.debug(
                            "Found example workflow folder '%s' for custom node '%s', consider renaming it to 'example_workflows'",
                            folder_name, module_name)

                    webapp.add_routes(
                        [
                            web.static(
                                "/api/workflow_templates/" + module_name, workflows_dir
                            )
                        ]
                    )

        @routes.get("/i18n")
        async def get_i18n(request):
            """Returns translations from all custom nodes' locales folders."""
            return web.json_response(self.build_translations())
