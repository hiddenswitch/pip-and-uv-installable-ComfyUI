import os
from typing import TYPE_CHECKING, Optional

from pylint.checkers import BaseChecker

if TYPE_CHECKING:
    from pylint.lint import PyLinter


class RootComfyExtrasNodesChecker(BaseChecker):
    name = "root-comfy-extras-nodes"
    priority = -1
    msgs = {
        "W8002": (
            "File %s must be moved under comfy_extras/nodes/",
            "root-comfy-extras-nodes-file",
            "Root-level comfy_extras/nodes_*.py files must live in comfy_extras/nodes/.",
        ),
    }

    def __init__(self, linter: Optional["PyLinter"] = None) -> None:
        super().__init__(linter)

    def visit_module(self, node) -> None:
        file_path = getattr(node, "file", None)
        if not file_path or not file_path.endswith(".py"):
            return

        normalized = os.path.normpath(file_path)
        directory = os.path.basename(os.path.dirname(normalized))
        filename = os.path.basename(normalized)

        if directory != "comfy_extras":
            return
        if filename == "__init__.py" or not filename.startswith("nodes_"):
            return

        self.add_message("root-comfy-extras-nodes-file", args=(normalized,), node=node)


def register(linter: "PyLinter") -> None:
    linter.register_checker(RootComfyExtrasNodesChecker(linter))
