import ast
from importlib.resources import files
from pathlib import Path


REPO_ROOT = Path(str(files("tests"))).parent
LINT_PATHS = (
    "comfy",
    "comfy_extras",
    "comfy_api",
    "comfy_api_nodes",
    "comfy_compatibility",
    "comfy_execution",
)
EXCLUDED_PARTS = {
    "alembic_db",
    "api",
    "controlnet_aux",
    "vendor",
}


def _iter_python_files():
    for lint_path in LINT_PATHS:
        root = REPO_ROOT / lint_path
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in EXCLUDED_PARTS for part in path.relative_to(REPO_ROOT).parts):
                continue
            yield path


def _module_import_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
    return names


def _target_names(node: ast.AST):
    if isinstance(node, ast.Name):
        yield node.id, node.lineno
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            yield from _target_names(element)
    elif isinstance(node, ast.Starred):
        yield from _target_names(node.value)


class _NestedBindingVisitor(ast.NodeVisitor):
    def __init__(self, import_names: set[str]):
        self.import_names = import_names
        self.scope_stack: list[dict[str, object]] = []
        self.violations: list[tuple[int, str]] = []

    def _record(self, name: str, lineno: int) -> None:
        if self.scope_stack and name in self.import_names:
            bindings = self.scope_stack[-1]["bindings"]
            assert isinstance(bindings, dict)
            bindings.setdefault(name, lineno)

    def _push_scope(self) -> None:
        self.scope_stack.append({"bindings": {}, "attr_bases": set()})

    def _pop_scope(self) -> None:
        scope = self.scope_stack.pop()
        bindings = scope["bindings"]
        attr_bases = scope["attr_bases"]
        assert isinstance(bindings, dict)
        assert isinstance(attr_bases, set)
        for name, lineno in bindings.items():
            if name in attr_bases:
                self.violations.append((lineno, name))

    def _record_arguments(self, args: ast.arguments) -> None:
        arguments = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if args.vararg is not None:
            arguments.append(args.vararg)
        if args.kwarg is not None:
            arguments.append(args.kwarg)
        for arg in arguments:
            self._record(arg.arg, arg.lineno)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node.name, node.lineno)
        self._push_scope()
        self._record_arguments(node.args)
        for child in node.body:
            self.visit(child)
        self._pop_scope()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._push_scope()
        self._record_arguments(node.args)
        self.visit(node.body)
        self._pop_scope()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node.name, node.lineno)
        self._push_scope()
        for child in node.body:
            self.visit(child)
        self._pop_scope()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.scope_stack and isinstance(node.value, ast.Name):
            attr_bases = self.scope_stack[-1]["attr_bases"]
            assert isinstance(attr_bases, set)
            attr_bases.add(node.value.id)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            for name, lineno in _target_names(target):
                self._record(name, lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        for name, lineno in _target_names(node.target):
            self._record(name, lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        for name, lineno in _target_names(node.target):
            self._record(name, lineno)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        for name, lineno in _target_names(node.target):
            self._record(name, lineno)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        for name, lineno in _target_names(node.target):
            self._record(name, lineno)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                for name, lineno in _target_names(item.optional_vars):
                    self._record(name, lineno)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self._record(node.name, node.lineno)
        self.generic_visit(node)


def _violations_for_source(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    visitor = _NestedBindingVisitor(_module_import_names(tree))
    visitor.visit(tree)
    return visitor.violations


def test_import_shadowing_detects_module_attribute_use():
    assert _violations_for_source(
        """
from comfy import conds


def build_conditioning():
    conds = {}
    return conds.CONDRegular
"""
    ) == [(6, "conds")]


def test_import_shadowing_allows_plain_local_shadowing():
    assert (
        _violations_for_source(
            """
from comfy import conds


def build_conditioning():
    conds = {}
    return conds
"""
        )
        == []
    )


def test_nested_bindings_do_not_shadow_module_imports():
    violations: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _NestedBindingVisitor(_module_import_names(tree))
        visitor.visit(tree)
        for lineno, name in visitor.violations:
            rel_path = path.relative_to(REPO_ROOT)
            violations.append(f"{rel_path}:{lineno}: local binding shadows imported name {name!r}")

    assert violations == []
