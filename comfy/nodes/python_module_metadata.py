from __future__ import annotations


def resolve_python_module_name(obj_class: type, default: str = "nodes") -> str:
    python_module = getattr(obj_class, "RELATIVE_PYTHON_MODULE", None)
    if isinstance(python_module, str) and python_module:
        return python_module

    python_module = getattr(obj_class, "__module__", None)
    if isinstance(python_module, str) and python_module:
        return python_module

    return default
