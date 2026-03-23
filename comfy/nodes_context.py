# todo: this should be defined in a common place, the fact that the nodes are imported by execution the way that they are is pretty radioactive
import threading

import lazy_object_proxy

from .execution_context import current_execution_context
from .nodes.package import import_all_nodes_in_workspace
from .nodes.package_typing import ExportedNodes, exported_nodes_view

_nodes_local = threading.local()
_nodes_global_lock = threading.Lock()
_nodes_global: ExportedNodes | None = None


def _load_nodes_once() -> ExportedNodes:
    global _nodes_global
    if _nodes_global is not None:
        return _nodes_global
    with _nodes_global_lock:
        if _nodes_global is not None:
            return _nodes_global
        _nodes_global = import_all_nodes_in_workspace()
        return _nodes_global


def invalidate():
    global _nodes_global
    _nodes_global = None
    _nodes_local.nodes = lazy_object_proxy.Proxy(_load_nodes_once)


def get_nodes() -> ExportedNodes:
    current_ctx = current_execution_context()
    try:
        nodes = _nodes_local.nodes
    except (LookupError, AttributeError):
        nodes = _nodes_local.nodes = lazy_object_proxy.Proxy(_load_nodes_once)

    if len(current_ctx.custom_nodes) == 0:
        return nodes
    return exported_nodes_view(nodes, current_ctx.custom_nodes)
