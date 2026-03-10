from __future__ import annotations

import logging
import os
import traceback
import types
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

_suppress_error_stack: ContextVar[bool] = ContextVar("_suppress_error_stack", default=False)

_COMFY_LOGGER_PREFIXES = ("comfy.", "comfy_", "comfy_execution.")


class NodeExecutionErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not _suppress_error_stack.get():
            return True
        if record.levelno < logging.ERROR:
            return True
        name = record.name
        if any(name.startswith(p) for p in _COMFY_LOGGER_PREFIXES) or name == "comfy":
            return True
        return False


@contextmanager
def suppress_error_stack_trace():
    token = _suppress_error_stack.set(True)
    try:
        yield
    finally:
        _suppress_error_stack.reset(token)

_INFRASTRUCTURE_MARKERS = (
    os.sep + "asyncio" + os.sep,
    os.sep + "concurrent" + os.sep,
    os.sep + "threading.py",
    os.sep + "contextlib.py",
    os.sep + "runners.py",
    os.sep + "opentelemetry" + os.sep,
    os.sep + "torch" + os.sep + "_dynamo" + os.sep,
    os.sep + "torch" + os.sep + "_inductor" + os.sep,
    os.sep + "torch" + os.sep + "nn" + os.sep + "modules" + os.sep + "module.py",
    os.sep + "triton" + os.sep + "compiler" + os.sep,
)

_EXECUTION_ENGINE_MARKERS = (
    os.sep + "comfy" + os.sep + "cmd" + os.sep + "execution.py",
    os.sep + "comfy" + os.sep + "client" + os.sep + "embedded_comfy_client.py",
)


def _is_infrastructure(filename: str) -> bool:
    return (
        any(marker in filename for marker in _INFRASTRUCTURE_MARKERS)
        or any(marker in filename for marker in _EXECUTION_ENGINE_MARKERS)
    )


def filter_traceback(tb: types.TracebackType | None) -> list[traceback.FrameSummary]:
    if tb is None:
        return []

    all_frames = traceback.extract_tb(tb)
    if not all_frames:
        return []

    raising_frame = all_frames[-1]

    relevant = []
    for frame in all_frames[:-1]:
        if not _is_infrastructure(frame.filename):
            relevant.append(frame)

    if not relevant or relevant[-1] != raising_frame:
        relevant.append(raising_frame)

    return relevant


def format_node_exception(
    ex: BaseException,
    tb: types.TracebackType | None,
    node_id: Optional[str] = None,
    class_type: Optional[str] = None,
    input_data_formatted: Optional[dict] = None,
) -> str:
    filtered = filter_traceback(tb)

    parts: list[str] = []

    header_items = ["Node execution error"]
    if class_type:
        header_items.append(f"class_type={class_type}")
    if node_id:
        header_items.append(f"node_id={node_id}")
    parts.append(" | ".join(header_items))

    exc_type_name = type(ex).__qualname__
    exc_module = type(ex).__module__
    if exc_module and exc_module != "builtins":
        exc_type_name = f"{exc_module}.{exc_type_name}"
    parts.append(f"{exc_type_name}: {ex}")

    if filtered:
        parts.append("Traceback (filtered):")
        for line in traceback.format_list(filtered):
            parts.append(line.rstrip())

    if input_data_formatted:
        compact_inputs = {}
        for k, v in input_data_formatted.items():
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "..."
            compact_inputs[k] = s
        parts.append(f"Inputs: {compact_inputs}")

    return "\n".join(parts)
