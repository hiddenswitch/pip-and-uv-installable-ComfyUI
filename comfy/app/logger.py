import io
import logging
import os
import sys
import threading
from collections import deque
from datetime import datetime

from ..component_model.node_traceback import NodeExecutionErrorFilter
from .. import internal_logging

ANSI_NAMED_COLORS = {
    'black':   '\033[30m',
    'red':     '\033[31m',
    'green':   '\033[32m',
    'yellow':  '\033[33m',
    'blue':    '\033[34m',
    'magenta': '\033[35m',
    'cyan':    '\033[36m',
    'white':   '\033[37m',
}

ANSI_LEVEL_COLORS = {
    'DEBUG':    ANSI_NAMED_COLORS['cyan'],
    'DETAIL':   ANSI_NAMED_COLORS['blue'],
    'INFO':     ANSI_NAMED_COLORS['green'],
    'WARNING':  ANSI_NAMED_COLORS['yellow'],
    'ERROR':    ANSI_NAMED_COLORS['red'],
    'CRITICAL': ANSI_NAMED_COLORS['magenta'],
}

ANSI_RESET = '\033[0m'
ANSI_BOLD  = '\033[1m'


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = ANSI_LEVEL_COLORS.get(record.levelname, '')
        bold  = ANSI_BOLD if record.levelno >= logging.WARNING else ''
        level_tag = f"{bold}{color}[{record.levelname}]{ANSI_RESET} "
        message = super().format(record)
        line_color = ANSI_NAMED_COLORS.get(getattr(record, 'color', ''), '')
        if line_color:
            return f"{level_tag}{line_color}{message}{ANSI_RESET}"
        return level_tag + message

logs = None
stdout_interceptor = None
stderr_interceptor = None

# initialize with sane defaults
logs = deque(maxlen=1000)
stdout_interceptor = sys.stdout
stderr_interceptor = sys.stderr
_initialized = False

logger = logging.getLogger(__name__)

class LogInterceptor(io.TextIOWrapper):
    def __init__(self, stream, *args, **kwargs):
        buffer = stream.buffer
        encoding = stream.encoding
        # Use 'replace' error handling to avoid Unicode encoding errors on Windows
        super().__init__(buffer, *args, **kwargs, encoding=encoding, errors='replace', line_buffering=stream.line_buffering)
        self._lock = threading.Lock()
        self._flush_callbacks = []
        self._logs_since_flush = []

    def write(self, data):
        entry = {"t": datetime.now().isoformat(), "m": data}
        with self._lock:
            self._logs_since_flush.append(entry)

            # Simple handling for cr to overwrite the last output if it isnt a full line
            # else logs just get full of progress messages
            if isinstance(data, str) and data.startswith("\r") and len(logs) > 0 and not logs[-1]["m"].endswith("\n"):
                logs.pop()
            logs.append(entry)
        if not self.closed:
            try:
                super().write(data)
            except UnicodeEncodeError:
                # some random bs in custom nodes will trigger errors on Windows
                super().write(data.encode(self.encoding, errors='replace').decode(self.encoding))

    def flush(self):
        if not self.closed:
            super().flush()
        for cb in self._flush_callbacks:
            cb(self._logs_since_flush)
            self._logs_since_flush = []

    def on_flush(self, callback):
        self._flush_callbacks.append(callback)


def get_logs():
    return logs


def on_flush(callback):
    if stdout_interceptor is not None and hasattr(stdout_interceptor, "on_flush"):
        stdout_interceptor.on_flush(callback)
    if stderr_interceptor is not None and hasattr(stderr_interceptor, "on_flush"):
        stderr_interceptor.on_flush(callback)


class StackTraceLogger(logging.Logger):
    pass


def get_log_level(level):
    if level == "DETAIL":
        return internal_logging.DETAIL
    get_level_names = getattr(logging, "getLevelNamesMapping", None)
    if get_level_names is not None:
        return get_level_names()[level]
    resolved = logging.getLevelName(level)
    if isinstance(resolved, int):
        return resolved
    raise KeyError(level)


def setup_logger(log_level: str = 'INFO', file_outputs=None, capacity: int = 300, use_stdout: bool = False):
    global logs, _initialized
    if _initialized:
        return
    _initialized = True

    # workaround for google colab
    if not hasattr(sys.stdout, "buffer") or not hasattr(sys.stdout, "encoding") or not hasattr(sys.stdout, "line_buffering"):
        return

    # Override output streams and log to buffer
    logs = deque(maxlen=capacity)

    global stdout_interceptor
    global stderr_interceptor
    stdout_interceptor = sys.stdout = LogInterceptor(sys.stdout)
    stderr_interceptor = sys.stderr = LogInterceptor(sys.stderr)

    # Setup default global logger
    logging.setLoggerClass(StackTraceLogger)
    logger = logging.getLogger()
    if file_outputs is None:
        file_outputs = [("DETAIL", "comfyui_detail.log")]
    console_level = get_log_level(log_level)
    logger.setLevel(min([console_level, *(get_log_level(level) for level, _ in file_outputs)]))

    node_error_filter = NodeExecutionErrorFilter()
    formatter = ColoredFormatter("%(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(console_level)
    stream_handler.addFilter(node_error_filter)

    if use_stdout:
        # Only errors and critical to stderr
        stream_handler.addFilter(lambda record: not record.levelno < logging.ERROR)

        # Lesser to stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(console_level)
        stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
        stdout_handler.addFilter(node_error_filter)
        logger.addHandler(stdout_handler)

    logger.addHandler(stream_handler)

    for output_level, output_path in file_outputs:
        output_path = os.path.abspath(output_path)
        try:
            output_handler = logging.FileHandler(output_path, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not open %s log %s: %s", output_level, output_path, exc)
            continue
        output_handler.setLevel(get_log_level(output_level))
        output_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(output_handler)
        logger.info("%s log: %s", output_level.title(), output_path)

    _patch_tqdm_write()


def _patch_tqdm_write():
    try:
        import tqdm

        @staticmethod
        def _logging_write(s, file=None, end="\n", nolock=False):
            if s:
                logging.getLogger("tqdm").info(s)

        tqdm.tqdm.write = _logging_write
    except ImportError:
        pass


STARTUP_WARNINGS = []


def log_startup_warning(msg):
    logger.warning(msg)
    STARTUP_WARNINGS.append(msg)


def print_startup_warnings():
    for s in STARTUP_WARNINGS:
        logger.warning(s)
    STARTUP_WARNINGS.clear()
