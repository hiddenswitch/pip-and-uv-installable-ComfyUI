"""Thread subclass that inherits the caller's contextvars.

Plain ``threading.Thread`` does NOT propagate ``contextvars`` —
the child thread gets an empty context, so execution-context-scoped
values like ``folder_names_and_paths`` resolve to defaults.

Use ``ContextThread`` (a drop-in replacement for ``threading.Thread``)
whenever the thread body needs to read from the current execution context.
"""

import contextvars
import threading
from typing import Any


class ContextThread(threading.Thread):
    """A ``threading.Thread`` that captures and runs inside the caller's context."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ctx = contextvars.copy_context()

    def run(self) -> None:
        self._ctx.run(super().run)
