"""Subprocess helpers for tests that spawn ``comfyui serve`` subprocesses.

Vanilla ``subprocess.Popen(..., stdout=PIPE)`` + not reading the pipe is a
deadlock waiting to happen: Linux pipe buffers are ~64 KB, and a chatty
ComfyUI server blows past that in the first few seconds of logging.
Once the pipe fills, the subprocess blocks on ``write()``, ``terminate()``
can't be honored because the SIGTERM handler can't flush its own
teardown log, ``wait()`` hangs, and ``kill()`` + ``wait()`` also hangs
because the zombie child is stuck mid-write.

The right pattern is to drain the pipe continuously into a bounded
in-memory buffer. That way:

  * The subprocess never blocks writing to stdout.
  * SIGTERM is honored immediately on ``terminate()``.
  * On failure we still have the tail of the output to report.
  * On success we simply drop the buffer.
"""
from __future__ import annotations

import collections
import subprocess
import threading
from typing import List, Optional


class _DrainingProcess:
    """A thin wrapper around ``subprocess.Popen`` that continuously drains
    stdout in a background thread and keeps the last N lines in memory.

    Attributes match Popen's: ``pid``, ``returncode``, ``poll()``,
    ``wait()``, ``terminate()``, ``kill()``. Adds ``tail()`` which returns
    the recent stdout as a string for pytest failure reports.
    """

    def __init__(
        self,
        args: list[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        max_lines: int = 2048,
    ):
        self._proc = subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered — every log line flushes
        )
        # deque with a maxlen drops old lines without blocking — we don't
        # want to grow unboundedly for a long-running server.
        self._lines: collections.deque[str] = collections.deque(maxlen=max_lines)
        self._reader = threading.Thread(
            target=self._drain, name="subproc-drainer", daemon=True,
        )
        self._reader.start()

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def returncode(self):  # type: ignore[no-untyped-def]
        return self._proc.returncode

    def _drain(self) -> None:
        stdout = self._proc.stdout
        if stdout is None:
            return
        try:
            for line in iter(stdout.readline, ""):
                if not line:
                    break
                self._lines.append(line)
        finally:
            try:
                stdout.close()
            except Exception:  # noqa: BLE001
                pass

    def poll(self):  # type: ignore[no-untyped-def]
        return self._proc.poll()

    def wait(self, timeout: Optional[float] = None) -> int:
        return self._proc.wait(timeout=timeout)

    def terminate(self) -> None:
        self._proc.terminate()

    def kill(self) -> None:
        self._proc.kill()

    def tail(self, n: int = 200) -> str:
        """Return the last *n* lines of output. Cheap; just slices the deque."""
        lines: List[str] = list(self._lines)
        return "".join(lines[-n:])

    def shutdown(self, *, term_timeout: float = 10.0, kill_timeout: float = 5.0) -> None:
        """Terminate, then escalate to kill. Reaps the reader thread too.

        Always safe to call in a ``finally:`` — swallows all errors rather
        than masking an earlier test failure.
        """
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=term_timeout)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    try:
                        self._proc.wait(timeout=kill_timeout)
                    except subprocess.TimeoutExpired:
                        # If even SIGKILL + wait is stuck, give up — the
                        # reader thread is a daemon and the OS will reap
                        # the zombie eventually. Don't block the test run.
                        return
            # Give the reader thread a short window to flush any lines
            # written just before termination.
            self._reader.join(timeout=2.0)
        except Exception:  # noqa: BLE001 - finally-path, swallow all
            pass


def spawn_comfyui_serve(
    executable: str,
    *,
    port: int,
    cwd: str,
    env: Optional[dict] = None,
    extra_args: Optional[list[str]] = None,
) -> _DrainingProcess:
    """Spawn ``python -m comfy.cmd.main`` with deadlock-free stdout draining."""
    args = [
        executable, "-m", "comfy.cmd.main",
        "--listen", "127.0.0.1",
        "--port", str(port),
        "--cpu",
        "--dont-print-server",
    ]
    if extra_args:
        args.extend(extra_args)
    return _DrainingProcess(args, cwd=cwd, env=env)
