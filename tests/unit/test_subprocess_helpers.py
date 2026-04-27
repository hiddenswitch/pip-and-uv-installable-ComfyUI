"""Unit tests for _subprocess_helpers._DrainingProcess.

Why these exist: the CI 6h-hang bug was caused by
``subprocess.Popen(..., stdout=PIPE)`` + no reader. Linux pipe buffers
are ~64 KB; once full the subprocess blocks on ``write()``, and
``terminate() + wait()`` can't reap it because the SIGTERM handler
tries to flush logs through the same blocked pipe. ``process.stdout.read()``
called *after* the test gave up also hangs forever because EOF only
arrives when the subprocess actually exits.

These tests reproduce the deadlock by spawning a subprocess that emits
hundreds of KB of stdout on startup, then asserting the helper's
invariants:

  * ``shutdown()`` always returns within seconds, even when the child is
    producing output faster than anyone reads it.
  * ``tail()`` returns recent output without needing the subprocess to
    exit first.
  * A subprocess that prints N KB quickly is fully observable through
    ``tail()`` — the drainer doesn't drop output on the happy path.
"""
from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def chatty_program_path(tmp_path) -> str:
    """Write a tiny Python script that emits ~200 KB of stdout quickly
    then hangs forever. 200 KB is well over Linux's 64 KB pipe buffer so
    if nobody drains it the subprocess will block on ``write()``.

    The final ``while True: time.sleep(...)`` loop means the script will
    not exit on its own — shutdown() must terminate it.
    """
    src = textwrap.dedent("""
        import sys, time
        # ~200 KB: 2000 lines × 100 chars each.
        line = "x" * 99 + "\\n"
        for i in range(2000):
            sys.stdout.write(line)
        sys.stdout.flush()
        while True:
            time.sleep(60)
    """).strip() + "\n"
    path = tmp_path / "chatty.py"
    path.write_text(src)
    return str(path)


@pytest.fixture
def quick_exit_program_path(tmp_path) -> str:
    """A subprocess that exits cleanly on its own after a short delay.
    Used to verify shutdown() is a no-op when the process is already dead.
    """
    src = textwrap.dedent("""
        import sys
        sys.stdout.write("bye\\n")
        sys.stdout.flush()
        sys.exit(0)
    """).strip() + "\n"
    path = tmp_path / "quick.py"
    path.write_text(src)
    return str(path)


class TestDrainingProcess:
    def test_shutdown_on_chatty_hung_child_returns_promptly(self, chatty_program_path):
        """The scenario that hung CI for 6 hours: subprocess emits far more
        than pipe buffer, then never exits on its own. shutdown() must
        still return within seconds."""
        from tests.unit._subprocess_helpers import _DrainingProcess

        p = _DrainingProcess([sys.executable, chatty_program_path])
        try:
            start = time.monotonic()
            p.shutdown(term_timeout=2.0)
            elapsed = time.monotonic() - start
        finally:
            if p.poll() is None:
                p.kill()
        assert elapsed < 3.0, (
            f"shutdown() took {elapsed:.1f}s on a chatty hung child — "
            f"this is the CI hang pattern, pipe drainer is broken"
        )
        assert p.returncode is not None

    def test_tail_captures_output_from_chatty_child(self, chatty_program_path):
        """The drainer keeps the last N lines so the pytest failure report
        has context."""
        from tests.unit._subprocess_helpers import _DrainingProcess

        p = _DrainingProcess([sys.executable, chatty_program_path],
                              max_lines=500)
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and p.tail().count("\n") < 500:
                time.sleep(0.01)
            tail = p.tail(n=10)
            assert tail, "tail() returned empty despite 2000 lines from child"
            assert tail.count("\n") <= 10
            tail_lines = tail.splitlines()
            chatty_lines = [line for line in tail_lines if line.strip() == "x" * 99]
            unexpected = [line for line in tail_lines if line.strip() != "x" * 99]
            assert chatty_lines, (
                f"no chatty-pattern lines in tail. tail_lines={tail_lines!r}"
            )
            assert not unexpected, (
                f"non-chatty lines mixed into tail: {unexpected!r}"
            )
        finally:
            p.shutdown()

    def test_shutdown_when_child_already_exited(self, quick_exit_program_path):
        """shutdown() on a process that already exited by itself is a
        harmless no-op and must not raise."""
        from tests.unit._subprocess_helpers import _DrainingProcess

        p = _DrainingProcess([sys.executable, quick_exit_program_path])
        # Wait for self-exit.
        p.wait(timeout=5.0)
        assert p.returncode == 0
        p.shutdown()  # must not raise
        assert "bye" in p.tail()

    def test_pid_accessible(self, chatty_program_path):
        from tests.unit._subprocess_helpers import _DrainingProcess

        p = _DrainingProcess([sys.executable, chatty_program_path])
        try:
            assert isinstance(p.pid, int) and p.pid > 0
        finally:
            p.shutdown()

    def test_bounded_buffer_does_not_grow_unbounded(self, chatty_program_path):
        """Default max_lines=2048 must hold regardless of how long the
        subprocess produces output. We pass a smaller max_lines to make
        this cheap to verify."""
        from tests.unit._subprocess_helpers import _DrainingProcess

        p = _DrainingProcess([sys.executable, chatty_program_path],
                              max_lines=100)
        try:
            # Wait briefly until drainer has observed the flood; the child
            # emits 2000 lines eagerly then sleeps forever.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and p.tail().count("\n") < 100:
                time.sleep(0.01)
            tail = p.tail(n=10_000)  # ask for more than we should have
            lines = tail.splitlines(keepends=True)
            # Deque bounds at 100; some trailing content may be partial.
            assert len(lines) <= 101, (
                f"buffer grew past max_lines=100: got {len(lines)}"
            )
        finally:
            p.shutdown()
