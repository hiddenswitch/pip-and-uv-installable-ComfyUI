"""Daemon support: background process, PID file, log file, stop."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

_OS_FORK = getattr(os, "fork", None)
_OS_SETSID = getattr(os, "setsid", None)
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _default_dir() -> Path:
    return Path.home() / ".comfyui"


def default_pid_file() -> str:
    return str(_default_dir() / "comfyui.pid")


def default_log_file() -> str:
    return str(_default_dir() / "comfyui.log")


def _daemonize_posix(pid_file: str, log_file: str) -> None:
    if _OS_FORK is None or _OS_SETSID is None:
        raise RuntimeError("POSIX daemon mode is not supported on this platform")

    pid = _OS_FORK()
    if pid > 0:
        Path(pid_file).write_text(str(pid))
        sys.stdout.write(f"ComfyUI daemon started (PID {pid}), logging to {log_file}\n")
        sys.exit(0)

    _OS_SETSID()
    log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, sys.stdout.fileno())
    os.dup2(log_fd, sys.stderr.fileno())
    os.close(log_fd)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, sys.stdin.fileno())
    os.close(devnull)


def _daemonize_windows(pid_file: str, log_file: str) -> None:
    log_fh = open(log_file, "a")
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    DETACHED_PROCESS = 0x00000008
    proc = subprocess.Popen(
        [sys.executable] + sys.argv,
        stdout=log_fh,
        stderr=log_fh,
        stdin=subprocess.DEVNULL,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    log_fh.close()
    Path(pid_file).write_text(str(proc.pid))
    sys.stdout.write(f"ComfyUI daemon started (PID {proc.pid}), logging to {log_file}\n")
    sys.exit(0)


def daemonize(pid_file: str, log_file: str) -> None:
    _default_dir().mkdir(parents=True, exist_ok=True)
    Path(pid_file).parent.mkdir(parents=True, exist_ok=True)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        _daemonize_windows(pid_file, log_file)
    else:
        _daemonize_posix(pid_file, log_file)


def read_pid(pid_file: str) -> Optional[int]:
    path = Path(pid_file)
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
    except (ValueError, OSError):
        return None
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        path.unlink(missing_ok=True)
        return None


def stop_daemon(pid_file: str) -> bool:
    pid = read_pid(pid_file)
    if pid is None:
        return False
    if sys.platform == "win32":
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except OSError:
                Path(pid_file).unlink(missing_ok=True)
                return True
        os.kill(pid, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except OSError:
                Path(pid_file).unlink(missing_ok=True)
                return True
        os.kill(pid, _SIGKILL)
    Path(pid_file).unlink(missing_ok=True)
    return True
