"""Daemon support: fork, PID file, log file, stop."""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional


def _default_dir() -> Path:
    return Path.home() / ".comfyui"


def default_pid_file() -> str:
    return str(_default_dir() / "comfyui.pid")


def default_log_file() -> str:
    return str(_default_dir() / "comfyui.log")


def daemonize(pid_file: str, log_file: str) -> None:
    if sys.platform == "win32":
        raise RuntimeError("Daemon mode is not supported on Windows")

    _default_dir().mkdir(parents=True, exist_ok=True)
    Path(pid_file).parent.mkdir(parents=True, exist_ok=True)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    pid = os.fork()
    if pid > 0:
        Path(pid_file).write_text(str(pid))
        sys.stdout.write(f"ComfyUI daemon started (PID {pid}), logging to {log_file}\n")
        sys.exit(0)

    os.setsid()
    log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, sys.stdout.fileno())
    os.dup2(log_fd, sys.stderr.fileno())
    os.close(log_fd)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, sys.stdin.fileno())
    os.close(devnull)


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
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except OSError:
            Path(pid_file).unlink(missing_ok=True)
            return True
    os.kill(pid, signal.SIGKILL)
    Path(pid_file).unlink(missing_ok=True)
    return True
