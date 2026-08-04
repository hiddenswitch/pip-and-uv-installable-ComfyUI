from __future__ import annotations

import sys

from .distributed import worker_main


if __name__ == "__main__":
    worker_main(*sys.argv[1:])
