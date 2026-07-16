"""Minimal subprocess entry point for integration-test ComfyUI servers."""

import asyncio
import pickle
import sys
from pathlib import Path

from comfy.cmd.main import _start_comfyui


def main() -> None:
    with Path(sys.argv[1]).open("rb") as config_file:
        configuration = pickle.load(config_file)
    asyncio.run(_start_comfyui(configuration=configuration))


if __name__ == "__main__":
    main()
