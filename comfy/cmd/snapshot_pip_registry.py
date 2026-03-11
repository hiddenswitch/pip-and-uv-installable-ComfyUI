from __future__ import annotations

import asyncio
import logging

from ..component_model.configuration import Configuration
from ..custom_node_facade.snapshot import snapshot_facade_registry


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


def run_snapshot_pip_registry(configuration: Configuration) -> None:
    _configure_logging(configuration.logging_level)
    try:
        path = asyncio.run(snapshot_facade_registry(configuration))
    except KeyboardInterrupt:
        return
    print(path)
