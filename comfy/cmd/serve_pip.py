from __future__ import annotations

import asyncio
import logging

from ..custom_node_facade import run_facade_server
from ..component_model.configuration import Configuration


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


def run_serve_pip(configuration: Configuration) -> None:
    _configure_logging(configuration.logging_level)
    try:
        asyncio.run(run_facade_server(configuration=configuration))
    except KeyboardInterrupt:
        pass
