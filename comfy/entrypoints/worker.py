import asyncio
import warnings

from ..cli_args_types import Configuration
from ..component_model.file_counter import cleanup_temp
from ..component_model.entrypoints_common import configure_application_paths, executor_from_args


async def run_worker(config: Configuration):
    """Core worker logic, called by the Typer worker command."""
    from ..distributed.distributed_prompt_worker import DistributedPromptWorker

    if config.block_runtime_package_installation is None:
        config.block_runtime_package_installation = True

    assert config.distributed_queue_connection_uri is not None, \
        "Set the --distributed-queue-connection-uri argument to your RabbitMQ server"

    configure_application_paths(config)
    executor = await executor_from_args(config)

    async with (
        DistributedPromptWorker(
            connection_uri=config.distributed_queue_connection_uri,
            queue_name=config.distributed_queue_name,
            executor=executor,
        ),
    ):
        with cleanup_temp():
            stop = asyncio.Event()
            try:
                await stop.wait()
            except (asyncio.CancelledError, InterruptedError, KeyboardInterrupt):
                pass


def entrypoint():
    """Legacy entrypoint. Delegates to ``comfyui worker``."""
    warnings.warn(
        "comfyui-worker is deprecated. Use: comfyui worker",
        DeprecationWarning,
        stacklevel=1,
    )
    import sys
    from ..cmd.cli import app
    sys.argv = [sys.argv[0], "worker"] + sys.argv[1:]
    app()


if __name__ == "__main__":
    entrypoint()
