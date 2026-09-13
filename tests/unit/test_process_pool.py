import contextvars
import multiprocessing
import os

import pytest

from comfy.component_model import cvpickle
from comfy.distributed.process_pool_executor import ProcessPoolExecutor

# Example context variable
my_var = contextvars.ContextVar('my_var', default=None)
cvpickle.register_contextvar(my_var, module=__name__)


def worker_function():
    """Function that runs in worker process and accesses context"""
    return my_var.get()


@pytest.mark.asyncio
async def test_context_preservation():
    # Set context in parent
    my_var.set("test_value")

    # Create pool and submit work
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker_function)
        result = future.result()

        # Verify context was preserved
        assert result == "test_value"


def test_shutdown_wait_reaps_worker():
    executor = ProcessPoolExecutor(max_workers=1)
    try:
        worker_pid = executor.submit(os.getpid).result(timeout=60)
        executor.shutdown(wait=True)
        assert worker_pid not in {child.pid for child in multiprocessing.active_children()}
    finally:
        executor.stop()
        executor.join()
