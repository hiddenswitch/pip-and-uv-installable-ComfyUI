import concurrent.futures
import contextvars
import multiprocessing
import os
import pickle
import logging
from functools import partial
from typing import Callable, Any

from pebble import ProcessPool, ProcessFuture

from ..component_model.executor_types import Executor

logger = logging.getLogger(__name__)

def _wrap_with_context(context_data: bytes, func: Callable, *args, **kwargs) -> Any:
    new_ctx: contextvars.Context = pickle.loads(context_data)
    return new_ctx.run(func, *args, **kwargs)


def _run_with_environment(
    environment: dict[str, str],
    function: Callable,
    args: tuple,
    kwargs: dict,
) -> Any:
    os.environ.update(environment)
    return function(*args, **kwargs)


class ProcessPoolExecutor(ProcessPool, Executor):
    def __init__(self,
                 max_workers: int = 1,
                 max_tasks: int = 0,
                 initializer: Callable = None,
                 initargs: list | tuple = (),
                 context: multiprocessing.context.BaseContext = None):
        if context is not None:
            logger.warning(f"A context was passed to a ProcessPoolExecutor when only spawn is supported (context={context})")
        context = multiprocessing.get_context('spawn')
        super().__init__(max_workers=max_workers, max_tasks=max_tasks, initializer=initializer, initargs=initargs, context=context)

    def shutdown(self, wait=True, *, cancel_futures=False):
        if cancel_futures:
            raise NotImplementedError("cannot cancel futures in this implementation")
        if wait:
            self.close()
        else:
            self.stop()
        return

    def schedule(self, function: Callable,
                 args: list | tuple = (),
                 kwargs=None,
                 timeout: float = None) -> ProcessFuture:
        return self._schedule_with_context(
            contextvars.copy_context(),
            function,
            args,
            kwargs,
            timeout,
        )

    def _schedule_with_context(
        self,
        context: contextvars.Context,
        function: Callable,
        args: list | tuple = (),
        kwargs=None,
        timeout: float = None,
    ) -> ProcessFuture:
        if kwargs is None:
            kwargs = {}
        context_bin = pickle.dumps(context)
        unpack_context_then_run_function = partial(_wrap_with_context, context_bin, function)

        return super().schedule(unpack_context_then_run_function, args=args, kwargs=kwargs, timeout=timeout)

    def submit(self, fn, /, *args, **kwargs) -> concurrent.futures.Future:
        return self.schedule(fn, args=list(args), kwargs=kwargs, timeout=None)

    def submit_with_environment(
        self,
        environment: dict[str, str],
        function: Callable,
        /,
        *args,
        detach_request_state: bool = False,
        **kwargs,
    ) -> concurrent.futures.Future:
        """Run a child task with the caller's context and canonical environment."""
        if detach_request_state:
            from ..execution_context import copy_process_context

            context = copy_process_context()
        else:
            context = contextvars.copy_context()
        return self._schedule_with_context(
            context,
            _run_with_environment,
            args=(dict(environment), function, tuple(args), kwargs),
        )
