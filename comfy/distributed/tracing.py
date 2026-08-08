from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from opentelemetry import context, propagate, trace


def inject_trace_context(message: Mapping) -> dict:
    message = dict(message)
    carrier = {}
    propagate.inject(carrier, context.get_current())
    message["trace_context"] = carrier
    return message


@contextmanager
def distributed_command_span(
    message: Mapping,
    name: str,
    attributes: Mapping[str, object] | None = None,
) -> Iterator[None]:
    token = context.attach(propagate.extract(message.get("trace_context", {})))
    try:
        with trace.get_tracer(__name__).start_as_current_span(
            name,
            attributes=dict(attributes or ()),
        ):
            yield
    finally:
        context.detach(token)
