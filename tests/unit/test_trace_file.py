import json

import pytest
from opentelemetry import trace

from comfy.client.embedded_comfy_client import Comfy
from comfy.cli_args_types import Configuration


@pytest.mark.asyncio
async def test_trace_file_is_flushed_at_comfy_shutdown(tmp_path):
    trace_file = tmp_path / "traces" / "sampling.jsonl"
    configuration = Configuration(
        otel_exporter_otlp_endpoint=f"file://{trace_file}",
    )

    async with Comfy(configuration=configuration):
        with trace.get_tracer(__name__).start_as_current_span("Sampler Invoke") as span:
            span.set_attribute("sampling.steps", 20)
            span.set_attribute("sampling.steps_per_second", 0.5)

    spans = [json.loads(line) for line in trace_file.read_text().splitlines()]
    sampler = next(span for span in spans if span["name"] == "Sampler Invoke")
    assert sampler["attributes"]["sampling.steps"] == 20
    assert sampler["attributes"]["sampling.steps_per_second"] == 0.5
