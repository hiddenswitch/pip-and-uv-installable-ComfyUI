import json
import os
import subprocess
import sys


def test_trace_file_is_flushed_at_shutdown(tmp_path):
    trace_file = tmp_path / "traces" / "sampling.jsonl"
    program = """
from opentelemetry import trace
from comfy.cli_args_types import Configuration
from comfy.component_model.setup import setup_tracing, shutdown_tracing

configuration = Configuration()
setup_tracing(configuration)
with trace.get_tracer("trace-file-test").start_as_current_span("Sampler Invoke") as span:
    span.set_attribute("sampling.steps", 20)
    span.set_attribute("sampling.steps_per_second", 0.5)
shutdown_tracing()
"""
    env = os.environ.copy()
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"file://{trace_file}"
    subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        check=True,
    )

    spans = [json.loads(line) for line in trace_file.read_text().splitlines()]
    sampler = next(span for span in spans if span["name"] == "Sampler Invoke")
    assert sampler["attributes"]["sampling.steps"] == 20
    assert sampler["attributes"]["sampling.steps_per_second"] == 0.5
