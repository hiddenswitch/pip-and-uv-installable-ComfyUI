from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter


class FileSpanExporter(SpanExporter):
    """Append OpenTelemetry spans as JSONL batches for local inspection."""

    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self.path.open("a", encoding="utf-8") as trace_file:
            for span in spans:
                trace_file.write(json.dumps(_span_to_json(span), separators=(",", ":")))
                trace_file.write("\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def _span_to_json(span: ReadableSpan) -> dict[str, Any]:
    context = span.get_span_context()
    parent = span.parent
    return {
        "name": span.name,
        "trace_id": f"{context.trace_id:032x}",
        "span_id": f"{context.span_id:016x}",
        "parent_span_id": f"{parent.span_id:016x}" if parent else None,
        "start_time_unix_nano": span.start_time,
        "end_time_unix_nano": span.end_time,
        "duration_ms": (
            (span.end_time - span.start_time) / 1_000_000
            if span.start_time is not None and span.end_time is not None
            else None
        ),
        "kind": span.kind.name,
        "status": {
            "status_code": span.status.status_code.name,
            "description": span.status.description,
        },
        "attributes": _json_value(dict(span.attributes or {})),
        "resource": _json_value(dict(span.resource.attributes or {})),
        "events": [
            {
                "name": event.name,
                "timestamp": event.timestamp,
                "attributes": _json_value(dict(event.attributes or {})),
            }
            for event in span.events
        ],
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
