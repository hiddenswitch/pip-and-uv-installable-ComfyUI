import asyncio
import json

try:
    from collections.abc import Buffer
except ImportError:
    from typing_extensions import Buffer
from io import BytesIO
from pathlib import Path
from typing import Literal, AsyncGenerator

import fsspec
import ijson
import aiofiles
import sys
import shlex


from .uris import is_uri as _is_uri


def load_workflow_json(source: str) -> dict:
    """Load a single workflow JSON from a file path, URI, or literal JSON string.

    Supports the same input types as :func:`stream_json_objects` except stdin.
    """
    if source.lstrip().startswith("{"):
        return json.loads(source)
    if _is_uri(source):
        with fsspec.open(source, mode="rb") as f:
            return json.load(f)
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _is_workflow_shaped(parsed) -> bool:
    if not isinstance(parsed, dict) or not parsed:
        return False
    if "nodes" in parsed and "links" in parsed:
        return True
    return all(isinstance(v, dict) and "class_type" in v for v in parsed.values())


def _workflow_from_png_bytes(body: bytes) -> bytes | None:
    """Extract a ComfyUI workflow JSON from a PNG's tEXt chunks.

    ComfyUI saves the UI graph under the `workflow` keyword and the API form
    under `prompt` (see ComfyUI/comfy/cli_args.py and the SaveImage node).
    Returns the JSON bytes, or None if neither key is present.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(BytesIO(body)) as im:
            text = getattr(im, "text", {}) or {}
    except Exception:  # noqa: BLE001
        return None
    for key in ("workflow", "prompt"):
        value = text.get(key)
        if not isinstance(value, str):
            continue
        try:
            parsed = json.loads(value)
        except Exception:  # noqa: BLE001
            continue
        if _is_workflow_shaped(parsed):
            return value.encode("utf-8")
    return None


def _maybe_extract_workflow_json(data: bytes) -> BytesIO:
    """Best-effort unwrap a workflow JSON from common Civitai upload shapes.

    Handles:
    * raw bytes already containing JSON (default passthrough)
    * zip archives with one or more .json graphs (picks the first
      workflow-shaped one)
    * zip archives of PNG screenshots only — extracts the workflow from the
      first PNG's tEXt chunks (the "Workflow-in-a-PNG" pattern that many
      Civitai authors use as their distribution format)
    * raw PNG bytes with embedded workflow chunks
    """
    # Raw PNG with embedded workflow tEXt chunks.
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        body = _workflow_from_png_bytes(data)
        if body is not None:
            return BytesIO(body)
        return BytesIO(data)
    if not data.startswith(b"PK\x03\x04"):
        return BytesIO(data)
    import zipfile
    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            names = zf.namelist()
            json_names = [n for n in names if n.lower().endswith(".json")]
            for name in json_names:
                with zf.open(name) as inner:
                    body = inner.read()
                try:
                    parsed = json.loads(body)
                except Exception:  # noqa: BLE001
                    continue
                if _is_workflow_shaped(parsed):
                    return BytesIO(body)
            # No workflow-shaped .json — try PNG-embedded workflows.
            png_names = [n for n in names if n.lower().endswith(".png")]
            for name in png_names:
                with zf.open(name) as inner:
                    body = inner.read()
                extracted = _workflow_from_png_bytes(body)
                if extracted is not None:
                    return BytesIO(extracted)
            # Last resort: the first .json entry verbatim.
            if json_names:
                with zf.open(json_names[0]) as inner:
                    return BytesIO(inner.read())
    except (zipfile.BadZipFile, Exception):  # noqa: BLE001
        pass
    return BytesIO(data)


async def stream_json_objects(source_path_or_stdin: str | Literal["-"]) -> AsyncGenerator[dict, None]:
    """
    Asynchronously yields JSON objects from a given source.
    The source can be a file path, "-" for stdin, a literal JSON string starting with ``{``,
    or a URI supported by fsspec (``https://``, ``s3://``, ``hf://``, etc.).
    Assumes the input stream contains concatenated JSON objects (e.g., {}{}{}).
    """
    if source_path_or_stdin is None or len(source_path_or_stdin) == 0:
        return
    elif source_path_or_stdin == "-":
        async for obj in ijson.items_async(aiofiles.stdin_bytes, '', multiple_values=True, use_float=True):
            yield obj
    else:
        # Handle literal JSON
        if "{" in source_path_or_stdin[:2]:
            encode: Buffer = source_path_or_stdin.encode("utf-8")
            source_path_or_stdin = BytesIO(encode)
            for obj in ijson.items(source_path_or_stdin, '', multiple_values=True, use_float=True):
                yield obj
        elif _is_uri(source_path_or_stdin):
            # URIs: https://, s3://, hf://, gcs://, etc. — delegate to fsspec.
            # Civitai workflow uploads are commonly zip archives containing
            # one or more .json graphs; transparently extract the first
            # workflow-shaped JSON instead of failing to parse zip bytes.
            data = fsspec.open(source_path_or_stdin, mode='rb').open().read()
            stream = _maybe_extract_workflow_json(data)
            for obj in ijson.items(stream, '', multiple_values=True, use_float=True):
                yield obj
        else:
            async with aiofiles.open(source_path_or_stdin, mode='rb') as f:
                async for obj in ijson.items_async(f, '', multiple_values=True, use_float=True):
                    yield obj
