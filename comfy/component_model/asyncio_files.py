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


from .uris import is_uri as _is_uri


def load_workflow_json(source: str) -> dict:
    """Load a single workflow JSON from a file path, URI, or literal JSON string.

    Supports the same input types as :func:`stream_json_objects` except stdin.
    Transparently extracts workflows from zip archives or PNG tEXt chunks
    (so Civitai workflow uploads — typically zips of PNG screenshots — load
    via the same code path as bare JSON).

    For non-ComfyUI formats (A1111/Forge dumps, Fooocus presets) the source
    is translated into an equivalent ComfyUI API-form workflow. For shapes
    we cannot translate (SwarmUI, InvokeAI, Krita-AI, unknown JSON),
    :class:`comfy.component_model.foreign_workflow.UnsupportedWorkflowFormatError`
    is raised with a clear explanation.
    """
    if source.lstrip().startswith("{"):
        return json.loads(source)
    if _is_uri(source):
        with fsspec.open(source, mode="rb") as fh:
            data = fh.read()
    else:
        data = Path(source).read_bytes()
    extracted = _maybe_extract_workflow_json(data, source=source)
    body = extracted.read()
    # If extraction produced ComfyUI-shaped JSON, parse and return; otherwise
    # delegate to the foreign-workflow translator (handles A1111/Fooocus or
    # raises a typed error for unsupported formats).
    try:
        parsed = json.loads(body)
    except Exception:  # noqa: BLE001
        parsed = None
    if isinstance(parsed, dict) and _is_workflow_shaped(parsed):
        return parsed
    from .foreign_workflow import translate_foreign_workflow
    return translate_foreign_workflow(parsed if parsed is not None else body, source=source)


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


def _maybe_extract_workflow_json(data: bytes, *, source: str | None = None) -> BytesIO:
    """Best-effort unwrap a workflow from common Civitai upload shapes.

    Handles:
    * raw bytes already containing JSON (default passthrough)
    * zip archives with one or more .json graphs (picks the first
      workflow-shaped one, falls back to A1111/Fooocus JSON or a .txt
      parameter dump if no ComfyUI-shaped JSON is present)
    * zip archives of PNG screenshots only — extracts the workflow from the
      first PNG's tEXt chunks (the "Workflow-in-a-PNG" pattern that many
      Civitai authors use as their distribution format), or returns the
      A1111 ``parameters`` chunk as a .txt body if that's all we find
    * raw PNG bytes with embedded workflow chunks (or A1111 ``parameters``)
    * raw A1111 ``.txt`` parameter dumps (passthrough for the loader to sniff)
    """
    # Raw PNG with embedded workflow tEXt chunks.
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        body = _workflow_from_png_bytes(data)
        if body is not None:
            return BytesIO(body)
        a1111 = _a1111_text_from_png_bytes(data)
        if a1111 is not None:
            return BytesIO(a1111)
        return BytesIO(data)
    if not data.startswith(b"PK\x03\x04"):
        return BytesIO(data)
    import zipfile
    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            names = zf.namelist()
            json_names = [n for n in names if n.lower().endswith(".json")]
            json_bodies: list[bytes] = []
            for name in json_names:
                with zf.open(name) as inner:
                    body = inner.read()
                try:
                    parsed = json.loads(body)
                except Exception:  # noqa: BLE001
                    continue
                if _is_workflow_shaped(parsed):
                    return BytesIO(body)
                json_bodies.append(body)
            # No ComfyUI-shaped .json — try PNG-embedded workflows.
            png_names = [n for n in names if n.lower().endswith(".png")]
            for name in png_names:
                with zf.open(name) as inner:
                    body = inner.read()
                extracted = _workflow_from_png_bytes(body)
                if extracted is not None:
                    return BytesIO(extracted)
            # Try .txt parameter dumps (A1111/Forge style).
            txt_names = [n for n in names if n.lower().endswith(".txt")]
            for name in txt_names:
                with zf.open(name) as inner:
                    body = inner.read()
                if _looks_like_a1111_bytes(body):
                    return BytesIO(body)
            # PNG with A1111 'parameters' chunk only (no ComfyUI workflow).
            for name in png_names:
                with zf.open(name) as inner:
                    body = inner.read()
                a1111 = _a1111_text_from_png_bytes(body)
                if a1111 is not None:
                    return BytesIO(a1111)
            # Surface the first non-ComfyUI JSON body so the foreign-workflow
            # translator can classify it (Fooocus / SwarmUI / InvokeAI / ...).
            if json_bodies:
                return BytesIO(json_bodies[0])
    except (zipfile.BadZipFile, Exception):  # noqa: BLE001
        pass
    return BytesIO(data)


def _a1111_text_from_png_bytes(body: bytes) -> bytes | None:
    """Extract A1111's ``parameters`` chunk from a PNG.

    A1111/Forge encode the full prompt + metadata block under the ``parameters``
    tEXt chunk. Returns the raw text bytes, or None if absent.
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
    value = text.get("parameters")
    if isinstance(value, str) and _looks_like_a1111_text(value):
        return value.encode("utf-8")
    return None


def _looks_like_a1111_bytes(body: bytes) -> bool:
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return False
    return _looks_like_a1111_text(text)


def _looks_like_a1111_text(text: str) -> bool:
    from .foreign_workflow import _looks_like_a1111
    return _looks_like_a1111(text)


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
            with fsspec.open(source_path_or_stdin, mode='rb') as fh:
                data = fh.read()
            stream = _maybe_extract_workflow_json(data)
            for obj in ijson.items(stream, '', multiple_values=True, use_float=True):
                yield obj
        else:
            async with aiofiles.open(source_path_or_stdin, mode='rb') as f:
                async for obj in ijson.items_async(f, '', multiple_values=True, use_float=True):
                    yield obj
