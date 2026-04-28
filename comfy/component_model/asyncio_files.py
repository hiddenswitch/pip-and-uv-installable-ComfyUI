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


def _maybe_extract_workflow_json(data: bytes) -> BytesIO:
    """If *data* is a zip archive, return a BytesIO over the first workflow-
    shaped .json inside (UI form with `nodes`+`links`, or API form). Otherwise
    return BytesIO over *data* unchanged.

    Civitai workflow uploads are commonly zips with one or more .json graphs.
    """
    import zipfile
    if not data.startswith(b"PK\x03\x04"):
        return BytesIO(data)
    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
            if not json_names:
                return BytesIO(data)
            for name in json_names:
                with zf.open(name) as inner:
                    body = inner.read()
                try:
                    parsed = json.loads(body)
                except Exception:
                    continue
                if isinstance(parsed, dict) and (
                    ("nodes" in parsed and "links" in parsed)
                    or all(isinstance(v, dict) and "class_type" in v for v in parsed.values())
                ):
                    return BytesIO(body)
            # No clear workflow-shaped JSON; fall back to the first .json entry.
            with zf.open(json_names[0]) as inner:
                return BytesIO(inner.read())
    except (zipfile.BadZipFile, Exception):
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
