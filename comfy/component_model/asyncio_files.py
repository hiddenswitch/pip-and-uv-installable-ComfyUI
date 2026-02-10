import asyncio

try:
    from collections.abc import Buffer
except ImportError:
    from typing_extensions import Buffer
from io import BytesIO
from typing import Literal, AsyncGenerator

import fsspec
import ijson
import aiofiles
import sys
import shlex


from .uris import is_uri as _is_uri


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
            # URIs: https://, s3://, hf://, gcs://, etc. — delegate to fsspec
            with fsspec.open(source_path_or_stdin, mode='rb') as f:
                for obj in ijson.items(f, '', multiple_values=True, use_float=True):
                    yield obj
        else:
            async with aiofiles.open(source_path_or_stdin, mode='rb') as f:
                async for obj in ijson.items_async(f, '', multiple_values=True, use_float=True):
                    yield obj
