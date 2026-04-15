"""Prompt envelope: attach per-job metadata to a workflow dict.

The ComfyUI prompt dict is ``{node_id: {"class_type": ..., "inputs": {...}}, ...}``
— every top-level entry is expected to be a node. Any extra key will fail
vanilla ``validate_prompt`` with ``missing_node_type``. That's the
behavior we want: malformed envelopes must fail loudly.

We attach a single well-known key, ``__metadata_v1__``, whose value is an
arbitrary dict controlled by the envelope schema. Callers on the server
side pop it with :func:`extract_metadata` *before* handing the clean
prompt to ``validate_prompt``. Callers on the client side attach it with
:func:`wrap_with_metadata` or the higher-level :func:`set_configuration`.

The envelope is deliberately flat: a prompt dict plus a metadata dict at
the same level, not a two-field outer object. That way a vanilla
``/prompt`` envelope that already unwraps ``{"prompt": {...}}`` needs no
changes — the inner prompt dict carries the metadata itself.

Any special-casing of ``__metadata_v1__`` must live in this module (and
the two server endpoints that call into it). ``validate_prompt`` stays
upstream-vanilla.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping


METADATA_KEY = "__metadata_v1__"
_CONFIGURATION_FIELD = "configuration"


def extract_metadata(prompt: Mapping[str, Any]) -> tuple[dict, Any]:
    """Return ``(clean_prompt, metadata)``.

    ``clean_prompt`` is a shallow copy of ``prompt`` with ``__metadata_v1__``
    stripped. If there was no metadata key, ``metadata`` is an empty dict
    (never ``None`` — callers commonly destructure fields off it without
    guarding).

    The input ``prompt`` is never mutated.
    """
    if METADATA_KEY not in prompt:
        # No-op: still return a dict (not a view) so callers can mutate
        # the result without surprising the caller.
        return dict(prompt), {}

    clean: dict = {k: v for k, v in prompt.items() if k != METADATA_KEY}
    metadata = prompt[METADATA_KEY]
    return clean, metadata


def wrap_with_metadata(prompt: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict:
    """Return a shallow copy of ``prompt`` with ``__metadata_v1__`` set.

    An empty ``metadata`` dict produces an unchanged copy — we don't write
    an empty envelope key that would just confuse later tooling.
    """
    out = dict(prompt)
    if metadata:
        # Shallow-copy the metadata so the caller can't reach in and mutate
        # the wrapped prompt's envelope after the fact.
        out[METADATA_KEY] = copy.deepcopy(dict(metadata))
    return out


def get_configuration(prompt: Mapping[str, Any]) -> dict:
    """Return the ``configuration`` sub-field of the metadata envelope.

    Empty dict when absent — the natural sentinel for "no per-job overrides".
    """
    meta = prompt.get(METADATA_KEY) or {}
    if not isinstance(meta, dict):
        return {}
    conf = meta.get(_CONFIGURATION_FIELD) or {}
    if not isinstance(conf, dict):
        return {}
    return dict(conf)


def set_configuration(prompt: Mapping[str, Any], configuration: Mapping[str, Any]) -> dict:
    """Return a shallow copy of ``prompt`` with the envelope's
    ``configuration`` field set.

    Other sibling fields under ``__metadata_v1__`` are preserved; only
    ``configuration`` is overwritten. Callers that want to merge with an
    existing configuration should read + compose explicitly.
    """
    out = dict(prompt)
    existing = out.get(METADATA_KEY)
    if not isinstance(existing, dict):
        existing = {}
    new_meta = dict(existing)
    new_meta[_CONFIGURATION_FIELD] = dict(configuration)
    out[METADATA_KEY] = new_meta
    return out
