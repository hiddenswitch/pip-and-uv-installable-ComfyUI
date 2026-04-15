"""Probe the behavior of ``validate_prompt`` to understand what can safely
travel inside a workflow dict alongside real nodes.

The goal is to anchor a forward-compatible metadata envelope
(``__metadata_v1__``) that:

  1. **Fails loudly** if submitted to vanilla ``validate_prompt`` without
     being stripped — we want a visible error, not silent dropping.
  2. Is **trivially removable** on the server side so the clean prompt
     passes vanilla validation unchanged.
  3. Round-trips through a dumb client that doesn't know about metadata
     without corrupting real node ids.

These tests document the validate_prompt contract empirically so later
refactors don't regress the envelope semantics.
"""
from __future__ import annotations

import asyncio
import json

import pytest


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _minimal_valid_prompt() -> dict:
    """Smallest prompt vanilla ``validate_prompt`` accepts: single output node.

    The ``SaveImage`` node needs an ``images`` input; we use ``PreviewImage``
    instead which is ``OUTPUT_NODE=True`` and accepts any IMAGE input.
    """
    return {
        "1": {"class_type": "EmptyImage",
              "inputs": {"width": 8, "height": 8, "batch_size": 1, "color": 0}},
        "2": {"class_type": "PreviewImage",
              "inputs": {"images": ["1", 0]}},
    }


class TestValidatePromptBaseline:
    """Sanity checks that establish what the validator accepts before we
    start poking the envelope."""

    def test_minimal_prompt_is_valid(self):
        from comfy.cmd.execution import validate_prompt
        result = _run(validate_prompt("probe-1", _minimal_valid_prompt()))
        assert result.valid, f"minimal prompt should validate cleanly: {result.error!r}"

    def test_missing_class_type_errors_loudly(self):
        """If we drop a node-like key with no class_type into the prompt,
        validate_prompt raises ``missing_node_type`` with the id embedded
        in the error message. This is the behavior the envelope must be
        compatible with when it's NOT stripped by the server."""
        from comfy.cmd.execution import validate_prompt
        prompt = _minimal_valid_prompt()
        prompt["__metadata_v1__"] = {"configuration": {"novram": True}}
        result = _run(validate_prompt("probe-2", prompt))
        assert not result.valid
        assert result.error is not None
        assert result.error.get("type") == "missing_node_type"
        extra = result.error.get("extra_info", {}) or {}
        assert extra.get("node_id") == "__metadata_v1__"

    def test_unknown_class_type_also_errors(self):
        """A fake class_type that isn't a registered node is rejected the
        same way — you cannot just hide metadata under a sentinel node type."""
        from comfy.cmd.execution import validate_prompt
        prompt = _minimal_valid_prompt()
        prompt["__metadata_v1__"] = {
            "class_type": "__MetadataV1__",
            "inputs": {"configuration": {"novram": True}},
        }
        result = _run(validate_prompt("probe-3", prompt))
        assert not result.valid
        assert result.error.get("type") == "missing_node_type"
        # The class_type is surfaced in the error so the caller knows
        # which node failed.
        assert result.error["extra_info"]["class_type"] == "__MetadataV1__"

    def test_integer_node_ids_work(self):
        """Numeric-string node ids are legal (they match the format
        convert_ui_to_api produces). Reserved keys like __metadata_v1__
        only collide if the caller chose that name deliberately."""
        from comfy.cmd.execution import validate_prompt
        # Copy the minimal prompt keyed by int-strings to confirm.
        prompt = {str(i + 10): node for i, node in enumerate(_minimal_valid_prompt().values())}
        # Rewire the link.
        for nid, node in prompt.items():
            for key, val in list(node.get("inputs", {}).items()):
                if isinstance(val, list) and len(val) == 2 and val[0] == "1":
                    node["inputs"][key] = ["10", val[1]]
        result = _run(validate_prompt("probe-4", prompt))
        assert result.valid, result.error

    def test_extra_whole_prompt_key_named_like_metadata_is_not_special_cased(self):
        """Vanilla ComfyUI does not know about ``__metadata_v1__`` — it
        treats it as any other key. Any special-casing must live OUTSIDE
        validate_prompt so upstream compatibility is preserved."""
        from comfy.cmd import execution as execution_mod
        # validate_prompt does not reference __metadata_v1__ anywhere in
        # its source. This test documents that invariant so upstream merges
        # don't silently sneak in a special case.
        src = execution_mod.__file__
        assert "__metadata_v1__" not in open(src).read(), (
            "validate_prompt should remain upstream-vanilla; __metadata_v1__ "
            "handling must live in the envelope helper."
        )


# ---------------------------------------------------------------------------
# Envelope helper contract (implementation tracked in a follow-up edit).
# ---------------------------------------------------------------------------

from comfy.component_model.prompt_envelope import (  # noqa: E402
    METADATA_KEY,
    extract_metadata,
    wrap_with_metadata,
)


class TestEnvelopeHelper:
    def test_metadata_key_is_stable(self):
        """If this changes we need a v2 key and a backward-compat path."""
        assert METADATA_KEY == "__metadata_v1__"

    def test_extract_strips_metadata_and_returns_it(self):
        prompt = _minimal_valid_prompt()
        prompt[METADATA_KEY] = {"configuration": {"novram": True}}
        clean, meta = extract_metadata(prompt)
        assert METADATA_KEY not in clean
        assert meta == {"configuration": {"novram": True}}

    def test_extract_is_a_copy_not_mutation(self):
        """The caller's prompt dict must not be modified in place — the
        same prompt may be handed to multiple queues or retried."""
        prompt = _minimal_valid_prompt()
        prompt[METADATA_KEY] = {"configuration": {"novram": True}}
        snapshot = json.dumps(prompt, sort_keys=True)
        extract_metadata(prompt)
        assert json.dumps(prompt, sort_keys=True) == snapshot

    def test_extract_with_no_metadata_returns_empty(self):
        prompt = _minimal_valid_prompt()
        clean, meta = extract_metadata(prompt)
        assert clean == prompt
        assert meta == {}

    def test_wrap_then_extract_is_identity(self):
        prompt = _minimal_valid_prompt()
        meta = {"configuration": {"novram": True, "compile": True}}
        wrapped = wrap_with_metadata(prompt, meta)
        assert METADATA_KEY in wrapped
        clean, extracted = extract_metadata(wrapped)
        assert clean == prompt
        assert extracted == meta

    def test_wrap_preserves_prompt_keys(self):
        """Wrapping doesn't clobber any node in the prompt dict."""
        prompt = _minimal_valid_prompt()
        wrapped = wrap_with_metadata(prompt, {"configuration": {"cpu": True}})
        for nid in prompt:
            assert wrapped[nid] == prompt[nid]

    def test_wrap_with_empty_metadata_is_a_no_op(self):
        prompt = _minimal_valid_prompt()
        wrapped = wrap_with_metadata(prompt, {})
        assert METADATA_KEY not in wrapped
        assert wrapped == prompt

    def test_cleaned_prompt_passes_vanilla_validation(self):
        """Round-trip test: wrap + extract + validate_prompt succeeds."""
        from comfy.cmd.execution import validate_prompt
        prompt = _minimal_valid_prompt()
        wrapped = wrap_with_metadata(prompt, {"configuration": {"lowvram": True}})
        clean, meta = extract_metadata(wrapped)
        result = _run(validate_prompt("probe-envelope", clean))
        assert result.valid, result.error
        assert meta == {"configuration": {"lowvram": True}}

    def test_metadata_with_non_dict_value_is_preserved_as_is(self):
        """The envelope shape is unconstrained other than ``__metadata_v1__`` being
        the key. If a future schema hangs a list or string off it, extract
        returns it unchanged."""
        prompt = _minimal_valid_prompt()
        prompt[METADATA_KEY] = ["v1", {"configuration": {}}]
        clean, meta = extract_metadata(prompt)
        assert meta == ["v1", {"configuration": {}}]


class TestEnvelopeConfigurationAccessors:
    """Higher-level helpers for the common case: reading/writing the
    ``configuration`` sub-field of the metadata envelope."""

    def test_get_configuration_returns_empty_when_absent(self):
        from comfy.component_model.prompt_envelope import get_configuration
        assert get_configuration({}) == {}
        assert get_configuration({METADATA_KEY: {}}) == {}

    def test_get_configuration_returns_dict_when_set(self):
        from comfy.component_model.prompt_envelope import get_configuration
        prompt = _minimal_valid_prompt()
        prompt[METADATA_KEY] = {"configuration": {"novram": True, "compile": False}}
        assert get_configuration(prompt) == {"novram": True, "compile": False}

    def test_set_configuration_on_empty_prompt(self):
        from comfy.component_model.prompt_envelope import set_configuration
        prompt = _minimal_valid_prompt()
        out = set_configuration(prompt, {"novram": True})
        assert out[METADATA_KEY]["configuration"] == {"novram": True}
        # Original prompt unchanged.
        assert METADATA_KEY not in prompt

    def test_set_configuration_merges_with_existing_metadata(self):
        from comfy.component_model.prompt_envelope import set_configuration
        prompt = _minimal_valid_prompt()
        prompt[METADATA_KEY] = {"something_else": 1, "configuration": {"cpu": True}}
        out = set_configuration(prompt, {"novram": True})
        assert out[METADATA_KEY]["something_else"] == 1
        # New configuration replaces old entirely; callers that want to
        # merge should read + compose explicitly.
        assert out[METADATA_KEY]["configuration"] == {"novram": True}
