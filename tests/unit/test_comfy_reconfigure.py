"""``Comfy.reconfigure()`` cycles the subprocess only when
``MODEL_MANAGEMENT_ARGS`` fingerprints differ, and never interrupts a
running job because ``max_workers=1`` serializes execution.

Tests exercise both the in-process ContextVarExecutor path and the
ProcessPoolExecutor path where reconfigure has real teeth (restarting
the subprocess is the only way to flip ``lowvram`` / ``novram`` /
precision / attention settings because model_management latches those
at import).
"""
from __future__ import annotations

import asyncio
import os
import uuid
from unittest import mock

import pytest

from comfy.cli_args_types import Configuration
from comfy.component_model.configuration import model_management_fingerprint


def _trivial_workflow() -> dict:
    return {
        "1": {"class_type": "EmptyImage",
              "inputs": {"width": 8, "height": 8, "batch_size": 1, "color": 0}},
        "2": {"class_type": "PreviewImage",
              "inputs": {"images": ["1", 0]}},
    }


class TestFingerprint:
    def test_none_and_empty_match(self):
        assert model_management_fingerprint(None) == model_management_fingerprint({})

    def test_irrelevant_keys_ignored(self):
        """Fields outside MODEL_MANAGEMENT_ARGS don't affect the
        fingerprint — changing ``prompt`` or ``output_directory`` doesn't
        force a subprocess swap."""
        a = Configuration(prompt="hello", output_directory="/tmp")
        b = Configuration(prompt="world", output_directory="/home")
        assert model_management_fingerprint(a) == model_management_fingerprint(b)

    def test_model_management_args_change_fingerprint(self):
        a = Configuration()
        b = Configuration(novram=True)
        assert model_management_fingerprint(a) != model_management_fingerprint(b)

    def test_set_valued_fast_field_stable(self):
        """``fast`` is a set; the fingerprint must be stable regardless
        of set-iteration order (which is non-deterministic in general)."""
        from comfy.cli_args_types import PerformanceFeature
        a = Configuration(fast={PerformanceFeature.CublasOps,
                                 PerformanceFeature.Fp16Accumulation})
        b = Configuration(fast={PerformanceFeature.Fp16Accumulation,
                                 PerformanceFeature.CublasOps})
        assert model_management_fingerprint(a) == model_management_fingerprint(b)


class TestReconfigureContextVar:
    """Reconfigure against the default in-process ContextVarExecutor.
    The main thing we verify here is bookkeeping (fingerprint updates,
    no subprocess swap) since there's no subprocess to cycle."""

    @pytest.mark.asyncio
    async def test_reconfigure_same_fingerprint_is_no_op(self):
        from comfy.client.embedded_comfy_client import Comfy

        async with Comfy() as client:
            executor_before = client._executor
            swapped = await client.reconfigure(Configuration(prompt="different"))
            assert swapped is False
            assert client._executor is executor_before

    @pytest.mark.asyncio
    async def test_reconfigure_none_matches_empty(self):
        from comfy.client.embedded_comfy_client import Comfy
        async with Comfy() as client:
            swapped = await client.reconfigure(None)
            assert swapped is False

    @pytest.mark.asyncio
    async def test_fingerprint_property_tracks_current_config(self):
        from comfy.client.embedded_comfy_client import Comfy
        async with Comfy(configuration=Configuration()) as client:
            fp0 = client.fingerprint
            # A no-op (model-management-wise) reconfigure leaves the fp alone.
            await client.reconfigure(Configuration(prompt="x"))
            assert client.fingerprint == fp0


class TestReconfigureProcessPool:
    """Reconfigure ACROSS the ProcessPool boundary: the old subprocess
    must be torn down and a new one started with the new configuration."""

    @pytest.mark.asyncio
    async def test_reconfigure_cycles_the_subprocess(self):
        """Submit a probe job that records its PID, reconfigure to a new
        MODEL_MANAGEMENT_ARGS fingerprint, submit another probe, assert
        the PIDs differ — confirming the subprocess was actually recycled.
        """
        from comfy.client.embedded_comfy_client import Comfy
        # Start with a config that forces ProcessPoolExecutor.
        cfg_a = Configuration(cpu=True)
        async with Comfy(configuration=cfg_a) as client:
            # A cheap workflow that returns SaveImage/PreviewImage outputs.
            # But to read the PID we bypass queue_prompt and call a tiny
            # function through the pool directly — the goal is to prove
            # the subprocess was recycled, not to run diffusion.
            pid_a = await asyncio.get_event_loop().run_in_executor(
                client._executor, os.getpid
            )

            cfg_b = Configuration(cpu=True, disable_smart_memory=True)
            swapped = await client.reconfigure(cfg_b)
            assert swapped is True, (
                "disable_smart_memory is in MODEL_MANAGEMENT_ARGS; "
                "reconfigure should have cycled the executor"
            )

            pid_b = await asyncio.get_event_loop().run_in_executor(
                client._executor, os.getpid
            )
            assert pid_a != pid_b, (
                f"subprocess was not recycled after reconfigure: "
                f"pid_a={pid_a} pid_b={pid_b}"
            )

    @pytest.mark.asyncio
    async def test_reconfigure_same_fingerprint_same_process(self):
        """Reconfiguring to the same MODEL_MANAGEMENT_ARGS values must NOT
        restart the subprocess — we want the happy path to be free."""
        from comfy.client.embedded_comfy_client import Comfy
        cfg_a = Configuration(cpu=True)
        async with Comfy(configuration=cfg_a) as client:
            pid_a = await asyncio.get_event_loop().run_in_executor(
                client._executor, os.getpid
            )
            # Change a field that isn't in MODEL_MANAGEMENT_ARGS.
            swapped = await client.reconfigure(
                Configuration(cpu=True, prompt="irrelevant_change")
            )
            assert swapped is False
            pid_b = await asyncio.get_event_loop().run_in_executor(
                client._executor, os.getpid
            )
            assert pid_a == pid_b


class TestQueuePromptPerJobConfiguration:
    """``queue_prompt`` reads ``__metadata_v1__.configuration`` from the
    envelope and triggers ``reconfigure`` when the job's config differs
    from the client's currently-installed configuration."""

    @pytest.mark.asyncio
    async def test_envelope_configuration_triggers_reconfigure(self):
        from comfy.client.embedded_comfy_client import Comfy
        from comfy.component_model.prompt_envelope import set_configuration

        # Start the client with defaults.
        async with Comfy() as client:
            # Sentinel wrapping of reconfigure so we can observe the call.
            with mock.patch.object(client, "reconfigure",
                                    wraps=client.reconfigure) as spy:
                prompt = set_configuration(_trivial_workflow(),
                                            {"disable_smart_memory": True})
                await client.queue_prompt(prompt, prompt_id=str(uuid.uuid4()))
                assert spy.called, "queue_prompt should call reconfigure when the envelope carries a new configuration"

    @pytest.mark.asyncio
    async def test_envelope_without_configuration_does_not_reconfigure(self):
        from comfy.client.embedded_comfy_client import Comfy
        from comfy.component_model.prompt_envelope import wrap_with_metadata

        async with Comfy() as client:
            with mock.patch.object(client, "reconfigure",
                                    wraps=client.reconfigure) as spy:
                prompt = wrap_with_metadata(_trivial_workflow(),
                                             {"labels": ["regression"]})
                await client.queue_prompt(prompt, prompt_id=str(uuid.uuid4()))
                assert not spy.called, (
                    "envelope without 'configuration' sub-field must not trigger reconfigure"
                )
