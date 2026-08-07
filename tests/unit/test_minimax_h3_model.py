from __future__ import annotations

import torch

from comfy.ldm.minimax.model import (
    _timestep_rows,
    prepare_audio_carry,
    restore_audio_carry,
    time_shift_sigma,
)


def test_timestep_rows_match_h3_video_and_audio_schedules():
    sigma = torch.tensor(0.37)

    rows, values = _timestep_rows(
        sigma,
        shift_v=12.0,
        shift_a=3.0,
        visual_augmentation=0.999,
        audio_augmentation=1.0,
        has_visual_conditioning=False,
        has_audio_conditioning=False,
    )

    assert rows == {"video": 0, "audio": 1}
    torch.testing.assert_close(values[rows["video"]], 1.0 - sigma)
    torch.testing.assert_close(
        values[rows["audio"]],
        1.0 - time_shift_sigma(sigma, 12.0, 3.0),
    )


def test_timestep_rows_include_conditioning_rows_by_semantic_name():
    sigma = torch.tensor(0.5)

    rows, values = _timestep_rows(
        sigma,
        shift_v=12.0,
        shift_a=3.0,
        visual_augmentation=0.999,
        audio_augmentation=1.0,
        has_visual_conditioning=True,
        has_audio_conditioning=True,
    )

    assert rows == {
        "video": 0,
        "audio": 1,
        "visual_condition": 2,
        "audio_condition": 3,
    }
    torch.testing.assert_close(values[rows["visual_condition"]], torch.tensor(0.999))
    torch.testing.assert_close(values[rows["audio_condition"]], torch.tensor(1.0))


def test_timestep_rows_compile_once_across_sampler_sigma_values():
    graphs = []

    def backend(graph, _example_inputs):
        graphs.append(graph)
        return graph.forward

    def timestep_values(sigma):
        return _timestep_rows(
            sigma,
            shift_v=12.0,
            shift_a=3.0,
            visual_augmentation=0.999,
            audio_augmentation=1.0,
            has_visual_conditioning=False,
            has_audio_conditioning=False,
        )[1]

    compiled = torch.compile(timestep_values, backend=backend, fullgraph=True)
    first = compiled(torch.tensor(0.9))
    second = compiled(torch.tensor(0.1))

    assert len(graphs) == 1
    assert not torch.equal(first, second)


def test_audio_carry_round_trip_matches_sampler_schedule_conversion():
    video = torch.zeros(1, 1)
    carried_audio = torch.ones(1, 1)
    timestep = torch.tensor([500.0])
    prepared, state = prepare_audio_carry(
        [video, carried_audio],
        timestep,
        {},
        {"audio_scale": 4.0},
        12.0,
        3.0,
    )

    sigma_a = time_shift_sigma(torch.tensor(0.5), 12.0, 3.0)
    expected_audio = carried_audio * (sigma_a / 0.5)
    torch.testing.assert_close(prepared[1], expected_audio)

    network_output = [torch.zeros_like(video), torch.full_like(carried_audio, 2.0)]
    restored = restore_audio_carry(network_output, state)
    torch.testing.assert_close(
        restored[1],
        -3.0 * expected_audio + (1.0 + 3.0 * sigma_a) * 2.0,
    )
