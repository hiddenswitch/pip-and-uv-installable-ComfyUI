from importlib.resources import as_file, files

import numpy as np

from comfy_extras.nodes.nodes_paligemma import _get_reconstruct_masks, extract_objs


def test_reconstruct_masks_matches_reference():
    with as_file(files("tests.language") / "paligemma_mask_reference.npz") as f:
        fixture = np.load(f)
        indices = fixture["indices"]
        expected = fixture["output"]

    reconstruct = _get_reconstruct_masks()
    actual = reconstruct(indices)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_extract_objs_produces_mask():
    text = (
        "<loc0100><loc0200><loc0500><loc0600>"
        "<seg010><seg020><seg030><seg040>"
        "<seg050><seg060><seg070><seg080>"
        "<seg090><seg100><seg110><seg120>"
        "<seg013><seg024><seg035><seg046> cat"
    )
    result = extract_objs(text, 640, 480)

    assert len(result) == 1
    obj = result[0]
    assert obj["name"] == "cat"
    assert obj["mask"].shape == (480, 640)
    assert obj["mask"].max() > 0
