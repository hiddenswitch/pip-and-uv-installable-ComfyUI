from unittest import mock

from comfy import storage


def test_fast_nvme_link_thresholds():
    cases = [
        ("8.0 GT/s PCIe", "4", True),
        ("16.0 GT/s PCIe", "4", True),
        ("32.0 GT/s PCIe", "2", True),
        ("8.0 GT/s PCIe", "2", False),
    ]
    for speed, width, expected in cases:
        with mock.patch.object(storage, "_read", side_effect=[speed, width]):
            assert storage._fast_nvme("nvme0n1") is expected


def test_non_nvme_is_not_fast():
    assert storage._fast_nvme("sda") is False


def test_every_model_file_must_be_on_fast_storage():
    with mock.patch.object(storage, "fast_storage", side_effect=[True, False]):
        assert storage.model_fast_disk(["first", "second"]) is False
    with mock.patch.object(storage, "fast_storage", side_effect=[True, True]):
        assert storage.model_fast_disk(["first", "second"]) is True
