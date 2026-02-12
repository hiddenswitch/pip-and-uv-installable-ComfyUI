import pytest
from comfy.cli_args_types import LatentPreviewMethod, PerformanceFeature, Configuration
from comfy.cmd.cli import _build_config, app, _validate_mutex

import typer


def _parse_test_args(args_list: list[str]) -> Configuration:
    # Parse the args list into a dict
    params = _defaults_dict()
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg.startswith("--"):
            key = arg.lstrip("-").replace("-", "_")
            # Check if next arg is a value or another flag
            if i + 1 < len(args_list) and not args_list[i + 1].startswith("--"):
                value = args_list[i + 1]
                # Try to convert to appropriate type
                if value.lower() in ("true", "false"):
                    params[key] = value.lower() == "true"
                else:
                    try:
                        params[key] = int(value)
                    except ValueError:
                        try:
                            params[key] = float(value)
                        except ValueError:
                            # Check if this is a list arg
                            if key in params and isinstance(params[key], list):
                                params[key].append(value)
                            else:
                                params[key] = value
                i += 2
            else:
                # Flag without value (store_true)
                params[key] = True
                i += 1
        else:
            i += 1
    return _build_config(params)


def _defaults_dict() -> dict:
    return {
        "listen": "127.0.0.1",
        "port": 8188,
        "auto_launch": False,
        "disable_auto_launch": False,
        "preview_method": "auto",
        "logging_level": "INFO",
        "fast": [],
        "base_paths": [],
        "extra_model_paths_config": [],
        "panic_when": [],
        "whitelist_custom_nodes": [],
        "blacklist_custom_nodes": [],
        "workflows": [],
        "image": [],
        "video": [],
        "audio": [],
    }


def _config_from(**kwargs) -> Configuration:
    defaults = {
        "listen": "127.0.0.1",
        "port": 8188,
        "auto_launch": False,
        "disable_auto_launch": False,
        "preview_method": "auto",
        "logging_level": "INFO",
        "fast": [],
        "base_paths": [],
        "extra_model_paths_config": [],
        "panic_when": [],
        "whitelist_custom_nodes": [],
        "blacklist_custom_nodes": [],
        "workflows": [],
        "image": [],
        "video": [],
        "audio": [],
    }
    params = {**defaults, **kwargs}
    return _build_config(params)


def test_default_values():
    config = _config_from()
    assert config.listen == "127.0.0.1"
    assert config.port == 8188
    assert config.auto_launch is False
    assert config.extra_model_paths_config == []
    assert config.preview_method == LatentPreviewMethod.Auto
    assert config.logging_level == "INFO"
    assert config.multi_user is False
    assert config.disable_xformers is False
    assert config.gpu_only is False
    assert config.highvram is False
    assert config.lowvram is False
    assert config.normalvram is False
    assert config.novram is False
    assert config.cpu is False


def test_listen_and_port():
    config = _config_from(listen="0.0.0.0", port=8000)
    assert config.listen == "0.0.0.0"
    assert config.port == 8000


def test_auto_launch_flags():
    config_auto = _config_from(auto_launch=True)
    assert config_auto.auto_launch is True

    config_disable = _config_from(disable_auto_launch=True)
    assert config_disable.auto_launch is False

    # disable_auto_launch overrides auto_launch
    config_both = _config_from(auto_launch=True, disable_auto_launch=True)
    assert config_both.auto_launch is False


def test_windows_standalone_build_enables_auto_launch():
    config = _config_from(windows_standalone_build=True)
    assert config.windows_standalone_build is True
    assert config.auto_launch is True


def test_windows_standalone_build_with_disable_auto_launch():
    config = _config_from(windows_standalone_build=True, disable_auto_launch=True)
    assert config.windows_standalone_build is True
    assert config.auto_launch is False


def test_force_fp16_enables_fp16_unet():
    config = _config_from(force_fp16=True)
    assert config.force_fp16 is True
    assert config.fp16_unet is True


@pytest.mark.parametrize("vram_arg, expected_true_field", [
    ("gpu_only", "gpu_only"),
    ("highvram", "highvram"),
    ("normalvram", "normalvram"),
    ("lowvram", "lowvram"),
    ("novram", "novram"),
    ("cpu", "cpu"),
])
def test_vram_modes(vram_arg, expected_true_field):
    config = _config_from(**{vram_arg: True})
    all_vram_fields = ["gpu_only", "highvram", "normalvram", "lowvram", "novram", "cpu"]
    for field in all_vram_fields:
        if field == expected_true_field:
            assert getattr(config, field) is True
        else:
            assert getattr(config, field) is False


def test_vram_mutex_violation():
    with pytest.raises(typer.BadParameter):
        _config_from(gpu_only=True, highvram=True)


def test_preview_method():
    config = _config_from(preview_method="taesd")
    assert config.preview_method == LatentPreviewMethod.TAESD


def test_preview_method_case_insensitive():
    # LatentPreviewMethod values are lowercase
    config = _config_from(preview_method="none")
    assert config.preview_method == LatentPreviewMethod.NoPreviews


def test_logging_level():
    config = _config_from(logging_level="DEBUG")
    assert config.logging_level == "DEBUG"


def test_multi_user():
    config = _config_from(multi_user=True)
    assert config.multi_user is True


def test_disable_xformers():
    config = _config_from(disable_xformers=True)
    assert config.disable_xformers is True


class TestFastArg:
    def test_not_provided(self):
        config = _config_from(fast=[])
        assert config.fast == set()

    def test_single_value(self):
        config = _config_from(fast=["fp16_accumulation"])
        assert PerformanceFeature.Fp16Accumulation in config.fast

    def test_comma_separated(self):
        config = _config_from(fast=["fp16_accumulation,fp8_matrix_mult"])
        assert PerformanceFeature.Fp16Accumulation in config.fast
        assert PerformanceFeature.Fp8MatrixMultiplication in config.fast
        assert len(config.fast) == 2

    def test_comma_separated_with_spaces(self):
        config = _config_from(fast=["fp16_accumulation , fp8_matrix_mult"])
        assert PerformanceFeature.Fp16Accumulation in config.fast
        assert PerformanceFeature.Fp8MatrixMultiplication in config.fast

    def test_multiple_values(self):
        config = _config_from(fast=["fp16_accumulation", "fp8_matrix_mult"])
        assert PerformanceFeature.Fp16Accumulation in config.fast
        assert PerformanceFeature.Fp8MatrixMultiplication in config.fast
        assert len(config.fast) == 2

    def test_mixed_comma_and_multiple(self):
        config = _config_from(fast=["fp16_accumulation,fp8_matrix_mult", "cublas_ops"])
        assert PerformanceFeature.Fp16Accumulation in config.fast
        assert PerformanceFeature.Fp8MatrixMultiplication in config.fast
        assert PerformanceFeature.CublasOps in config.fast
        assert len(config.fast) == 3

    def test_all_features(self):
        all_values = ",".join(f.value for f in PerformanceFeature)
        config = _config_from(fast=[all_values])
        for feature in PerformanceFeature:
            assert feature in config.fast

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            _config_from(fast=["not_a_real_feature"])


class TestListFields:
    def test_extra_model_paths_comma(self):
        config = _config_from(extra_model_paths_config=["a,b"])
        assert config.extra_model_paths_config == ["a", "b"]

    def test_extra_model_paths_multiple(self):
        config = _config_from(extra_model_paths_config=["a", "b"])
        assert config.extra_model_paths_config == ["a", "b"]

    def test_extra_model_paths_mixed(self):
        config = _config_from(extra_model_paths_config=["a,b", "c"])
        assert config.extra_model_paths_config == ["a", "b", "c"]

    def test_blacklist_comma(self):
        config = _config_from(blacklist_custom_nodes=["foo,bar"])
        assert config.blacklist_custom_nodes == ["foo", "bar"]

    def test_whitelist_comma(self):
        config = _config_from(whitelist_custom_nodes=["foo,bar"])
        assert config.whitelist_custom_nodes == ["foo", "bar"]

    def test_workflows_comma(self):
        config = _config_from(workflows=["a.json,b.json"])
        assert config.workflows == ["a.json", "b.json"]

    def test_workflows_multiple(self):
        config = _config_from(workflows=["a.json", "b.json"])
        assert config.workflows == ["a.json", "b.json"]

    def test_image_comma(self):
        config = _config_from(image=["a.png,b.png"])
        assert config.image == ["a.png", "b.png"]

    def test_image_uri(self):
        config = _config_from(image=["https://example.com/img.png,s3://bucket/img.png"])
        assert config.image == ["https://example.com/img.png", "s3://bucket/img.png"]


class TestMutexValidation:
    def test_precision_mutex(self):
        with pytest.raises(typer.BadParameter):
            _config_from(force_fp32=True, force_fp16=True)

    def test_attention_mutex(self):
        with pytest.raises(typer.BadParameter):
            _config_from(use_split_cross_attention=True, use_sage_attention=True)

    def test_upcast_mutex(self):
        with pytest.raises(typer.BadParameter):
            _config_from(force_upcast_attention=True, dont_upcast_attention=True)


def test_cli_args_default_configuration():
    from comfy.cli_args import default_configuration
    config = default_configuration()
    assert isinstance(config, Configuration)
    assert config.listen == "127.0.0.1"


def test_cli_args_cli_args_configuration():
    from comfy.cli_args import cli_args_configuration
    config = cli_args_configuration()
    assert isinstance(config, Configuration)


def test_cli_args_enables_dynamic_vram():
    from comfy.cli_args import enables_dynamic_vram
    # With default config (no fast features), should return False
    result = enables_dynamic_vram()
    assert result is False


def test_cli_args_imports():
    from comfy.cli_args import DEFAULT_VERSION_STRING
    assert "comfyanonymous" in DEFAULT_VERSION_STRING

    from comfy.cli_args import PerformanceFeature as PF
    assert PF.Fp16Accumulation is not None

    from comfy.cli_args import LatentPreviewMethod as LPM
    assert LPM.Auto is not None

    from comfy.cli_args import EnumAction
    assert EnumAction is not None
