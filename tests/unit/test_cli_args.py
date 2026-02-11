import pytest
from unittest.mock import patch
from comfy import cli_args
from comfy.cli_args_types import LatentPreviewMethod, PerformanceFeature

# Helper function to parse args and return the Configuration object
def _parse_test_args(args_list):
    parser = cli_args._create_parser()
    # The `args_parsing=True` makes it use the provided list.
    with patch.object(parser, 'parse_known_args_with_config_files', return_value=(parser.parse_known_args(args_list)[0], [], [])):
        return cli_args._parse_args(parser, args_parsing=True)

@pytest.mark.parametrize("args, expected", [
    ([], []),
    (['--extra-model-paths-config', 'a'], ['a']),
    (['--extra-model-paths-config', 'a', '--extra-model-paths-config', 'b'], ['a', 'b']),
    (['--extra-model-paths-config', 'a,b'], ['a', 'b']),
    (['--extra-model-paths-config', 'a,b', '--extra-model-paths-config', 'c'], ['a', 'b', 'c']),
    (['--extra-model-paths-config', ' a , b ', '--extra-model-paths-config', 'c'], ['a', 'b', 'c']),
    (['--extra-model-paths-config', 'a,b', 'c'], ['a', 'b', 'c']),
])
def test_extra_model_paths_config(args, expected):
    """Test that extra_model_paths_config is parsed correctly."""
    config = _parse_test_args(args)
    assert config.extra_model_paths_config == expected

def test_default_values():
    """Test that default values are set correctly when no args are provided."""
    config = _parse_test_args([])
    assert config.listen == "127.0.0.1"
    assert config.port == 8188
    assert config.auto_launch is False
    assert config.extra_model_paths_config == []
    assert config.preview_method == LatentPreviewMethod.Auto
    assert config.logging_level == 'INFO'
    assert config.multi_user is False
    assert config.disable_xformers is False
    assert config.gpu_only is False
    assert config.highvram is False
    assert config.lowvram is False
    assert config.normalvram is False
    assert config.novram is False
    assert config.cpu is False

def test_listen_and_port():
    """Test --listen and --port arguments."""
    config = _parse_test_args(['--listen', '0.0.0.0', '--port', '8000'])
    assert config.listen == '0.0.0.0'
    assert config.port == 8000

def test_listen_no_arg():
    """Test --listen without an argument."""
    config = _parse_test_args(['--listen'])
    assert config.listen == '0.0.0.0,::'

def test_auto_launch_flags():
    """Test --auto-launch and --disable-auto-launch flags."""
    config_auto = _parse_test_args(['--auto-launch'])
    assert config_auto.auto_launch is True

    config_disable = _parse_test_args(['--disable-auto-launch'])
    assert config_disable.auto_launch is False

    # Test that --disable-auto-launch overrides --auto-launch if both are present
    # The order matters, argparse behavior. Last one wins for store_true/false.
    config_both_1 = _parse_test_args(['--auto-launch', '--disable-auto-launch'])
    assert config_both_1.auto_launch is False

    config_both_2 = _parse_test_args(['--disable-auto-launch', '--auto-launch'])
    assert config_both_2.auto_launch is False

def test_windows_standalone_build_enables_auto_launch():
    """Test that --windows-standalone-build enables auto-launch."""
    config = _parse_test_args(['--windows-standalone-build'])
    assert config.windows_standalone_build is True
    assert config.auto_launch is True

def test_windows_standalone_build_with_disable_auto_launch():
    """Test that --disable-auto-launch overrides --windows-standalone-build's auto-launch."""
    config = _parse_test_args(['--windows-standalone-build', '--disable-auto-launch'])
    assert config.windows_standalone_build is True
    assert config.auto_launch is False

def test_force_fp16_enables_fp16_unet():
    """Test that --force-fp16 enables --fp16-unet."""
    config = _parse_test_args(['--force-fp16'])
    assert config.force_fp16 is True
    assert config.fp16_unet is True

@pytest.mark.parametrize("vram_arg, expected_true_field", [
    ('--gpu-only', 'gpu_only'),
    ('--highvram', 'highvram'),
    ('--normalvram', 'normalvram'),
    ('--lowvram', 'lowvram'),
    ('--novram', 'novram'),
    ('--cpu', 'cpu'),
])
def test_vram_modes(vram_arg, expected_true_field):
    """Test mutually exclusive VRAM mode arguments."""
    config = _parse_test_args([vram_arg])
    all_vram_fields = ['gpu_only', 'highvram', 'normalvram', 'lowvram', 'novram', 'cpu']
    for field in all_vram_fields:
        if field == expected_true_field:
            assert getattr(config, field) is True
        else:
            assert getattr(config, field) is False

def test_preview_method():
    """Test the --preview-method argument."""
    config = _parse_test_args(['--preview-method', 'TAESD'])
    assert config.preview_method == LatentPreviewMethod.TAESD

def test_logging_level():
    """Test the --logging-level argument."""
    config = _parse_test_args(['--logging-level', 'debug'])
    assert config.logging_level == 'DEBUG'

def test_multi_user():
    """Test the --multi-user flag."""
    config = _parse_test_args(['--multi-user'])
    assert config.multi_user is True

def test_disable_xformers():
    """Test the --disable-xformers flag."""
    config = _parse_test_args(['--disable-xformers'])
    assert config.disable_xformers is True


# ---------------------------------------------------------------------------
# --fast: comma-separated, space-separated, and mixed
# ---------------------------------------------------------------------------

class TestFastArg:
    def test_not_provided(self):
        config = _parse_test_args([])
        assert config.fast == set()

    def test_no_values(self):
        """--fast with no values should produce an empty list."""
        config = _parse_test_args(['--fast'])
        assert list(config.fast) == []

    def test_single_value(self):
        config = _parse_test_args(['--fast', 'fp16_accumulation'])
        assert PerformanceFeature.Fp16Accumulation in config.fast

    def test_space_separated(self):
        config = _parse_test_args(['--fast', 'fp16_accumulation', 'fp8_matrix_mult'])
        assert PerformanceFeature.Fp16Accumulation in config.fast
        assert PerformanceFeature.Fp8MatrixMultiplication in config.fast
        assert len(config.fast) == 2

    def test_comma_separated(self):
        config = _parse_test_args(['--fast', 'fp16_accumulation,fp8_matrix_mult'])
        assert PerformanceFeature.Fp16Accumulation in config.fast
        assert PerformanceFeature.Fp8MatrixMultiplication in config.fast
        assert len(config.fast) == 2

    def test_comma_separated_with_spaces(self):
        config = _parse_test_args(['--fast', 'fp16_accumulation , fp8_matrix_mult'])
        assert PerformanceFeature.Fp16Accumulation in config.fast
        assert PerformanceFeature.Fp8MatrixMultiplication in config.fast

    def test_mixed_comma_and_space(self):
        config = _parse_test_args(['--fast', 'fp16_accumulation,fp8_matrix_mult', 'cublas_ops'])
        assert PerformanceFeature.Fp16Accumulation in config.fast
        assert PerformanceFeature.Fp8MatrixMultiplication in config.fast
        assert PerformanceFeature.CublasOps in config.fast
        assert len(config.fast) == 3

    def test_all_features_comma(self):
        all_values = ','.join(f.value for f in PerformanceFeature)
        config = _parse_test_args(['--fast', all_values])
        for feature in PerformanceFeature:
            assert feature in config.fast

    def test_invalid_value_raises(self):
        with pytest.raises(SystemExit):
            _parse_test_args(['--fast', 'not_a_real_feature'])

    def test_invalid_in_comma_list_raises(self):
        with pytest.raises(SystemExit):
            _parse_test_args(['--fast', 'fp16_accumulation,bogus'])


# ---------------------------------------------------------------------------
# --image: comma-separated, space-separated, and mixed
# ---------------------------------------------------------------------------

class TestImageArg:
    def test_not_provided(self):
        config = _parse_test_args([])
        assert config.image is None

    def test_single_value(self):
        config = _parse_test_args(['--image', 'photo.png'])
        assert config.image == ['photo.png']

    def test_space_separated(self):
        config = _parse_test_args(['--image', 'a.png', 'b.png'])
        assert config.image == ['a.png', 'b.png']

    def test_comma_separated(self):
        config = _parse_test_args(['--image', 'a.png,b.png'])
        assert config.image == ['a.png', 'b.png']

    def test_comma_separated_with_spaces(self):
        config = _parse_test_args(['--image', ' a.png , b.png '])
        assert config.image == ['a.png', 'b.png']

    def test_mixed_comma_and_space(self):
        config = _parse_test_args(['--image', 'a.png,b.png', 'c.png'])
        assert config.image == ['a.png', 'b.png', 'c.png']

    def test_uri_values(self):
        config = _parse_test_args(['--image', 'https://example.com/img.png,s3://bucket/img.png'])
        assert config.image == ['https://example.com/img.png', 's3://bucket/img.png']


# ---------------------------------------------------------------------------
# --workflows: verify comma + space still works (backward compat)
# ---------------------------------------------------------------------------

class TestWorkflowsArg:
    def test_not_provided(self):
        config = _parse_test_args([])
        assert config.workflows == []

    def test_space_separated(self):
        config = _parse_test_args(['--workflows', 'a.json', 'b.json'])
        assert config.workflows == ['a.json', 'b.json']

    def test_comma_separated(self):
        config = _parse_test_args(['--workflows', 'a.json,b.json'])
        assert config.workflows == ['a.json', 'b.json']

    def test_mixed(self):
        config = _parse_test_args(['--workflows', 'a.json,b.json', 'c.json'])
        assert config.workflows == ['a.json', 'b.json', 'c.json']

    def test_repeated(self):
        config = _parse_test_args(['--workflows', 'a.json', '--workflows', 'b.json'])
        assert config.workflows == ['a.json', 'b.json']


# ---------------------------------------------------------------------------
# --blacklist-custom-nodes / --whitelist-custom-nodes: backward compat
# ---------------------------------------------------------------------------

class TestCustomNodeListArgs:
    def test_blacklist_comma(self):
        config = _parse_test_args(['--blacklist-custom-nodes', 'foo,bar'])
        assert config.blacklist_custom_nodes == ['foo', 'bar']

    def test_blacklist_space(self):
        config = _parse_test_args(['--blacklist-custom-nodes', 'foo', 'bar'])
        assert config.blacklist_custom_nodes == ['foo', 'bar']

    def test_whitelist_comma(self):
        config = _parse_test_args(['--whitelist-custom-nodes', 'foo,bar'])
        assert config.whitelist_custom_nodes == ['foo', 'bar']

    def test_whitelist_repeated(self):
        config = _parse_test_args(['--whitelist-custom-nodes', 'a', '--whitelist-custom-nodes', 'b,c'])
        assert config.whitelist_custom_nodes == ['a', 'b', 'c']
