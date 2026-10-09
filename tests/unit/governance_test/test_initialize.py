import builtins
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pytest

from comfy.app import governance
from comfy.cli_args import default_configuration
from comfy.cli_args_types import Configuration
from comfy.execution_context import context_configuration


POLICY_MESSAGE = "organization's policy"


@pytest.fixture
def configuration() -> Configuration:
    """The configuration governance.initialize reads, current for the test."""
    configuration = default_configuration()
    configuration.disabled_nodes_config = None
    configuration.extra_model_paths_config = None
    configuration.enable_manager = False
    with context_configuration(configuration):
        yield configuration


@pytest.fixture
def governed(configuration: Configuration, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    policy_path = tmp_path / "policy.signed.json"
    monkeypatch.setattr(governance, "GOVERNANCE_REQUIRED", True, raising=False)
    monkeypatch.setattr(governance, "_POLICY_PATH", policy_path, raising=False)
    monkeypatch.setattr(governance, "_policy", None, raising=False)
    monkeypatch.setattr(governance, "_disabled_nodes", frozenset(), raising=False)
    # A custom-node policy switches off bytecode writes for the process; undo that and the pack policy after each test.
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", os.environ.get("PYTHONDONTWRITEBYTECODE", ""))
    monkeypatch.setattr(governance, "_custom_node_mode", None, raising=False)
    monkeypatch.setattr(governance, "_denied_packs", frozenset(), raising=False)
    monkeypatch.setattr(governance, "_allowed_packs", MappingProxyType({}), raising=False)
    return policy_path


def _assert_policy_exit(exc_info: pytest.ExceptionInfo[SystemExit], log_text: str) -> None:
    assert exc_info.value.code != 0
    assert POLICY_MESSAGE in log_text


def _run_governed(
    working_directory: Path,
    timeout: float,
    *arguments: str,
    import_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the fork's comfyui entry point in a governed build whose signed policy is missing."""
    setup = [
        "from pathlib import Path",
        "import sys",
        "from comfy.app import governance",
        "governance.GOVERNANCE_REQUIRED = True",
        f"governance._POLICY_PATH = Path({str(working_directory / 'missing-policy.signed.json')!r})",
    ]
    if import_path is not None:
        setup.append(f"sys.path.insert(0, {str(import_path)!r})")
    setup.extend(
        (
            f"sys.argv = ['comfyui', *{list(arguments)!r}]",
            "from comfy.cmd.main import entrypoint",
            "entrypoint()",
        )
    )

    return subprocess.run(
        [sys.executable, "-c", "\n".join(setup)],
        cwd=working_directory,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_build_constants_have_upstream_safe_defaults() -> None:
    assert governance.GOVERNANCE_REQUIRED is False
    assert governance.GOVERNANCE_BUILD_IDENTITY == ""
    assert governance.GOVERNANCE_PUBLIC_KEY == ""
    assert governance.GOVERNANCE_MIN_POLICY_GENERATION == 0
    assert governance.GOVERNANCE_CAPABILITY_VERSION == 1


def test_governance_import_does_not_require_policy_dependencies(process_startup_timeout_seconds: float) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins

original_import = builtins.__import__

def import_without_policy_dependencies(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'blake3', 'cryptography'}:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_policy_dependencies
from comfy.app import governance
assert governance.GOVERNANCE_REQUIRED is False
""",
        ],
        capture_output=True,
        text=True,
        timeout=process_startup_timeout_seconds,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_initialize_is_noop_without_filesystem_access(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_filesystem_access(*_args, **_kwargs):
        pytest.fail("initialize accessed the filesystem")

    monkeypatch.setattr(governance, "GOVERNANCE_REQUIRED", False, raising=False)
    monkeypatch.setattr(builtins, "open", unexpected_filesystem_access)
    monkeypatch.setattr(os.path, "isfile", unexpected_filesystem_access)
    monkeypatch.setattr(Path, "is_file", unexpected_filesystem_access)
    monkeypatch.setattr(Path, "read_bytes", unexpected_filesystem_access)

    governance.initialize()


def test_initialize_exits_when_manifest_is_missing(governed: Path, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc_info:
        governance.initialize()

    _assert_policy_exit(exc_info, caplog.text)


def test_initialize_exits_on_internal_exception(
    governed: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    governed.write_bytes(b"signed policy")

    def fail_verification(_envelope_bytes: bytes):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(governance, "verify_and_load", fail_verification, raising=False)

    with pytest.raises(SystemExit) as exc_info:
        governance.initialize()

    _assert_policy_exit(exc_info, caplog.text)


def test_initialize_rejects_disabled_nodes_config(
    configuration: Configuration,
    governed: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given a valid policy, so the unsigned config is the only reason to stop
    _use_policy(governed, monkeypatch, {})
    configuration.disabled_nodes_config = str(governed.parent / "disabled.yaml")

    with pytest.raises(SystemExit) as exc_info:
        governance.initialize()

    _assert_policy_exit(exc_info, caplog.text)
    assert "unsigned disabled-node config is not allowed" in caplog.text


def test_initialize_does_not_gate_on_capability_version(
    governed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    governed.write_bytes(b"signed policy")
    monkeypatch.setattr(governance, "GOVERNANCE_CAPABILITY_VERSION", -1, raising=False)
    monkeypatch.setattr(governance, "verify_and_load", lambda envelope_bytes: {}, raising=False)

    governance.initialize()


@pytest.mark.parametrize("active_forms", [["model"], ["customNode", "model"], ["unknownForm"]])
def test_initialize_exits_on_form_this_build_cannot_enforce(
    governed: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    active_forms: list[str],
) -> None:
    governed.write_bytes(b"signed policy")
    monkeypatch.setattr(governance, "verify_and_load", lambda envelope_bytes: {"activeForms": active_forms}, raising=False)

    with pytest.raises(SystemExit) as exc_info:
        governance.initialize()

    _assert_policy_exit(exc_info, caplog.text)
    assert governance._policy is None


def test_initialize_applies_policy_limited_to_enforced_forms(
    governed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    governed.write_bytes(b"signed policy")
    policy = {"activeForms": ["nodeId"], "disabledNodes": ["SomeNode"]}
    monkeypatch.setattr(governance, "verify_and_load", lambda envelope_bytes: policy, raising=False)

    governance.initialize()

    assert governance._disabled_nodes == frozenset({"SomeNode"})


def _use_policy(governed: Path, monkeypatch: pytest.MonkeyPatch, policy: dict) -> None:
    governed.write_bytes(b"signed policy")
    monkeypatch.setattr(governance, "verify_and_load", lambda envelope_bytes: policy, raising=False)


def test_initialize_prunes_partner_nodes_with_disabled_nodes(governed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a policy disabling one node id and one partner node
    _use_policy(
        governed,
        monkeypatch,
        {
            "activeForms": ["nodeId", "partnerNode"],
            "disabledNodes": ["SomeNode"],
            "disabledPartnerNodes": [{"nodeId": "PartnerNode", "providerId": "acme"}],
        },
    )

    # When the policy is applied
    governance.initialize()

    # Then both ids are pruned
    assert governance._disabled_nodes == frozenset({"SomeNode", "PartnerNode"})


def _custom_node_policy(mode: str) -> dict:
    return {"activeForms": ["customNode"], "customNodeMode": mode, "packs": [], "deniedPacks": []}


@pytest.mark.parametrize("mode", ["allowlist", "blocklist"])
def test_initialize_turns_manager_off_under_custom_node_policy(
    configuration: Configuration,
    governed: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mode: str,
) -> None:
    # Given Manager is enabled, whose prestartup runs scheduled install scripts before any pack is checked
    _use_policy(governed, monkeypatch, _custom_node_policy(mode))
    configuration.enable_manager = True

    # When the policy is applied
    governance.initialize()

    # Then startup continues with Manager off, and the log says why
    assert configuration.enable_manager is False
    assert "ComfyUI-Manager is turned off" in caplog.text


@pytest.mark.parametrize("mode", ["allowlist", "blocklist"])
def test_initialize_applies_custom_node_policy_without_manager(governed: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    # Given a custom-node policy with Manager off
    _use_policy(governed, monkeypatch, _custom_node_policy(mode))

    # When the policy is applied
    governance.initialize()

    # Then the pack gate takes the policy's mode
    assert governance._custom_node_mode == mode


def test_initialize_allows_manager_without_custom_node_policy(configuration: Configuration, governed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given Manager is enabled and the policy governs node ids only
    _use_policy(governed, monkeypatch, {"activeForms": ["nodeId"], "disabledNodes": ["SomeNode"]})
    configuration.enable_manager = True

    # When the policy is applied, then startup continues
    governance.initialize()

    assert governance._custom_node_mode is None
    assert configuration.enable_manager is True


def test_initialize_accepts_extra_model_paths_that_add_custom_nodes(configuration: Configuration, governed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a policy and an extra paths file that adds a custom_nodes folder; packs found there pass the same pack gate at load
    governed.write_bytes(b"signed policy")
    monkeypatch.setattr(governance, "verify_and_load", lambda envelope_bytes: {}, raising=False)
    extra_paths = governed.parent / "extra-paths.yaml"
    extra_paths.write_text("test:\n  base_path: .\n  checkpoints: models\n  custom_nodes: nodes\n", encoding="utf-8")
    configuration.extra_model_paths_config = [str(extra_paths)]

    # When the policy is applied, then startup continues
    governance.initialize()


def test_manager_import_waits_until_after_governance(tmp_path: Path, process_startup_timeout_seconds: float) -> None:
    sentinel_path = tmp_path / "manager-imported"
    manager_package = tmp_path / "comfyui_manager"
    manager_package.mkdir()
    (manager_package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel_path)!r}).write_text('imported', encoding='utf-8')\n",
        encoding="utf-8",
    )

    result = _run_governed(
        tmp_path,
        process_startup_timeout_seconds,
        "--enable-manager",
        "--cpu",
        "--quick-test-for-ci",
        import_path=tmp_path,
    )

    assert result.returncode != 0
    assert POLICY_MESSAGE in result.stdout + result.stderr
    assert not sentinel_path.exists()


def test_governance_failure_precedes_prestartup_scripts(tmp_path: Path, process_startup_timeout_seconds: float) -> None:
    sentinel_path = tmp_path / "prestartup-ran"
    pack_path = tmp_path / "custom_nodes" / "sentinel_pack"
    pack_path.mkdir(parents=True)
    (pack_path / "prestartup_script.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel_path)!r}).write_text('ran', encoding='utf-8')\n",
        encoding="utf-8",
    )

    result = _run_governed(
        tmp_path,
        process_startup_timeout_seconds,
        "--base-directory",
        str(tmp_path),
        "--cpu",
        "--quick-test-for-ci",
    )

    assert result.returncode != 0
    assert POLICY_MESSAGE in result.stdout + result.stderr
    assert not sentinel_path.exists()
