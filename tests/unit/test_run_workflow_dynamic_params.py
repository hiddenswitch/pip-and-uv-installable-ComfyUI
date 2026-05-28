from __future__ import annotations

import click
import pytest

import comfy.cmd.cli as cli_module
from comfy.cmd.cli import _RunWorkflowCommand
from comfy.entrypoints.workflow_params import Param


def _ctx() -> click.Context:
    command = click.Command(
        "run-workflow",
        params=[
            click.Option(["--all"], is_flag=True),
            click.Option(["--video"], multiple=True),
            click.Option(["--prompt"]),
            click.Option(["--set"], multiple=True),
        ],
    )
    return click.Context(command)


def _patch_params(monkeypatch: pytest.MonkeyPatch, params: list[Param]) -> None:
    monkeypatch.setattr(cli_module, "_discover_from_ref", lambda ref: params)


def test_rewrites_discovered_workflow_flag_to_set(monkeypatch):
    _patch_params(
        monkeypatch,
        [
            Param(
                node_id="266:300",
                class_type="PrimitiveInt",
                widget_name="value",
                value=81,
                type="INT",
                flag_name="num-frames-in-the-middle",
            )
        ],
    )

    args = _RunWorkflowCommand._rewrite_workflow_param_args(
        ["workflow.json", "--video", "input.mp4", "--num-frames-in-the-middle", "17"],
        ctx=_ctx(),
    )

    assert args == [
        "workflow.json",
        "--video",
        "input.mp4",
        "--set",
        "266:300.inputs.value=17",
    ]


def test_rewrites_discovered_workflow_flag_equals_form(monkeypatch):
    _patch_params(
        monkeypatch,
        [
            Param(
                node_id="266:300",
                class_type="PrimitiveInt",
                widget_name="value",
                value=81,
                type="INT",
                flag_name="num-frames-in-the-middle",
            )
        ],
    )

    args = _RunWorkflowCommand._rewrite_workflow_param_args(
        ["workflow.json", "--num-frames-in-the-middle=17"],
        ctx=_ctx(),
    )

    assert args == ["workflow.json", "--set", "266:300.inputs.value=17"]


def test_rewrites_discovered_boolean_flags(monkeypatch):
    _patch_params(
        monkeypatch,
        [
            Param(
                node_id="10",
                class_type="PrimitiveBoolean",
                widget_name="value",
                value=False,
                type="BOOLEAN",
                flag_name="enable-loop",
            )
        ],
    )

    assert _RunWorkflowCommand._rewrite_workflow_param_args(
        ["workflow.json", "--enable-loop"],
        ctx=_ctx(),
    ) == ["workflow.json", "--set", "10.inputs.value=true"]
    assert _RunWorkflowCommand._rewrite_workflow_param_args(
        ["workflow.json", "--no-enable-loop"],
        ctx=_ctx(),
    ) == ["workflow.json", "--set", "10.inputs.value=false"]


def test_builtin_option_collision_gets_x_prefixed_workflow_flag(monkeypatch):
    _patch_params(
        monkeypatch,
        [
            Param(
                node_id="12",
                class_type="PrimitiveString",
                widget_name="value",
                value="old",
                type="STRING",
                flag_name="prompt",
            )
        ],
    )

    args = _RunWorkflowCommand._rewrite_workflow_param_args(
        ["workflow.json", "--prompt", "builtin", "--x-prompt", "workflow"],
        ctx=_ctx(),
    )

    assert args == [
        "workflow.json",
        "--prompt",
        "builtin",
        "--set",
        "12.inputs.value=workflow",
    ]


def test_workflow_flag_for_param_uses_x_prefix_on_collision():
    param = Param(
        node_id="12",
        class_type="PrimitiveString",
        widget_name="value",
        value="old",
        type="STRING",
        flag_name="prompt",
    )

    assert _RunWorkflowCommand._workflow_flag_for_param(param, ctx=_ctx()) == "--x-prompt"


def test_extract_workflow_ref_uses_click_option_arity():
    assert _RunWorkflowCommand._extract_workflow_ref(
        ["--all", "--video", "input.mp4", "workflow.json", "--help"],
        ctx=_ctx(),
    ) == "workflow.json"
