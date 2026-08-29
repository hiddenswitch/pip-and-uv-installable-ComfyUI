"""`comfyui workflows run --no-<flag>` must accept the inverse of every
affirmative boolean option we declare.

Typer does not auto-generate ``--no-<flag>`` for plain bool options — you
have to declare the flag with the ``--flag/--no-flag`` slash syntax. This
test guards against someone adding a new bool option without the slash
form and then being surprised that Typer rejects ``--no-<flag>`` at the
command line.
"""
from __future__ import annotations

import re
from importlib.resources import files

import pytest


CLI_PY = files("comfy.cmd").joinpath("cli.py")

# Flags for which ``--no-<flag>`` would be a linguistic double-negative and
# the CLI intentionally does not provide the inverse. A positive counterpart
# is usually available (e.g. --cuda-malloc vs --disable-cuda-malloc).
_DOUBLE_NEGATIVES_PREFIXES = ("--disable-", "--dont-", "--no-")


def _affirmative_bool_flags_in_cli_py() -> list[str]:
    """Scan cli.py for every affirmative bool flag declaration."""
    src = CLI_PY.read_text()
    # Matches ``typer.Option(True|False, ..., "--flag...", ...)`` — captures
    # the last long flag in the decls (short ones are single-letter).
    pattern = re.compile(
        r'typer\.Option\((?:True|False),\s*(?:"-[a-z]",\s*)*"(--[a-z0-9][a-z0-9_/-]*)"'
    )
    flags: set[str] = set()
    for m in pattern.finditer(src):
        # The match may be "--flag/--no-flag"; take the affirmative part only.
        raw = m.group(1).split("/", 1)[0]
        if raw.startswith(_DOUBLE_NEGATIVES_PREFIXES):
            continue
        flags.add(raw)
    return sorted(flags)


def _has_no_inverse_registered(src: str, flag: str) -> bool:
    """Return True if ``--<flag>/--no-<flag>`` appears in *src* in the same
    ``typer.Option`` call — i.e., Typer's slash syntax is wired up."""
    tail = flag.lstrip("-")
    # Match "...--<tail>/--no-<tail>..." on the same line as typer.Option.
    pattern = re.compile(
        rf'typer\.Option\([^)]*"[^"]*--{re.escape(tail)}/--no-{re.escape(tail)}"',
        re.DOTALL,
    )
    return pattern.search(src) is not None


@pytest.mark.parametrize("flag", _affirmative_bool_flags_in_cli_py())
def test_affirmative_bool_flag_has_no_inverse(flag):
    """Every affirmative bool flag in cli.py must declare its inverse via
    Typer's ``--flag/--no-flag`` slash syntax."""
    src = CLI_PY.read_text()
    assert _has_no_inverse_registered(src, flag), (
        f"{flag} lacks its ``--no-<flag>`` inverse in cli.py. "
        f"Change ``typer.Option(..., \"{flag}\", ...)`` to "
        f"``typer.Option(..., \"{flag}/--no-{flag.lstrip('-')}\", ...)``."
    )


def test_guess_settings_default_is_true():
    """Regression guard: the switch to guess-settings-by-default should not
    be reverted silently."""
    from comfy.cli_args_types import Configuration
    c = Configuration()
    assert c.guess_settings is True


def test_guess_settings_has_both_forms():
    src = CLI_PY.read_text()
    # The short form -g is preserved, and the slash inverse is declared.
    assert "--guess-settings/--no-guess-settings" in src
