import subprocess
import sys
from importlib.resources import files
from pathlib import Path


PROJECT_ROOT = Path(str(files("tests"))).parent


def test_gemma4_import_does_not_require_torchaudio():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torchaudio'] = None; import comfy.text_encoders.gemma4",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
