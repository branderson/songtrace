import shutil
import subprocess
from pathlib import Path

import pytest


def test_static_js_unit_tests():
    """The page's own logic, run under Node's test runner."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    tests = sorted(str(p) for p in (Path(__file__).parent / "js").glob("*.test.mjs"))
    proc = subprocess.run([node, "--test", *tests], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
