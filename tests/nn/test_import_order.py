"""Importing ``mowl.nn`` before ``mowl.datasets`` must work (issue #176)."""

import subprocess
import sys


def test_mowl_nn_can_be_imported_first():
    code = ("import mowl; mowl.init_jvm('1g'); import mowl.nn; "
            "from mowl.nn import ELEmModule; print('ok')")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")
