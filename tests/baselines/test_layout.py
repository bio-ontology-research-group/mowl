"""The library is the combinations of parts; direct implementations are only a reference."""
import ast
import os

import pytest

import mowl
from tests.baselines.harness import all_exempt, all_pairs

ROOT = os.path.dirname(mowl.__file__)


def _python_files():
    for folder, _, files in os.walk(ROOT):
        if os.path.join(ROOT, "baselines") in folder + os.sep or "__pycache__" in folder:
            continue
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(folder, name)


def test_nothing_outside_the_baselines_imports_them():
    offenders = []
    for path in _python_files():
        if os.path.relpath(path, ROOT).startswith("baselines"):
            continue
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any(n == "mowl.baselines" or n.startswith("mowl.baselines.") for n in names):
                offenders.append(f"{os.path.relpath(path, ROOT)}:{node.lineno}")
    assert not offenders, offenders


def _catalogue_keys():
    from mowl.catalog.registry import list_methods
    return sorted(card.key for card in list_methods(implemented=True))


@pytest.mark.parametrize("key", _catalogue_keys())
def test_every_implemented_method_is_compared_or_exempt(key):
    compared = {p.method for p in all_pairs()}
    exempt = all_exempt()
    assert key in compared or key in exempt, (
        f"{key}: no comparison in tests/baselines/pairs_*.py and no stated exemption")
    assert not (key in compared and key in exempt), f"{key}: both compared and exempt"


def test_exemptions_give_a_reason():
    for key, reason in all_exempt().items():
        assert isinstance(reason, str) and len(reason) > 30, key
