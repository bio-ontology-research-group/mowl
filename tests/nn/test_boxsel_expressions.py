"""``compile_expression``: names to structure and leaves."""
import pytest

from mowl.nn.el.boxsel.expressions import compile_expression

CLASSES = {"A": 0, "B": 1, "C": 2}
ROLES = {"r": 0, "s": 1}


def test_atoms_conjunctions_and_existentials():
    assert compile_expression("B", CLASSES, ROLES) == (("atom",), [1])
    assert compile_expression(("and", "A", "C"), CLASSES, ROLES) == (
        ("and", ("atom",), ("atom",)), [0, 2])
    assert compile_expression(("some", "s", ("and", "A", ("some", "r", "B"))), CLASSES, ROLES) == (
        ("some", ("and", ("atom",), ("some", ("atom",)))), [1, 0, 0, 1])


def test_unknown_names_and_malformed_expressions_are_refused():
    with pytest.raises(KeyError):
        compile_expression("Z", CLASSES, ROLES)
    with pytest.raises(KeyError):
        compile_expression(("some", "q", "A"), CLASSES, ROLES)
    for bad in (("and", "A"), ("some", "r"), ("or", "A", "B"), (), 3):
        with pytest.raises(ValueError):
            compile_expression(bad, CLASSES, ROLES)


def test_agrees_with_the_direct_function_of_the_baseline():
    from mowl.baselines.nn.el.boxsel.losses import compile_expression as direct
    expr = ("and", ("some", "r", "B"), "A", ("some", "s", ("some", "r", "C")))
    assert compile_expression(expr, CLASSES, ROLES) == direct(expr, CLASSES, ROLES)
