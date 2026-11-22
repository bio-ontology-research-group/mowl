"""The general scoring helpers of the query geometries, used on their own."""
import torch as th

from mowl.evaluation.query_scoring import axiom_query, top_answers
from mowl.nn.query.expression import signature
from mowl.projection.query_graph import ISA_INV


def test_axiom_query_columns_per_normal_form():
    rows = th.tensor([[3, 1, 5], [4, 0, 2]])
    sig, ids, subjects = axiom_query("gci2", rows, num_classes=10, extension=True)
    assert ids.tolist() == [[5, ISA_INV, 5], [2, ISA_INV, 3]]       # D, isa^-1, r^-1
    assert subjects.tolist() == [3, 4]
    sig0, ids0, _ = axiom_query("gci0", th.tensor([[3, 5]]), 10, extension=False)
    assert ids0.tolist() == [[5]] and sig0 != sig
    _, _, individuals = axiom_query("class_assertion", th.tensor([[2, 7]]), 10, True)
    assert individuals.tolist() == [12]                             # after the classes


def test_top_answers_are_named_best_first():
    out = top_answers(th.tensor([0.1, 0.9, 0.5]), ["a", "b", "c"], 2)
    assert [n for n, _ in out] == ["b", "c"]
    assert len(top_answers(th.tensor([1.0]), ["a"], 5)) == 1
