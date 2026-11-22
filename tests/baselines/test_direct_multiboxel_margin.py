"""The direct MultiBoxEL module names the margin of its objective by a property of the
class (:mod:`mowl.baselines`)."""
from mowl.baselines.nn.el.multiboxel.module import MultiBoxELModule


def test_the_direct_module_reads_the_margin_semantics_from_its_objective():
    assert MultiBoxELModule.margin_semantics.fget(
        type("S", (), {"objective": "structural"})()) == "strict"
    assert MultiBoxELModule.margin_semantics.fget(
        type("S", (), {"objective": "axiom"})()) == "slack"
