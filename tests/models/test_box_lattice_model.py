"""Box lattice embeddings (Vilnis et al., ACL 2018) over the class taxonomy."""
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.evaluation.model_scoring import axiom_scoring_of  # noqa: E402
from mowl.models.box_lattice import BoxLattice  # noqa: E402
from mowl.nn.el.box_lattice import BoxLatticeModule, join, meet, volume  # noqa: E402
from tests.models.taxonomy_helpers import (check_trained_taxonomy, total_loss,  # noqa: E402
                                           train)


def box(lo, hi):
    return th.tensor([lo], dtype=th.double), th.tensor([hi], dtype=th.double)


def test_lattice_operations_and_non_distributivity():
    # Appendix B.1 (Vilnis et al.): the box lattice is non-distributive
    x, y, z = box([0.0], [0.3]), box([0.2], [0.6]), box([0.5], [1.0])

    def is_empty(b):
        return bool((b[1] <= b[0]).any())

    def m(a, b):
        r = meet(*a, *b)
        return None if is_empty(r) else r

    def j(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return join(*a, *b)

    # The 1-D example printed in Appendix B.1 does not give different sides with these
    # numbers (both are [.2, .3]); the lattice is nevertheless non-distributive, shown in
    # two dimensions below.
    assert m(x, j(y, z))[0].item() == pytest.approx(0.2)
    # non-distributivity in two dimensions: the join of two boxes covers area that neither
    # has, and the meet with a third box can sit entirely in that area
    a_, b_, c_ = box([0.0, 0.0], [0.4, 0.4]), box([0.6, 0.6], [1.0, 1.0]), \
        box([0.45, 0.45], [0.55, 0.55])
    assert m(a_, c_) is None and m(b_, c_) is None
    left2 = m(c_, j(a_, b_))                   # c ∧ (a ∨ b) = c
    right2 = j(m(c_, a_), m(c_, b_))           # ⊥ ∨ ⊥ = ⊥
    assert left2 is not None and right2 is None
    assert m(x, z) is None                       # disjoint: meet is bottom
    assert volume(*x).item() == pytest.approx(0.3)


def test_volume_is_probability_and_conditional_is_volume_ratio():
    module = BoxLatticeModule(3, embed_dim=2)
    with th.no_grad():     # [min | delta]: a=[0,.5]^2, b=[0,1]x[0,.5], c=[.6,1]^2
        module.class_embed.weight.copy_(th.tensor([
            [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 1.0, 0.5], [0.6, 0.6, 0.4, 0.4]]))
    idx = th.tensor([0, 1, 2])
    assert module.marginal(idx).tolist() == pytest.approx([0.25, 0.5, 0.16])
    p = lambda s, t: module.conditional(th.tensor([s]), th.tensor([t])).item()  # noqa: E731
    assert p(0, 1) == pytest.approx(1.0)           # a inside b: P(b|a) = 1
    assert p(1, 0) == pytest.approx(0.5)           # P(a|b) = vol(a)/vol(b)
    assert p(0, 2) <= 0.0                          # disjoint: lower bound, non-positive
    # losses: containment costs nothing, the reverse and the disjoint pair cost more
    loss = lambda s, t: module(th.tensor([[s, t]]), "gci0").item()  # noqa: E731
    assert abs(loss(0, 1)) < 1e-5
    assert loss(1, 0) > 0.5 and loss(0, 2) > 10
    # negatives (target P = 0): zero when disjoint, positive when contained
    neg = lambda s, t: module(th.tensor([[s, t]]), "gci0", neg=True).item()  # noqa: E731
    assert neg(0, 2) == 0.0 and neg(0, 1) > 5.0


def test_projection_keeps_boxes_in_the_unit_cube():
    module = BoxLatticeModule(4, embed_dim=3)
    with th.no_grad():
        module.class_embed.weight.mul_(5).sub_(1)
    module.project()
    mn, mx = module.corners(th.arange(4))
    assert (mn >= 0).all() and (mx <= 1 + 1e-6).all() and (mx - mn >= 1e-6 - 1e-9).all()


def test_negative_pairs_do_not_update_delta():
    module = BoxLatticeModule(2, embed_dim=2, freeze_delta_negatives=True)
    with th.no_grad():
        module.class_embed.weight.copy_(th.tensor([[0.1, 0.1, 0.5, 0.5], [0.4, 0.4, 0.5, 0.5]]))
    module(th.tensor([[0, 1]]), "gci0", neg=True).sum().backward()
    g = module.class_embed.weight.grad
    assert g[:, :2].abs().sum() > 0 and g[:, 2:].abs().sum() == 0


def test_trains_on_the_taxonomy_and_declares_scoring():
    model = train(BoxLattice, epochs=600, embed_dim=8, learning_rate=0.01)
    assert axiom_scoring_of(model) == "module" and model.scored_forms == ("gci0",)
    check_trained_taxonomy(model)
    first = train(BoxLattice, epochs=1, embed_dim=8, learning_rate=0.01)
    assert total_loss(model) < total_loss(first)


def test_unary_marginals_follow_descendant_counts():
    model = train(BoxLattice, epochs=2, embed_dim=2, unary_weight=1.0)
    frac = model.descendant_fractions()
    idx = model.class_index_dict
    n = len(model.named_class_index_dict)
    assert frac[idx["http://mowl.borg/bench#C"]] == pytest.approx(4 / n)   # C, B, A, D
    assert frac[idx["http://mowl.borg/bench#A"]] == pytest.approx(1 / n)
    assert model.additional_training_loss(0) is not None
