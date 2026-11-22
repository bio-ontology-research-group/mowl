"""Query embedding geometries and the shared operator tree (mowl.nn.query).

The properties tested are the ones the papers state for the operators; they are checked
on hand-set parameters, independently of training.
"""
import math

import pytest
import torch as th

from mowl.nn.query.expression import (NEGATION_ID, answers, anchor, from_nested, from_signature,
                                      ids, intersect, negate, project, signature,
                                      template_from_nested, to_dnf, union)
from mowl.nn.query.geometry import (AConEGeometry, BetaEGeometry, ConEGeometry,
                                    Query2BoxGeometry, UnsupportedOperator)

N, R, D = 6, 4, 3
pi = math.pi


class Graph:
    """0 -r2-> 1, 0 -r2-> 2, 1 -r3-> 4, 2 -r3-> 5."""

    def __init__(self):
        self.edges = {(0, 2): {1, 2}, (1, 3): {4}, (2, 3): {5}}

    def successors(self, e, r):
        return self.edges.get((e, r), set())


# -- operator tree ---------------------------------------------------------------------

def test_nested_format_two_hop_intersection_and_negation():
    q = from_nested((("e", ("r", "r")), ("e", ("r", "n"))), ((0, (2, 3)), (1, (2, NEGATION_ID))))
    assert q.kind == "i"
    assert signature(q) == ("i", ("p", ("p", ("a",))), ("n", ("p", ("a",))))
    assert ids(q) == [0, 2, 3, 1, 2]
    rebuilt = from_signature(signature(q), iter(ids(q)))
    assert rebuilt == q
    assert signature(template_from_nested((("e", ("r", "r")), ("e", ("r", "n"))))) == signature(q)


def test_exact_answers():
    g, universe = Graph(), set(range(6))
    two_hop = from_nested(("e", ("r", "r")), (0, (2, 3)))
    assert answers(two_hop, g, universe) == {4, 5}
    neg = negate(project(2, anchor(0)))
    assert answers(neg, g, universe) == universe - {1, 2}
    assert answers(union(anchor(3), anchor(4)), g, universe) == {3, 4}
    assert answers(intersect(project(2, anchor(0)), anchor(2)), g, universe) == {2}


def test_dnf_moves_unions_to_the_top_and_uses_de_morgan():
    q = project(2, union(anchor(0), anchor(1)))                      # ∃(C ⊔ D)
    assert set(to_dnf(q)) == {project(2, anchor(0)), project(2, anchor(1))}
    q = intersect(union(anchor(0), anchor(1)), anchor(2))            # (C ⊔ D) ⊓ E
    assert set(to_dnf(q)) == {intersect(anchor(0), anchor(2)), intersect(anchor(1), anchor(2))}
    q = negate(union(anchor(0), anchor(1)))                          # ¬(C ⊔ D) = ¬C ⊓ ¬D
    assert to_dnf(q) == [intersect(negate(anchor(0)), negate(anchor(1)))]
    assert to_dnf(negate(negate(anchor(0)))) == [anchor(0)]
    # the DNF has the same answers
    g, universe = Graph(), set(range(6))
    q = intersect(negate(union(project(2, anchor(0)), anchor(5))), union(anchor(3), anchor(4)))
    dnf = set()
    for b in to_dnf(q):
        dnf |= answers(b, g, universe)
    assert dnf == answers(q, g, universe)


def test_class_expression_front_end():
    from mowl.owlapi import OWLAPIAdapter
    from mowl.nn.query.expression import (UnsupportedClassExpression, Vocabulary,
                                          class_expression_to_query)
    ad = OWLAPIAdapter()
    ns = "http://mowl/qe#"
    A, B = ad.create_class(ns + "A"), ad.create_class(ns + "B")
    r = ad.create_object_property(ns + "r")
    voc = Vocabulary(class_id={ns + "A": 0, ns + "B": 1}.__getitem__,
                     individual_id=lambda iri: 9, relation_id=lambda iri, inv: 10 + inv,
                     isa_inverse=1, extension=False)
    some = class_expression_to_query(ad.create_object_some_values_from(
        r, ad.create_object_intersection_of(A, ad.create_complement_of(B))), voc)
    assert some == project(11, intersect(anchor(0), negate(anchor(1))))
    every = class_expression_to_query(ad.create_object_all_values_from(r, A), voc)
    assert every == negate(project(11, negate(anchor(0))))          # ∀r.A = ¬∃r.¬A
    assert class_expression_to_query(ad.create_object_union_of(A, B), voc) == \
        union(anchor(0), anchor(1))
    ext = Vocabulary(voc.class_id, voc.individual_id, voc.relation_id, 1, extension=True)
    assert class_expression_to_query(A, ext) == project(1, anchor(0))
    card = ad.data_factory.getOWLObjectMinCardinality(2, r)
    with pytest.raises(UnsupportedClassExpression):
        class_expression_to_query(card, voc)


# -- Query2Box ---------------------------------------------------------------------------

def _q2b():
    th.manual_seed(0)
    return Query2BoxGeometry(N, R, D, gamma=4.0, center_weight=0.5)


def test_q2b_intersection_box_is_not_larger_than_its_operands():
    g = _q2b()
    a = (th.randn(5, D), th.rand(5, D))
    b = (th.randn(5, D), th.rand(5, D))
    c, o = g.intersect([a, b])
    assert (o <= th.minimum(a[1], b[1]) + 1e-6).all() and (o >= 0).all()
    lo = th.minimum(a[0], b[0])
    hi = th.maximum(a[0], b[0])
    assert ((c >= lo - 1e-6) & (c <= hi + 1e-6)).all()                # a convex combination


def test_q2b_has_no_negation():
    g = _q2b()
    assert not g.supports_negation
    with pytest.raises(UnsupportedOperator):
        g.negate(g.anchor(th.tensor([0])))


# -- BetaE -------------------------------------------------------------------------------

def test_betae_negation_is_an_involution_and_intersection_idempotent():
    th.manual_seed(0)
    g = BetaEGeometry(N, R, D, gamma=4.0)
    s = g.project(g.anchor(th.tensor([0, 1])), th.tensor([2, 3]))
    nn_ = g.negate(g.negate(s))
    assert all(th.allclose(x, y, atol=1e-5) for x, y in zip(nn_, s))
    i = g.intersect([s, s, s])
    assert all(th.allclose(x, y, atol=1e-5) for x, y in zip(i, s))
    assert all((x >= 0.05 - 1e-6).all() for x in s)


def test_betae_score_is_gamma_minus_kl_and_negation_flips_density():
    th.manual_seed(0)
    g = BetaEGeometry(N, R, D, gamma=4.0)
    s = g.anchor(th.tensor([0]))
    assert float(g.distance(s, th.tensor([[0]]))) == pytest.approx(0.0, abs=1e-5)
    assert float(g.logit(s, th.tensor([[0]]))) == pytest.approx(4.0, abs=1e-5)
    other = float(g.distance(s, th.tensor([[1]])))
    assert other > 0
    # KL against torch's reference
    a, b = g.anchor(th.tensor([1]))
    ref = th.distributions.kl_divergence(th.distributions.Beta(a, b),
                                         th.distributions.Beta(*s)).sum()
    assert other == pytest.approx(float(ref), rel=1e-4)
    # the negated query is nearer to a distribution with inverted parameters
    ns = g.negate(s)
    assert th.allclose(ns[0], 1 / s[0])


# -- ConE and AConE ----------------------------------------------------------------------

def test_cone_intersection_aperture_is_not_larger_than_its_operands():
    th.manual_seed(0)
    g = ConEGeometry(N, R, D, gamma=4.0)
    a = (th.rand(5, D) * 2 - 1, th.rand(5, D) * 3)
    b = (th.rand(5, D) * 2 - 1, th.rand(5, D) * 3)
    axis, aperture = g.intersect([a, b])
    assert (aperture <= th.minimum(a[1], b[1]) + 1e-6).all()
    # the intersection of a cone with itself has its axis
    same_axis, _ = g.intersect([a, a])
    assert th.allclose(th.cos(same_axis), th.cos(a[0]), atol=1e-5)


def test_every_geometry_embeds_a_query_tree_in_batch():
    for cls in (Query2BoxGeometry, BetaEGeometry, ConEGeometry, AConEGeometry):
        th.manual_seed(0)
        g = cls(N, R, D, gamma=4.0)
        q = [intersect(project(2, anchor(0)), project(3, project(2, anchor(1))))] * 2
        sig = signature(q[0])
        rows = th.tensor([ids(x) for x in q])
        state = g.embed(sig, rows)
        logit = g.logit(state, th.tensor([[0, 1, 2], [3, 4, 5]]))
        assert logit.shape == (2, 3) and th.isfinite(logit).all()


# -- properties the papers state, one test each -------------------------------------------

def test_cone_negation_of_a_negative_aperture_is_the_complement():
    """A cone with aperture -a is the cone of aperture a: its complement has aperture
    pi - a (not pi + a), and agrees with the complement of the cone of aperture a."""
    th.manual_seed(0)
    g = ConEGeometry(N, R, D, gamma=4.0)
    axis = th.tensor([[0.5, -1.0, 2.0]])
    ap = th.tensor([[0.4, 1.0, 3.0]])
    n_axis, n_ap = g.negate((axis, -ap))
    assert (n_ap >= 0).all() and (n_ap <= pi + 1e-6).all()
    ref_axis, ref_ap = g.negate((axis, ap))
    assert th.allclose(n_ap, ref_ap) and th.allclose(n_axis, ref_axis)
    # the old behaviour is selectable
    old = ConEGeometry(N, R, D, gamma=4.0, signed_negation=True)
    _, o_ap = old.negate((axis, -ap))
    assert th.allclose(o_ap, pi + ap) and (o_ap > pi).any()
    _, o_ap = old.negate((axis, ap))
    assert th.allclose(o_ap, pi - ap)
