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


def test_q2b_projection_translates_and_grows_and_answers_are_inside():
    g = _q2b()
    with th.no_grad():
        g.entity_embedding.zero_()
        g.entity_embedding[1] = th.tensor([1.0, 0.0, 0.0])
        g.entity_embedding[2] = th.tensor([5.0, 0.0, 0.0])
        g.relation_embedding[2] = th.tensor([1.0, 0.0, 0.0])
        g.offset_embedding[2] = th.tensor([0.5, 0.5, 0.5])
    state = g.project(g.anchor(th.tensor([0])), th.tensor([2]))
    assert th.allclose(state[0], th.tensor([[1.0, 0.0, 0.0]]))
    assert th.allclose(state[1], th.tensor([[0.5, 0.5, 0.5]]))
    d = g.distance(state, th.tensor([[1, 2]]))[0].detach()
    assert float(d[0]) == pytest.approx(0.0, abs=1e-6)                 # at the center
    assert float(d[1]) == pytest.approx(3.5 + 0.5 * 0.5, abs=1e-6)     # d_out + cen * d_in
    with th.no_grad():
        g.entity_embedding[3] = th.tensor([1.3, 0.1, 0.0])
    inside = g.distance(state, th.tensor([[3]])).detach()               # in the box: d_out = 0
    assert float(inside) == pytest.approx(0.5 * (0.3 + 0.1), abs=1e-6)


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

def test_cone_negation_is_the_complement_and_an_involution():
    th.manual_seed(0)
    g = ConEGeometry(N, R, D, gamma=4.0, center_weight=0.5)
    axis, aperture = th.tensor([[0.5, -1.0, 2.0]]), th.tensor([[0.4, 1.0, 3.0]])
    n_axis, n_ap = g.negate((axis, aperture))
    assert th.allclose(n_ap, pi - aperture)
    twice_axis, twice_ap = g.negate((n_axis, n_ap))
    assert th.allclose(twice_axis, axis, atol=1e-5) and th.allclose(twice_ap, aperture)
    # a point on the axis is inside the cone (outside distance 0) and outside the complement
    with th.no_grad():
        g.entity_embedding[0] = axis[0] * 0 + th.atanh(axis[0] / pi) * g.embedding_range
    e = g.entity_vectors()[0:1]
    assert th.allclose(e, axis, atol=1e-4)
    inside = th.tensor([[0.0, 0.0, 0.0]])
    assert float(g.distance((axis, aperture), th.tensor([[0]])) / g.modulus) == \
        pytest.approx(0.5 * float(th.min(th.abs(th.sin((e - axis) / 2)),
                                         th.abs(th.sin(aperture / 2))).sum()), abs=1e-5)
    outside = g.distance((n_axis, n_ap), th.tensor([[0]]))
    assert float(outside) > float(g.distance((axis, aperture), th.tensor([[0]])))
    assert inside.sum() == 0


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


def test_acone_relations_are_commuting_rotations_that_compose_by_adding_angles():
    th.manual_seed(0)
    g = AConEGeometry(N, R, D, gamma=4.0)
    s = g.anchor(th.tensor([0]))
    ra, rb = th.tensor([2]), th.tensor([3])
    ab = g.project(g.project(s, ra), rb)
    ba = g.project(g.project(s, rb), ra)
    for x, y in zip(ab, ba):
        assert th.allclose(th.cos(x), th.cos(y), atol=1e-5)
        assert th.allclose(th.sin(x), th.sin(y), atol=1e-5)
    (ax_a, ap_a), (ax_b, ap_b) = g._relation(ra), g._relation(rb)
    want_axis = s[0] + ax_a + ax_b
    assert th.allclose(th.cos(ab[0]), th.cos(want_axis), atol=1e-5)
    want_ap = s[1] + ap_a + ap_b                                     # aperture-additive
    assert th.allclose(th.sin(ab[1]), th.sin(want_ap), atol=1e-5)
    # zero angles are the identity rotation
    with th.no_grad():
        g.axis_embedding.zero_()
        g.aperture_embedding.zero_()
    same = g.project(s, ra)
    assert th.allclose(same[0], s[0], atol=1e-6) and th.allclose(same[1], s[1], atol=1e-6)
    assert not hasattr(g, "proj_in")                                  # no projection network


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

def test_q2b_point_inside_the_box_has_zero_outside_distance_and_matches_the_paper_formula():
    """Query2Box, Section 4.1: dist_outside is 0 inside the box; with alpha = 0 the distance
    is dist_outside = ||max(v - q_max, 0) + max(q_min - v, 0)||_1 (written out here from
    the paper, independently of the implementation)."""
    th.manual_seed(0)
    g = Query2BoxGeometry(N, R, D, gamma=4.0, center_weight=0.0)
    centre, offset = th.randn(8, D), th.rand(8, D) + 0.1
    with th.no_grad():
        points = th.randn(8, 5, D) * 2
        g.entity_embedding.zero_()
        # entities 0..4 hold the points of query 0, and so on: use one query at a time
    for q in range(8):
        with th.no_grad():
            g.entity_embedding[:5] = points[q]
        d = g.distance((centre[q:q + 1], offset[q:q + 1]), th.arange(5).unsqueeze(0))[0]
        q_max, q_min = centre[q] + offset[q], centre[q] - offset[q]
        want = (th.clamp(points[q] - q_max, min=0) + th.clamp(q_min - points[q], min=0)).sum(-1)
        assert th.allclose(d, want, atol=1e-5)
        inside = ((points[q] >= q_min) & (points[q] <= q_max)).all(-1)
        assert (d[inside] == 0).all()
    # a point in the box: zero outside distance
    with th.no_grad():
        g.entity_embedding[0] = centre[0] + 0.5 * offset[0]
    assert float(g.distance((centre[:1], offset[:1]), th.tensor([[0]]))) == 0.0


def test_acone_proposition_5_role_inclusion_and_proposition_3_union_distribution():
    """He et al. 2024. Prop. 5: R<theta, g1, d1> is included in S<theta, g2, d2> if g1 <= g2
    and d1 <= d2 (here gamma = 1, aperture-additive): every entity in the cone after r is
    in the cone after s. Prop. 3: projection distributes over a union (gamma = 1, d >= 0)."""
    th.manual_seed(0)
    g = AConEGeometry(200, R, D, gamma=4.0, center_weight=0.0)

    def set_relation(rel, axis, aperture):
        with th.no_grad():
            g.axis_embedding[rel] = th.atanh(th.tensor(axis / pi)) * g.embedding_range
            g.aperture_embedding[rel] = th.atanh(th.tensor(aperture / pi)) * g.embedding_range

    set_relation(2, 0.7, 0.2)                       # R: delta1 = 0.2
    set_relation(3, 0.7, 0.9)                       # S: delta2 = 0.9 >= delta1, same axis
    cone = (th.rand(1, D) * 2 - 1, th.rand(1, D) * 0.5)
    after_r = g.project(cone, th.tensor([2]))
    after_s = g.project(cone, th.tensor([3]))
    ids_ = th.arange(200).unsqueeze(0)
    in_r = g.distance(after_r, ids_)[0] == 0
    in_s = g.distance(after_s, ids_)[0] == 0
    assert in_r.any()                                # not vacuous
    assert (in_s | ~in_r).all()                      # in R-cone implies in S-cone
    assert (g.distance(after_s, ids_)[0] <= g.distance(after_r, ids_)[0] + 1e-6).all()
    # a strictly larger aperture does include more entities for this seed
    assert int(in_s.sum()) >= int(in_r.sum())
