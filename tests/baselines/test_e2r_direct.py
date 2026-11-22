"""Tests of the direct E2R module (mowl.baselines.nn.alc.e2r): subspace operations against linear-algebra
ground truth, loss semantics on hand-built subspaces, non-distributivity of quantum logic,
distributivity of the axis-parallel restriction, and the T3 pattern."""
import itertools

import numpy as np
import pytest
import torch as th

import mowl.datasets  # noqa: F401  (import order, see COMMON circular-import note)
from mowl.baselines.nn.alc.e2r import losses as L
from mowl.baselines.nn.alc.e2r.module import (E2RModule, indicator_projector, is_subspace,
                                    join_projector, meet_projector, orthocomplement,
                                    projector, rank, residual_length)


def _np_rank(m):
    return int(np.linalg.matrix_rank(np.asarray(m), tol=1e-8))


def _random_basis(rng, n, k):
    return rng.standard_normal((n, k))


def _same(p, q):
    return bool(th.allclose(p, q, atol=1e-8))


def _bits(n):
    return [th.tensor(b, dtype=th.float64) for b in itertools.product([0.0, 1.0], repeat=n)]


# -- general subspaces vs numpy --------------------------------------------------------

class TestGeneralSubspaces:

    @pytest.mark.parametrize("seed", range(5))
    def test_meet_join_complement_against_ranks(self, seed):
        rng = np.random.default_rng(seed)
        n = 6
        k1, k2 = rng.integers(1, n, size=2)
        b1, b2 = _random_basis(rng, n, k1), _random_basis(rng, n, k2)
        p1, p2 = projector(b1), projector(b2)
        assert rank(p1) == _np_rank(b1) and rank(p2) == _np_rank(b2)
        # span: rank of the stacked bases
        pj = join_projector(p1, p2)
        assert rank(pj) == _np_rank(np.hstack([b1, b2]))
        assert is_subspace(p1, pj) and is_subspace(p2, pj)
        # intersection: dim = dim1 + dim2 - dim(sum)
        pm = meet_projector(p1, p2)
        assert rank(pm) == _np_rank(b1) + _np_rank(b2) - _np_rank(np.hstack([b1, b2]))
        assert is_subspace(pm, p1) and is_subspace(pm, p2)
        # complement: orthogonal and spanning
        pc = orthocomplement(p1)
        assert rank(pc) == n - rank(p1)
        assert th.allclose(p1 @ pc, th.zeros(n, n, dtype=th.float64), atol=1e-10)
        assert _same(join_projector(p1, pc), th.eye(n, dtype=th.float64))

    def test_intersection_with_shared_vectors(self):
        rng = np.random.default_rng(0)
        shared = rng.standard_normal((7, 2))
        b1 = np.hstack([shared, rng.standard_normal((7, 2))])
        b2 = np.hstack([shared, rng.standard_normal((7, 1))])
        pm = meet_projector(projector(b1), projector(b2))
        assert rank(pm) == 2
        assert _same(pm, projector(shared))

    def test_inclusion(self):
        rng = np.random.default_rng(1)
        b = rng.standard_normal((5, 3))
        sub = b @ rng.standard_normal((3, 2))  # columns inside span(b)
        assert is_subspace(projector(sub), projector(b))
        assert not is_subspace(projector(b), projector(sub))
        assert not is_subspace(projector(rng.standard_normal((5, 2))), projector(b))

    def test_residual_length_is_distance_to_subspace(self):
        p = projector(np.eye(4)[:, :2])
        assert residual_length(p, [3.0, 4.0, 0.0, 0.0]) == pytest.approx(0.0)
        assert residual_length(p, [1.0, 0.0, 3.0, 4.0]) == pytest.approx(5.0)


# -- axis-parallel subspaces: indicator operations ----------------------------------------

class TestAxisParallel:

    def test_indicator_operations_match_projectors(self):
        for y1, y2 in itertools.product(_bits(3), repeat=2):
            p1, p2 = indicator_projector(y1), indicator_projector(y2)
            assert _same(indicator_projector(L.meet(y1, y2)), meet_projector(p1, p2))
            assert _same(indicator_projector(L.join(y1, y2)), join_projector(p1, p2))
            assert _same(indicator_projector(L.complement(y1)), orthocomplement(p1))
            assert (float(L.inclusion_loss(y1, y2)) == 0) == is_subspace(p1, p2)
            # Supplementary Theorem 10: axis-parallel projectors commute.
            assert _same(p1 @ p2, p2 @ p1)

    def test_distributive_law_holds(self):
        """Supplementary Theorem 11, exhaustively for all axis-parallel subspaces of R^3."""
        for a, b, c in itertools.product(_bits(3), repeat=3):
            pa, pb, pc = (indicator_projector(v) for v in (a, b, c))
            lhs = join_projector(pa, meet_projector(pb, pc))
            rhs = meet_projector(join_projector(pa, pb), join_projector(pa, pc))
            assert _same(lhs, rhs)
            lhs = meet_projector(pa, join_projector(pb, pc))
            rhs = join_projector(meet_projector(pa, pb), meet_projector(pa, pc))
            assert _same(lhs, rhs)


class TestNonDistributivity:

    def test_three_lines_in_the_plane(self):
        """General subspaces (Sec. 3): a ∧ (b ∨ c) = a but (a ∧ b) ∨ (a ∧ c) = {0}."""
        a = projector([[1.0], [0.0]])
        b = projector([[0.0], [1.0]])
        c = projector([[1.0], [1.0]])
        lhs = meet_projector(a, join_projector(b, c))
        rhs = join_projector(meet_projector(a, b), meet_projector(a, c))
        assert rank(lhs) == 1 and _same(lhs, a)
        assert rank(rhs) == 0
        assert not _same(a @ c, c @ a)  # the projectors do not commute

    def test_disjunction_is_span_not_union(self):
        """x ∈ C ⊔ D does not imply x ∈ C or x ∈ D, even for axis-parallel subspaces."""
        y_c = th.tensor([1.0, 0.0, 0.0])
        y_d = th.tensor([0.0, 1.0, 0.0])
        x = th.tensor([1.0, 1.0, 0.0]) / 2 ** 0.5
        assert float(L.residual_sq(L.join(y_c, y_d), x)) == 0
        assert float(L.residual_sq(y_c, x)) > 0 and float(L.residual_sq(y_d, x)) > 0
        # and x lies neither in C nor in ¬C
        assert float(L.residual_sq(L.complement(y_c), x)) > 0


# -- T3 pattern -------------------------------------------------------------------------

def _t3_holds(p1, p2, p3):
    """A = X1∧X2, B = X2∧X3, C = X1∧X3 non-trivial and pairwise meeting in {0}."""
    a, b, c = meet_projector(p1, p2), meet_projector(p2, p3), meet_projector(p1, p3)
    nontrivial = all(rank(p) > 0 for p in (a, b, c))
    disjoint = all(rank(meet_projector(p, q)) == 0 for p, q in ((a, b), (b, c), (a, c)))
    return nontrivial and disjoint, (a, b, c)


class TestT3:

    def test_axis_parallel_planes_in_r3(self):
        """X1 = span(e1, e3), X2 = span(e1, e2), X3 = span(e2, e3): A, B, C are the three
        axes; they meet pairwise in {0} and are even orthogonal."""
        e = np.eye(3)
        x1, x2, x3 = (projector(e[:, cols]) for cols in ([0, 2], [0, 1], [1, 2]))
        ok, (a, b, c) = _t3_holds(x1, x2, x3)
        assert ok
        for p, q in ((a, b), (b, c), (a, c)):
            assert th.allclose(p @ q, th.zeros(3, 3, dtype=th.float64))
        # same with the indicator operations used by the losses
        y1, y2, y3 = (th.tensor(v) for v in ([1.0, 0, 1], [1.0, 1, 0], [0.0, 1, 1]))
        ya, yb, yc = L.meet(y1, y2), L.meet(y2, y3), L.meet(y1, y3)
        for y, ops in ((ya, (y1, y2)), (yb, (y2, y3)), (yc, (y1, y3))):
            assert float(L.conjunction_loss(y, *ops)) == 0 and float(y.sum()) == 1
        for p, q in ((ya, yb), (yb, yc), (ya, yc)):
            assert float(L.disjointness_loss(p, q)) == 0

    def test_oblique_planes_in_r3(self):
        """Three generic planes through 0 meet pairwise in three distinct lines, which
        meet pairwise in {0} but are not orthogonal."""
        rng = np.random.default_rng(3)
        x1, x2, x3 = (projector(rng.standard_normal((3, 2))) for _ in range(3))
        ok, (a, b, c) = _t3_holds(x1, x2, x3)
        assert ok
        assert not th.allclose(a @ b, th.zeros(3, 3, dtype=th.float64), atol=1e-6)

    def test_impossible_with_two_axes(self):
        """Axis-parallel subspaces of R^2 (all 16 triples) never realise T3; for general
        subspaces of R^2 see docs/source/methods/e2r.rst."""
        for y1, y2, y3 in itertools.product(_bits(2), repeat=3):
            ok, _ = _t3_holds(*(indicator_projector(y) for y in (y1, y2, y3)))
            assert not ok

    def test_impossible_in_the_plane_random(self):
        """Random lines/planes of R^2 never realise T3 (consistent with the proof)."""
        rng = np.random.default_rng(4)
        choices = [projector(np.zeros((2, 1))), projector(np.eye(2))]
        lines = [projector(rng.standard_normal((2, 1))) for _ in range(3)]
        for x1, x2, x3 in itertools.product(choices + lines, repeat=3):
            assert not _t3_holds(x1, x2, x3)[0]

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_helly_number_of_subspaces_is_n(self, n):
        """n coordinate hyperplanes of R^n: every n - 1 of them meet non-trivially, all n
        meet in {0} (so the bound n of the docs is tight)."""
        e = np.eye(n)
        hyper = [projector(np.delete(e, i, axis=1)) for i in range(n)]
        for sub in itertools.combinations(hyper, n - 1):
            assert rank(meet_projector(*sub)) > 0
        assert rank(meet_projector(*hyper)) == 0

    @pytest.mark.parametrize("seed", range(5))
    def test_intersection_attained_by_n_members(self, seed):
        """The intersection of any finite family of subspaces of R^n is the intersection
        of at most n of its members (proof in the docs)."""
        rng = np.random.default_rng(seed)
        n = 4
        fam = [projector(rng.standard_normal((n, rng.integers(2, n + 1)))) for _ in range(8)]
        full = meet_projector(*fam)
        chosen, cur = [], th.eye(n, dtype=th.float64)
        for p in fam:  # greedy: keep members that strictly decrease the dimension
            nxt = meet_projector(cur, p)
            if rank(nxt) < rank(cur):
                chosen.append(p)
                cur = nxt
        assert len(chosen) <= n
        assert _same(cur, full)


# -- loss semantics on hand-built subspaces ----------------------------------------------

D = 3  # concept space R^3, vectors in R^6


def _concept(bits_real):
    return th.cat([th.tensor(bits_real, dtype=th.float64), th.zeros(D, dtype=th.float64)])


def _real_bits():
    return [_concept(b) for b in itertools.product([0.0, 1.0], repeat=D)]


def _support(y):
    return frozenset(int(i) for i in th.nonzero(y).flatten())


class TestLossSemantics:

    def test_inclusion_zero_iff_subspace(self):
        for a, b in itertools.product(_real_bits(), repeat=2):
            zero = float(L.inclusion_loss(a, b)) == 0
            assert zero == is_subspace(indicator_projector(a), indicator_projector(b))

    def test_conjunction_disjunction_zero_iff_meet_join(self):
        bits = _real_bits()
        for y, a, b in itertools.product(bits, repeat=3):
            meet = meet_projector(indicator_projector(a), indicator_projector(b))
            join = join_projector(indicator_projector(a), indicator_projector(b))
            assert (float(L.conjunction_loss(y, a, b)) == 0) == _same(
                indicator_projector(y), meet)
            assert (float(L.disjunction_loss(y, a, b)) == 0) == _same(
                indicator_projector(y), join)

    def test_negation_zero_iff_complement_in_concept_space(self):
        universe = _concept([1.0] * D)
        pu = indicator_projector(universe)
        for y, z in itertools.product(_real_bits(), repeat=2):
            # orthogonal complement of z inside the concept universe
            comp = pu @ orthocomplement(indicator_projector(z)) @ pu
            assert (float(L.negation_loss(y, z, dim=D)) == 0) == _same(
                indicator_projector(y), comp)

    def test_relation_negation_full_space(self):
        for y, z in itertools.product(_bits(4), repeat=2):
            comp = orthocomplement(indicator_projector(z))
            assert (float(L.negation_loss(y, z)) == 0) == _same(indicator_projector(y), comp)

    def test_disjointness_zero_iff_meet_is_zero_subspace(self):
        for a, b in itertools.product(_real_bits(), repeat=2):
            meet = meet_projector(indicator_projector(a), indicator_projector(b))
            assert (float(L.disjointness_loss(a, b)) == 0) == (rank(meet) == 0)

    def test_membership_zero_iff_unit_vector_in_subspace(self):
        rng = np.random.default_rng(0)
        for y in _real_bits():
            p = indicator_projector(y)
            for _ in range(5):
                v = np.zeros(2 * D)
                v[:D] = rng.standard_normal(D) * rng.integers(0, 2, D)
                if np.linalg.norm(v) == 0:
                    continue
                v /= np.linalg.norm(v)
                x = th.tensor(v)
                zero = float(L.membership_loss(y, x)) < 1e-20
                assert zero == (residual_length(p, v) < 1e-12)
            # scaled vectors violate the unit-length term
            x = 2 * th.tensor(np.r_[np.ones(D) / D ** 0.5, np.zeros(D)])
            assert float(L.membership_loss(_concept([1.0] * D), x)) > 0

    def test_role_membership_zero_iff_pair_vector_in_subspace(self):
        xp = th.tensor([1.0, 0, 0, 0, 0, 0], dtype=th.float64)
        xq = th.tensor([0, 1.0, 0, 0, 0, 0], dtype=th.float64)
        pair = L.pair_vector(xp, xq, D)
        assert pair.tolist() == [1, 0, 0, 0, 1, 0]
        for z in _bits(2 * D):
            zero = float(L.role_membership_loss(z, xp, xq, D)) == 0
            assert zero == (residual_length(indicator_projector(z), pair) < 1e-12)

    def test_negative_membership_bounded_below(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            y = th.tensor(rng.uniform(0, 1, 2 * D))
            x = th.tensor(rng.standard_normal(2 * D) * rng.uniform(0, 3))
            assert float(L.negative_membership_loss(y, x)) >= -1.25 - 1e-9

    def test_range_zero_iff_object_axes_avoid_concept(self):
        for yk in _real_bits():
            for z in _bits(2 * D):
                obj_axes = {t - D for t in _support(z) if t >= D}
                zero = float(L.range_loss(yk, z, D)) == 0
                assert zero == (not (obj_axes & _support(yk)))

    def test_structure_and_binary_losses(self):
        assert float(L.entity_structure_loss(th.tensor([1.0, 2, 0, 0]), 2)) == 0
        assert float(L.entity_structure_loss(th.tensor([1.0, 2, 0, 1]), 2)) > 0
        assert float(L.concept_structure_loss(th.tensor([1.0, 0, 0, 0]), 2)) == 0
        assert float(L.concept_structure_loss(th.tensor([1.0, 0, 1, 0]), 2)) > 0
        assert float(L.relation_structure_loss(th.tensor([1.0, 0, 0, 1]), 2)) == 0
        assert float(L.relation_structure_loss(th.tensor([1.0, 1, 0, 0]), 2)) > 0
        assert float(L.binary_loss(th.tensor([1.0, 0, 1]))) == 0
        assert float(L.binary_loss(th.tensor([1.0, 0.5, 1]))) > 0

    def test_regularisers(self):
        a, b = th.tensor([1.0, 0, 0, 0]), th.tensor([1.0, 1, 1, 0])
        assert float(L.orthogonality_loss(a, th.tensor([0.0, 1, 0, 0]))) == 0
        assert float(L.orthogonality_loss(a, b)) == 1
        assert float(L.separation_loss(a, b, 2.0)) == 0
        assert float(L.separation_loss(a, b, 1.0)) == 1


# -- the module ---------------------------------------------------------------------------

class TestModule:

    def _module(self):
        m = E2RModule(num_entities=2, num_classes=4, num_relations=1, dim=2,
                      top_index=2, bottom_index=3)
        with th.no_grad():
            m.c_embedding.weight[0] = th.tensor([1.0, 0, 0, 0])
            m.c_embedding.weight[1] = th.tensor([1.0, 1, 0, 0])
            m.r_embedding.weight[0] = th.tensor([1.0, 0, 0, 1])
            m.e_embedding.weight[0] = th.tensor([1.0, 0, 0, 0])
            m.e_embedding.weight[1] = th.tensor([0.0, 1, 0, 0])
        return m

    def test_initialisation_and_projection(self):
        m = E2RModule(5, 4, 2, dim=3)
        x = m.e_embedding.weight
        assert th.allclose(x[:, 3:], th.zeros_like(x[:, 3:]))
        assert th.allclose(x.norm(dim=1), th.ones(5))
        assert float(m.c_embedding.weight.detach()[:, 3:].abs().sum()) == 0
        with th.no_grad():
            m.c_embedding.weight[0, 0] = 2.0
            m.r_embedding.weight[0, 0] = -1.0
        m.project_()
        assert float(m.c_embedding.weight.detach().max()) <= 1
        assert float(m.r_embedding.weight.detach().min()) >= 0

    def test_top_and_bottom_are_fixed(self):
        m = self._module()
        y = m.concept(th.tensor([2, 3]))
        assert y[0].tolist() == [1, 1, 0, 0] and y[1].tolist() == [0, 0, 0, 0]

    def test_axiom_losses(self):
        m = self._module().requires_grad_(False)
        t = lambda *rows: th.tensor(rows)  # noqa: E731
        assert float(m("class_assertion", t([0, 0]))) == 0          # e1 in span(e1)
        assert float(m("class_assertion", t([1, 0]))) > 0           # e2 not in span(e1)
        assert float(m("class_assertion", t([1, 1]))) == 0
        assert float(m("role_assertion", t([0, 0, 1]))) == 0        # (e1, e2) -> axes 0, 3
        assert float(m("role_assertion", t([1, 0, 0]))) > 0
        assert float(m("concept_inclusion", t([0, 1]))) == 0
        assert float(m("concept_inclusion", t([1, 0]))) > 0
        assert float(m("concept_inclusion", t([1, 2]))) == 0        # C ⊑ owl:Thing
        assert float(m("concept_disjointness", t([0, 3]))) == 0     # C ⊓ owl:Nothing
        assert float(m("concept_conjunction", t([0, 0, 1]))) == 0
        assert float(m("concept_disjunction", t([1, 0, 1]))) == 0
        assert float(m("concept_negation", t([0, 3]))) > 0
        assert float(m("range", t([0, 0]))) == 0                    # object axis 1 ∉ {0}
        assert float(m("range", t([0, 1]))) > 0
        assert float(m("class_assertion", t([1, 0]), neg=True)) < 0
        with pytest.raises(ValueError):
            m("unknown", t([0, 0]))
        with pytest.raises(NotImplementedError):
            m.forward_fs(None, None, None)
