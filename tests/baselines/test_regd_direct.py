"""Tests of the RegD dissimilarities and modules (Yang & Chen, KR 2026, arXiv 2501.17518v3)."""
import math
from unittest import TestCase

import numpy as np
import pytest
import torch as th

# mowl.datasets first: importing mowl.nn first hits a circular import
import mowl.datasets  # noqa: F401
import mowl.baselines.nn.el.regd.dissimilarity as D
from mowl.baselines.nn.el.regd.module import RegDELBEModule, RegDELEmModule


def t(*values):
    return th.tensor(values, dtype=th.float64)


class TestHandComputedDissimilarities(TestCase):
    """Values computed by hand from Definitions 1 and 2 and Examples 1-3."""

    def test_ball_contained(self):
        # ball1 = ball((0,0), 2), ball2 = ball((1,0), 0.5): distance 1 + 0.5 = 1.5 <= 2.
        c1, r1, c2, r2 = t(0, 0), t(2), t(1, 0), t(0.5)
        # minimal distance of ball2 to the complement of ball1 is 2 - 1.5 = 0.5
        self.assertAlmostEqual(D.ball_boundary_dissimilarity(c1, r1, c2, r2).item(), -0.5)
        # (1 + 0.5^2... (|c1-c2|^2 + (r1-r2)^2) / (r1 r2) = (1 + 2.25) / 1
        self.assertAlmostEqual(D.ball_depth_dissimilarity(c1, r1, c2, r2, p=2).item(), 3.25)
        # p = 1: (1 + 1.5) / 1
        self.assertAlmostEqual(D.ball_depth_dissimilarity(c1, r1, c2, r2, p=1).item(), 2.5)
        self.assertAlmostEqual(D.ball_energy(c1, r1, c2, r2, lamda=0.5).item(), -0.5 + 0.5 * 3.25)

    def test_ball_not_contained(self):
        # ball2 = ball((3,0), 1): its farthest point (4,0) is 2 away from ball1.
        c1, r1, c2, r2 = t(0, 0), t(2), t(3, 0), t(1)
        self.assertAlmostEqual(D.ball_boundary_dissimilarity(c1, r1, c2, r2).item(), 2.0)
        self.assertAlmostEqual(D.ball_depth_dissimilarity(c1, r1, c2, r2).item(), (9 + 1) / 2)

    def test_ball_internally_tangent_is_zero(self):
        # ball((1,0),1) touches ball((0,0),2) from inside at (2,0).
        value = D.ball_boundary_dissimilarity(t(0, 0), t(2), t(1, 0), t(1))
        self.assertAlmostEqual(value.item(), 0.0)

    def test_box_contained(self):
        # box1: centre (0,0), offset (2,1); box2: centre (1,0.5), offset (0.5,0.25)
        c1, o1, c2, o2 = t(0, 0), t(2, 1), t(1, 0.5), t(0.5, 0.25)
        # a = |c1-c2| + o2 - o1 = (-0.5, -0.25); the child is 0.25 from the nearest side
        self.assertAlmostEqual(D.box_boundary_dissimilarity(c1, o1, c2, o2).item(), -0.25)
        # p = 1: (1.5 + 2.25) / (sqrt(5) * sqrt(5/16)) = 3.75 / 1.25
        self.assertAlmostEqual(D.box_depth_dissimilarity(c1, o1, c2, o2, p=1).item(), 3.0)
        # p = 2: ((1 + 0.25) + (1.5^2 + 0.75^2)) / 1.25
        self.assertAlmostEqual(D.box_depth_dissimilarity(c1, o1, c2, o2, p=2).item(),
                               ((1 + 0.25) + (2.25 + 0.5625)) / 1.25)

    def test_box_not_contained(self):
        # child offset (1, 0.25), centre (3, 0.5): a = (3+1-2, 0.5+0.25-1) = (2, -0.25)
        c1, o1, c2, o2 = t(0, 0), t(2, 1), t(3, 0.5), t(1, 0.25)
        self.assertAlmostEqual(D.box_boundary_dissimilarity(c1, o1, c2, o2).item(), 2.0)
        # two dimensions outside: a = (2, 3) -> sqrt(13)
        c2 = t(3, 3.0)
        o2 = t(1, 1.0)
        self.assertAlmostEqual(D.box_boundary_dissimilarity(c1, o1, c2, o2).item(),
                               math.sqrt((3 + 1 - 2) ** 2 + (3 + 1 - 1) ** 2))

    def test_box_energy_and_depth_on_containment_flag(self):
        c1, o1, c2, o2 = t(0, 0), t(2, 1), t(1, 0.5), t(0.5, 0.25)
        paper = D.box_energy(c1, o1, c2, o2, lamda=2.0)
        self.assertAlmostEqual(paper.item(), -0.25 + 2.0 * 3.0)
        # authors' code: no depth term when the child is contained
        code = D.box_energy(c1, o1, c2, o2, lamda=2.0, depth_on_containment=False)
        self.assertAlmostEqual(code.item(), -0.25)
        # ... but it is added when the child is not contained
        c2, o2 = t(3, 0.5), t(1, 0.25)
        outside = D.box_energy(c1, o1, c2, o2, lamda=2.0, depth_on_containment=False)
        expected = 2.0 + 2.0 * D.box_depth_dissimilarity(c1, o1, c2, o2).item()
        self.assertAlmostEqual(outside.item(), expected)

    def test_batched_shapes(self):
        c, o = th.rand(4, 3, 2), th.rand(4, 3, 2) + 0.1
        self.assertEqual(D.box_boundary_dissimilarity(c, o, c, o).shape, (4, 3))
        self.assertEqual(D.box_depth_dissimilarity(c, o, c, o).shape, (4, 3))
        r = th.rand(4, 3) + 0.1
        self.assertEqual(D.ball_boundary_dissimilarity(c, r, c, r).shape, (4, 3))
        self.assertEqual(D.ball_depth_dissimilarity(c, r, c, r).shape, (4, 3))

    def test_depth_dissimilarity_is_symmetric_and_zero_only_for_equal_regions(self):
        c1, r1, c2, r2 = t(0, 0), t(2), t(1, 0), t(0.5)
        self.assertAlmostEqual(D.ball_depth_dissimilarity(c1, r1, c2, r2).item(),
                               D.ball_depth_dissimilarity(c2, r2, c1, r1).item())
        self.assertEqual(D.ball_depth_dissimilarity(c1, r1, c1, r1).item(), 0.0)
        self.assertGreater(D.ball_depth_dissimilarity(c1, r1, c2, r1).item(), 0.0)

    def test_boundary_dissimilarity_is_asymmetric(self):
        c1, r1, c2, r2 = t(0, 0), t(2), t(1, 0), t(0.5)
        self.assertLess(D.ball_boundary_dissimilarity(c1, r1, c2, r2).item(), 0)
        self.assertGreater(D.ball_boundary_dissimilarity(c2, r2, c1, r1).item(), 0)

    def test_contrastive_loss_hand_value(self):
        positive = t(0.5, -1.0)
        # hinge values of the negatives: max(0.1 - 1.0, 0) = 0, max(0.1 + 0.5, 0) = 0.6
        negatives = D.negative_hinge(t(1.0, -0.5).reshape(1, 2).expand(2, 2), gamma2=0.1)
        loss = D.contrastive_loss(positive, negatives, gamma1=0.0)
        neg_term = math.log(math.exp(0.0) + math.exp(0.6))
        self.assertAlmostEqual(loss.item(), (0.5 + neg_term + 0.0 + neg_term) / 2)


class TestBoundaryDissimilarityProperties(TestCase):
    """Remark after Definition 2 and Proposition 3 (the proofs hold for any region)."""

    @classmethod
    def setUpClass(cls):
        g = th.Generator().manual_seed(0)
        cls.n = 400
        cls.bc1 = th.randn(cls.n, 3, generator=g, dtype=th.float64)
        cls.bo1 = th.rand(cls.n, 3, generator=g, dtype=th.float64) * 2 + 0.2
        # half of the children are built inside the parent, half are arbitrary
        inside = th.rand(cls.n, 3, generator=g, dtype=th.float64)
        co = cls.bo1 * inside * 0.9 + 1e-3
        shift = (th.rand(cls.n, 3, generator=g, dtype=th.float64) * 2 - 1) * (cls.bo1 - co)
        random_c = th.randn(cls.n, 3, generator=g, dtype=th.float64)
        random_o = th.rand(cls.n, 3, generator=g, dtype=th.float64) + 0.1
        use_inside = (th.arange(cls.n) % 2 == 0).unsqueeze(-1)
        cls.bc2 = th.where(use_inside, cls.bc1 + shift, random_c)
        cls.bo2 = th.where(use_inside, co, random_o)

    def test_box_nonpositive_iff_contained(self):
        value = D.box_boundary_dissimilarity(self.bc1, self.bo1, self.bc2, self.bo2)
        # containment of axis-aligned boxes, checked on the corners, independently of the loss
        contained = ((self.bc2 - self.bo2 >= self.bc1 - self.bo1)
                     & (self.bc2 + self.bo2 <= self.bc1 + self.bo1)).all(dim=-1)  # noqa: W503
        self.assertTrue(contained.any() and (~contained).any())
        self.assertTrue(th.equal(value <= 0, contained))
        self.assertTrue((value[contained] < 0).all())  # strictly inside: strictly negative
        self.assertTrue((value[~contained] > 0).all())

    def test_ball_nonpositive_iff_contained(self):
        c1, c2 = self.bc1[:, :2], self.bc2[:, :2]
        r1, r2 = self.bo1[:, 0], self.bo2[:, 0] * 0.5
        value = D.ball_boundary_dissimilarity(c1, r1, c2, r2)
        # the child ball is inside the parent iff all its boundary points are
        angles = th.linspace(0, 2 * math.pi, 721, dtype=th.float64)[:-1]
        circle = th.stack([th.cos(angles), th.sin(angles)], dim=-1)
        points = c2[:, None, :] + r2[:, None, None] * circle[None]
        far = th.linalg.norm(points - c1[:, None, :], dim=-1).max(dim=1).values
        contained = far <= r1 + 1e-12
        self.assertTrue(contained.any() and (~contained).any())
        self.assertTrue(th.equal(value <= 0, contained))

    def test_ball_value_is_signed_distance_to_complement_or_farthest_point(self):
        """Definition 2 checked against a brute-force evaluation on the child's boundary."""
        c1, c2 = self.bc1[:, :2], self.bc2[:, :2]
        r1, r2 = self.bo1[:, 0], self.bo2[:, 0] * 0.5
        value = D.ball_boundary_dissimilarity(c1, r1, c2, r2)
        angles = th.linspace(0, 2 * math.pi, 7201, dtype=th.float64)[:-1]
        circle = th.stack([th.cos(angles), th.sin(angles)], dim=-1)
        points = c2[:, None, :] + r2[:, None, None] * circle[None]
        dist_to_centre = th.linalg.norm(points - c1[:, None, :], dim=-1)
        # outside: the largest distance to the parent ball is max(|x - c1| - r1)
        farthest = (dist_to_centre - r1[:, None]).max(dim=1).values
        # inside: minus the smallest distance to the complement, min(r1 - |x - c1|)
        nearest_complement = (r1[:, None] - dist_to_centre).min(dim=1).values
        expected = th.where(farthest <= 0, -nearest_complement, farthest)
        self.assertTrue(th.allclose(value, expected, atol=1e-4))

    def test_zero_iff_internally_tangent(self):
        # box2 touches the left side of box1 from inside, in dimension 0
        c1, o1 = t(0, 0), t(2, 1)
        c2, o2 = t(-1, 0.2), t(1, 0.3)
        self.assertAlmostEqual(D.box_boundary_dissimilarity(c1, o1, c2, o2).item(), 0.0)
        # moving it out by a little makes the value positive, in by a little negative
        self.assertGreater(D.box_boundary_dissimilarity(c1, o1, c2 - t(0.01, 0), o2).item(), 0)
        self.assertLess(D.box_boundary_dissimilarity(c1, o1, c2 + t(0.01, 0), o2).item(), 0)

    def test_proposition_3_monotone_along_inclusion_chains(self):
        # reg3 inside reg2 inside reg1: d_bd(reg1, reg3) <= d_bd(reg1, reg2)
        g = th.Generator().manual_seed(1)
        c1 = th.randn(200, 3, generator=g, dtype=th.float64)
        o1 = th.rand(200, 3, generator=g, dtype=th.float64) + 2.0
        o2 = o1 * (th.rand(200, 3, generator=g, dtype=th.float64) * 0.8 + 0.05)
        c2 = c1 + (th.rand(200, 3, generator=g, dtype=th.float64) * 2 - 1) * (o1 - o2)
        o3 = o2 * (th.rand(200, 3, generator=g, dtype=th.float64) * 0.9 + 0.05)
        c3 = c2 + (th.rand(200, 3, generator=g, dtype=th.float64) * 2 - 1) * (o2 - o3)
        d12 = D.box_boundary_dissimilarity(c1, o1, c2, o2)
        d13 = D.box_boundary_dissimilarity(c1, o1, c3, o3)
        self.assertTrue((d12 <= 0).all() and (d13 <= d12 + 1e-12).all())
        # the same for balls
        r1, r2, r3 = t(5.0), t(3.0), t(1.0)
        a, b, c = t(0, 0), t(1, 0), t(1.5, 0.5)
        self.assertLessEqual(D.ball_boundary_dissimilarity(a, r1, c, r3).item(),
                             D.ball_boundary_dissimilarity(a, r1, b, r2).item())


def hyperbolic_distance(x, y):
    """Distance of the upper half-space model of curvature -1, points ``(centre, height)``."""
    num = np.sum((x - y) ** 2)
    return math.acosh(1 + num / (2 * x[-1] * y[-1]))


class TestDepthDissimilarityTheorems(TestCase):
    """Theorem 1, Propositions 1 and 2 of the paper, checked numerically."""

    def test_theorem_1_item_1_smaller_children_are_further_away(self):
        ball1 = (t(0, 0), t(1.0))
        ball2 = (t(2, 0), t(1.0))
        base = D.ball_depth_dissimilarity(*ball1, *ball2).item()
        delta = 50.0
        # a ball inside ball2 with a small enough radius is more than `delta` further away
        for radius in [1e-1, 1e-2, 1e-3, 1e-4]:
            child = D.ball_depth_dissimilarity(*ball1, t(2, 0), t(radius)).item()
            if radius <= 1e-3:
                self.assertGreater(child, base + delta)
        # and the dissimilarity is monotone in the shrinking
        values = [D.ball_depth_dissimilarity(*ball1, t(2, 0), t(r)).item()
                  for r in [0.5, 0.1, 0.01, 0.001]]
        self.assertEqual(values, sorted(values))

    def test_theorem_1_item_1_boxes(self):
        box1 = (t(0, 0), t(1.0, 1.0))
        base = D.box_depth_dissimilarity(*box1, t(2, 0), t(1.0, 1.0)).item()
        small = D.box_depth_dissimilarity(*box1, t(2, 0), t(1e-4, 1e-4)).item()
        self.assertGreater(small, base + 1000)

    def test_theorem_1_item_2_many_pairwise_far_children(self):
        # n children inside a ball whose pairwise depth dissimilarity exceeds M
        n, M = 12, 100.0
        angles = th.arange(n, dtype=th.float64) * 2 * math.pi / n
        centers = 0.5 * th.stack([th.cos(angles), th.sin(angles)], dim=-1)
        radius = th.full((n,), 1e-3, dtype=th.float64)
        # all children lie inside ball((0,0), 1) ...
        inside = D.ball_boundary_dissimilarity(t(0, 0), t(1.0), centers, radius)
        self.assertTrue((inside < 0).all())
        # ... and are pairwise far apart in depth dissimilarity
        pair = D.ball_depth_dissimilarity(centers[:, None], radius[:, None],
                                          centers[None], radius[None])
        off_diagonal = pair[~th.eye(n, dtype=th.bool)]
        self.assertTrue((off_diagonal > M).all())
        # in the same box, the same holds for boxes
        offsets = th.full((n, 2), 1e-3, dtype=th.float64)
        pair = D.box_depth_dissimilarity(centers[:, None], offsets[:, None],
                                         centers[None], offsets[None])
        self.assertTrue((pair[~th.eye(n, dtype=th.bool)] > M).all())

    def test_proposition_1_isometry_with_hyperbolic_space(self):
        """With p = 2, g(x) = arcosh(x + 1) and f(r) = sqrt(2) r, the depth dissimilarity of
        balls is the hyperbolic distance of the points (c, r) of the upper half-space."""
        g = th.Generator().manual_seed(2)
        for _ in range(50):
            c1 = th.randn(3, generator=g, dtype=th.float64)
            c2 = th.randn(3, generator=g, dtype=th.float64)
            r1 = th.rand(1, generator=g, dtype=th.float64)[0] + 0.1
            r2 = th.rand(1, generator=g, dtype=th.float64)[0] + 0.1
            # depth dissimilarity with f = sqrt(2) r: numerator / (2 r1 r2)
            ddep = D.ball_depth_dissimilarity(c1, r1, c2, r2, p=2) / 2.0
            x = np.concatenate([c1.numpy(), [r1.item()]])
            y = np.concatenate([c2.numpy(), [r2.item()]])
            self.assertAlmostEqual(math.acosh(1 + ddep.item()), hyperbolic_distance(x, y),
                                   places=9)

    def test_proposition_2_order_of_hyperbolic_distances_is_preserved(self):
        g = th.Generator().manual_seed(3)
        n = 80
        centers = th.randn(n, 2, generator=g, dtype=th.float64)
        radii = th.rand(n, generator=g, dtype=th.float64) + 0.05
        pairs = th.randint(0, n, (300, 2), generator=g)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        hyp = np.array([hyperbolic_distance(
            np.concatenate([centers[i].numpy(), [radii[i].item()]]),
            np.concatenate([centers[j].numpy(), [radii[j].item()]])) for i, j in pairs])
        ddep = D.ball_depth_dissimilarity(centers[pairs[:, 0]], radii[pairs[:, 0]],
                                          centers[pairs[:, 1]], radii[pairs[:, 1]]).numpy()
        self.assertTrue(np.array_equal(np.argsort(hyp, kind="stable"),
                                       np.argsort(ddep, kind="stable")))


def _convex_hull_vertices_contain(points, query):
    """Whether ``query`` lies in the convex hull of ``points`` (2D), by linear programming."""
    from scipy.optimize import linprog
    points = np.asarray(points)
    k = len(points)
    # find weights w >= 0, sum w = 1, points^T w = query
    a_eq = np.vstack([points.T, np.ones((1, k))])
    b_eq = np.concatenate([query, [1.0]])
    result = linprog(np.zeros(k), A_eq=a_eq, b_eq=b_eq, bounds=[(0, None)] * k, method="highs")
    return result.status == 0


class TestFaithfulnessConstruction(TestCase):
    """Theorem 4.2: every partial order has a faithful embedding in R^2.

    "Faithful" (Definition 4.1) means ``v < v'  iff  f(v) is a subset of f(v')`` for all
    distinct ``v, v'``: the embedding captures the *partial order* exactly. It is a statement
    about a hierarchy only; it says nothing about conjunctions, existential restrictions or
    disjointness. We check the construction of the proof (vertices of a regular m-gon, the
    region of ``v`` is the convex hull of ``v`` and its descendants) on a small DAG. The
    embedding is by convex polygons, not by boxes or balls, which have to be learned."""

    def test_convex_hull_embedding_of_a_dag_is_faithful(self):
        # a small DAG (not a tree): w is below x and y; x, y below z; u below y only
        below = {"w": {"x", "y", "z"}, "x": {"z"}, "y": {"z"}, "u": {"y", "z"}, "z": set()}
        names = list(below)
        m = len(names)
        angles = [2 * math.pi * i / m for i in range(m)]
        point = {v: np.array([math.cos(a), math.sin(a)]) for v, a in zip(names, angles)}
        # v is a descendant of v' when v' is in below[v]; S_{v'} = {v'} and its descendants
        region = {vp: [point[vp]] + [point[v] for v in names if vp in below[v]] for vp in names}
        for v in names:
            for vp in names:
                if v == vp:
                    continue
                included = all(_convex_hull_vertices_contain(region[vp], p) for p in region[v])
                self.assertEqual(included, vp in below[v], (v, vp))


class _ModuleFixture:
    """Hand-built embeddings: class 0 is a big region, 1 is inside 0, 2 is outside 0."""

    def setup_box(self, **kwargs):
        module = RegDELBEModule(4, 1, None, embed_dim=2, margin=0, **kwargs)
        centers = th.tensor([[0.0, 0.0], [0.5, 0.0], [5.0, 0.0], [0.5, 0.5]])
        offsets = th.tensor([[2.0, 1.0], [0.5, 0.25], [1.0, 1.0], [2.0, 1.0]])
        with th.no_grad():
            module.class_embed.weight.copy_(centers)
            module.class_offset.weight.copy_(offsets)
            module.rel_embed.weight.copy_(th.tensor([[1.0, 0.0]]))
        return module

    def setup_ball(self, **kwargs):
        module = RegDELEmModule(4, 1, None, embed_dim=2, margin=0, **kwargs)
        centers = th.tensor([[0.0, 0.0], [0.5, 0.0], [5.0, 0.0], [0.0, 0.0]])
        radii = th.tensor([[2.0], [0.5], [1.0], [1.0]])
        with th.no_grad():
            module.class_embed.weight.copy_(centers)
            module.class_rad.weight.copy_(radii)
            module.rel_embed.weight.copy_(th.tensor([[1.0, 0.0]]))
        return module


class TestRegDModules(_ModuleFixture, TestCase):

    def test_gci0_box_values(self):
        module = self.setup_box(lamda=0.0)
        # classes 1 inside 0: d_bd = max(0.5+0.5-2, 0+0.25-1) = -0.75, loss max(E, 0) = 0
        loss = module.gci0_loss(th.tensor([[1, 0]]))
        self.assertEqual(loss.shape, (1, 1))
        self.assertAlmostEqual(loss.item(), 0.0)
        with module.scoring():
            self.assertAlmostEqual(module.gci0_loss(th.tensor([[1, 0]])).item(), -0.75, places=5)
            # 2 is not inside 0: a = (5+1-2, 0+1-1) = (4, 0) -> 4
            self.assertAlmostEqual(module.gci0_loss(th.tensor([[2, 0]])).item(), 4.0, places=5)
        self.assertAlmostEqual(module.gci0_loss(th.tensor([[2, 0]])).item(), 4.0, places=5)

    def test_gci0_box_with_depth_term(self):
        module = self.setup_box(lamda=1.0, p=1)
        with module.scoring():
            value = module.gci0_loss(th.tensor([[1, 0]])).item()
        # d_dep = (0.5+0 + 1.5+0.75) / (sqrt(5) * sqrt(0.3125)) = 2.75 / 1.25 = 2.2
        self.assertAlmostEqual(value, -0.75 + 2.2, places=5)

    def test_negative_loss_is_zero_iff_pushed_out_by_the_margin(self):
        module = self.setup_box(lamda=1.0, gamma2=0.5)
        # class 2 is 4 away from class 0: no loss; class 1 is inside: loss gamma2 - d_bd
        far = module.gci0_loss(th.tensor([[2, 0]]), neg=True)
        inside = module.gci0_loss(th.tensor([[1, 0]]), neg=True)
        self.assertAlmostEqual(far.item(), 0.0)
        self.assertAlmostEqual(inside.item(), 0.5 + 0.75, places=5)

    def test_gci2_translates_the_child(self):
        module = self.setup_box(lamda=0.0)
        # C=class 1 shifted by +1 along x: centre (1.5, 0), offset (0.5, 0.25) inside class 0:
        # a = (1.5+0.5-2, 0.25-1) = (0, -0.75) -> tangent, value 0
        with module.scoring():
            self.assertAlmostEqual(module.gci2_loss(th.tensor([[1, 0, 0]])).item(), 0.0, places=5)
            # shifted twice as far, it leaves the parent: use class 3 as the child instead
            value = module.gci2_loss(th.tensor([[3, 0, 0]])).item()
        # class 3 centre (0.5,0.5)+(1,0) = (1.5, 0.5), offset (2,1); a = (1.5+2-2, 0.5+1-1)
        # = (1.5, 0.5) -> sqrt(1.5^2 + 0.5^2)
        self.assertAlmostEqual(value, math.sqrt(1.5 ** 2 + 0.5 ** 2), places=5)

    def test_gci3_translates_the_child_backwards(self):
        module = self.setup_box(lamda=0.0)
        # exists R.C subclass D with C = class 1: centre (0.5,0) - (1,0) = (-0.5, 0)
        # a = (0.5+0.5-2, 0+0.25-1) = (-1, -0.75): contained, value -0.75
        with module.scoring():
            self.assertAlmostEqual(module.gci3_loss(th.tensor([[0, 1, 0]])).item(), -0.75,
                                   places=5)

    def test_gci1_uses_intersection_box_and_nonempty_term(self):
        module = self.setup_box(lamda=0.0)
        # classes 0 and 3 intersect in x in [-1.5... compute: starts max((-2,-1),(-1.5,-0.5)) =
        # (-1.5,-0.5); ends min((2,1),(2.5,1.5)) = (2, 1): non empty, centre (0.25, 0.25),
        # offset (1.75, 0.75). Parent: class 0 (centre (0,0), offset (2,1)):
        # a = (0.25+1.75-2, 0.25+0.75-1) = (0, 0) -> tangent, value 0
        with module.scoring():
            self.assertAlmostEqual(module.gci1_loss(th.tensor([[0, 3, 0]])).item(), 0.0, places=5)
        # disjoint boxes: classes 0 and 2 do not intersect; the non-empty term is positive
        with module.scoring():
            value = module.gci1_loss(th.tensor([[0, 2, 0]])).item()
        # start = (4,-1); end = (2,1); relu(start - end) = (2,0)
        self.assertGreater(value, 2.0 - 1e-5)

    def test_bottom_forms_keep_the_host_losses(self):
        module = self.setup_box()
        host = RegDELBEModule.__mro__[2]  # ELBEModule
        data = th.tensor([[0, 1]])
        self.assertTrue(th.equal(module.gci0_bot_loss(data), host.gci0_bot_loss(module, data)))
        data3 = th.tensor([[0, 1, 2]])
        self.assertTrue(th.equal(module.gci1_bot_loss(data3), host.gci1_bot_loss(module, data3)))

    def test_ball_modules(self):
        module = self.setup_ball(lamda=0.0)
        # class 1 = ball((0.5,0), 0.5) inside class 0 = ball((0,0), 2): d_bd = 0.5+0.5-2 = -1
        with module.scoring():
            self.assertAlmostEqual(module.gci0_loss(th.tensor([[1, 0]])).item(), -1.0, places=5)
            # class 2 = ball((5,0),1): 5 + 1 - 2 = 4
            self.assertAlmostEqual(module.gci0_loss(th.tensor([[2, 0]])).item(), 4.0, places=5)
            # translation by (1,0): ball((1.5,0),0.5): 1.5+0.5-2 = 0
            self.assertAlmostEqual(module.gci2_loss(th.tensor([[1, 0, 0]])).item(), 0.0, places=5)
            # exists R.C: ball((-0.5,0),0.5): 0.5+0.5-2 = -1
            self.assertAlmostEqual(module.gci3_loss(th.tensor([[0, 1, 0]])).item(), -1.0, places=5)
        self.assertAlmostEqual(module.gci0_loss(th.tensor([[1, 0]])).item(), 0.0)

    def test_ball_depth_term(self):
        module = self.setup_ball(lamda=1.0, p=2)
        with module.scoring():
            value = module.gci0_loss(th.tensor([[1, 0]])).item()
        # d_dep = (0.25 + 2.25) / (2 * 0.5) = 2.5
        self.assertAlmostEqual(value, -1.0 + 2.5, places=5)

    def test_ball_intersection_form_keeps_elem_loss(self):
        module = self.setup_ball()
        host = RegDELEmModule.__mro__[2]  # ELEmModule
        data = th.tensor([[0, 1, 2]])
        self.assertTrue(th.equal(module.gci1_loss(data), host.gci1_loss(module, data)))

    def test_declared_capabilities(self):
        for cls in (RegDELBEModule, RegDELEmModule):
            self.assertEqual(cls.neg_capable_gcis,
                             frozenset({"gci0", "gci2", "object_property_assertion"}))
            self.assertEqual(cls.contrastive_gcis, frozenset({"gci0", "gci2"}))
            self.assertFalse(cls.role_axiom_capable)

    def test_gradients_flow_and_are_finite_at_degenerate_pairs(self):
        for module in (self.setup_box(), self.setup_ball()):
            # a class compared with itself: zero distance, the norm must not give NaN
            data = th.tensor([[0, 0], [1, 0], [2, 0]])
            loss = module.gci0_loss(data).sum() + module.gci0_loss(data, neg=True).sum()
            loss.backward()
            for parameter in module.parameters():
                if parameter.grad is not None:
                    self.assertTrue(th.isfinite(parameter.grad).all())

    def test_forward_records_the_normal_form(self):
        module = self.setup_box()
        module(th.tensor([[1, 0]]), "gci0")
        self.assertEqual(module.last_gci_name, "gci0")

    def test_invalid_hyperparameters(self):
        with self.assertRaises(ValueError):
            RegDELBEModule(2, 1, None, embed_dim=2, lamda=-1)
        with self.assertRaises(ValueError):
            RegDELEmModule(2, 1, None, embed_dim=2, eps=0)


def _tree(depth):
    n = 2 ** (depth + 1) - 1
    ancestors = {i: set() for i in range(n)}
    for i in range(1, n):
        j = i
        while j > 0:
            j = (j - 1) // 2
            ancestors[i].add(j)
    return n, ancestors


class TestTreeEmbedding(TestCase):
    """A binary tree of 15 classes embedded in the plane: after training, a class is inside
    another class exactly if it is a descendant of it (a faithful embedding of the hierarchy
    in the sense of Definition 4.1, found by the RegD contrastive loss)."""

    def _train_and_count_violations(self, module, n, ancestors, steps=800, lr=0.01):
        positives = th.tensor([(a, b) for a in range(n) for b in sorted(ancestors[a])])
        # every other class is a negative parent of a; masked entries get a hinge of -inf
        mask = th.zeros(len(positives), n, dtype=th.bool)
        for row, (a, _) in enumerate(positives.tolist()):
            for c in range(n):
                mask[row, c] = c != a and c not in ancestors[a]
        negatives = th.stack([positives[:, :1].expand(-1, n),
                              th.arange(n).expand(len(positives), -1)], dim=-1).reshape(-1, 2)
        optimiser = th.optim.Adam(module.parameters(), lr=lr)
        for _ in range(steps):
            optimiser.zero_grad()
            positive_loss = module(positives, "gci0").reshape(-1)
            negative_loss = module(negatives, "gci0", neg=True).reshape(len(positives), n)
            negative_loss = th.where(mask, negative_loss, th.full_like(negative_loss, -1e9))
            D.contrastive_loss(positive_loss, negative_loss).backward()
            optimiser.step()

        pairs = th.tensor([(a, b) for a in range(n) for b in range(n) if a != b])
        with th.no_grad():
            parent = module._region(pairs[:, 1])
            child = module._region(pairs[:, 0])
            boundary = module.boundary(*parent, *child).reshape(-1)
        truth = th.tensor([b in ancestors[a] for a, b in pairs.tolist()])
        return int(((boundary <= 0) != truth).sum())

    def test_box_tree_in_two_dimensions(self):
        th.manual_seed(0)
        n, ancestors = _tree(3)
        module = RegDELBEModule(n, 1, None, embed_dim=2, margin=0, gamma2=0.1)
        self.assertEqual(self._train_and_count_violations(module, n, ancestors), 0)

    def test_ball_tree_in_two_dimensions(self):
        th.manual_seed(0)
        n, ancestors = _tree(3)
        module = RegDELEmModule(n, 1, None, embed_dim=2, margin=0, gamma2=0.1)
        self.assertEqual(self._train_and_count_violations(module, n, ancestors), 0)

    def test_box_and_ball_tree_in_three_dimensions_fast(self):
        """The fast part of the depth-four test below, which runs in CI: the same tree of
        15 classes in three dimensions."""
        th.manual_seed(0)
        n, ancestors = _tree(3)
        for cls in (RegDELBEModule, RegDELEmModule):
            module = cls(n, 1, None, embed_dim=3, margin=0, gamma2=0.1)
            self.assertEqual(self._train_and_count_violations(module, n, ancestors), 0)

    @pytest.mark.slow
    def test_box_and_ball_tree_in_five_dimensions_with_depth_four(self):
        th.manual_seed(0)
        n, ancestors = _tree(4)
        for cls in (RegDELBEModule, RegDELEmModule):
            module = cls(n, 1, None, embed_dim=5, margin=0, gamma2=0.1)
            self.assertEqual(
                self._train_and_count_violations(module, n, ancestors, steps=2500), 0)


class TestSoftplusRadiusConsistency:
    """With radius="softplus" every form reads the same radius of a class."""

    @staticmethod
    def _module(**kw):
        th.manual_seed(0)
        m = RegDELEmModule(6, 2, 3, embed_dim=4, radius="softplus", margin=0.1, **kw)
        with th.no_grad():
            m.class_rad.weight.copy_(th.tensor([[-2.0], [-1.0], [0.5], [1.5], [-0.3], [2.0]]))
        return m

    def test_bottom_forms_and_assertion_use_softplus(self):
        m = self._module()
        sp = th.nn.functional.softplus(m.class_rad.weight)
        d2 = th.tensor([[0, 1], [4, 2]])
        d3 = th.tensor([[0, 1, 2], [1, 4, 5]])
        assert th.allclose(m.gci0_bot_loss(d2), sp[d2[:, 0]])
        assert th.allclose(m.gci3_bot_loss(d3), sp[d3[:, 1]])
        ca = th.tensor([[0, 1], [2, 4]])
        got = m.class_assertion_loss(ca)
        c = m.ind_embed(ca[:, 0])
        d = m.class_embed(ca[:, 1])
        ri = th.abs(m.ind_rad(ca[:, 0]))
        expect = th.relu(th.linalg.norm(c - d, dim=1, keepdim=True) + ri - sp[ca[:, 1]] - 0.1)
        assert th.allclose(got, expect)

    def test_gci1_negative_uses_softplus_and_old_option_does_not(self):
        data = th.tensor([[0, 1, 2], [4, 1, 5]])
        new = self._module()
        old = self._module(consistent_radius=False)
        # the same loss as the positive form (ELEm ignores neg for gci1) in both, which the
        # earlier fall-through option strict_negatives=False selects
        new.strict_negatives = old.strict_negatives = False
        assert th.allclose(new.gci1_loss(data, neg=True), new.gci1_loss(data))
        assert th.allclose(old.gci1_loss(data), new.gci1_loss(data))
        assert not th.allclose(old.gci1_loss(data, neg=True), new.gci1_loss(data, neg=True))

    def test_old_option_keeps_absolute_radius_in_bottom_forms(self):
        old = self._module(consistent_radius=False)
        d = th.tensor([[0, 1], [4, 2]])
        assert th.allclose(old.gci0_bot_loss(d), old.class_rad(d[:, 0]))  # raw, as before
        assert old.consistent_radius is False

    def test_abs_radius_unchanged(self):
        th.manual_seed(0)
        m = RegDELEmModule(6, 2, 3, embed_dim=4)
        assert m.radius == "abs" and m.consistent_radius is True
        d = th.tensor([[0, 1], [4, 2]])
        assert th.allclose(m.gci0_bot_loss(d), m.class_rad(d[:, 0]))
