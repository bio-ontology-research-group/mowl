"""Query2Box, BetaE, ConE and AConE from parts: equal to the existing geometries, and the
common interface."""
import math
from unittest import TestCase

import torch as th

from mowl.nn.query.geometry import (Query2BoxGeometry, BetaEGeometry, ConEGeometry,
                                    AConEGeometry, UnsupportedOperator)
from mowl.representation import (geometry_of, Geometry, Box, ConstructorUnavailable, METHODS,
                                 CLASS_MODELS, RELATION_MODELS, MEASURES)
from mowl.representation.library.query_methods import (
    ComposedQueryGeometry, query2box, betae, cone_qe, acone)
from mowl.representation.library.query_regions import BetaRegion, SectorCone, ConeBounds

E, R, D = 12, 7, 6          # entities, relation rows (isa, isa inverse, two roles), dimension
NUM_CLASSES = 8

SIGNATURES = {
    "1p": ("p", ("a",)),
    "2p": ("p", ("p", ("a",))),
    "3p": ("p", ("p", ("p", ("a",)))),
    "2i": ("i", ("p", ("a",)), ("p", ("a",))),
    "3i": ("i", ("p", ("a",)), ("p", ("a",)), ("p", ("a",))),
    "ip": ("p", ("i", ("p", ("a",)), ("p", ("a",)))),
    "pi": ("i", ("p", ("p", ("a",))), ("p", ("a",))),
}
NEGATIVE = {
    "2in": ("i", ("p", ("a",)), ("n", ("p", ("a",)))),
    "inp": ("p", ("i", ("p", ("a",)), ("n", ("p", ("a",))))),
    "pin": ("i", ("p", ("p", ("a",))), ("n", ("p", ("a",)))),
}


def _ids(signature, batch, generator):
    """An id matrix for ``signature`` (depth-first, children first, as ``embed`` reads)."""
    columns = []

    def walk(sig):
        for child in sig[1:]:
            walk(child)
        if sig[0] == "a":
            columns.append(th.randint(0, E, (batch,), generator=generator))
        elif sig[0] == "p":
            columns.append(th.randint(0, R, (batch,), generator=generator))
    walk(signature)
    return th.stack(columns, 1)


def _modules():
    th.manual_seed(0)
    return {
        "q2b": Query2BoxGeometry(E, R, D),
        "q2b softplus": Query2BoxGeometry(E, R, D, activation="softplus"),
        "q2b none": Query2BoxGeometry(E, R, D, activation="none"),
        "betae": BetaEGeometry(E, R, D, hidden=16, layers=2),
        "cone": ConEGeometry(E, R, D, hidden=16, layers=2),
        "acone": AConEGeometry(E, R, D),
    }


class TestEqualToExisting(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.modules = _modules()

    def assertSameEverywhere(self, module, geometry):
        generator = th.Generator().manual_seed(1)
        candidates = th.randint(0, E, (5, 4), generator=generator)
        signatures = dict(SIGNATURES)
        if module.supports_negation:
            signatures.update(NEGATIVE)
        for name, sig in signatures.items():
            matrix = _ids(sig, 5, generator)
            with th.no_grad():
                a = module.embed(sig, matrix)
                b = geometry.embed(sig, matrix)
                for x, y in zip(a, b):
                    self.assertTrue(th.allclose(x, y, atol=1e-6), name)
                self.assertTrue(th.allclose(module.logit(a, candidates),
                                            geometry.logit(b, candidates), atol=1e-5), name)
                self.assertTrue(th.allclose(module.distance(a, candidates),
                                            geometry.distance(b, candidates), atol=1e-5), name)

    def test_each_geometry_is_rebuilt_with_shared_parameters(self):
        for name, module in self.modules.items():
            geometry = geometry_of(module)
            self.assertIsInstance(geometry, ComposedQueryGeometry, name)
            self.assertIsInstance(geometry, Geometry, name)
            self.assertSameEverywhere(module, geometry)
            owned = {id(p) for p in module.parameters()}
            self.assertTrue({id(p) for p in geometry.parameters()} <= owned, name)
            self.assertEqual(geometry.supports_negation, module.supports_negation, name)

    def test_gradients_reach_the_original_parameters(self):
        for name, module in self.modules.items():
            module.zero_grad()
            geometry = geometry_of(module)
            sig = SIGNATURES["2i"]
            matrix = _ids(sig, 3, th.Generator().manual_seed(2))
            geometry.logit(geometry.embed(sig, matrix), th.tensor([[1, 2], [3, 4], [5, 6]])
                           ).sum().backward()
            self.assertIsNotNone(module.entity_embedding.grad, name)

    def test_parts(self):
        parts = {name: geometry_of(m).parts for name, m in self.modules.items()}
        self.assertEqual(parts["q2b"], {
            "classes": "box", "relations": "box of translations",
            "measure": "bounds per coordinate", "conjunction": "approximate",
            "negation": "none"})
        self.assertEqual(parts["betae"]["relations"], "neural network")
        self.assertEqual(parts["betae"]["measure"], "divergence")
        self.assertEqual((parts["cone"]["classes"], parts["cone"]["relations"],
                          parts["cone"]["measure"]),
                         ("sector cone", "neural network", "angle"))
        self.assertEqual(parts["acone"]["relations"], "rotation")

    def test_registered(self):
        self.assertTrue({"query2box", "betae", "cone_qe", "acone"} <= set(METHODS))
        self.assertIn("sector cone", CLASS_MODELS)
        self.assertIn("shift and widen", RELATION_MODELS)
        self.assertIn("angle rotation", RELATION_MODELS)
        self.assertIn("angle", MEASURES)

    def test_fresh_geometries_have_the_conventional_sizes(self):
        for build in (query2box, betae, cone_qe, acone):
            geometry = build(5, 2, 3, embed_dim=4)
            self.assertEqual(geometry.num_entities, 8)
            self.assertEqual(geometry.num_relations, 6)       # isa, its inverse, 2 roles x 2
            state = geometry.embed(SIGNATURES["2i"], th.tensor([[0, 2, 1, 3]]))
            self.assertEqual(geometry.logit(state, th.tensor([[1, 2]])).shape, (1, 2))


class TestSwappableParts(TestCase):

    def test_acone_is_cone_regions_with_the_rotation(self):
        """ConE's regions, intersection, negation and distance with AConE's relation model
        equal the existing AConE geometry, with copied parameters."""
        th.manual_seed(3)
        cone = ConEGeometry(E, R, D, hidden=16)
        aconeg = AConEGeometry(E, R, D)
        aconeg.load_state_dict(cone.state_dict(), strict=False)    # all but the network
        rebuilt = cone_qe(tables=cone, relations="rotation")
        self.assertEqual(rebuilt.parts["relations"], "rotation")
        generator = th.Generator().manual_seed(4)
        for name, sig in {**SIGNATURES, **NEGATIVE}.items():
            matrix = _ids(sig, 4, generator)
            cand = th.randint(0, E, (4, 3), generator=generator)
            with th.no_grad():
                a, b = aconeg.embed(sig, matrix), rebuilt.embed(sig, matrix)
                for x, y in zip(a, b):
                    self.assertTrue(th.allclose(x, y, atol=1e-6), name)
                self.assertTrue(th.allclose(aconeg.logit(a, cand), rebuilt.logit(b, cand),
                                            atol=1e-5), name)

    def test_acone_function_equals_cone_with_rotation(self):
        th.manual_seed(5)
        module = AConEGeometry(E, R, D)
        one, two = acone(tables=module), cone_qe(tables=module, relations="rotation")
        sig, matrix = SIGNATURES["pi"], _ids(SIGNATURES["pi"], 3, th.Generator().manual_seed(6))
        with th.no_grad():
            for x, y in zip(one.embed(sig, matrix), two.embed(sig, matrix)):
                self.assertTrue(th.equal(x, y))

    def test_network_needs_a_network(self):
        with self.assertRaisesRegex(ValueError, "rotation"):
            cone_qe(tables=AConEGeometry(E, R, D), relations="network")

    def test_query2box_with_the_exact_box_intersection(self):
        th.manual_seed(7)
        module = Query2BoxGeometry(E, R, D)
        exact = query2box(tables=module, intersection="exact")
        self.assertEqual(exact.parts["conjunction"], "exact")
        self.assertEqual(list(exact.intersection.parameters()), [])
        sig2p = SIGNATURES["2p"]
        matrix = _ids(sig2p, 4, th.Generator().manual_seed(8))
        arms = [module.embed(s, _ids(s, 4, th.Generator().manual_seed(9)))
                for s in (SIGNATURES["1p"],) * 3]
        with th.no_grad():
            c, o = exact.intersect(arms)
            lower = th.stack([a[0] - a[1] for a in arms]).max(0)[0]
            upper = th.stack([a[0] + a[1] for a in arms]).min(0)[0]
            nonempty = upper >= lower
            self.assertTrue(th.allclose((c - o)[nonempty], lower[nonempty], atol=1e-6))
            self.assertTrue(th.allclose((c + o)[nonempty], upper[nonempty], atol=1e-6))
            self.assertTrue(bool((o >= 0).all()))
            # the rest of the operators are those of Query2Box
            self.assertTrue(th.equal(exact.embed(sig2p, matrix)[0],
                                     module.embed(sig2p, matrix)[0]))
        # and it is the learned operator that is replaced, not the exact one
        learned = module.intersect(arms)
        self.assertFalse(th.allclose(learned[1], exact.intersect(arms)[1]))

    def test_unknown_intersection(self):
        with self.assertRaises(ValueError):
            query2box(3, 1, intersection="soft")


class TestInterface(TestCase):

    def setUp(self):
        th.manual_seed(10)
        self.g = {"q2b": query2box(NUM_CLASSES, 2, 4, embed_dim=D),
                  "betae": betae(NUM_CLASSES, 2, 4, embed_dim=D, hidden=16),
                  "cone": cone_qe(NUM_CLASSES, 2, 4, embed_dim=D, hidden=16),
                  "acone": acone(NUM_CLASSES, 2, 4, embed_dim=D)}
        for g in self.g.values():
            g._num_classes = NUM_CLASSES

    def test_classes_individuals_exists_follow_the_query_reading(self):
        """``exists r.C`` is the projection of the class along the inverse of ``r``."""
        ids, roles = th.tensor([0, 3, 5]), th.tensor([0, 1, 1])
        for name, g in self.g.items():
            with th.no_grad():
                c = g.classes(ids)
                self.assertIsInstance(c, g.region_class)
                query = g.project(g.project(g.anchor(ids), th.ones_like(ids)), 3 + 2 * roles)
                got = g.exists(roles, c)
                self.assertTrue(all(th.allclose(x, y, atol=1e-6) for x, y in zip(
                    query, tuple(getattr(got, f) for f in got.__dataclass_fields__))), name)
                self.assertEqual(len(g.anchors(ids)), 3)
                ind = g.individuals(th.tensor([0, 1]))
                want = g.anchors(th.tensor([NUM_CLASSES, NUM_CLASSES + 1]))
                for f in ind.__dataclass_fields__:
                    self.assertTrue(th.equal(getattr(ind, f), getattr(want, f)), name)
            self.assertEqual((g.num_classes, g.num_individuals), (NUM_CLASSES, 4))

    def test_extension_false_gives_the_anchors(self):
        g = self.g["q2b"]
        g.extension = False
        ids = th.tensor([1, 2])
        self.assertTrue(th.equal(g.classes(ids).centre, g.anchors(ids).centre))

    def test_member_is_the_distance_of_the_model(self):
        for name, g in self.g.items():
            ids = th.tensor([0, 2, 3])
            with th.no_grad():
                region = g.classes(th.tensor([1, 2, 4]))
                got = g.member(ids, region)
                want = g.distance(tuple(getattr(region, f) for f in region.__dataclass_fields__),
                                  (ids + NUM_CLASSES).unsqueeze(1))
            self.assertEqual(got.shape, (3, 1), name)
            self.assertTrue(th.allclose(got, want, atol=1e-6), name)
            self.assertTrue(bool((got >= 0).all()), name)

    def test_individuals_need_the_number_of_classes(self):
        g = query2box(3, 1, 2, embed_dim=3)
        with self.assertRaisesRegex(ConstructorUnavailable, "number of classes"):
            g.individuals(th.tensor([0]))
        with self.assertRaises(ConstructorUnavailable):
            g.member(th.tensor([0]), g.classes(th.tensor([0])))

    def test_conjunction_and_negation(self):
        ids = th.tensor([0, 1])
        for name, g in self.g.items():
            a, b = g.classes(ids), g.classes(ids + 2)
            region, gap = g.conjunction(a, b)
            self.assertIsInstance(region, g.region_class)
            self.assertEqual(gap.shape, (2, 1))
            if name == "q2b":
                with self.assertRaises(ConstructorUnavailable):
                    g.negation(a)
                with self.assertRaises(UnsupportedOperator):
                    g.negate((a.centre, a.offset))
            else:
                twice = g.negation(g.negation(a))
                for f in a.__dataclass_fields__:
                    self.assertTrue(th.allclose(getattr(twice, f), getattr(a, f), atol=1e-5), name)
            with self.assertRaises(ConstructorUnavailable):
                g.disjunction(a, b)

    def test_boxes_inside_and_apart(self):
        g = self.g["q2b"]
        small = Box(th.zeros(1, 2), th.full((1, 2), 0.1))
        big = Box(th.tensor([[0.05, 0.0]]), th.full((1, 2), 0.5))
        far = Box(th.tensor([[3.0, 0.0]]), th.full((1, 2), 0.5))
        self.assertAlmostEqual(float(g.inside(small, big)), 0.0, places=6)
        self.assertGreater(float(g.inside(big, small)), 0.1)
        self.assertAlmostEqual(float(g.apart(small, far)), 0.0, places=6)
        self.assertGreater(float(g.apart(small, big)), 0.0)

    def test_cones_inside_and_apart(self):
        g = self.g["cone"]
        pi = math.pi

        def cone(axis, aperture):
            return SectorCone(th.tensor([[axis]]), th.tensor([[aperture]]))
        self.assertAlmostEqual(float(g.inside(cone(0.0, 0.3), cone(0.1, 0.5))), 0.0, places=6)
        self.assertAlmostEqual(float(g.inside(cone(0.0, 0.3), cone(0.3, 0.5))), 0.1, places=5)
        # across the cut at pi: the axes are 0.2 apart
        self.assertAlmostEqual(float(g.inside(cone(pi - 0.1, 0.1), cone(-pi + 0.1, 0.4))), 0.0,
                               places=5)
        # the full circle contains everything, and only the full circle contains it
        self.assertAlmostEqual(float(g.inside(cone(2.0, 1.0), cone(-1.0, pi))), 0.0, places=6)
        self.assertGreater(float(g.inside(cone(0.0, pi), cone(0.0, 3.0))), 0.0)
        self.assertAlmostEqual(float(g.apart(cone(0.0, 0.3), cone(2.0, 0.3))), 0.0, places=6)
        self.assertGreater(float(g.apart(cone(0.0, 0.3), cone(0.4, 0.3))), 0.0)
        # a negative (wrapped) aperture counts by its absolute value
        self.assertAlmostEqual(float(g.inside(cone(0.0, -0.3), cone(0.1, 0.5))), 0.0, places=6)
        self.assertIsInstance(g.measure, ConeBounds)

    def test_beta_inside_is_the_divergence(self):
        g = self.g["betae"]
        a = BetaRegion(th.tensor([[2.0, 3.0]]), th.tensor([[4.0, 1.0]]))
        b = BetaRegion(th.tensor([[1.0, 1.0]]), th.tensor([[1.0, 1.0]]))
        self.assertAlmostEqual(float(g.inside(a, a)), 0.0, places=5)
        self.assertGreater(float(g.inside(a, b)), 0.0)
        with th.no_grad():
            point = g.anchors(th.tensor([2]))
            self.assertAlmostEqual(float(g.inside(point, point)), 0.0, places=5)
            # the divergence of the model's distance: entity against query
            self.assertTrue(th.allclose(
                g.inside(point, a.__class__(a.alpha.new_ones(1, D), a.alpha.new_ones(1, D))),
                g.member(th.tensor([-NUM_CLASSES + 2]), a.__class__(
                    a.alpha.new_ones(1, D), a.alpha.new_ones(1, D))), atol=1e-4))
        with self.assertRaisesRegex(ConstructorUnavailable, "divergence"):
            g.apart(a, b)

    def test_exact_box_conjunction_reports_emptiness(self):
        g = query2box(NUM_CLASSES, 2, 0, embed_dim=2, intersection="exact")
        a = Box(th.zeros(1, 2), th.ones(1, 2))
        near = Box(th.tensor([[1.0, 0.0]]), th.ones(1, 2))
        far = Box(th.tensor([[5.0, 0.0]]), th.ones(1, 2))
        region, gap = g.conjunction(a, near)
        self.assertEqual(float(gap), 0.0)
        reference, _ = a.intersect(near)
        self.assertTrue(th.allclose(region.centre, reference.centre))
        self.assertTrue(th.allclose(region.offset, reference.offset))
        _, gap = g.conjunction(a, far)
        self.assertAlmostEqual(float(gap), 3.0, places=5)
