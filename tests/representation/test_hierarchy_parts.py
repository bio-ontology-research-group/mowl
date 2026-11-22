"""The methods that embed a hierarchy, rebuilt from parts, give the numbers of the existing
modules, and their parts combine with parts of other methods."""
from unittest import TestCase

import torch as th

from mowl.models import ComposedEL
from mowl.baselines.nn.el.box_lattice import BoxLatticeModule
from mowl.baselines.nn.el.gumbel_box import GumbelBoxModule
from mowl.baselines.nn.el.order_embeddings import OrderEmbeddingsModule
from mowl.baselines.nn.graph.bcbox import BinaryCodeBoxModule
from mowl.baselines.nn.graph.bcbox.losses import neg_log1mexp as bcbox_neg_loss
from mowl.baselines.nn.graph.cate import CatEModule
from mowl.nn.lm import BagOfWordsEncoder, HiTModule, OnTModule
from mowl.nn.order import (DiskEmbeddingsModule, HyperbolicConesModule, RiemannianSGD,
                           ShadowConesModule)
from mowl.nn.order.manifolds import Euclidean, PoincareBall, Sphere, UpperHalfSpace
from mowl.representation import (BallClasses, BoxClasses, Bounds, CentreDistance,
                                 ClassModel, ComposedGeometry, ConstructorUnavailable,
                                 Measure, Objective, geometry_of)
from mowl.representation import methods as M
from mowl.representation.library import hierarchy_methods as HM
from mowl.representation.library.hierarchy_classes import (
    DiskClasses, PointClasses, UnitCubeBoxes)
from mowl.representation.library.hierarchy_geometry import (
    hierarchy_geometry, TextGeometry)
from mowl.representation.library.hierarchy_measures import (
    ConeAngle, Protrusion, ShadowCone, LogVolumeRatio)
from tests.datasetFactory import FamilyDataset

N = 14


def pairs(n=60, seed=0, size=N):
    g = th.Generator().manual_seed(seed)
    return th.randint(0, size, (n, 2), generator=g)


def perturb(module, scale=0.3, seed=0):
    """Move the parameters of ``module`` so that its pairs are not all in the same state."""
    g = th.Generator().manual_seed(seed)
    with th.no_grad():
        for p in module.parameters():
            p.add_(scale * th.randn(p.shape, generator=g))
    return module


class Same:
    """Checks: the geometry of a module has the module's scores, positive and negative."""

    def assertClose(self, got, expected, msg=None, atol=1e-6):
        expected = expected.reshape(-1, 1)
        self.assertEqual(got.shape, expected.shape, msg)
        self.assertTrue(th.allclose(got.float(), expected.float(), atol=atol, rtol=1e-5),
                        (msg, (got - expected).abs().max().item()))

    def assertScalar(self, got, expected, msg=None):
        self.assertTrue(th.allclose(th.as_tensor(got, dtype=th.float).reshape(()),
                                    th.as_tensor(expected, dtype=th.float).reshape(()),
                                    atol=1e-6), msg)

    def assertSharesParameters(self, geometry, module):
        before = {id(p) for p in module.parameters()}
        self.assertTrue({id(p) for p in geometry.parameters()} <= before)


class TestOrderEmbeddings(Same, TestCase):

    def test_scores(self):
        th.manual_seed(0)
        module = OrderEmbeddingsModule(N, 5, margin=0.7)
        g = geometry_of(module)
        rows = pairs()
        self.assertClose(g(rows, "gci0"), module(rows, "gci0"))
        self.assertClose(g(rows, "gci0", neg=True), module(rows, "gci0", neg=True))
        self.assertSharesParameters(g, module)
        self.assertIsInstance(g, ComposedGeometry)

    def test_fresh_parts_have_the_same_scores(self):
        th.manual_seed(0)
        g = HM.order_embeddings(N, embed_dim=5, margin=0.7)
        module = OrderEmbeddingsModule(N, 5, margin=0.7)
        module.class_embed.weight.data.copy_(g.class_embed.weight.data)
        rows = pairs()
        self.assertClose(g(rows, "gci0", neg=True), module(rows, "gci0", neg=True))

    def test_other_forms_need_parts_that_are_missing(self):
        g = HM.order_embeddings(N, embed_dim=5)
        with self.assertRaisesRegex(ConstructorUnavailable, "relation"):
            g(th.zeros(3, 3, dtype=th.long), "gci2")
        self.assertEqual(g.parts["conjunction"], "none")


class TestCatE(Same, TestCase):

    def make(self):
        th.manual_seed(1)
        module = CatEModule(N, 5)
        module.set_existentials([0, 1, 0], [2, 3, 4], [10, 11, 12])
        return module

    def test_scores(self):
        module = self.make()
        g = geometry_of(module)
        rows = pairs()
        self.assertClose(g(rows, "gci0"), module(rows, "gci0"))
        self.assertClose(g(rows, "gci0", neg=True),
                         th.relu(1.0 - module.order_violation(rows[:, 0], rows[:, 1])))

    def test_existentials_by_lookup(self):
        module = self.make()
        g = geometry_of(module)
        rows = th.tensor([[1, 0, 2], [5, 1, 3], [7, 0, 4]])
        expected = module(rows, "gci2")
        c, filler = g.classes(rows[:, 0]), g.classes(rows[:, 2])
        self.assertClose(g.inside(c, g.exists(rows[:, 1], filler)), expected)
        with self.assertRaisesRegex(LookupError, "not an element"):
            g.exists(th.tensor([1]), g.classes(th.tensor([2])))

    def test_conjunctions_by_lookup(self):
        module = self.make()
        g = geometry_of(module)
        g._conjunction_ids[(2, 3)] = 9
        region, gap = g.conjunction(g.classes(th.tensor([3])), g.classes(th.tensor([2])))
        self.assertTrue(th.equal(region.centre, module.embed(th.tensor([9]))))
        self.assertEqual(float(gap), 0.0)
        with self.assertRaisesRegex(LookupError, "not an element"):
            g.conjunction(g.classes(th.tensor([0])), g.classes(th.tensor([1])))
        self.assertEqual(g.parts["conjunction"], "own embedding")

    def test_individuals_are_elements(self):
        g = geometry_of(self.make())
        self.assertTrue(th.equal(g.individuals(th.tensor([4])).centre,
                                 g.classes(th.tensor([4])).centre))


class TestBoxes(Same, TestCase):

    def check(self, module, kind):
        g = geometry_of(module)
        self.assertIsInstance(g, ComposedGeometry)
        rows = pairs()
        for seed in range(3):
            rows = pairs(seed=seed)
            self.assertClose(g(rows, "gci0"), module(rows, "gci0"), (kind, seed, "pos"))
            self.assertClose(g(rows, "gci0", neg=True), module(rows, "gci0", neg=True),
                             (kind, seed, "neg"))
        self.assertSharesParameters(g, module)
        reg = module.regularization_loss()
        self.assertScalar(g.regularization_loss(), reg)

    def test_box_lattice(self):
        for kwargs in ({}, {"init_side": (0.05, 0.2)}, {"eps": 1e-3},
                       {"freeze_delta_negatives": False}, {"disjoint_weight": 1.0},
                       {"max_regularization": 0.05},
                       # the earlier guard: -log(1 + eps) for an included pair
                       {"zero_at_inclusion": False},
                       {"zero_at_inclusion": False, "eps": 1e-3}):
            th.manual_seed(3)
            module = BoxLatticeModule(N, 3, **kwargs)
            self.check(module, f"box_lattice {kwargs}")
            self.assertEqual(geometry_of(module).measure.zero_at_inclusion,
                             module.zero_at_inclusion)

    def test_box_lattice_default_is_exactly_zero_for_an_included_pair(self):
        g = HM.box_lattice(N, embed_dim=3)
        self.assertTrue(g.measure.zero_at_inclusion)
        self.assertFalse(HM.box_lattice(N, embed_dim=3,
                                        zero_at_inclusion=False).measure.zero_at_inclusion)

    def test_box_lattice_covers_disjoint_and_nested_pairs(self):
        th.manual_seed(3)
        module = BoxLatticeModule(N, 3, init_side=(0.05, 0.2))
        mn, mx = module.corners(th.arange(N))
        inter = ((th.minimum(mx[:, None], mx[None]) > th.maximum(mn[:, None], mn[None]))
                 .all(-1))
        self.assertTrue(inter.any() and (~inter).any())

    def test_freezing_the_sizes_of_negatives_is_part_of_the_objective(self):
        th.manual_seed(3)
        g = HM.box_lattice(N, embed_dim=3)
        rows = pairs()
        g(rows, "gci0", neg=True).sum().backward()
        d = g.class_embed.weight.grad[:, 3:]
        self.assertEqual(float(d.abs().sum()), 0.0)
        g.zero_grad()
        g.objective.freeze_size_negatives = False
        g(rows, "gci0", neg=True).sum().backward()
        self.assertGreater(float(g.class_embed.weight.grad[:, 3:].abs().sum()), 0.0)

    def test_gumbel_box(self):
        for kwargs in ({}, {"beta": 0.1}, {"beta": 0.002}, {"max_regularization": 0.02},
                       {"log_guard": "epsilon"}, {"log_guard": "epsilon", "beta": 0.002}):
            th.manual_seed(4)
            module = GumbelBoxModule(N, 3, **kwargs)
            self.check(module, f"gumbel {kwargs}")
            self.assertEqual(geometry_of(module).measure.tail,
                             "linear" if module.log_guard == "stable" else "floor")

    def test_gumbel_is_the_box_lattice_with_another_measure(self):
        th.manual_seed(5)
        lattice = BoxLatticeModule(N, 3)
        g = geometry_of(lattice)
        soft = hierarchy_geometry(g.class_model, None, LogVolumeRatio(0.01),
                                  Objective.hierarchy())
        gumbel = GumbelBoxModule(N, 3, beta=0.01, min_delta=lattice.min_delta)
        gumbel.class_embed.weight.data.copy_(lattice.class_embed.weight.data)
        rows = pairs()
        self.assertClose(soft(rows, "gci0"), gumbel(rows, "gci0"))
        self.assertTrue(isinstance(g.measure, LogVolumeRatio) and not g.measure.smoothed)
        self.assertEqual(g.class_model.part, "box")

    def test_hard_limit_of_the_soft_measure(self):
        th.manual_seed(6)
        g = HM.box_lattice(N, embed_dim=3, init_side=(0.5, 0.9))
        rows = pairs()
        a, b = g.classes(rows[:, 0]), g.classes(rows[:, 1])
        p, gap = g.measure.conditional(a, b)
        meets = (gap == 0).reshape(-1)
        hard = LogVolumeRatio(0.0, 0.0, gumbel_shift=False).energy(a, b)
        self.assertTrue(meets.any())
        self.assertTrue(th.allclose(hard[meets].double(), -th.log(p[meets]), atol=1e-4))

    def test_project_keeps_the_cube(self):
        g = HM.box_lattice(N, embed_dim=3)
        with th.no_grad():
            g.class_embed.weight.mul_(5).sub_(2)
        g.project()
        low, delta = g.class_embed.weight[:, :3], g.class_embed.weight[:, 3:]
        inside = (low >= 0).all() and (low + delta <= 1 + 1e-6).all()
        self.assertTrue(bool(inside and (delta >= g.class_model.min_delta - 1e-9).all()))

    def test_bcbox(self):
        for kwargs in ({"variant": "gbox"}, {"variant": "vbc"}, {"variant": "gbc"},
                       {"variant": "vbc", "code_dims": 2},
                       {"variant": "gbc", "num_codes": 3, "code_dims": 2},
                       {"variant": "vbc", "intersection_temp": 0.0, "volume_temp": 0.0},
                       {"variant": "vbc", "intersection_temp": 0.3, "volume_temp": 0.05}):
            th.manual_seed(7)
            module = BinaryCodeBoxModule(N, 4, **kwargs)
            g = geometry_of(module)
            rows = pairs()
            sub, sup = rows[:, 0], rows[:, 1]
            energy = module.energy(sup, sub)
            self.assertClose(g(rows, "gci0"), energy, kwargs)
            self.assertClose(g(rows, "gci0", neg=True), bcbox_neg_loss(energy), kwargs)
            self.assertSharesParameters(g, module)
            self.assertScalar(g.class_model.code_regularization(),
                              module.code_regularization(), kwargs)

    def test_bcbox_codes_belong_to_the_class_model_and_the_measure(self):
        g = HM.bcbox(N, embed_dim=4, variant="vbc")
        self.assertEqual(g.class_model.part, "soft box")
        self.assertIsNotNone(g.classes(th.tensor([0, 1])).code)
        self.assertIsNone(HM.bcbox(N, embed_dim=4, variant="gbox").classes(
            th.tensor([0])).code)
        g = HM.bcbox(N, embed_dim=4, variant="vbc", code_weight=0.5)
        self.assertGreater(float(g.regularization_loss()), 0.0)


class TestManifolds(Same, TestCase):

    def test_hyperbolic_cones(self):
        th.manual_seed(8)
        module = HyperbolicConesModule(N, 4, K=0.1, margin=0.2, seed=0)
        g = geometry_of(module)
        rows = pairs()
        self.assertClose(g(rows, "gci0"), module(rows, "gci0"))
        self.assertClose(g(rows, "gci0", neg=True), module(rows, "gci0", neg=True))
        self.assertGreater(float(g(rows, "gci0").max()), 0.0)
        self.assertSharesParameters(g, module)
        self.assertEqual(g.class_model.part, "geodesic cone")

    def test_shadow_cones_all_formulations(self):
        for formulation in ("umbral-half-space", "penumbral-half-space", "umbral-ball",
                            "penumbral-ball"):
            # the hinge (default) and the raw energy of the earlier module, with its own margin
            for options in ({}, {"negative": "energy"}, {"margin": 0.3},
                            {"margin": 0.3, "negative": "energy"}):
                module = ShadowConesModule(N, 3, formulation=formulation, init_scale=0.1,
                                           seed=1, **options)
                g = geometry_of(module)
                rows = pairs()
                self.assertClose(g(rows, "gci0"), module(rows, "gci0"), formulation)
                self.assertClose(g(rows, "gci0", neg=True), module(rows, "gci0", neg=True),
                                 (formulation, options))
                self.assertSharesParameters(g, module)
                self.assertEqual(g.measure.negative_form, module.negative)

    def test_shadow_cones_default_negative_is_not_negative(self):
        g = HM.shadow_cones(N, embed_dim=3)
        self.assertEqual(g.measure.negative_form, "hinge")
        self.assertTrue(bool((g(pairs(), "gci0", neg=True) >= 0).all()))

    def test_disk_embeddings_all_spaces(self):
        for space in ("hyperbolic", "spherical", "euclidean"):
            module = DiskEmbeddingsModule(N, 3, geometry=space, margin=0.3, seed=2)
            perturb(module, 0.2)
            with th.no_grad():
                module.class_embed.weight.copy_(module.manifold.project(
                    module.class_embed.weight))
            g = geometry_of(module)
            rows = pairs()
            self.assertClose(g(rows, "gci0"), module(rows, "gci0"), space)
            self.assertClose(g(rows, "gci0", neg=True), module(rows, "gci0", neg=True), space)
            self.assertSharesParameters(g, module)

    def test_fresh_tables_are_valid_points(self):
        for formulation in ("umbral-half-space", "penumbral-ball"):
            g = HM.shadow_cones(N, embed_dim=3, formulation=formulation)
            self.assertTrue(bool(th.isfinite(g(pairs(), "gci0")).all()))
        for space in ("hyperbolic", "spherical", "euclidean"):
            g = HM.disk_embeddings(N, embed_dim=3, geometry=space)
            self.assertTrue(bool(th.isfinite(g(pairs(), "gci0")).all()))
        g = HM.hyperbolic_cones(N, embed_dim=3)
        self.assertGreaterEqual(float(g.class_model.centres.weight.norm(dim=1).min()),
                                g.class_model.epsilon - 1e-6)


class TestTextModels(Same, TestCase):

    texts = ["heart", "organ", "heart valve", "valve", "body part", "lung", "thing", "cell"]

    def setUp(self):
        th.manual_seed(9)
        self.encoder = BagOfWordsEncoder(dim=8, num_buckets=32)

    def test_hit_scores(self):
        module = HiTModule(self.encoder, clustering_margin=2.0, centripetal_margin=0.3)
        g = geometry_of(module)
        self.assertIsInstance(g, TextGeometry)
        child, parent, negative = (self.texts[:4], self.texts[4:], self.texts[2:6])
        x, y, z = (module.embed(t) for t in (child, parent, negative))
        self.assertClose(g.measure.inside(g.classes(child), g.classes(parent)),
                         -module.score(x, y, 1.0))
        g = HM.hit(tables=module, centri_weight=0.4)
        self.assertClose(g.measure.inside(g.classes(child), g.classes(parent)),
                         -module.score(x, y, 0.4))
        self.assertClose(g.hierarchy_loss(g.classes(child), g.classes(parent),
                                          g.classes(negative)),
                         module.hierarchy_loss(x, y, z))
        # classes that were never seen are regions too, and ids work with a list of texts
        g = HM.hit(encoder=self.encoder, texts=self.texts)
        rows = th.tensor([[0, 1], [2, 3]])
        self.assertClose(g(rows, "gci0"),
                         g.measure.inside(g.classes(["heart", "heart valve"]),
                                          g.classes(["organ", "valve"])))
        self.assertEqual(g.classes(["an unseen name"]).centre.shape, (1, 8))

    def test_ont_relation_model(self):
        module = OnTModule(self.encoder)
        g = geometry_of(module)
        self.assertEqual(g.parts["relations"], "rotation")
        filler = ["organ", "valve"]
        roles = ["part of", "has part"]
        image = g.exists(roles, g.classes(filler))
        expected = module.role_image(module.embed(filler), roles)
        self.assertTrue(th.allclose(image.centre, expected, atol=1e-6))
        # Eq. 6: both directions of the order between the compound class and f_r(D)
        compound = ["part of organ", "has part valve"]
        n1, n2 = module.embed(self.texts[:2]), module.embed(self.texts[2:4])
        got = g.existential_loss(g.compound(compound), roles, g.classes(filler),
                                 g.classes(self.texts[:2]), g.classes(self.texts[2:4]))
        want = module.existential_loss(compound, roles, filler, n1, n2)
        self.assertClose(got, want)
        with self.assertRaises(ConstructorUnavailable):
            g.relation_model.image(roles, g.classes(filler))

    def test_ont_without_roles_has_no_existentials(self):
        g = geometry_of(OnTModule(self.encoder, use_roles=False))
        self.assertIsNone(g.relation_model)
        with self.assertRaises(ConstructorUnavailable):
            g.exists(["part of"], g.classes(["organ"]))


class TestRegistry(TestCase):

    def test_methods_and_parts_are_registered(self):
        for key in ("order_embeddings", "cate", "box_lattice", "gumbel_box", "bcbox",
                    "hyperbolic_cones", "shadow_cones", "disk_embeddings", "hit", "ont"):
            self.assertIn(key, M.METHODS)
        for name in ("log volume ratio", "order violation", "angle",
                     "protrusion", "distance and depth", "shadow cone"):
            self.assertIn(name, M.MEASURES)
        for name in ("unit box", "order cone", "point", "geodesic cone", "disk"):
            self.assertIn(name, M.CLASS_MODELS)

    def test_parts_name_vocabulary_entries(self):
        geometries = [HM.order_embeddings(N, embed_dim=3), HM.box_lattice(N, embed_dim=3),
                      HM.gumbel_box(N, embed_dim=3), HM.bcbox(N, embed_dim=3),
                      HM.hyperbolic_cones(N, embed_dim=3), HM.shadow_cones(N, embed_dim=3),
                      HM.disk_embeddings(N, embed_dim=3), HM.cate(N, embed_dim=3)]
        for g in geometries:
            ClassModel(g.parts["classes"])
            Measure(g.parts["measure"])

    def test_the_catalogue_agrees_with_the_parts(self):
        from mowl.representation import REPRESENTATIONS
        for key, geometry in (("order_embeddings", HM.order_embeddings(N, embed_dim=3)),
                              ("box_lattice", HM.box_lattice(N, embed_dim=3)),
                              ("gumbel_box", HM.gumbel_box(N, embed_dim=3)),
                              ("bcbox", HM.bcbox(N, embed_dim=3)),
                              ("hyperbolic_cones", HM.hyperbolic_cones(N, embed_dim=3)),
                              ("disk_embeddings", HM.disk_embeddings(N, embed_dim=3))):
            rep = REPRESENTATIONS[key]
            self.assertTrue(rep.has(geometry.parts["classes"]), key)
            self.assertTrue(rep.has(geometry.parts["measure"]), key)


def train(geometry, steps=60, lr=0.05, optimizer=None, seed=0):
    th.manual_seed(seed)
    rows = th.tensor([[1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 2], [7, 3], [8, 3], [9, 4]])
    neg = th.tensor([[1, 2], [2, 1], [3, 4], [4, 3], [5, 6], [6, 5], [7, 8], [8, 7], [9, 8]])

    def loss():
        return geometry(rows, "gci0").mean() + geometry(neg, "gci0", neg=True).mean()
    opt = optimizer or th.optim.Adam(geometry.parameters(), lr=lr)
    first = float(loss())
    for _ in range(steps):
        opt.zero_grad()
        loss().backward()
        opt.step()
        geometry.project()
    return first, float(loss())


class TestParts(TestCase):
    """Parts work outside their paper: swapped, combined with parts of other methods, trained."""

    def test_protrusion_on_every_space_with_the_same_class_model(self):
        for manifold in (Euclidean(), Sphere()):
            classes = DiskClasses(N, 3, manifold, init_scale=0.3)
            g = hierarchy_geometry(classes, None, Protrusion(), Objective.hierarchy(0.1))
            self.assertTrue(bool((g(pairs(), "gci0") >= 0).all()))
            first, last = train(g, 40)
            self.assertLess(last, first, type(manifold).__name__)

    def test_protrusion_of_balls_is_the_measure_of_balls(self):
        th.manual_seed(1)
        classes = BallClasses(N, 3)
        rows = pairs()
        rows = rows[rows[:, 0] != rows[:, 1]]
        a, b = classes(rows[:, 0]), classes(rows[:, 1])
        for margin in (0.0, 0.15):
            self.assertTrue(th.allclose(Protrusion().inside(a, b, margin),
                                        CentreDistance().inside(a, b, margin), atol=1e-6))

    def test_shadow_cone_on_the_half_space_and_on_the_ball(self):
        for formulation, manifold in (("umbral-half-space", UpperHalfSpace()),
                                      ("penumbral-ball", PoincareBall(min_norm=0.1))):
            classes = PointClasses(N, 3, manifold, init_scale=0.2)
            g = hierarchy_geometry(classes, None, ShadowCone(
                formulation, signed=True, negative_form="energy"), Objective.hierarchy())
            opt = RiemannianSGD(g.param_groups(), lr=0.02)
            first, last = train(g, 30, optimizer=opt)
            self.assertLess(last, first, formulation)

    def test_a_cone_measure_needs_a_matching_manifold(self):
        classes = PointClasses(N, 3, PoincareBall(min_norm=0.1))
        g = hierarchy_geometry(classes, None, ShadowCone("umbral-half-space"),
                               Objective.hierarchy())
        with self.assertRaisesRegex(TypeError, "does not live on"):
            g(pairs(), "gci0")
        with self.assertRaisesRegex(TypeError, "compares"):
            hierarchy_geometry(classes, None, ConeAngle(), Objective.hierarchy())

    def test_measures_swap_on_the_box_lattice_class_model(self):
        for measure in (LogVolumeRatio(), LogVolumeRatio(0.02), Bounds()):
            th.manual_seed(0)
            margin = 0.0 if not isinstance(measure, Bounds) else 0.05
            g = hierarchy_geometry(UnitCubeBoxes(N, 3), None, measure,
                                   Objective.hierarchy(margin))
            first, last = train(g, 150, lr=0.02)
            self.assertLess(last, first, type(measure).__name__)

    def test_volume_ratio_on_the_boxes_of_elbe(self):
        classes = BoxClasses(N, 3)
        g = hierarchy_geometry(classes, None, LogVolumeRatio(), Objective.hierarchy())
        self.assertTrue(bool(th.isfinite(g(pairs(), "gci0")).all()))
        self.assertTrue(bool(th.isfinite(g(pairs(), "gci0", neg=True)).all()))

    def test_methods_train(self):
        for name, make in (
                ("order_embeddings", lambda: HM.order_embeddings(N, embed_dim=4)),
                ("cate", lambda: HM.cate(N, embed_dim=4)),
                ("box_lattice", lambda: HM.box_lattice(N, embed_dim=4)),
                ("gumbel_box", lambda: HM.gumbel_box(N, embed_dim=4, beta=0.05)),
                ("bcbox", lambda: HM.bcbox(N, embed_dim=4)),
                ("disk", lambda: HM.disk_embeddings(N, embed_dim=3, geometry="euclidean"))):
            th.manual_seed(0)
            first, last = train(make(), 150, lr=0.02)
            self.assertLess(last, first, name)

    def test_cones_train_with_riemannian_sgd(self):
        g = HM.hyperbolic_cones(N, embed_dim=3, margin=0.05)
        opt = RiemannianSGD(g.param_groups(), lr=0.05)
        first, last = train(g, 40, optimizer=opt)
        self.assertLessEqual(last, first)


class TestComposedWithOtherMethods(TestCase):
    """The unit-cube class model and the volume measures in ComposedEL, with the relation
    model of ELEmbeddings and ELBE."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def test_trains(self):
        for kwargs in ({"classes": "unit box"},
                       {"classes": "unit box", "measure": LogVolumeRatio(0.05)},
                       {"classes": "unit box", "measure": Bounds()}):
            th.manual_seed(0)
            model = ComposedEL(self.dataset, embed_dim=8, objective=Objective(0.0),
                               learning_rate=0.01, **kwargs)
            self.assertIsInstance(model.module, ComposedGeometry)
            before = sum(float(model.module(d[:], f).mean())
                         for f, d in model.training_datasets.items() if len(d))
            model.train(epochs=60)
            after = sum(float(model.module(d[:], f).mean())
                        for f, d in model.training_datasets.items() if len(d))
            self.assertLess(after, before, kwargs)
            self.assertEqual(model.module.parts["relations"], "translation")
