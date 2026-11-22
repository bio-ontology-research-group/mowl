"""The structural objective of the prism module rebuilt from parts, and the structural reading
and the definitions with the regions of other methods (octagons, boxes)."""
from unittest import TestCase

import torch as th

import mowl
mowl.init_jvm("1g")

from mowl.baselines.nn.el.prism.module import PrismModule
from mowl.nn.el.structural import DefinitionTable, ExistenceTargets
from mowl.representation import Bounds, BoxClasses, ComposedGeometry, geometry_of
from mowl.representation.library.shapes_classes import OctagonClasses, PrismClasses
from mowl.representation.library.shapes_measures import OctagonBounds, PrismBounds
from mowl.representation.library.shapes_regions import OctagonProduct, Prism
from mowl.representation.library.structural_methods import StructuralGeometry
from mowl.representation.library.structural_reading import (ExistencePenalty,
                                                            StructuralReading, size_of)
from mowl.representation.library.structural_regions import DefinedClasses
from tests.representation.test_structural_parts import GCI0, GCI1, N, R, ROWS, TOP

D, NI = 5, 3


def prism_module(seed=0, tri="overlap", perturb=0.3, **options):
    th.manual_seed(seed)
    module = PrismModule(N, R, NI, embed_dim=D, margin=0.05, objective="structural",
                         existence_weight=0.7, size_reg=1e-2, gci2_negative_triangle=tri,
                         **options)
    table = DefinitionTable.from_normal_forms(
        gci0=GCI0, gci1=GCI1, n_classes=N, k=1, max_components=4, top_idx=TOP,
        bottom_idx=None, enabled=True)
    targets = ExistenceTargets(classes=th.tensor([0, 1, 2, 3, 4, 5, 7]),
                               pairs=th.tensor([[0, 1], [1, 2], [0, 2]]))
    module.set_structural_tables(table, targets, top_idx=TOP)
    g = th.Generator().manual_seed(seed + 1)
    with th.no_grad():
        for p in module.parameters():
            p.add_(perturb * th.randn(p.shape, generator=g, dtype=p.dtype))
    return module


class TestStructuralPrism(TestCase):

    def compare(self, module, geometry):
        nonzero = 0
        for form, rows in ROWS.items():
            for neg in (False, True):
                try:
                    old = getattr(module, form + "_loss")(rows, neg=neg).reshape(-1, 1)
                except NotImplementedError:
                    # a module refuses a negative loss it does not have: so does the geometry
                    with self.assertRaises(NotImplementedError):
                        getattr(geometry, form + "_loss")(rows, neg=neg)
                    # and both agree on the earlier reading (the positive loss)
                    module.strict_negatives = geometry.strict_negatives = False
                    try:
                        old = getattr(module, form + "_loss")(rows, neg=neg).reshape(-1, 1)
                        new = getattr(geometry, form + "_loss")(rows, neg=neg)
                    finally:
                        module.strict_negatives = geometry.strict_negatives = True
                    self.assertTrue(th.allclose(old, new, atol=1e-6), (form, neg, old, new))
                    nonzero += float(old.abs().sum() > 0)
                    continue
                new = getattr(geometry, form + "_loss")(rows, neg=neg)
                self.assertEqual(new.shape, old.shape)
                self.assertTrue(th.allclose(old, new, atol=1e-6), (form, neg, old, new))
                nonzero += float(old.abs().sum() > 0)
        return nonzero

    def test_every_form_positive_and_negative(self):
        for tri in ("overlap", "inradius"):
            module = prism_module(tri=tri)
            geometry = geometry_of(module)
            self.assertIsInstance(geometry, StructuralGeometry)
            self.assertIsInstance(geometry.class_model, DefinedClasses)
            self.assertGreater(self.compare(module, geometry), 10)

    def test_every_form_with_the_absolute_reading_of_an_empty_box(self):
        # the old option of the module equals the parts with empty_box_reading="absolute"
        module = prism_module(empty_box_reading="absolute")
        geometry = geometry_of(module)
        self.assertEqual(geometry.objective.empty_regions, "abs")
        self.assertGreater(self.compare(module, geometry), 10)

    def test_regularisation_is_the_modules_extra_loss(self):
        for options in ({}, {"existence_weight": 0.0}, {"size_reg": 0.0}):
            module = prism_module()
            for key, value in options.items():
                setattr(module, key, value)
            geometry = geometry_of(module)
            expected = module.regularization_loss() + module.structural_extra_loss()
            self.assertTrue(th.allclose(geometry.regularization_loss(), expected, atol=1e-6))
        self.assertGreater(float(expected + 1), 0)

    def test_size_term_before_the_tables_default_and_old_option(self):
        # default: no size term before the tables (owl:Thing is unknown), as the module;
        # size_reg_without_tables=True: every class is shrunk, as the module with that option
        for flag in (False, True):
            th.manual_seed(0)
            module = PrismModule(N, R, NI, embed_dim=D, objective="structural",
                                 existence_weight=0.0, size_reg=1.0,
                                 size_reg_without_tables=flag)
            geometry = geometry_of(module)
            self.assertEqual(geometry.size_reg_without_tables, flag)
            expected = module.regularization_loss() + module.structural_extra_loss()
            self.assertEqual(float(expected) > 0, flag)
            self.assertTrue(th.allclose(geometry.regularization_loss(), expected, atol=1e-6))
        from mowl.representation.library.shapes_methods import prism
        self.assertFalse(prism(N, R, reading="structural").size_reg_without_tables)
        self.assertTrue(prism(N, R, reading="structural", size_reg_without_tables=True)
                        .size_reg_without_tables)

    def test_defined_classes_are_prisms_with_signed_half_widths(self):
        module = prism_module()
        geometry = geometry_of(module)
        ids = th.arange(N)
        old, new = module.class_regions(ids), geometry.classes(ids)
        self.assertIsInstance(new, Prism)
        self.assertTrue(th.equal(old.h, new.supports))
        self.assertTrue(th.equal(old.cen, new.centre))
        self.assertTrue(th.equal(old.w, new.offset))

    def empty_box_setup(self, **options):
        # Class 3 is defined as 0 and 1. Move the two boxes apart in one coordinate: the box
        # part of the defined class is empty (negative half-width), its triangle is not.
        module = prism_module(**options)
        with th.no_grad():
            module.box_center.weight[0, 0] = -5.0
            module.box_center.weight[1, 0] = 5.0
            module.box_offset.weight[:2, 0] = 0.5
        module.set_structural_tables(module.definition_table, module.existence_targets, TOP)
        return module

    def empty_rows(self):
        rows = {form: ROWS[form][:3].clone() for form in
                ("class_assertion", "gci2", "gci3", "gci3_bot")}
        rows["class_assertion"][:, 1] = 3        # the defined (empty) class is the class
        rows["gci2"][:, 0] = 3                   # ... the source
        rows["gci3"][:, 1] = 3                   # ... the filler
        rows["gci3_bot"][:, 1] = 3               # ... the filler
        return rows

    def test_module_and_parts_agree_on_an_empty_box_by_default(self):
        # the default reading of both is the signed one: the empty box is empty
        module = self.empty_box_setup()
        self.assertEqual(module.empty_box_reading, "signed")
        geometry = geometry_of(module)
        self.assertEqual(geometry.objective.empty_regions, "keep")
        self.assertLess(float(geometry.classes(th.tensor([3])).offset[0, 0]), 0)
        nonzero = 0
        for form, rows in self.empty_rows().items():
            for neg in (False, True):
                try:
                    old = getattr(module, form + "_loss")(rows, neg=neg).reshape(-1, 1)
                except NotImplementedError:
                    continue
                new = getattr(geometry, form + "_loss")(rows, neg=neg)
                self.assertTrue(th.allclose(old, new, atol=1e-6), (form, neg, old, new))
                nonzero += float(old.abs().sum() > 0)
        # the rows with the empty class as source or filler are vacuous: zero cost
        for form in ("gci2", "gci3", "gci3_bot"):
            rows = self.empty_rows()[form]
            self.assertEqual(float(geometry(rows, form).abs().sum()), 0.0, form)
        self.assertEqual(float(geometry(self.empty_rows()["gci2"], "gci2", neg=True).sum()), 0)

    def test_the_absolute_reading_of_an_empty_box_is_the_old_option(self):
        # empty_box_reading="absolute" (module) = empty_regions="abs" (parts): a phantom box,
        # in both; and it differs from the default reading.
        module = self.empty_box_setup(empty_box_reading="absolute")
        geometry = geometry_of(module)
        self.assertEqual(geometry.objective.empty_regions, "abs")
        default = geometry_of(self.empty_box_setup())
        phantom = 0
        for form, rows in self.empty_rows().items():
            old = getattr(module, form + "_loss")(rows).reshape(-1, 1)
            new = getattr(geometry, form + "_loss")(rows)
            self.assertTrue(th.allclose(old, new, atol=1e-6), (form, old, new))
            phantom += float(not th.allclose(new, getattr(default, form + "_loss")(rows),
                                             atol=1e-6))
        self.assertGreaterEqual(phantom, 3)
        # the reading without the option (the parts' "keep" over the same module) differs
        keep = ComposedGeometry(
            geometry.class_model, geometry.relation_model, geometry.measure,
            StructuralReading(0.05, empty_regions="keep", gci3="overlap", gci3_bot="filler",
                              empty_margin=True, negatives="gci2", strict_negatives=False),
            individuals=geometry.individual_model)
        rows = self.empty_rows()["class_assertion"]
        self.assertFalse(th.allclose(keep.class_assertion_loss(rows),
                                     geometry.class_assertion_loss(rows), atol=1e-6))
        # the builder: the old option is a keyword, the default is signed
        from mowl.representation.library.shapes_methods import prism
        self.assertEqual(prism(N, R, reading="structural").objective.empty_regions, "keep")
        self.assertEqual(prism(N, R, reading="structural",
                               empty_box_reading="absolute").objective.empty_regions, "abs")
        with self.assertRaises(ValueError):
            prism(N, R, reading="structural", empty_box_reading="other")


class TestOtherFamilies(TestCase):

    def octagon_geometry(self, seed=0):
        th.manual_seed(seed)
        classes = OctagonClasses(N, 4)
        table = DefinitionTable.from_normal_forms(
            gci0=GCI0, gci1=GCI1, n_classes=N, k=1, max_components=4, top_idx=TOP,
            bottom_idx=None, enabled=True)
        return ComposedGeometry(DefinedClasses(classes, table), None, OctagonBounds(),
                                StructuralReading(0.0, definitions=False),
                                supported_forms=("gci0", "gci0_bot", "gci1", "gci1_bot"))

    def test_defined_classes_on_octagons_are_the_intersections(self):
        geometry = self.octagon_geometry()
        ids = th.arange(N)
        region = geometry.classes(ids)
        self.assertIsInstance(region, OctagonProduct)
        raw = geometry.class_model.base(ids)
        self.assertTrue(th.equal(region.bounds[3], th.minimum(raw.bounds[0], raw.bounds[1])))
        self.assertTrue(th.equal(region.bounds[0], raw.bounds[0]))
        # 5 = 0 and 2
        self.assertTrue(th.equal(region.bounds[5], th.minimum(raw.bounds[0], raw.bounds[2])))

    def test_the_reading_holds_at_the_intersection_of_octagons(self):
        geometry = self.octagon_geometry()
        # a definition holds without being masked: A in X and A in Y, and X and Y in the
        # conjunction in A
        for row in ([3, 0], [3, 1], [4, 1], [4, 2]):
            self.assertAlmostEqual(float(geometry.gci0_loss(th.tensor([row]))), 0.0, places=6)
        self.assertAlmostEqual(float(geometry.gci1_loss(th.tensor([[0, 1, 3]]))), 0.0, places=6)
        # an unrelated inclusion is not
        self.assertGreater(float(geometry.gci0_loss(th.tensor([[0, 2]]))), 0.0)
        self.assertEqual(size_of(geometry.classes(th.arange(N))).shape, (N,))

    def test_octagons_with_the_reading_are_trained(self):
        geometry = self.octagon_geometry(1)
        rows = {"gci0": th.tensor([[0, 2], [1, 7], [7, 6]]), "gci1": th.tensor([[0, 1, 2]]),
                "gci0_bot": th.tensor([[2, 6]]), "gci1_bot": th.tensor([[0, 7], [1, 2]])}
        targets = ExistenceTargets(classes=th.tensor([0, 1]), pairs=th.tensor([[0, 1]]))
        penalty = ExistencePenalty(targets.classes, targets.pairs, 0.05)

        def total():
            return sum(getattr(geometry, f + "_loss")(r).mean() for f, r in rows.items()) \
                + penalty.penalty(geometry)

        before = float(total())
        optimizer = th.optim.Adam(geometry.parameters(), lr=0.05)
        for _ in range(100):
            optimizer.zero_grad()
            total().backward()
            optimizer.step()
        self.assertLess(float(total()), before)

    def test_defined_classes_on_prisms_without_a_module(self):
        th.manual_seed(2)
        table = DefinitionTable.from_normal_forms(
            gci0=GCI0, gci1=GCI1, n_classes=N, k=1, max_components=4, top_idx=TOP,
            bottom_idx=None, enabled=True)
        geometry = ComposedGeometry(DefinedClasses(PrismClasses(N, D), table), None,
                                    PrismBounds(), StructuralReading(0.0))
        region = geometry.classes(th.arange(N))
        raw = geometry.class_model.base(th.arange(N))
        self.assertTrue(th.equal(region.supports[3], th.minimum(raw.supports[0],
                                                                raw.supports[1])))
        self.assertGreater(float(geometry.gci0_loss(th.tensor([[0, 2]]))), 0.0)

    def test_the_same_definitions_on_plain_boxes(self):
        th.manual_seed(3)
        table = DefinitionTable.from_normal_forms(
            gci0=GCI0, gci1=GCI1, n_classes=N, k=1, max_components=4, top_idx=TOP,
            bottom_idx=None, enabled=True)
        geometry = ComposedGeometry(DefinedClasses(BoxClasses(N, 3), table), None, Bounds(),
                                    StructuralReading(0.0, definitions=False))
        region = geometry.classes(th.arange(N))
        raw = geometry.class_model.base(th.arange(N))
        lower = th.maximum(raw.lower[0], raw.lower[1])
        upper = th.minimum(raw.upper[0], raw.upper[1])
        self.assertTrue(th.allclose(region.centre[3], (lower + upper) / 2))
        self.assertTrue(th.allclose(region.offset[3], (upper - lower) / 2))   # signed
        self.assertAlmostEqual(float(geometry.gci0_loss(th.tensor([[3, 0]]))), 0.0, places=6)
