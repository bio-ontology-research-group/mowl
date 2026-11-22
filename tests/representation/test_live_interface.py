"""What the geometry of an existing module shares with it: the margin (live, in both
directions), the raw parameter tables, and the answer to "is this region empty"."""
from unittest import TestCase

import torch as th

from mowl.baselines.nn.el.boxel.module import BoxELModule   # the direct module has no margin
from mowl.nn import (ELBEModule, ELEmModule, ELKEmModule, OctagonELModule,
                     PrismModule, TransBoxModule)
from mowl.nn.el.regd import RegDELEmModule
from mowl.representation import (Ball, Box, CornerBox, ConstructorUnavailable,
                                 geometry_of)
from mowl.representation.library.boxrel_boxes import FixableBoxes
from mowl.representation.library.multibox_parts import UnionOfBoxes
from mowl.representation.library.shapes_regions import OctagonProduct, Prism

N, R, I, D = 12, 3, 5, 6


def _rows(form, seed=0):
    g = th.Generator().manual_seed(seed)
    high = {"c": N, "r": R, "i": I}
    columns = {"gci0": "cc", "gci1": "ccc", "gci2": "crc", "gci3": "rcc",
               "gci0_bot": "cc", "class_assertion": "ic"}[form]
    return th.stack([th.randint(0, high[k], (30,), generator=g) for k in columns], dim=1)


def _modules():
    th.manual_seed(0)
    return {
        "ELEm": ELEmModule(N, R, I, embed_dim=D, margin=0.1),
        "ELBE": ELBEModule(N, R, I, embed_dim=D, margin=0.1),
        "RegD-ELEm": RegDELEmModule(N, R, I, embed_dim=D, margin=0.1),
        "TransBox": TransBoxModule(N, R, I, embed_dim=D, margin=0.1),
        "Octagon": OctagonELModule(N, R, I, embed_dim=2 * (D // 2), margin=0.1),
        "Prism": PrismModule(N, R, I, embed_dim=D + 1, margin=0.1),
    }


class TestLiveMargin(TestCase):

    def assertSameLosses(self, module, geometry, message):
        for form in ("gci0", "gci2", "gci3"):
            rows = _rows(form)
            expected = module(rows, form).reshape(-1, 1)
            self.assertTrue(th.allclose(geometry(rows, form), expected, atol=1e-5),
                            (message, form))

    def test_the_margin_is_one_number_in_both_directions(self):
        for name, module in _modules().items():
            geometry = geometry_of(module)
            self.assertEqual(geometry.margin, module.margin, name)
            if module.margin_semantics is not None:   # OctagonELModule declares none
                self.assertEqual(geometry.margin_semantics, module.margin_semantics, name)
            forms = ("gci0", "gci1", "gci2", "gci3")
            before = [geometry(_rows(f), f).detach().clone() for f in forms]
            geometry.margin = 0.37                       # through the geometry
            self.assertEqual(module.margin, 0.37, name)
            self.assertEqual(geometry.objective.margin, 0.37, name)
            self.assertSameLosses(module, geometry, name)
            after = [geometry(_rows(f), f) for f in forms]
            self.assertTrue(any(not th.allclose(a, b) for a, b in zip(after, before)), name)
            module.margin = 0.02                         # through the module
            self.assertEqual(geometry.margin, 0.02, name)
            self.assertSameLosses(module, geometry, name)
            module.margin = -0.2                         # the sign is the sense, as in
            self.assertEqual(geometry.margin, -0.2, name)  # mowl.training.margin
            self.assertSameLosses(module, geometry, name)

    def test_a_training_regime_that_sets_the_module_margin_changes_the_geometry(self):
        from mowl.training.margin import SlackMargin, StrictMargin

        class Model:
            pass
        for name, module in _modules().items():
            if module.margin_semantics is None:
                continue          # a regime cannot set the margin of that module by meaning
            geometry = geometry_of(module)
            model = Model()
            model.module = module
            regime = SlackMargin(0.15) if module.margin_semantics == "slack" \
                else StrictMargin(0.15)
            regime.attach(model)
            self.assertEqual(geometry.margin, 0.15, name)
            self.assertSameLosses(module, geometry, name)

    def test_a_geometry_of_its_own_keeps_its_own_margin(self):
        from mowl.representation import METHODS
        g = METHODS["elbe"](N, R, I, embed_dim=D, margin=0.2)
        g.margin = 0.4
        self.assertEqual(g.objective.margin, 0.4)

    def test_modules_without_a_margin_are_not_linked(self):
        module = BoxELModule(N, R, I, embed_dim=D)
        geometry = geometry_of(module)
        geometry.margin = 0.3
        self.assertFalse(hasattr(module, "margin"))
        self.assertEqual(geometry.margin, 0.3)


class TestRawTables(TestCase):

    def test_the_tables_are_the_modules_parameters(self):
        module = _modules()["ELEm"]
        tables = geometry_of(module).tables()
        self.assertIs(tables["classes"]["centres"], module.class_embed.weight)
        self.assertIs(tables["classes"]["radii"], module.class_rad.weight)
        self.assertIs(tables["relations"]["vectors"], module.rel_embed.weight)
        self.assertIs(tables["individuals"]["centres"], module.ind_embed.weight)
        module = _modules()["ELBE"]
        tables = geometry_of(module).tables()
        self.assertIs(tables["classes"]["offsets"], module.class_offset.weight)

    def test_a_class_model_gives_its_tables_directly(self):
        g = geometry_of(_modules()["TransBox"])
        self.assertEqual(set(g.class_model.tables()), {"centres", "offsets"})
        self.assertEqual(set(g.relation_model.tables()), {"vectors", "offsets"})
        self.assertIs(g.class_model.tables()["offsets"],
                      g.class_model.offsets.weight)

    def test_a_geometry_without_individuals_or_relations_leaves_them_out(self):
        from mowl.representation import METHODS
        g = METHODS["box_lattice"](N, embed_dim=3)
        self.assertEqual(set(g.tables()), {"classes"})


class TestEmptyRegions(TestCase):

    def test_boxes(self):
        box = Box(th.zeros(3, 2), th.tensor([[1., 1], [0, 0], [-1, 1]]))
        self.assertEqual(box.is_empty().tolist(), [False, False, True])
        corners = CornerBox.from_corners(th.tensor([[0., 0], [2, 0]]),
                                         th.tensor([[1., 1], [1, 1]]))
        self.assertEqual(corners.is_empty().tolist(), [False, True])

    def test_balls_are_never_empty(self):
        self.assertFalse(bool(Ball(th.zeros(2, 3), th.zeros(2, 1)).is_empty().any()))

    def test_the_fixed_classes_of_elkem(self):
        th.manual_seed(1)
        module = ELKEmModule(6, 2, 2, embed_dim=3)
        module.bot_mask[1] = True
        module.top_mask[2] = True
        geometry = geometry_of(module)
        regions = geometry.classes(th.arange(6))
        self.assertEqual(regions.is_empty().tolist(), [False, True, False, False, False, False])
        # the full box is as big as the number says and is not empty
        self.assertGreater(float(regions.offset[2].min()), 1e5)
        fixable = FixableBoxes(4, 2)
        fixable.fix([0], [3])
        self.assertEqual(fixable(th.arange(4)).is_empty().tolist(), [True, False, False, False])

    def test_unions_prisms_octagons(self):
        union = UnionOfBoxes(th.zeros(2, 2, 3), th.tensor([[[1.] * 3, [1.] * 3],
                                                          [[-1.] * 3, [-1.] * 3]]))
        self.assertEqual(union.is_empty().tolist(), [False, True])
        prism = Prism(th.tensor([[1., 1, 1], [-1, -1, -1], [1, 1, 1]]), th.zeros(3, 2),
                      th.tensor([[1., 1], [1, 1], [-1, 1]]))
        self.assertEqual(prism.is_empty().tolist(), [False, True, True])
        point = OctagonProduct.of_point(th.zeros(2, 4))
        self.assertEqual(point.is_empty().tolist(), [False, False])
        empty = OctagonProduct(-th.ones(1, 2, 8))
        self.assertEqual(empty.is_empty().tolist(), [True])

    def test_a_family_that_cannot_say_is_an_error(self):
        from mowl.representation.library.logic_fuzzy import FuzzySet, Product
        with self.assertRaises(ConstructorUnavailable):
            FuzzySet(th.rand(2, 4), Product()).is_empty()
