"""One named part per formula and per meaning: the volume-ratio measures, the union-of-boxes
regions, the head-and-tail-box relation models; no registered alias."""
from unittest import TestCase

import torch as th

from mowl.representation import (Box, CLASS_MODELS, MEASURES, METHODS, RELATION_MODELS,
                                 ConstructorUnavailable)
from mowl.representation.library.boxrel_measures import VolumeShare
from mowl.representation.library.hierarchy_measures import LogVolumeRatio
from mowl.representation.library.multibox_parts import HeadTailRelations, UnionOfBoxes
from mowl.representation.library.other_graph import BoxERelations
from mowl.representation.library.other_regions import MaskedUnionOfBoxes
from mowl.representation.library.shapes_measures import GumbelVolumeRatio


def _boxes():
    return (Box(th.tensor([[0.5, 0.5]]), th.tensor([[0.5, 0.5]])),
            Box(th.tensor([[0.5, 0.25]]), th.tensor([[0.5, 0.25]])))


class TestVolumeRatios(TestCase):

    def test_three_formulas_three_measures(self):
        self.assertIs(MEASURES["volume share"], VolumeShare)
        self.assertIs(MEASURES["log volume ratio"], LogVolumeRatio)
        self.assertIs(MEASURES["gumbel volume ratio"], GumbelVolumeRatio)
        for gone in ("softplus volume ratio", "exact volume ratio", "volume ratio",
                     "smoothed volume ratio"):
            self.assertNotIn(gone, MEASURES)

    def test_each_says_which_methods_use_it(self):
        for cls, methods in ((VolumeShare, ("boxel", "boxsel")),
                             (LogVolumeRatio, ("box_lattice", "gumbel_box", "bcbox")),
                             (GumbelVolumeRatio, ("concept2box",))):
            self.assertIn("Used by", cls.__doc__)
            for method in methods:
                self.assertIn(method, cls.__doc__, (cls.__name__, method))
                self.assertIn(method, METHODS)

    def test_the_formulas_are_different_costs_of_one_pair(self):
        child, parent = _boxes()[1], _boxes()[0]
        share = float(VolumeShare(temperature=None).inside(parent, child))
        log = float(LogVolumeRatio().inside(parent, child))
        gumbel = float(GumbelVolumeRatio(beta=0.001).inside(parent, child))
        # half of the parent lies in the child: 1 - 1/2, -log(1/2), (1 - 1/2)
        self.assertAlmostEqual(share, 0.5, places=5)
        self.assertAlmostEqual(log, 0.6931, places=3)
        self.assertAlmostEqual(gumbel, 0.5, places=2)

    def test_one_log_volume_ratio_for_hard_and_smoothed_boxes(self):
        a, b = _boxes()
        hard, smooth = LogVolumeRatio(), LogVolumeRatio(0.0, 0.0, gumbel_shift=False)
        self.assertFalse(hard.smoothed)
        self.assertTrue(smooth.smoothed)
        self.assertEqual((hard.part, smooth.part), ("volume ratio", "smoothed volume ratio"))
        # the hard limit of the smoothed score is the hard score of intersecting boxes
        self.assertTrue(th.allclose(hard.inside(a, b), smooth.inside(a, b), atol=1e-4))
        with self.assertRaises(ConstructorUnavailable):
            smooth.conditional(a, b)
        with self.assertRaises(ConstructorUnavailable):
            hard.energy(a, b)
        with self.assertRaises(ValueError):
            LogVolumeRatio(volume_temp=0.1)         # a smoothed-box parameter, hard boxes

    def test_the_guards_are_the_corrected_ones(self):
        measure = LogVolumeRatio(0.01)
        self.assertEqual(measure.energy_floor, 1e-12)
        self.assertTrue(LogVolumeRatio().zero_at_inclusion)


class TestNoAliases(TestCase):

    def test_registered_names_are_parts(self):
        for gone in ("shift and widen", "entity point", "point with bump"):
            self.assertNotIn(gone, CLASS_MODELS)
            self.assertNotIn(gone, RELATION_MODELS)
        self.assertIn("box of translations", RELATION_MODELS)
        self.assertIn("point", CLASS_MODELS)


class TestUnionsAndHeadTail(TestCase):

    def test_two_unions_of_boxes_name_their_difference(self):
        self.assertIsNot(UnionOfBoxes, MaskedUnionOfBoxes)
        self.assertIn("MaskedUnionOfBoxes", UnionOfBoxes.__doc__)
        self.assertIn("UnionOfBoxes", MaskedUnionOfBoxes.__doc__)
        self.assertIn("inactive", MaskedUnionOfBoxes.__doc__)

    def test_two_head_and_tail_box_relation_models_name_their_difference(self):
        self.assertIs(RELATION_MODELS["head and tail boxes"], HeadTailRelations)
        self.assertIs(RELATION_MODELS["head and tail boxes (distance score)"],
                      BoxERelations)
        self.assertIn("distance score", HeadTailRelations.__doc__)
        self.assertIn("containment", BoxERelations.__doc__)

    def test_the_boxe_method_uses_the_boxe_score(self):
        g = METHODS["boxe"](5, 2, 0, embed_dim=4)
        self.assertIsInstance(g.relation_model.relation_model
                              if hasattr(g.relation_model, "relation_model")
                              else g.relation_model, BoxERelations)
