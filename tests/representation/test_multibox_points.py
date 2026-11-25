"""The point objectives of MultiBoxEL and MultiELBE (``point``, ``mixed``, ``hybrid``) from
parts: the soft membership measure, the point terms, the reading that switches the forms, and
the point loss as a penalty. They give the losses of the modules and work with the parts of
other methods."""
from unittest import TestCase

import numpy as np
import torch as th

import mowl
mowl.init_jvm("1g")

from mowl.datasets.el.points import PointSupervision
from mowl.baselines.nn.el.multiboxel.module import MultiBoxELModule
from mowl.baselines.nn.el.multielbe.module import MultiELBEModule
from mowl.baselines.nn.el.multiboxel.pointloss import volume_penalty
from mowl.representation import Ball, Box, Objective, compose, geometry_of, MEASURES
from mowl.representation.library.structural_points import (PointPenalty, PointReading,
                                                           PointTerms, SoftMembership)

N, R, I, D = 6, 2, 3, 3
ROWS = {"gci0": th.tensor([[0, 1], [2, 3]]), "gci0_bot": th.tensor([[0, 5]]),
        "gci1": th.tensor([[0, 1, 2]]), "gci1_bot": th.tensor([[0, 1]]),
        "gci2": th.tensor([[0, 1, 2]]), "gci3": th.tensor([[1, 2, 3]]),
        "gci3_bot": th.tensor([[1, 2]]), "class_assertion": th.tensor([[0, 3], [1, 2]]),
        "object_property_assertion": th.tensor([[0, 1, 2]])}


def supervision():
    E = 5       # elements 0..3 stand for classes, 4 for the individual 1
    return PointSupervision(
        n_elements=E, element_names=[f"e{i}" for i in range(E)],
        element_class_index=th.tensor([0, 1, 2, 3, 1]),
        element_is_individual=th.tensor([False] * 4 + [True]),
        pos_pairs=th.tensor([[0, 0], [0, 1], [1, 1], [2, 2], [3, 5], [4, 2]]),
        hard_neg_pairs=th.tensor([[0, 2], [1, 3], [2, 0]]),
        role_pairs=th.tensor([[0, 1, 1], [2, 0, 3], [1, 0, 4], [4, 1, 0]]))


BATCH = {"pos": supervision().pos_pairs, "hard": supervision().hard_neg_pairs,
         "rand": th.tensor([[0, 3], [1, 4]]), "roles": supervision().role_pairs}


def make(cls, mode, seed=1, **options):
    th.manual_seed(seed)
    if cls is MultiELBEModule:
        module = cls(N, R, I, embed_dim=D, n_components=2, objective=mode, beta=0.2,
                     neg_weight=0.3, margin=0.1, **options)
    else:
        module = cls(N, R, I, embed_dim=D, n_components=2, objective=mode, beta=0.2,
                     neg_weight=0.3, gamma=0.05, **options)
    module.set_point_supervision(supervision())
    with th.no_grad():
        for p in module.parameters():
            p.add_(0.3 * th.randn_like(p))
    return module


class TestPointObjectives(TestCase):

    def test_losses_on_sampled_points_equal_the_modules(self):
        for cls in (MultiBoxELModule, MultiELBEModule):
            for mode in ("point", "mixed", "hybrid"):
                module = make(cls, mode)
                geometry = geometry_of(module)
                terms = geometry.point_terms
                self.assertIsInstance(geometry.objective, PointReading)
                for key in ("pos", "hard", "rand", "roles"):
                    one = {key: BATCH[key]}
                    old = module.point_membership_loss(one) + module.point_role_loss(one)
                    new = terms.membership_loss(geometry, one) + terms.role_loss(geometry, one)
                    self.assertTrue(th.allclose(old, new, atol=1e-6), (cls, mode, key))
                old = module.point_membership_loss(BATCH)
                new = terms.membership_loss(geometry, BATCH)
                self.assertTrue(th.allclose(old, new, atol=1e-6))
                self.assertGreater(float(old), 0)
                old = module.point_role_loss(BATCH)
                new = terms.role_loss(geometry, BATCH)
                self.assertTrue(th.allclose(old, new, atol=1e-6))
                self.assertGreater(float(old), 0)
                pairs = BATCH["pos"]
                for neg in (False, True):
                    self.assertTrue(th.allclose(
                        module.point_membership_loss(pairs, neg=neg),
                        terms.membership_loss(geometry, pairs, neg=neg), atol=1e-6))
                    self.assertTrue(th.allclose(
                        module.point_role_loss(BATCH["roles"], neg=neg),
                        terms.role_loss(geometry, BATCH["roles"], neg=neg), atol=1e-6))

    def test_forms_in_training_and_evaluation_mode(self):
        for cls in (MultiBoxELModule, MultiELBEModule):
            for mode in ("point", "mixed", "hybrid"):
                module = make(cls, mode)
                geometry = geometry_of(module)
                nonzero = 0
                for training in (True, False):
                    module.train(training)
                    geometry.train(training)
                    for form, rows in ROWS.items():
                        for neg in (False, True):
                            try:
                                old = getattr(module, form + "_loss")(rows, neg=neg)
                            except NotImplementedError:
                                # a module refuses a negative loss it does not have: so
                                # does the geometry
                                with self.assertRaises(NotImplementedError):
                                    getattr(geometry, form + "_loss")(rows, neg=neg)
                                continue
                            old = old.reshape(-1, 1)
                            new = getattr(geometry, form + "_loss")(rows, neg=neg)
                            self.assertTrue(th.allclose(old, new, atol=1e-6),
                                            (cls, mode, training, form, neg))
                            nonzero += float(old.abs().sum() > 0)
                self.assertGreater(nonzero, 10)

    def test_the_switches_are_the_modules(self):
        # point: every concept form is zero in training; mixed: the forms without bottom;
        # hybrid: none. The class assertion is the point score in every mode and every state.
        zero = {"point": {"gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
                          "object_property_assertion"},
                "mixed": {"gci0", "gci1", "gci2", "gci3", "object_property_assertion"},
                "hybrid": set()}
        for mode, forms in zero.items():
            geometry = geometry_of(make(MultiBoxELModule, mode))
            geometry.train()
            for form, rows in ROWS.items():
                loss = getattr(geometry, form + "_loss")(rows)
                self.assertEqual(bool((loss == 0).all()), form in forms, (mode, form))

    def test_volume_and_the_penalty(self):
        module = make(MultiBoxELModule, "point", volume_reg=0.4)
        geometry = geometry_of(module)
        terms = geometry.point_terms
        self.assertAlmostEqual(terms.volume_reg, 0.4)
        terms.volume_classes = [0, 1, 2, 3]
        _, offsets = module.class_boxes(th.tensor([0, 1, 2, 3]))
        self.assertTrue(th.allclose(terms.volume(geometry), volume_penalty(offsets)))
        expected = module.point_membership_loss(BATCH) + module.point_role_loss(BATCH) \
            + 0.4 * volume_penalty(offsets)
        self.assertTrue(th.allclose(terms.loss(geometry, BATCH), expected, atol=1e-6))
        penalty = PointPenalty(terms, sampler=lambda call: BATCH)
        self.assertTrue(th.allclose(penalty.penalty(geometry), expected, atol=1e-6))
        # the sampling of the model: a generator seeded by the epoch
        penalty = PointPenalty(terms, points_per_epoch=6, seed=11)
        rng = np.random.default_rng(11)
        batch = supervision().sample(6, rng=rng)
        expected = module.point_membership_loss(batch) + module.point_role_loss(batch) \
            + 0.4 * volume_penalty(offsets)
        self.assertTrue(th.allclose(penalty.penalty(geometry), expected, atol=1e-6))

    def test_parameters_are_shared_with_the_module(self):
        module = make(MultiBoxELModule, "point")
        geometry = geometry_of(module)
        geometry.point_terms.loss(geometry, BATCH).backward()
        for table in (module.elem_embed, module.class_center, module.head_center,
                      module.bump_classes):
            self.assertIsNotNone(table.weight.grad)
            self.assertGreater(float(table.weight.grad.abs().sum()), 0)

    def test_built_from_parts_without_a_module(self):
        from mowl.representation import METHODS
        for key in ("multiboxel", "multielbe"):
            geometry = METHODS[key](N, R, I, embed_dim=D, n_components=2, reading="mixed",
                                    supervision=supervision(), beta=0.2, neg_weight=0.3)
            self.assertGreater(float(geometry.point_terms.loss(geometry, BATCH)), 0)
            self.assertIn("elem_embed", dict(geometry.point_terms.named_children()))


class TestSoftMembershipWithOtherParts(TestCase):

    def test_measure_is_registered_and_works_for_balls_and_boxes(self):
        self.assertIn("soft membership", MEASURES)
        measure = SoftMembership(beta=0.1)
        centre = th.zeros(2, 3)
        ball = Ball(centre, th.tensor([[1.0], [0.1]]))
        point = th.tensor([[0.5, 0.0, 0.0]] * 2)
        m = measure.probability(point, ball)
        self.assertAlmostEqual(float(m[0]), float(th.sigmoid(th.tensor(5.0))), places=5)
        self.assertLess(float(m[1]), 0.05)
        box = Box(centre, th.tensor([[1.0] * 3, [0.1] * 3]))
        m = measure.probability(point, box)
        self.assertGreater(float(m[0]), 0.99)
        self.assertLess(float(m[1]), 0.05)
        # cost of a member, cost of a non-member
        self.assertTrue((measure.member(point, ball)[0] < measure.member(point, ball)[1]).all())
        self.assertTrue((measure.apart(point, ball)[1] < measure.apart(point, ball)[0]).all())

    def test_membership_loss_trains_balls(self):
        # the point terms with the classes of ELEmbeddings: the loss on sampled points falls
        th.manual_seed(0)
        geometry = compose(N, R, I, embed_dim=D, classes="ball", relations=None,
                           measure=SoftMembership(0.2), objective=Objective(0.0),
                           individuals=False)
        terms = PointTerms(supervision(), D, SoftMembership(0.2))
        geometry.point_terms = terms
        pairs = {"pos": supervision().pos_pairs[:4], "hard": supervision().hard_neg_pairs}
        before = float(terms.membership_loss(geometry, pairs))
        optimizer = th.optim.Adam(geometry.parameters(), lr=0.05)
        for _ in range(80):
            optimizer.zero_grad()
            terms.membership_loss(geometry, pairs).backward()
            optimizer.step()
        self.assertLess(float(terms.membership_loss(geometry, pairs)), before)

    def test_the_point_reading_with_plain_boxes_and_translations(self):
        th.manual_seed(0)
        terms = PointTerms(supervision(), D, SoftMembership(0.2))
        geometry = compose(N, R, I, embed_dim=D, classes="box", relations="translation",
                           objective=PointReading(terms, "mixed", 0.05))
        geometry.point_terms = terms
        geometry.train()
        self.assertTrue((geometry.gci0_loss(ROWS["gci0"]) == 0).all())          # switched off
        self.assertTrue((geometry.gci1_bot_loss(ROWS["gci1_bot"]) >= 0).all())  # not
        self.assertGreater(float(geometry.class_assertion_loss(ROWS["class_assertion"]).sum()),
                           0)
        before = float(terms.role_loss(geometry, BATCH))
        optimizer = th.optim.Adam(geometry.parameters(), lr=0.05)
        for _ in range(80):
            optimizer.zero_grad()
            (terms.role_loss(geometry, BATCH) + terms.membership_loss(geometry, BATCH)
             ).backward()
            optimizer.step()
        self.assertLess(float(terms.role_loss(geometry, BATCH)), before)
        self.assertIsInstance(geometry.classes(th.tensor([0])), Box)
