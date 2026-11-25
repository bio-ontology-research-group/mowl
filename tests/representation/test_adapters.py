"""The geometry of an existing module gives the module's own numbers."""
from unittest import TestCase

import torch as th

from mowl.nn import ELEmModule, ELBEModule
from mowl.representation import geometry_of, ComposedGeometry

N, R, I, D = 12, 3, 5, 6


def _rows(form, n=40, seed=0):
    g = th.Generator().manual_seed(seed)

    def col(high):
        return th.randint(0, high, (n,), generator=g)
    if form in ("gci0", "gci0_bot", "gci1_bot"):
        return th.stack([col(N), col(N)], dim=1)
    if form == "gci1":
        return th.stack([col(N), col(N), col(N)], dim=1)
    if form == "gci2":
        return th.stack([col(N), col(R), col(N)], dim=1)
    if form in ("gci3", "gci3_bot"):
        return th.stack([col(R), col(N), col(N)], dim=1)
    if form == "class_assertion":
        return th.stack([col(I), col(N)], dim=1)
    if form == "object_property_assertion":
        return th.stack([col(I), col(R), col(I)], dim=1)
    raise ValueError(form)


GCIS = ("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot")


class AdapterCase:
    """Shared checks; subclasses give ``make(**kwargs)``."""

    def assertSameLosses(self, module, forms, negs=(False,)):
        geometry = geometry_of(module)
        self.assertIsInstance(geometry, ComposedGeometry)
        for form in forms:
            rows = _rows(form)
            for neg in negs:
                expected = module(rows, form, neg=neg).reshape(-1, 1)
                got = geometry(rows, form, neg=neg)
                self.assertEqual(got.shape, expected.shape, (form, neg))
                self.assertTrue(th.allclose(got, expected, atol=1e-6), (form, neg))

    def test_positive_losses(self):
        self.assertSameLosses(self.make(), GCIS + ("class_assertion",
                                                   "object_property_assertion"))

    def test_original_negatives(self):
        self.assertSameLosses(self.make(), ("gci2", "object_property_assertion"),
                              negs=(True,))

    def test_dele_negatives(self):
        self.assertSameLosses(self.make(all_negative_losses=True), GCIS, negs=(True, False))

    def test_parameters_are_shared(self):
        module = self.make()
        geometry = geometry_of(module)
        self.assertIs(geometry.class_embed, module.class_embed)
        rows = _rows("gci0")
        geometry(rows, "gci0").sum().backward()
        self.assertIsNotNone(module.class_embed.weight.grad)


class TestELEmAdapter(AdapterCase, TestCase):

    def make(self, **kwargs):
        th.manual_seed(1)
        return ELEmModule(N, R, I, embed_dim=D, margin=0.07, **kwargs)

    def test_options(self):
        self.assertSameLosses(self.make(gci1_radius_term=False), ("gci1",))
        module = self.make()
        module.negative_loss_form = "linear"
        self.assertSameLosses(module, ("gci2",), negs=(True,))

    def test_parts(self):
        parts = geometry_of(self.make()).parts
        self.assertEqual((parts["classes"], parts["relations"], parts["measure"],
                          parts["conjunction"]),
                         ("ball", "translation", "centre distance", "approximate"))


class TestELBEAdapter(AdapterCase, TestCase):

    def make(self, **kwargs):
        th.manual_seed(2)
        return ELBEModule(N, R, I, embed_dim=D, margin=0.07, **kwargs)

    def test_options(self):
        self.assertSameLosses(self.make(gci3_form="inclusion"), ("gci3",))

    def test_parts(self):
        parts = geometry_of(self.make()).parts
        self.assertEqual((parts["classes"], parts["relations"], parts["measure"],
                          parts["conjunction"]),
                         ("box", "translation", "bounds per coordinate", "exact"))


class TestUnknown(TestCase):

    def test_unknown_module_says_so(self):
        with self.assertRaisesRegex(NotImplementedError, "ELEmModule"):
            geometry_of(th.nn.Linear(2, 2))
