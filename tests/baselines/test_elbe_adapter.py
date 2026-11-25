"""The geometry of the direct ELBE module (the baseline) gives the module's numbers, with the
module's parameters shared. (The module users get, ``mowl.nn.ELBEModule``, is a geometry
itself; ``geometry_of`` describes the direct module and the modules of earlier releases.)"""
from unittest import TestCase

import torch as th

from mowl.baselines.nn.el.elbe.module import ELBEModule
from mowl.representation import geometry_of
from tests.representation.test_adapters import AdapterCase, D, I, N, R


class TestDirectELBEAdapter(AdapterCase, TestCase):

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
