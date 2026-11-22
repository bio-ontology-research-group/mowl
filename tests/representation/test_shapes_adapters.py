"""Octagon, prism, EIKE and Concept2Box: the geometry of the existing module, and the
method rebuilt from parts, give the module's own losses."""
from unittest import TestCase

import torch as th

from mowl.nn.el.concept2box.module import Concept2BoxModule
from mowl.nn.el.eike.module import EIKEModule
from mowl.nn.el.octagon.module import OctagonELModule
from mowl.nn.el.prism.module import PrismModule
from mowl.representation import ComposedGeometry, METHODS, geometry_of
from tests.representation.test_adapters import _rows, GCIS, N, R, I

D = 6
ALL = GCIS + ("class_assertion", "object_property_assertion")


def _same(case, module, geometry, forms, negs=(False,), rows_of=_rows):
    for form in forms:
        rows = rows_of(form)
        for neg in negs:
            expected = module(rows, form, neg=neg).reshape(-1, 1)
            got = geometry(rows, form, neg=neg)
            case.assertEqual(got.shape, expected.shape, (form, neg))
            case.assertTrue(th.allclose(got, expected, atol=1e-5),
                            (form, neg, float((got - expected).abs().max())))


def _copy(dst, src):
    """Copy parameters between two modules, by shape and order of the given pairs."""
    with th.no_grad():
        for d, s in zip(dst, src):
            d.copy_(s)


class TestOctagon(TestCase):

    def make(self, **kwargs):
        th.manual_seed(3)
        return OctagonELModule(N, R, I, embed_dim=D, margin=0.07, **kwargs)

    def test_adapter_all_forms_positive_and_negative(self):
        module = self.make()
        geometry = geometry_of(module)
        self.assertIsInstance(geometry, ComposedGeometry)
        _same(self, module, geometry, ALL, (False, True))

    def test_options(self):
        module = self.make(disjoint_margin=0.3, neg_margin=0.4)
        _same(self, module, geometry_of(module), ("gci1_bot", "gci2"), (False, True))

    def test_regularisation_equals(self):
        module = self.make(min_width=0.2)
        geometry = geometry_of(module)
        self.assertTrue(th.allclose(module.regularization_loss(),
                                    geometry.regularization_loss(), atol=1e-6))
        self.assertGreater(float(geometry.regularization_loss()), 0)

    def test_rebuilt_from_parts_with_copied_parameters(self):
        module = self.make()
        geometry = METHODS["octagon"](N, R, I, embed_dim=D, margin=0.07)
        cm = geometry.class_model
        _copy([cm.centres.weight, cm.radii.weight, geometry.relation_model.vectors.weight,
               geometry.individual_model.centres.weight],
              [module.class_embed.weight, module.class_radius.weight,
               module.rel_embed.weight, module.ind_embed.weight])
        _same(self, module, geometry, ALL, (False, True))

    def test_parameters_are_shared(self):
        module = self.make()
        geometry = geometry_of(module)
        self.assertIs(geometry.class_embed, module.class_embed)
        geometry(_rows("gci0"), "gci0").sum().backward()
        self.assertIsNotNone(module.class_radius.weight.grad)
        self.assertEqual(geometry.neg_capable_gcis, module.neg_capable_gcis)

    def test_parts(self):
        parts = geometry_of(self.make()).parts
        self.assertEqual((parts["classes"], parts["relations"], parts["measure"],
                          parts["conjunction"]),
                         ("octagon product", "translation", "bounds per coordinate", "exact"))


class TestPrism(TestCase):

    def make(self, **kwargs):
        th.manual_seed(4)
        return PrismModule(N, R, I, embed_dim=D + 1, margin=0.07, **kwargs)

    def test_adapter_all_forms_positive_and_negative(self):
        module = self.make()
        _same(self, module, geometry_of(module), ALL, (False, True))

    def test_margins(self):
        for margin in (0.0, 0.3):
            th.manual_seed(5)
            module = PrismModule(N, R, I, embed_dim=D, margin=margin)
            _same(self, module, geometry_of(module), ALL, (False, True))

    def test_rebuilt_from_parts_with_copied_parameters(self):
        module = self.make()
        geometry = METHODS["prism"](N, R, I, embed_dim=D + 1, margin=0.07)
        cm = geometry.class_model
        _copy([cm.centres.weight, cm.inradii.weight, cm.box_centres.weight,
               cm.box_offsets.weight, geometry.relation_model.vectors.weight,
               geometry.individual_model.centres.weight],
              [module.tri_center.weight, module.tri_inradius.weight,
               module.box_center.weight, module.box_offset.weight, module.rel_embed.weight,
               module.ind_embed.weight])
        _same(self, module, geometry, ALL, (False, True))

    def test_parameters_are_shared_and_parts(self):
        module = self.make()
        geometry = geometry_of(module)
        geometry(_rows("gci2"), "gci2").sum().backward()
        self.assertIsNotNone(module.box_center.weight.grad)
        self.assertEqual(geometry.parts["classes"], "triangle x box")
        self.assertEqual(geometry.neg_capable_gcis, module.neg_capable_gcis)

    def test_structural_module_has_a_geometry_with_its_regions(self):
        # Class 0 is defined as the conjunction of classes 1 and 2: its region is theirs.
        # The losses are those of the parts, not the structural losses of the module.
        from mowl.nn.el.structural import DefinitionTable
        module = self.make(objective="structural")
        table = DefinitionTable.from_normal_forms(
            th.tensor([[0, 1], [0, 2]]), th.tensor([[1, 2, 0]]), N, 1, 4, None, None)
        module.definition_table = table
        geometry = geometry_of(module)
        region = geometry.classes(th.tensor([0]))
        r1, r2 = (geometry.classes(th.tensor([i])) for i in (1, 2))
        self.assertTrue(th.allclose(region.supports, th.minimum(r1.supports, r2.supports)))
        self.assertEqual(geometry(_rows("gci0"), "gci0").shape, (40, 1))


class TestEIKE(TestCase):

    def make(self, **kwargs):
        th.manual_seed(6)
        kwargs.setdefault("alpha", 0.7)
        kwargs.setdefault("margins", (0.3, 0.4, 0.5))
        return EIKEModule(N, R, I, embed_dim=D, **kwargs)

    forms = ("gci0", "class_assertion", "object_property_assertion")

    def test_adapter_both_instance_maps_positive_and_negative(self):
        for instance_map in ("eye", "matrix"):
            module = self.make(instance_map=instance_map)
            _same(self, module, geometry_of(module), self.forms, (False, True))

    def test_text_vectors(self):
        init = th.randn(N, D).numpy()
        init[2] = float("nan")
        module = self.make(intensional_init=init, freeze_intensional=True)
        _same(self, module, geometry_of(module), self.forms, (False, True))

    def test_rebuilt_from_parts_with_copied_parameters(self):
        module = self.make(instance_map="matrix")
        geometry = METHODS["eike"](N, R, I, embed_dim=D, alpha=0.7, margins=(0.3, 0.4, 0.5),
                                   instance_map="matrix")
        c, i = geometry.class_model, geometry.individual_model
        _copy([c.first.centres.weight, c.first.axes_raw.weight, c.first.radius_raw.weight,
               c.second.centres.weight, i.first.centres.weight, i.second.matrix,
               geometry.relation_model.vectors.weight],
              [module.class_center.weight, module.class_axes_raw.weight,
               module.class_radius_raw.weight, module.class_intensional.weight,
               module.ind_embed.weight, module.M, module.rel_embed.weight])
        _same(self, module, geometry, self.forms, (False, True))

    def test_parameters_are_shared(self):
        module = self.make(instance_map="matrix")
        geometry = geometry_of(module)
        loss = geometry(_rows("class_assertion"), "class_assertion").sum()
        loss.backward()
        for table in (module.class_center, module.class_axes_raw, module.class_intensional,
                      module.ind_embed):
            self.assertIsNotNone(table.weight.grad)
        self.assertIsNotNone(module.M.grad)

    def test_other_forms_are_available_through_the_parts(self):
        # Beyond the module: C subclass-of exists r.D reads the ellipsoid of C inside the
        # translated ellipsoid of D; the intensional space takes no part in it.
        geometry = geometry_of(self.make())
        cost = geometry(_rows("gci2"), "gci2")
        self.assertEqual(cost.shape, (40, 1))
        self.assertTrue(bool((cost >= 0).all()))


class TestConcept2Box(TestCase):

    forms = ("gci0", "gci2", "class_assertion", "object_property_assertion")

    def make(self, **kwargs):
        th.manual_seed(7)
        return Concept2BoxModule(N, R, I, embed_dim=D, **kwargs)

    def test_adapter_positive_and_negative(self):
        module = self.make()
        _same(self, module, geometry_of(module), self.forms, (False, True))

    def test_options(self):
        module = self.make(beta=0.5, alpha=0.3, kge_margin=0.6, cross_margin=0.2,
                           instance_weight=0.7, cross_weight=1.9)
        _same(self, module, geometry_of(module), self.forms, (False, True))

    def test_rebuilt_from_parts_with_copied_parameters(self):
        module = self.make()
        geometry = METHODS["concept2box"](N, R, I, embed_dim=D)
        cm = geometry.class_model
        _copy([cm.centres.weight, cm.contexts.weight, geometry.relation_model.vectors.weight,
               geometry.individual_model.centres.weight,
               *[p for p in cm.project.parameters()]],
              [module.class_embed.weight, module.ontology_rel_embed.weight,
               module.rel_embed.weight, module.ind_embed.weight,
               *[p for p in module.project.parameters()]])
        _same(self, module, geometry, self.forms, (False, True))

    def test_parameters_are_shared(self):
        module = self.make()
        geometry = geometry_of(module)
        geometry(_rows("gci2"), "gci2").sum().backward()
        self.assertIsNotNone(module.ontology_rel_embed.weight.grad)
        self.assertIsNotNone(module.project[0].weight.grad)
        self.assertEqual(geometry.neg_capable_gcis, module.neg_capable_gcis)

    def test_parts(self):
        parts = geometry_of(self.make()).parts
        self.assertEqual((parts["classes"], parts["measure"]),
                         ("soft box", "smoothed volume ratio"))
