"""The structural objective of MultiBoxEL rebuilt from parts.

* the structural reading (``StructuralReading``), slots of boxes (``SlotClasses``), the
  definitions as a class model wrapper (``DefinedClasses``), the measure that knows the mask
  (``SlotMeasure``) and the three penalties give the losses of ``MultiBoxELModule`` with
  ``objective="structural"``, positive and negative, for every form;
* the parts work with the parts of other methods (plain boxes, a ``ComposedEL``).
"""
import itertools
from unittest import TestCase

import numpy as np
import torch as th

import mowl
mowl.init_jvm("1g")

from mowl.models import ComposedEL
from mowl.baselines.nn.el.multiboxel.module import MultiBoxELModule
from mowl.nn.el.structural import (DefinitionTable, ExistenceTargets, WitnessTargets,
                                   build_structural_tables)
from mowl.representation import (Bounds, BoxClasses, ComposedGeometry, METHODS, geometry_of)
from mowl.representation.library.structural_methods import StructuralGeometry
from mowl.representation.library.structural_reading import (ExistencePenalty, StructuralReading,
                                                            WitnessPenalty, size_of)
from mowl.representation.library.structural_regions import DefinedClasses, SlotBoxes
from mowl.training import IdleComponentReseeding
from tests.datasetFactory import FamilyDataset

N, R, I, D = 8, 2, 3, 3
GCI0 = th.tensor([[3, 0], [3, 1], [4, 1], [4, 2], [5, 0], [5, 2], [3, 6], [7, 0]])
GCI1 = th.tensor([[0, 1, 3], [1, 2, 4], [0, 2, 5]])
TOP = 6

ROWS = {
    "gci0": th.tensor([[3, 0], [0, 1], [7, 0], [4, 2], [5, 6], [1, 2]]),
    "gci0_bot": th.tensor([[3, 6], [0, 6], [5, 6], [7, 6]]),
    "gci1": th.tensor([[0, 1, 3], [1, 2, 4], [3, 4, 7], [0, 2, 6], [7, 0, 1]]),
    "gci1_bot": th.tensor([[3, 4], [0, 1], [5, 7], [2, 2]]),
    "gci2": th.tensor([[3, 0, 4], [0, 1, 5], [7, 1, 1], [2, 0, 6]]),
    "gci3": th.tensor([[0, 3, 4], [1, 5, 0], [1, 7, 3], [0, 0, 2]]),
    "gci3_bot": th.tensor([[0, 3], [1, 4]]),
    "class_assertion": th.tensor([[0, 3], [1, 5], [2, 7], [1, 0]]),
    "object_property_assertion": th.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
}


def perturbed(module, seed=0, scale=0.3):
    g = th.Generator().manual_seed(seed)
    with th.no_grad():
        for p in module.parameters():
            p.add_(scale * th.randn(p.shape, generator=g, dtype=p.dtype))
    return module


def tables(k, max_components):
    table = DefinitionTable.from_normal_forms(
        gci0=GCI0, gci1=GCI1, n_classes=N, k=k, max_components=max_components, top_idx=TOP,
        bottom_idx=None, enabled=True)
    witness = WitnessTargets(
        classes=th.tensor([0, 1, 2]),
        pos=th.tensor([[0, 0], [1, 1], [2, 2], [0, 3], [1, 4]]),
        neg=th.tensor([[0, 2], [1, 0], [2, 1]]))
    targets = ExistenceTargets(classes=th.tensor([0, 1, 2, 3, 4, 5, 7]),
                               pairs=th.tensor([[0, 1], [1, 2], [0, 2], [3, 4]]),
                               witness=witness)
    return table, targets


def structural_module(k=2, max_components=4, seed=0, set_tables=True, **options):
    th.manual_seed(seed)
    options.setdefault("margin", 0.05)
    options.setdefault("witness_weight", 0.5)
    options.setdefault("size_reg", 1e-2)
    module = MultiBoxELModule(N, R, I, embed_dim=D, n_components=k, objective="structural",
                              max_components=max_components, **options)
    if set_tables:
        table, targets = tables(k, max_components)
        module.set_structural_tables(table, targets, top_idx=TOP)
    return perturbed(module, seed + 1)


class TestStructuralMultiBoxEL(TestCase):

    def assert_same_losses(self, module, geometry, forms=None, negs=(False, True)):
        compared = 0
        for form, rows in ROWS.items():
            if forms is not None and form not in forms:
                continue
            for neg in negs:
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
                    compared += float(old.abs().sum() > 0)
                    continue
                new = getattr(geometry, form + "_loss")(rows, neg=neg)
                self.assertEqual(new.shape, old.shape, (form, neg))
                self.assertTrue(th.allclose(old, new, atol=1e-6), (form, neg, old, new))
                compared += float(old.abs().sum() > 0)
        return compared

    def test_every_form_positive_and_negative(self):
        for k, slots, margin, agg in itertools.product((1, 2), (4, 8), (0.0, 0.05),
                                                       ("mean", "sum")):
            if slots < k:
                continue
            module = structural_module(k, slots, margin=margin, bot_agg=agg)
            geometry = geometry_of(module)
            self.assertIsInstance(geometry, StructuralGeometry)
            compared = self.assert_same_losses(module, geometry)
            # the comparison is not between zeros
            self.assertGreater(compared, 10, (k, slots, margin, agg))

    def test_margin_is_the_modules(self):
        module = structural_module()
        geometry = geometry_of(module)
        module.margin = 0.2
        self.assertAlmostEqual(geometry.margin, 0.2)
        self.assert_same_losses(module, geometry)

    def test_without_the_definition_rows_masked(self):
        module = structural_module(structural_definitions=True)
        geometry = geometry_of(module)
        # the rows that state a definition are zero ...
        self.assertTrue((geometry.gci0_loss(th.tensor([[3, 0], [3, 1], [4, 1]])) == 0).all())
        self.assertTrue((geometry.gci1_loss(th.tensor([[0, 1, 3], [1, 2, 4]])) == 0).all())
        # ... and not where the reading is told not to skip them
        geometry.objective.definitions = False
        self.assertTrue((geometry.gci0_loss(th.tensor([[3, 0], [3, 1]])) >= 0).all())
        self.assertGreater(float(geometry.gci1_loss(th.tensor([[0, 1, 3], [1, 2, 4]])).sum()),
                           0.0)

    def test_defined_classes_are_the_intersections_of_their_definers(self):
        module = structural_module()
        geometry = geometry_of(module)
        ids = th.arange(N)
        old = module.class_regions(ids)
        new = geometry.classes(ids)
        self.assertIsInstance(new, SlotBoxes)
        self.assertTrue(th.equal(old.cen, new.centre))
        self.assertTrue(th.equal(old.w, new.w))
        self.assertTrue(th.equal(old.active, new.active))
        # defined: the regions of 3 = 0 and 1 are not the row of 3
        self.assertFalse(th.equal(new.w[3, :2], module.class_boxes(th.tensor([3]))[1][0]))
        # the bump vector is the class's own
        self.assertTrue(th.equal(new.bump, module.bump_classes(ids)))

    def test_regularisation_is_the_modules_plus_the_structural_extra_loss(self):
        for options in ({}, {"existence_weight": 0.0}, {"size_reg": 0.0},
                        {"witness_weight": 0.0}, {"component_reg": 0.3, "diversity_reg": 0.2},
                        {"existence_weight": 2.5, "witness_weight": 1.5, "size_reg": 0.1}):
            module = structural_module(**options)
            geometry = geometry_of(module)
            expected = module.regularization_loss() + module.structural_extra_loss()
            self.assertTrue(th.allclose(geometry.regularization_loss(), expected, atol=1e-6),
                            (options, geometry.regularization_loss(), expected))
            self.assertGreater(float(expected), 0.0)

    def test_the_penalties_are_regularizers_of_their_own(self):
        module = structural_module()
        geometry = geometry_of(module)
        kinds = {type(r).__name__ for r in geometry.regularizers}
        self.assertTrue({"ExistencePenalty", "SizePenalty", "WitnessPenalty"} <= kinds)
        existence = next(r for r in geometry.regularizers if isinstance(r, ExistencePenalty))
        self.assertGreater(float(existence.penalty(geometry)), 0.0)
        with th.no_grad():                        # a class of size at least the margin is free
            module.class_offset.weight.fill_(2.0)
            module.class_center.weight.zero_()
        self.assertAlmostEqual(float(existence.penalty(geometry)), 0.0, places=6)

    def test_without_tables_there_is_only_the_size_penalty(self):
        module = structural_module(set_tables=False)
        geometry = geometry_of(module)
        self.assertTrue(th.allclose(
            geometry.regularization_loss(),
            module.regularization_loss() + module.structural_extra_loss(), atol=1e-6))
        self.assertNotIn("ExistencePenalty", {type(r).__name__ for r in geometry.regularizers})
        self.assert_same_losses(module, geometry)

    def test_size_penalty_without_tables_default_and_old_option(self):
        # default (module and parts): no size term before the tables say what owl:Thing and
        # the defined classes are; size_reg_without_tables=True (old): every class is shrunk.
        default = structural_module(set_tables=False)
        self.assertFalse(default.size_reg_without_tables)
        geometry = geometry_of(default)
        self.assertFalse(geometry.size_reg_without_tables)
        self.assertEqual(float(geometry.regularization_loss()
                               - default.regularization_loss()), 0.0)
        old = structural_module(set_tables=False, size_reg_without_tables=True)
        old_geometry = geometry_of(old)
        self.assertTrue(old_geometry.size_reg_without_tables)
        self.assertGreater(float(old.structural_extra_loss()), 0.0)
        self.assertTrue(th.allclose(
            old_geometry.regularization_loss(),
            old.regularization_loss() + old.structural_extra_loss(), atol=1e-6))
        self.assertGreater(float(old_geometry.regularization_loss()
                                 - geometry.regularization_loss()), 0.0)
        # with tables installed the two options agree, in module and parts
        for flag in (False, True):
            module = structural_module(size_reg_without_tables=flag)
            self.assertTrue(th.allclose(
                geometry_of(module).regularization_loss(),
                module.regularization_loss() + module.structural_extra_loss(), atol=1e-6))
        # the builder: the default is False
        self.assertFalse(METHODS["multiboxel"](N, R, reading="structural")
                         .size_reg_without_tables)
        self.assertTrue(METHODS["multiboxel"](N, R, reading="structural",
                                              size_reg_without_tables=True)
                        .size_reg_without_tables)

    def test_parameters_are_shared(self):
        module = structural_module()
        geometry = geometry_of(module)
        (geometry.gci2_loss(ROWS["gci2"]).sum() + geometry.regularization_loss()).backward()
        for table in (module.class_center, module.class_offset, module.bump_classes,
                      module.head_center, module.witness_embed):
            self.assertIsNotNone(table.weight.grad)
            self.assertGreater(float(table.weight.grad.abs().sum()), 0.0)

    def test_witness_points_and_effective_components(self):
        module = structural_module()
        geometry = geometry_of(module)
        self.assertTrue(th.equal(geometry.witness_points(), module.witness_points()))
        self.assertTrue(th.equal(geometry.effective_components(), module.effective_components()))

    def test_built_from_parts_with_fresh_tables(self):
        module = structural_module()
        # the builder gives the module's earlier reading (a positive loss for a negative of a
        # form without one); the module itself refuses
        module.strict_negatives = False
        geometry = METHODS["multiboxel"](N, R, I, embed_dim=D, margin=0.05, n_components=2,
                                         reading="structural", max_components=4,
                                         existence_weight=1.0, size_reg=1e-2,
                                         witness_weight=0.5)
        table, targets = tables(2, 4)
        geometry.set_structural_tables(table, targets, TOP)
        with th.no_grad():
            cm = geometry.class_model.base
            cm.centres.weight.copy_(module.class_center.weight)
            cm.offsets.weight.copy_(module.class_offset.weight)
            cm.bumps.weight.copy_(module.bump_classes.weight)
            rm = geometry.relation_model
            rm.head_centres.weight.copy_(module.head_center.weight)
            rm.head_offsets.weight.copy_(module.head_offset.weight)
            rm.tail_centres.weight.copy_(module.tail_center.weight)
            rm.tail_offsets.weight.copy_(module.tail_offset.weight)
            im = geometry.individual_model
            im.centres.weight.copy_(module.ind_center.weight)
            im.offsets.weight.copy_(module.ind_offset.weight)
            im.bumps.weight.copy_(module.bump_individuals.weight)
            geometry.witness_embed.weight.copy_(module.witness_embed.weight)
        self.assertGreater(self.assert_same_losses(module, geometry), 10)
        self.assertTrue(th.allclose(
            geometry.regularization_loss(),
            module.regularization_loss() + module.structural_extra_loss(), atol=1e-6))

    def test_soft_aggregation_is_refused(self):
        with self.assertRaises(ValueError):
            METHODS["multiboxel"](N, R, I, embed_dim=D, reading="structural",
                                  component_aggregation="soft")


class TestTheReadingWithOtherParts(TestCase):
    """The structural reading and the definitions work for the classes of other methods."""

    def test_plain_boxes_equal_the_module_with_one_slot(self):
        module = structural_module(k=1, max_components=1, margin=0.05, witness_weight=0.0)
        classes = BoxClasses(centres=module.class_center, offsets=module.class_offset)
        table, _ = tables(1, 1)
        geometry = ComposedGeometry(DefinedClasses(classes, table), None, Bounds(),
                                    StructuralReading(0.05), supported_forms=(
                                        "gci0", "gci0_bot", "gci1", "gci1_bot"))
        for form in ("gci0", "gci0_bot", "gci1", "gci1_bot"):
            old = getattr(module, form + "_loss")(ROWS[form]).reshape(-1, 1)
            new = getattr(geometry, form + "_loss")(ROWS[form])
            self.assertTrue(th.allclose(old, new, atol=1e-6), (form, old, new))
        self.assertGreater(float(geometry.gci0_loss(ROWS["gci0"]).sum()), 0)

    def test_the_existence_penalty_reads_the_size_of_any_family(self):
        module = structural_module(k=1, max_components=1, witness_weight=0.0)
        classes = DefinedClasses(
            BoxClasses(centres=module.class_center, offsets=module.class_offset),
            tables(1, 1)[0])
        geometry = ComposedGeometry(classes, None, Bounds(), StructuralReading(0.05))
        targets = tables(1, 1)[1]
        penalty = ExistencePenalty(targets.classes, targets.pairs, 0.05)
        size_term = module.size_reg * (2 * th.abs(
            module.class_offset.weight[classes.primitive_mask(TOP)])).mean()
        self.assertTrue(th.allclose(penalty.penalty(geometry),
                                    module.structural_extra_loss() - size_term, atol=1e-6))
        self.assertEqual(size_of(geometry.classes(th.arange(N))).shape, (N,))

    def test_composed_el_trains_with_the_reading_on_plain_boxes(self):
        dataset = FamilyDataset()
        th.manual_seed(0)
        model = ComposedEL(dataset, classes="box", relations="translation",
                           objective=StructuralReading(0.02), embed_dim=6)
        tables_ = build_structural_tables(model, k=1, max_components=1)
        classes = DefinedClasses(model.module.class_model, tables_[0])
        model.module.class_model = classes
        forms = [f for f in ("gci0", "gci0_bot", "gci1", "gci1_bot")
                 if f in model.training_datasets and len(model.training_datasets[f])]
        self.assertTrue(forms)

        def total():
            return sum(float(model.module(model.training_datasets[f][:], f).mean())
                       for f in forms)

        before = total()
        optimizer = th.optim.Adam(model.module.parameters(), lr=0.05)
        for _ in range(60):
            optimizer.zero_grad()
            loss = sum(model.module(model.training_datasets[f][:], f).mean() for f in forms)
            loss.backward()
            optimizer.step()
        self.assertLess(total(), before)


class TestDefinedClassesEdges(TestCase):

    TABLE = DefinitionTable.from_normal_forms(
        gci0=GCI0, gci1=GCI1, n_classes=N, k=1, max_components=4, top_idx=TOP, bottom_idx=None,
        enabled=True)

    def test_bumps_stay_those_of_the_class(self):
        from mowl.representation.library.multibox_parts import BumpedBoxClasses
        base = BumpedBoxClasses(N, 3)
        classes = DefinedClasses(base, self.TABLE)
        region, raw = classes(th.arange(N)), base(th.arange(N))
        self.assertTrue(th.equal(region.bump, raw.bump))              # the class's own bump
        self.assertFalse(th.equal(region.centre[3], raw.centre[3]))   # a defined box
        self.assertTrue(th.equal(region.centre[7], raw.centre[7]))    # a primitive one

    def test_a_family_that_does_not_intersect_says_so(self):
        from mowl.representation import BallClasses, ConstructorUnavailable
        with self.assertRaises(ConstructorUnavailable):
            DefinedClasses(BallClasses(N, 3), self.TABLE)(th.arange(N))

    def test_unions_without_slots_have_no_room_for_the_intersection(self):
        from mowl.representation import ConstructorUnavailable
        from mowl.representation.library.multibox_parts import UnionClasses
        classes = DefinedClasses(UnionClasses(N, 3, components=2), self.TABLE)
        with self.assertRaises(ConstructorUnavailable):
            classes(th.arange(N))

    def test_definitions_as_a_mapping(self):
        classes = DefinedClasses(BoxClasses(N, 3), definitions={3: (0, 1), 5: (3, 2)})
        region = classes(th.arange(N))
        raw = classes.base(th.arange(N))
        lower = th.maximum(raw.lower[0], raw.lower[1])
        upper = th.minimum(raw.upper[0], raw.upper[1])
        self.assertTrue(th.allclose(region.offset[3], (upper - lower) / 2))
        # nested: 5 = (0 and 1) and 2
        lower = th.maximum(lower, raw.lower[2])
        upper = th.minimum(upper, raw.upper[2])
        self.assertTrue(th.allclose(region.offset[5], (upper - lower) / 2, atol=1e-6))
        self.assertEqual(classes.masked_gci0(th.tensor([[3, 0], [3, 2], [5, 3]])).tolist(),
                         [True, False, True])
        self.assertEqual(classes.masked_gci1(th.tensor([[0, 1, 3], [1, 0, 3], [0, 2, 3]]))
                         .tolist(), [True, True, False])


class TestStructuralComposedEL(TestCase):
    """A ComposedEL of the method with the structural reading, and the re-seeding regime."""

    def make(self, **options):
        dataset = FamilyDataset()
        th.manual_seed(0)
        model = ComposedEL(dataset, method="multiboxel", embed_dim=4, margin=0.02,
                           learning_rate=0.05, reading="structural", n_components=2,
                           max_components=4, **options)
        table, targets, top = build_structural_tables(model, k=2, max_components=4)
        model.module.set_structural_tables(table, targets, top)
        return model

    def test_trains_and_the_loss_falls(self):
        model = self.make()
        self.assertIsInstance(model.module, StructuralGeometry)
        forms = [f for f, d in model.training_datasets.items() if len(d)]
        self.assertIn("gci0", forms)

        def total():
            with th.no_grad():
                return float(sum(model.module(model.training_datasets[f][:], f).mean()
                                 for f in forms) + model.module.regularization_loss())

        before = total()
        optimizer = th.optim.Adam(model.module.parameters(), lr=0.05)
        for _ in range(80):
            optimizer.zero_grad()
            loss = sum(model.module(model.training_datasets[f][:], f).mean() for f in forms) \
                + model.module.regularization_loss()
            loss.backward()
            optimizer.step()
        self.assertLess(total(), before)

    def test_the_reseeding_regime_works_with_the_composed_model(self):
        class Reseeding(ComposedEL):
            def reseed_components(self, epoch=0):
                return self.module.reseed_components(epoch)

        dataset = FamilyDataset()
        th.manual_seed(0)
        model = Reseeding(dataset, method="multiboxel", embed_dim=4, margin=0.02,
                          reading="structural", n_components=2, max_components=4)
        table, targets, top = build_structural_tables(model, k=2, max_components=4)
        model.module.set_structural_tables(table, targets, top)
        # collapse every class to a point: the existence targets are violated, so the
        # relocation rule has something to repair
        with th.no_grad():
            model.module.class_model.base.offsets.weight.fill_(1e-5)
        regime = IdleComponentReseeding(every=1, reset_optimizer=False)
        regime.attach(model)
        moved = model.reseed_components(0)
        self.assertGreater(moved, 0)
        self.assertEqual(model.module.last_reseed_count, moved)
        regime.on_epoch_end(0, {})
        self.assertEqual(regime.history[-1][0], 0)
        self.assertGreater(regime.history[-1][1], 0)

    def test_tables_come_from_the_dataset_without_a_subclass(self):
        dataset = FamilyDataset()
        th.manual_seed(0)
        model = ComposedEL(dataset, method="multiboxel", reading="structural", embed_dim=4,
                           margin=0.02, n_components=2, max_components=4)
        geometry = model.module
        self.assertIsInstance(geometry, StructuralGeometry)
        self.assertFalse(geometry.wants_structural_tables)
        table, targets, top = build_structural_tables(model, k=2, max_components=4)
        self.assertTrue(th.equal(geometry.definition_table.def_mask, table.def_mask))
        self.assertTrue(th.equal(geometry.existence_targets.classes, targets.classes))
        self.assertEqual(geometry.top_idx, top)
        self.assertIn("ExistencePenalty", {type(r).__name__ for r in geometry.regularizers})
        self.assertIsInstance(geometry.class_model, DefinedClasses)
        # building the module again keeps the tables
        model.init_module()
        self.assertIsInstance(model.module.class_model, DefinedClasses)
        # a geometry that does not read the structure is left alone
        plain = ComposedEL(dataset, method="multiboxel", embed_dim=4, n_components=2)
        self.assertFalse(getattr(plain.module, "wants_structural_tables", False))
        self.assertFalse(hasattr(plain, "reseed_components"))
        with self.assertRaises(TypeError):
            IdleComponentReseeding(every=1).attach(plain)

    def test_the_reseeding_regime_without_a_subclass(self):
        dataset = FamilyDataset()
        th.manual_seed(0)
        model = ComposedEL(dataset, method="multiboxel", reading="structural", embed_dim=4,
                           margin=0.02, n_components=2, max_components=4)
        with th.no_grad():
            model.module.class_model.base.offsets.weight.fill_(1e-5)
        regime = IdleComponentReseeding(every=1, reset_optimizer=False)
        regime.attach(model)
        regime.on_epoch_end(0, {})
        self.assertGreater(regime.history[-1][1], 0)
        self.assertEqual(model.module.last_reseed_count, regime.history[-1][1])

    def test_a_prism_has_no_components_to_reseed(self):
        model = ComposedEL(FamilyDataset(), method="prism", reading="structural", embed_dim=6,
                           margin=0.02)
        self.assertFalse(model.module.wants_structural_tables)    # tables came from the data
        self.assertFalse(hasattr(model, "reseed_components"))
        with self.assertRaises(TypeError):
            IdleComponentReseeding(every=1).attach(model)

    def test_trains_with_the_regime(self):
        dataset = FamilyDataset()
        th.manual_seed(0)
        model = ComposedEL(dataset, method="multiboxel", reading="structural", embed_dim=4,
                           margin=0.02, learning_rate=0.05, n_components=2, max_components=4)
        regime = IdleComponentReseeding(every=1)
        model.training_regime = regime
        model.train(epochs=3, validate_every=10)
        self.assertEqual([e for e, _ in regime.history], [0, 1, 2])

    def test_reseeding_moves_what_the_module_moves(self):
        module = structural_module(reseed_mode="dead")
        geometry = geometry_of(module)
        with th.no_grad():
            module.class_offset.weight.fill_(1e-5)
        before = module.class_offset.weight.detach().clone()
        moved_geometry = geometry.reseed(np.random.default_rng(3))
        after_geometry = module.class_offset.weight.detach().clone()
        with th.no_grad():
            module.class_offset.weight.copy_(before)
        moved_module = module.reseed(np.random.default_rng(3))
        self.assertEqual(moved_geometry, moved_module)
        self.assertTrue(th.equal(after_geometry, module.class_offset.weight.detach()))
        self.assertGreater(moved_module, 0)


class TestPenaltyParts(TestCase):

    def test_witness_penalty_needs_a_witness_table(self):
        module = structural_module()
        geometry = geometry_of(module)
        penalty = next(r for r in geometry.regularizers if isinstance(r, WitnessPenalty))
        self.assertGreater(float(penalty.penalty(geometry)), 0.0)
        geometry.witness_embed = None
        self.assertIsNone(penalty.penalty(geometry))
