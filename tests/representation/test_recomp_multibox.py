"""Box2EL, MultiBoxEL, MultiELBE and the volume-based MultiBoxEL are combinations of parts.

``mowl.models.<Method>`` builds its module from a class model, a relation model, a measure, an
objective and regularisers (the ``register_method`` builders of
:mod:`mowl.representation.library.multibox_methods`). These tests read the combination back, swap
its parts and its regimes, and check that the catalogue card and the parts agree.
"""
import math
import os
from unittest import TestCase

import torch as th

import mowl.catalog as catalog
from mowl.evaluation.el_scoring import centre_distance_scores, loss_scores
from mowl.models import (BoxSquaredEL, ComposedEL, MultiBoxEL, MultiELBE, VolumeMultiBoxEL)
from mowl.nn import (BoxSquaredELModule, MultiBoxELModule, MultiELBEModule,
                     VolumeMultiBoxELModule)
from mowl.representation import (Bounds, ComposedGeometry, Geometry, METHODS, Objective,
                                 REPRESENTATIONS, Translation)
from mowl.representation.library.multibox_methods import box2el, mcmultibox, multielbe, multiboxel
from mowl.representation.library.multibox_objective import ComponentSpread, DiversityPenalty
from mowl.representation.library.multibox_parts import (
    BumpedBoxClasses, HeadTailRelations, PackedUnionClasses, PairUnions, SampledVolume,
    UnionClasses, UnionMeasure)
from mowl.representation.library.structural_measure import SlotMeasure
from mowl.representation.library.structural_methods import StructuralGeometry
from mowl.representation.library.structural_points import PointReading
from mowl.representation.library.structural_reading import StructuralReading
from mowl.representation.library.structural_regions import SlotClasses
from mowl.training import (NormPenalty, PerNormalFormMiniBatches, TemperatureAnnealing)
from tests.datasetFactory import FamilyDataset
from tests.geometric.helpers import fixture_path


def finite(case, value):
    value = th.as_tensor(value, dtype=th.float)
    case.assertTrue(bool(th.isfinite(value).all()), value)
    return value


def agrees(case, key, module):
    """The parts of the module are those the representation catalogue states for the method."""
    rep = REPRESENTATIONS[key]
    for name in ("classes", "relations", "measure"):
        case.assertIn(module.parts[name], [part.value for part in getattr(rep, name)],
                      (key, name))


class Case(TestCase):
    """Tiny data; a model that trains for an epoch and is evaluated."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def trains_and_evaluates(self, model, epochs=1):
        th.manual_seed(0)
        model.train(epochs=epochs)
        n = min(len(model.class_index_dict), 6)
        pairs = th.cartesian_prod(th.arange(n), th.arange(n))
        with th.no_grad():
            finite(self, model.module(pairs, "gci0"))
            for form, data in model.training_datasets.items():
                if len(data) and form in model.module.supported_forms:
                    finite(self, model.module(data[:], form))
        finite(self, model.get_regularization_loss())
        return model


# =======================================================================================
# Box2EL
# =======================================================================================

class TestBox2EL(Case):

    def test_the_named_combination(self):
        model = BoxSquaredEL(self.dataset, embed_dim=4)
        module = model.module
        self.assertIsInstance(module, Geometry)
        self.assertIsInstance(module, ComposedGeometry)
        self.assertIsInstance(module, BoxSquaredELModule)
        self.assertIs(METHODS["box2el"], box2el)
        self.assertEqual(module.parts["classes"], "box")
        self.assertEqual(module.parts["relations"], "head and tail boxes")
        self.assertEqual(module.parts["measure"], "bounds per coordinate")
        self.assertIsInstance(module.class_model, BumpedBoxClasses)
        self.assertIsInstance(module.relation_model, HeadTailRelations)
        self.assertIsInstance(module.measure, Bounds)
        objective = module.objective
        self.assertIs(type(objective), Objective)
        self.assertTrue(objective.slack)
        self.assertEqual(objective.gci2_negative, "delta")
        self.assertEqual(objective.squared, ("gci0", "gci1", "gci3", "gci0_bot", "gci1_bot",
                                             "gci3_bot"))
        self.assertEqual([type(r) for r in module.regularizers], [NormPenalty])
        self.assertAlmostEqual(module.regularizers[0].weight, model.reg_factor)

    def test_the_builder_arguments_are_the_options_of_the_class(self):
        model = BoxSquaredEL(self.dataset, embed_dim=4, margin=0.3, delta=1.7, reg_factor=0.4)
        module = model.module
        self.assertAlmostEqual(module.objective.margin, 0.3)
        self.assertAlmostEqual(module.gamma, 0.3)
        self.assertAlmostEqual(module.objective.delta, 1.7)
        self.assertAlmostEqual(module.reg_factor, 0.4)
        module.gamma = 0.1
        self.assertAlmostEqual(module.objective.margin, 0.1)

    def test_swap_the_relation_model(self):
        model = ComposedEL(self.dataset, classes=lambda n, d: BumpedBoxClasses(n, d),
                           relations="translation", measure=Bounds(),
                           objective=Objective(0.02), embed_dim=4)
        self.assertIsInstance(model.module.relation_model, Translation)
        self.trains_and_evaluates(model)

    def test_swap_the_measure(self):
        model = BoxSquaredEL(self.dataset, embed_dim=4)
        model.module.measure = Bounds(separation="every", extent="norm", conjunction="exact")
        self.assertEqual(model.module.parts["measure"], "bounds per coordinate")
        self.trains_and_evaluates(model)

    def test_swap_an_objective_option(self):
        model = ComposedEL(self.dataset, method="box2el", all_negatives=True, delta=1.0,
                           embed_dim=4, margin=0.05)
        self.assertTrue({"gci0", "gci1", "gci3"} <= model.module.neg_capable_gcis)
        self.assertAlmostEqual(model.module.objective.delta, 1.0)
        self.trains_and_evaluates(model)

    def test_add_a_regulariser(self):
        extra = NormPenalty(parameters=("class_embed",), target=1.0, weight=0.1)
        model = ComposedEL(self.dataset, method="box2el", embed_dim=4, regularizers=(extra,))
        self.assertIn(extra, model.module.regularizers)
        self.assertEqual(len(model.module.regularizers), 2)
        self.trains_and_evaluates(model)

    def test_another_training_regime(self):
        model = BoxSquaredEL(self.dataset, embed_dim=4)
        model.training_regime = PerNormalFormMiniBatches(batch_size=3, order="cyclic")
        self.trains_and_evaluates(model, epochs=2)

    def test_another_evaluation_regime(self):
        model = BoxSquaredEL(self.dataset, embed_dim=4)
        model.train(epochs=1)
        rows = th.tensor([[0, 1], [1, 2], [2, 3]])
        for scores in (loss_scores, centre_distance_scores):
            finite(self, scores(model, "gci0", rows))

    def test_the_card_and_the_parts_agree(self):
        card = catalog.get_method("box2el")
        module = BoxSquaredEL(self.dataset, embed_dim=4).module
        self.assertEqual(card.region, module.parts["classes"])
        self.assertTrue(set(card.axioms) <= set(module.supported_forms))
        self.assertTrue(module.role_axiom_capable)
        self.assertEqual(card.implementation, f"{BoxSquaredEL.__module__}.BoxSquaredEL")
        self.assertEqual(card.module, f"{BoxSquaredELModule.__module__}.BoxSquaredELModule")
        agrees(self, "box2el", module)
        self.assertTrue(card.supported_expressivity.has_abox)
        self.assertEqual(BoxSquaredELModule(5, 2, 3, embed_dim=4).parts["individuals"],
                         "region")


# =======================================================================================
# MultiBoxEL
# =======================================================================================

class TestMultiBoxEL(Case):

    def test_the_named_combination(self):
        model = MultiBoxEL(self.dataset, embed_dim=4, n_components=3)
        module = model.module
        self.assertIsInstance(module, MultiBoxELModule)
        self.assertIsInstance(module, ComposedGeometry)
        self.assertIs(METHODS["multiboxel"], multiboxel)
        self.assertEqual(module.parts["classes"], "union of boxes")
        self.assertEqual(module.parts["relations"], "head and tail boxes")
        self.assertEqual(module.parts["measure"], "bounds per coordinate")
        self.assertIsInstance(module.class_model, UnionClasses)
        self.assertEqual(module.class_model.components, 3)
        self.assertIsInstance(module.relation_model, HeadTailRelations)
        self.assertIsInstance(module.measure, UnionMeasure)
        self.assertEqual(module.measure.aggregation, "hard")
        self.assertEqual(module.reading, "axiom")
        self.assertEqual([type(r) for r in module.regularizers], [NormPenalty])

    def test_the_regularisers_are_the_options(self):
        module = MultiBoxEL(self.dataset, embed_dim=4, component_reg=0.2,
                            diversity_reg=0.1).module
        kinds = [type(r) for r in module.regularizers]
        self.assertEqual(kinds, [NormPenalty, ComponentSpread, DiversityPenalty])
        self.assertAlmostEqual(module.component_reg, 0.2)
        module.component_reg = 0.0
        self.assertEqual(module.component_reg, 0.0)

    def test_the_structural_reading_is_a_combination_of_other_parts(self):
        model = MultiBoxEL(self.dataset, embed_dim=4, objective="structural",
                           max_components=4)
        module = model.module
        self.assertIsInstance(module, StructuralGeometry)
        self.assertIsInstance(module.class_model, SlotClasses)
        self.assertIsInstance(module.measure, SlotMeasure)
        self.assertIsInstance(module.objective, StructuralReading)
        self.assertEqual(module.margin_semantics, "strict")
        self.assertEqual(module.margin_attribute, "margin")

    def test_the_point_readings_are_an_objective(self):
        for reading in ("point", "mixed", "hybrid"):
            module = MultiBoxEL(self.dataset, embed_dim=4, objective=reading).module
            self.assertIsInstance(module.objective, PointReading)
            self.assertEqual(module.objective.mode, reading)
            self.assertIs(module.objective.terms, module.point_terms)

    def test_swap_the_aggregation_and_the_temperature_schedule(self):
        model = MultiBoxEL(self.dataset, embed_dim=4, component_aggregation="soft",
                           tau_start=1.0, tau_end=0.1, tau_anneal=True, diversity_reg=0.1)
        self.assertEqual(model.module.measure.aggregation, "soft")
        self.trains_and_evaluates(model, epochs=3)
        self.assertLess(float(model.module.tau), 1.0)

    def test_swap_the_relation_model(self):
        model = ComposedEL(self.dataset, classes=lambda n, d: UnionClasses(n, d, components=2),
                           relations="translation", measure=UnionMeasure(Bounds()),
                           objective=Objective(0.02), embed_dim=4)
        self.assertIsInstance(model.module.relation_model, Translation)
        self.trains_and_evaluates(model)

    def test_swap_the_class_model_and_the_measure_of_the_method(self):
        model = ComposedEL(self.dataset, method="multiboxel", embed_dim=4, n_components=1,
                           margin=0.05)
        self.assertEqual(model.module.class_model.components, 1)
        self.trains_and_evaluates(model)

    def test_add_a_regulariser_and_another_training_regime(self):
        extra = NormPenalty(parameters=("class_embed",), target=1.0, weight=0.1)
        model = ComposedEL(self.dataset, method="multiboxel", embed_dim=4,
                           regularizers=(extra,))
        model.training_regime = PerNormalFormMiniBatches(batch_size=3)
        self.trains_and_evaluates(model, epochs=2)

    def test_the_temperature_is_a_regime(self):
        model = MultiBoxEL(self.dataset, embed_dim=4, component_aggregation="soft")
        model.training_regime = TemperatureAnnealing(start=1.0, end=0.2)
        self.trains_and_evaluates(model, epochs=3)

    def test_another_evaluation_regime(self):
        model = MultiBoxEL(self.dataset, embed_dim=4)
        model.train(epochs=1)
        rows = th.tensor([[0, 1], [1, 2]])
        finite(self, loss_scores(model, "gci0", rows))
        finite(self, model.module.effective_components())

    def test_the_card_and_the_parts_agree(self):
        card = catalog.get_method("multiboxel")
        module = MultiBoxEL(self.dataset, embed_dim=4).module
        self.assertEqual(card.region, module.parts["classes"])
        self.assertTrue(set(card.axioms) <= set(module.supported_forms))
        self.assertEqual(card.module, f"{MultiBoxELModule.__module__}.MultiBoxELModule")
        agrees(self, "multiboxel", module)
        self.assertTrue(card.supported_expressivity.has_abox)


# =======================================================================================
# MultiELBE
# =======================================================================================

class TestMultiELBE(Case):

    def test_the_named_combination(self):
        model = MultiELBE(self.dataset, embed_dim=4, n_components=2)
        module = model.module
        self.assertIsInstance(module, MultiELBEModule)
        self.assertIsInstance(module, ComposedGeometry)
        self.assertIs(METHODS["multielbe"], multielbe)
        self.assertEqual(module.parts["classes"], "union of boxes")
        self.assertEqual(module.parts["relations"], "translation")
        self.assertEqual(module.parts["measure"], "bounds per coordinate")
        self.assertIsInstance(module.class_model, UnionClasses)
        self.assertIsInstance(module.relation_model, Translation)
        self.assertIsInstance(module.measure, UnionMeasure)
        objective = module.objective
        self.assertFalse(objective.slack)
        self.assertEqual(objective.gci3_reading, "overlap")
        self.assertEqual(module.regularizers, ())

    def test_swap_gci3_to_the_corrected_reading_and_the_aggregation(self):
        model = ComposedEL(self.dataset, method="multielbe", embed_dim=4, gci3="inclusion",
                           component_aggregation="soft", diversity_reg=0.1, n_components=2,
                           margin=0.05)
        self.assertEqual(model.module.objective.gci3_reading, "inclusion")
        self.trains_and_evaluates(model)

    def test_swap_the_relation_model(self):
        model = ComposedEL(self.dataset, classes=lambda n, d: UnionClasses(n, d, components=2),
                           relations=lambda n, d: HeadTailRelations(n, d),
                           measure=UnionMeasure(Bounds()), objective=Objective(0.02),
                           embed_dim=4)
        self.assertIsInstance(model.module.relation_model, HeadTailRelations)
        self.trains_and_evaluates(model)

    def test_add_a_regulariser(self):
        model = MultiELBE(self.dataset, embed_dim=4, n_components=2, component_reg=0.5)
        self.assertEqual([type(r) for r in model.module.regularizers], [ComponentSpread])
        self.trains_and_evaluates(model)

    def test_another_training_regime(self):
        model = MultiELBE(self.dataset, embed_dim=4, n_components=2)
        model.training_regime = PerNormalFormMiniBatches(batch_size=3, order="cyclic")
        self.trains_and_evaluates(model, epochs=2)

    def test_the_point_readings(self):
        for reading in ("point", "mixed", "hybrid"):
            module = MultiELBE(self.dataset, embed_dim=4, objective=reading).module
            self.assertIsInstance(module.objective, PointReading)

    def test_another_evaluation_regime(self):
        model = MultiELBE(self.dataset, embed_dim=4, n_components=2)
        model.train(epochs=1)
        finite(self, loss_scores(model, "gci0", th.tensor([[0, 1], [1, 2]])))

    def test_the_card_and_the_parts_agree(self):
        card = catalog.get_method("multielbe")
        module = MultiELBE(self.dataset, embed_dim=4).module
        self.assertEqual(card.region, module.parts["classes"])
        self.assertTrue(set(card.axioms) <= set(module.supported_forms))
        self.assertEqual(card.module, f"{MultiELBEModule.__module__}.MultiELBEModule")
        agrees(self, "multielbe", module)


# =======================================================================================
# the volume-based MultiBoxEL
# =======================================================================================

def volume(dataset, **options):
    options = {**dict(embed_dim=3, num_boxes=2, num_points=128, point_chunk_size=64,
                      num_batches=2), **options}
    return VolumeMultiBoxEL(dataset, **options)


class TestVolumeMultiBoxEL(Case):

    def test_the_named_combination(self):
        model = volume(self.dataset)
        module = model.module
        self.assertIsInstance(module, VolumeMultiBoxELModule)
        self.assertIsInstance(module, ComposedGeometry)
        self.assertIs(METHODS["mcmultibox"], mcmultibox)
        self.assertEqual(module.parts["classes"], "union of boxes")
        self.assertEqual(module.parts["relations"], "region over pairs")
        self.assertEqual(module.parts["measure"], "sampled volume")
        self.assertIsInstance(module.class_model, PackedUnionClasses)
        self.assertIsInstance(module.relation_model, PairUnions)
        self.assertIsInstance(module.measure, SampledVolume)
        self.assertEqual(module.regularizers, ())
        self.assertEqual(module.neg_capable_gcis, frozenset())

    def test_the_numbers_of_the_method_are_those_of_the_parts(self):
        module = volume(self.dataset, temperature=0.5, bound=2.0, aggregate="mean").module
        self.assertEqual(module.measure.temperature, 0.5)
        self.assertEqual(module.measure.bound, 2.0)
        self.assertEqual(module.relation_model.bound, 2.0)
        self.assertEqual(module.measure.aggregate, "mean")
        module.temperature = 0.25
        self.assertEqual(module.measure.temperature, 0.25)

    def test_swap_the_measure_aggregate_and_the_point_set(self):
        for options in ({"aggregate": "mean"}, {"num_points": 64, "temperature": 0.3}):
            model = volume(self.dataset, **options)
            model.train(epochs=1)
            data = model.training_datasets["gci0"][:]
            finite(self, model.module(data, "gci0"))

    def test_swap_the_class_model_and_measure_through_the_geometry(self):
        """The sampled measure compares any region that has a membership function: boxes of
        a Box2EL module through it."""
        from mowl.representation.library.multibox_parts import SampledVolume as Sampled
        module = BoxSquaredEL(self.dataset, embed_dim=3).module
        module.measure = Sampled(num_points=128, bound=1.5)
        rows = th.tensor([[0, 1], [1, 2]])
        finite(self, module(rows, "gci0"))
        self.assertEqual(module.parts["measure"], "sampled volume")

    def test_another_training_regime(self):
        """The batching of the thesis is the regime ShuffledSplits, so the model takes every
        regime (it consults all the hooks); the same parts train in the generic loop of
        ``ComposedEL`` with another regime."""
        self.assertTrue(frozenset(VolumeMultiBoxEL.training_hooks))
        model = ComposedEL(self.dataset, method="mcmultibox", embed_dim=3, num_boxes=2,
                           num_points=128, point_chunk_size=64)
        self.assertIsInstance(model.module, ComposedGeometry)
        self.assertEqual(model.module.parts["measure"], "sampled volume")
        model.training_regime = PerNormalFormMiniBatches(batch_size=3, order="cyclic")
        model.train(epochs=2)
        data = model.training_datasets["gci0"][:]
        finite(self, model.module(data, "gci0"))

    def test_another_evaluation_regime(self):
        model = volume(self.dataset)
        model.train(epochs=1)
        n = len(model.class_index_dict)
        ids = th.arange(min(n, 5))
        pairs = th.cartesian_prod(ids, ids)
        distance = model.module.distance(pairs, "gci0")
        finite(self, distance)
        self.assertTrue(bool(((distance >= 0) & (distance <= 1)).all()))
        finite(self, loss_scores(model, "gci0", pairs))

    def test_the_card_and_the_parts_agree(self):
        card = catalog.get_method("mcmultibox")
        module = volume(self.dataset).module
        self.assertEqual(card.region, module.parts["classes"])
        self.assertEqual(set(card.axioms), set(module.supported_forms))
        self.assertEqual(card.module,
                         f"{VolumeMultiBoxELModule.__module__}.VolumeMultiBoxELModule")
        agrees(self, "mcmultibox", module)
        self.assertEqual(card.expressivity, "ELH(∘)")


class TestSavedModels(Case):
    """Models saved under the names of the tables of the direct modules load, and the other
    way round."""

    def test_state_dictionaries_load_both_ways(self):
        from mowl.baselines.nn.el.boxsquaredel.module import BoxSquaredELModule as Direct
        old = Direct(5, 2, 3, embed_dim=4, gamma=0.1)
        new = BoxSquaredELModule(5, 2, 3, embed_dim=4, gamma=0.1)
        self.assertEqual(sorted(old.state_dict()), sorted(new.state_dict()))
        new.load_state_dict(old.state_dict())
        old2 = Direct(5, 2, 3, embed_dim=4, gamma=0.1)
        old2.load_state_dict(new.state_dict())
        for name, value in old.state_dict().items():
            self.assertTrue(th.equal(value, old2.state_dict()[name]), name)
            self.assertTrue(th.equal(value, new.state_dict()[name]), name)

    def test_the_table_names_are_views_of_the_parts(self):
        module = MultiBoxELModule(5, 2, 3, embed_dim=4, n_components=2)
        self.assertIs(module.class_center, module.class_model.centres)
        self.assertIs(module.head_offset, module.relation_model.head_offsets)
        self.assertIs(module.bump_classes, module.class_model.bumps)
        self.assertIs(module.ind_offset, module.individual_model.offsets)
        self.assertNotIn("class_model.centres.weight", module.state_dict())
        self.assertIn("class_center.weight", module.state_dict())

    def test_a_model_reloads_its_best_state(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "m.pt")
            model = BoxSquaredEL(self.dataset, embed_dim=4, model_filepath=path)
            model.train(epochs=2)
            th.save(model.module.state_dict(), path)
            other = BoxSquaredEL(self.dataset, embed_dim=4, model_filepath=path)
            other.from_pretrained(path)
            self.assertTrue(th.equal(model.module.class_center.weight,
                                     other.module.class_center.weight))


class TestStructuralPartsOfTheModule(Case):
    """Re-seeding is the relocation rule of the parts (the regime calls the geometry)."""

    def test_the_geometry_reseeds_without_a_module_shell(self):
        from mowl.datasets import PathDataset
        dataset = PathDataset(fixture_path("t3.owl"))
        model = MultiBoxEL(dataset, embed_dim=2, n_components=2, objective="structural",
                           max_components=8, neg_sampling_gcis=[])
        model.train(epochs=1)
        import numpy as np
        moves = model.module.reseed(np.random.default_rng(0))
        self.assertIsInstance(moves, int)
        self.assertEqual(moves, model.module.last_reseed_count)
        self.assertEqual(math.isfinite(float(model.module.structural_extra_loss())), True)
