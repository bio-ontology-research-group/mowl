"""DF-ALC, FALCON, Al-cone and E2R are combinations of parts: the combination can be read back,
its parts can be swapped (each swap trains for an epoch and evaluates to finite numbers), and the
catalogue card agrees with what the parts can do."""
import math
from unittest import TestCase

import torch as th

from mowl.catalog import card_for
from mowl.models import ALConeModel, ComposedALC, ComposedEL, DFALCModel, E2R, FALCONModel
from mowl.nn import ALCModule, ALConeModule, DFALCModule, E2RModule, FALCONModule
from mowl.representation import (ComposedGeometry, ConstructorUnavailable, Geometry, METHODS,
                                 METHOD_KINDS)
from mowl.representation.library import logic_fuzzy as F
from mowl.representation.library import logic_subspace as S
from mowl.representation.library.logic_alc import (AnnealedRelaxation, ExpressionGeometry,
                                                   ExpressionReading)
from mowl.training.regularization import NormPenalty
from tests.baselines.pairs_logic import alc_dataset, e2r_dataset

KEYS = ("dfalc", "falcon", "alcone", "e2r")


def finite(tensor):
    return bool(th.isfinite(th.as_tensor(tensor, dtype=th.float32)).all())


def all_pairs_of_classes(model):
    n = len(model.class_index_dict)
    return th.cartesian_prod(th.arange(n), th.arange(n))


class Reading(TestCase):
    """(a) The named combination can be read back from the model and from the builder."""

    def test_the_methods_are_registered_as_combinations(self):
        for key in KEYS:
            self.assertIn(key, METHODS)
        self.assertEqual(METHOD_KINDS["e2r"], "expressions")
        self.assertEqual([METHOD_KINDS[k] for k in ("dfalc", "falcon", "alcone")], ["axioms"] * 3)

    def test_dfalc(self):
        model = DFALCModel(alc_dataset(), anon_domain=1)
        m = model.module
        self.assertIsInstance(m, DFALCModule)
        self.assertIsInstance(m, ExpressionGeometry)
        self.assertIsInstance(m, ComposedGeometry)
        self.assertIsInstance(m, Geometry)
        self.assertIsInstance(m, ALCModule)
        self.assertIsInstance(m.class_model, F.TruthTableClasses)
        self.assertIsInstance(m.relation_model, F.TruthMatrix)
        self.assertIsInstance(m.measure, F.MeanViolation)
        self.assertIsInstance(m.objective, F.FuzzyObjective)
        self.assertIsInstance(m.objective, ExpressionReading)
        self.assertIsInstance(m.class_model.family, F.Goedel)
        self.assertFalse(m.objective.rule)
        self.assertEqual(m.parts["classes"], "fuzzy set")
        self.assertEqual(m.parts["relations"], "fuzzy relation")
        rule = DFALCModel(alc_dataset(), anon_domain=1, loss_type="rule")
        self.assertTrue(rule.module.objective.rule)
        self.assertEqual(rule.module.objective.threshold, 0.8)

    def test_falcon(self):
        m = FALCONModel(alc_dataset(), embed_dim=6).module
        self.assertIsInstance(m, FALCONModule)
        self.assertIsInstance(m.class_model, F.NetworkClasses)
        self.assertIsInstance(m.relation_model, F.PairNetworkRelations)
        self.assertIsInstance(m.measure, F.ConflictMeasure)
        self.assertEqual(m.measure.stabiliser, "add")        # the published reading
        self.assertIsInstance(m.class_model.family, F.Product)
        self.assertEqual(m.objective.abox, "sampled")
        self.assertIs(m.class_model.network, m.relation_model.network)
        self.assertIs(m.domain, m.relation_model.domain)
        self.assertIsInstance(m.individual_model, F.DomainElements)

    def test_alcone(self):
        model = ALConeModel(alc_dataset(), dim=3)
        m = model.module
        self.assertIsInstance(m, ALConeModule)
        self.assertIsInstance(m.class_model, F.HalfAxisClasses)
        self.assertIsInstance(m.relation_model, F.IncidenceMatrices)
        self.assertIsInstance(m.measure, F.ConeMeasure)
        self.assertIsInstance(m.class_model.family, F.LogitGoedel)
        self.assertEqual([type(r) for r in m.regularizers], [F.NonDegeneracy])
        self.assertIsInstance(m.schedule, AnnealedRelaxation)
        self.assertEqual((model.temperature, model.noise), (0.3, 0.3))
        self.assertTrue(m.objective.validity)

    def test_e2r(self):
        model = E2R(e2r_dataset(), embed_dim=4)
        m = model.module
        self.assertIsInstance(m, E2RModule)
        self.assertIsInstance(m, S.SubspaceGeometry)
        self.assertIsInstance(m, Geometry)
        self.assertIsInstance(m.class_model, S.SubspaceClasses)
        self.assertIsInstance(m.relation_model, S.SubspaceRelations)
        self.assertIsInstance(m.individual_model, S.PointIndividuals)
        self.assertIsInstance(m.measure, S.ResidualMeasure)
        self.assertIsInstance(m.objective, S.SubspaceObjective)
        self.assertEqual([type(r) for r in m.regularizers],
                         [S.SubspaceStructure, S.Orthogonality, S.Separation])
        self.assertEqual(m.parts["classes"], "subspace")

    def test_the_builders_make_the_same_parts(self):
        for key, parts in (("dfalc", ("TruthTableClasses", "TruthMatrix", "MeanViolation")),
                           ("falcon", ("NetworkClasses", "PairNetworkRelations",
                                       "ConflictMeasure")),
                           ("alcone", ("HalfAxisClasses", "IncidenceMatrices", "ConeMeasure"))):
            g = METHODS[key](5, 2, 3, embed_dim=4)
            self.assertIsInstance(g, ExpressionGeometry)
            self.assertEqual((type(g.class_model).__name__, type(g.relation_model).__name__,
                              type(g.measure).__name__), parts)
        g = METHODS["e2r"](5, 2, 3, embed_dim=4)
        self.assertEqual((type(g.class_model).__name__, type(g.measure).__name__),
                         ("SubspaceClasses", "ResidualMeasure"))

    def test_the_parts_do_not_wrap_the_direct_modules(self):
        for model in (DFALCModel(alc_dataset(), anon_domain=1), FALCONModel(alc_dataset(), 6),
                      ALConeModel(alc_dataset(), dim=3), E2R(e2r_dataset(), embed_dim=4)):
            for part in vars(model.module).get("_modules", {}).values():
                self.assertNotIn("mowl.baselines", type(part).__module__)
            self.assertNotIn("mowl.baselines", type(model.module).__module__)

    def test_a_seed_gives_the_numbers_of_the_direct_modules(self):
        from mowl.baselines.nn.alc.alcone.module import ALConeModule as DirectALCone
        from mowl.baselines.nn.alc.dfalc.module import DFALCModule as DirectDFALC
        from mowl.baselines.nn.alc.e2r.module import E2RModule as DirectE2R
        from mowl.baselines.nn.alc.falcon.module import FALCONModule as DirectFALCON

        def make(direct, part_built, *args, **kwargs):
            th.manual_seed(3)
            old = direct(*args, **kwargs)
            th.manual_seed(3)
            return old, part_built(*args, **kwargs)

        old, new = make(DirectDFALC, DFALCModule, 6, 2, 5, top_index=4, bottom_index=5)
        self.assertTrue(th.equal(old.c_embedding.weight, new.c_embedding.weight))
        self.assertTrue(th.equal(old.r_embedding.weight, new.r_embedding.weight))
        old, new = make(DirectALCone, ALConeModule, 6, 2, 3, dim=3)
        for name in ("c_embedding", "r_embedding", "e_embedding"):
            self.assertTrue(th.equal(getattr(old, name).weight, getattr(new, name).weight))
        old, new = make(DirectFALCON, FALCONModule, 6, 3, 2, {}, {}, embed_dim=5)
        for name in ("c_embedding", "r_embedding", "e_embedding"):
            self.assertTrue(th.equal(getattr(old, name).weight, getattr(new, name).weight))
        self.assertTrue(th.equal(old.fc_0.weight, new.fc_0.weight))
        self.assertTrue(th.equal(old.fc_1.bias, new.fc_1.bias))
        old, new = make(DirectE2R, E2RModule, 4, 6, 2, dim=5)
        for name in ("c_embedding", "r_embedding", "e_embedding"):
            self.assertTrue(th.equal(getattr(old, name).weight, getattr(new, name).weight))


def trains_and_scores(case, model, epochs=1):
    """One epoch of training, then the scores an evaluation regime ranks: finite numbers."""
    model.train(epochs=epochs)
    for parameter in model.module.parameters():
        case.assertTrue(finite(parameter))
    pairs = all_pairs_of_classes(model)
    case.assertTrue(finite(model.axiom_costs("gci0", pairs)))
    return model


class Swaps(TestCase):
    """(b) Parts can be swapped: another family, relation model, measure, objective option,
    an added regulariser, another training schedule, another reading of the assertions."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = alc_dataset()

    def alc(self, **kwargs):
        th.manual_seed(0)
        kwargs.setdefault("embed_dim", 6)
        return ComposedALC(self.dataset, **kwargs)

    # DF-ALC ---------------------------------------------------------------------------------
    def test_dfalc_with_another_family(self):
        for family in ("lukasiewicz", "product"):
            model = trains_and_scores(self, self.alc(method="dfalc", family=family))
            self.assertEqual(type(model.module.class_model.family).__name__.lower(), family)

    def test_dfalc_with_the_rule_based_objective(self):
        trains_and_scores(self, self.alc(method="dfalc", loss_type="rule", threshold=0.6))

    def test_dfalc_tables_with_the_conflict_measure_of_falcon(self):
        model = trains_and_scores(self, self.alc(
            classes="fuzzy set (product)", relations="fuzzy relation", measure="conflict",
            objective=F.FuzzyObjective()))
        self.assertIsInstance(model.module.measure, F.ConflictMeasure)

    def test_dfalc_with_an_added_regulariser(self):
        model = self.alc(method="dfalc", regularizers=(NormPenalty(parameters="c_embedding"),))
        trains_and_scores(self, model)
        self.assertEqual(len(model.module.regularizers), 1)

    # FALCON ---------------------------------------------------------------------------------
    def test_falcon_with_other_operators_and_pairs(self):
        for options in (dict(t_norm="minmax"), dict(t_norm="Łukasiewicz"),
                        dict(role_pairs="self")):
            trains_and_scores(self, self.alc(method="falcon", anon_e=2, **options))

    def test_falcon_with_the_correct_direction_of_role_assertions(self):
        model = trains_and_scores(self, self.alc(method="falcon", anon_e=2,
                                                 abox_head="subject"))
        self.assertEqual(model.module.objective.abox_head, "subject")

    def test_falcon_with_the_clamped_conflict_and_a_ranking_role_loss(self):
        trains_and_scores(self, self.alc(method="falcon", anon_e=2, stabiliser="clamp",
                                         role_loss="ranking"))

    def test_falcon_with_semantic_assertions_instead_of_sampled_ones(self):
        objective = F.FuzzyObjective(role_assertion="image")
        model = self.alc(classes="fuzzy set (product)", relations="fuzzy relation",
                         measure="conflict", objective=objective)
        trains_and_scores(self, model)

    # Al-cone --------------------------------------------------------------------------------
    def test_alcone_without_its_training_schedule(self):
        model = trains_and_scores(self, self.alc(method="alcone", embed_dim=3, anneal=None))
        self.assertIsNone(model.module.schedule)

    def test_alcone_with_a_schedule_of_our_own(self):
        model = self.alc(method="alcone", embed_dim=3, anneal=None,
                         schedule=AnnealedRelaxation(temperature=0.1, noise=0.0))
        trains_and_scores(self, model, epochs=2)
        self.assertEqual(model.module.temperature, 0.0)      # exact operators afterwards

    def test_alcone_with_another_margin_and_regulariser(self):
        model = self.alc(method="alcone", embed_dim=3, margin=0.3, nondegeneracy=0.0,
                         regularizers=(NormPenalty(parameters="c_embedding"),))
        trains_and_scores(self, model)
        self.assertEqual(model.module.objective.margin, 0.3)

    def test_alcone_parts_named_one_by_one(self):
        model = self.alc(classes="half-axis cone", relations="incidence matrix",
                         measure="half-axis hinge", embed_dim=3,
                         objective=F.FuzzyObjective(0.1, validity=True))
        trains_and_scores(self, model)
        self.assertIsInstance(model.module.measure, F.ConeMeasure)

    # E2R ------------------------------------------------------------------------------------
    def test_e2r_swaps(self):
        dataset = e2r_dataset()
        for options in (dict(role_unit_length=False), dict(num_negatives=0),
                        dict(reg_orth=0.0, reg_sep=0.0), dict(range_negatives="all"),
                        dict(sep_gap=2.0, negative_weight=0.5)):
            th.manual_seed(0)
            model = E2R(dataset, embed_dim=4, **options)
            history = model.train(epochs=1)
            self.assertTrue(all(math.isfinite(x) for x in history))
            self.assertTrue(math.isfinite(model.total_loss()))
            self.assertTrue(finite(model.score_membership("http://mowl.borg/logic#A")))

    def test_e2r_with_an_added_regulariser_and_a_measure_that_counts_the_length(self):
        th.manual_seed(0)
        model = E2R(e2r_dataset(), embed_dim=4)
        model.module.regularizers = model.module.regularizers + (
            NormPenalty(parameters="e_embedding"),)
        model.module.measure.unit_length = True
        model.train(epochs=1)
        penalty = model.module.regularization_loss()
        self.assertTrue(finite(penalty))
        ids = th.arange(len(model.class_index_dict))
        region = model.module.classes(ids)
        costs = model.module.member(th.zeros(len(ids), dtype=th.long), region)
        self.assertEqual(costs.shape, (len(ids), 1))
        self.assertTrue(finite(costs))

    def test_e2r_is_trained_on_expressions_not_normal_forms(self):
        with self.assertRaisesRegex(TypeError, "ComposedALC"):
            ComposedEL(e2r_dataset(), method="e2r")
        with self.assertRaises(TypeError):
            ComposedALC(e2r_dataset(), method="e2r")

    # the general model ----------------------------------------------------------------------
    def test_composed_el_still_trains_the_fuzzy_methods_on_normal_forms(self):
        for key in ("dfalc", "falcon", "alcone"):
            th.manual_seed(0)
            model = ComposedEL(alc_dataset(), method=key, embed_dim=4)
            model.train(epochs=1)
            self.assertIsInstance(model.module, ExpressionGeometry)

    def test_the_geometry_reads_both_normal_forms_and_alc_axioms(self):
        model = self.alc(method="dfalc")
        model._load_datasets()
        g = model.module
        rows = th.tensor([[0, 1], [1, 2]])
        self.assertEqual(g(rows, "gci0").shape, (2, 1))
        for axiom, data in model._training_datasets.items():
            self.assertTrue(finite(g(axiom, data.tensors[0])))


class CardsAndParts(TestCase):
    """(c) The catalogue card states what the parts do."""

    def test_the_cards_are_those_of_the_classes(self):
        for cls, key in ((DFALCModel, "dfalc"), (FALCONModel, "falcon"),
                         (ALConeModel, "alcone"), (E2R, "e2r"), (ComposedALC, "composed-alc")):
            self.assertEqual(card_for(cls).key, key)

    def test_alc_cards_agree_with_the_constructors_of_the_geometry(self):
        from tests.representation.test_logic_fuzzy import CLASS_IDS, EXPRESSIONS, IND_IDS, PROP_IDS
        from mowl.representation.library.logic_expressions import expression_region
        for key in ("dfalc", "falcon", "alcone"):
            card = card_for(METHOD_CLASS[key])
            self.assertIn("ALC", card.expressivity)
            g = METHODS[key](6, 2, 3, embed_dim=4)
            for expression in EXPRESSIONS:            # ⊓ ⊔ ¬ ∃ ∀ of the card, nested
                region = expression_region(g, expression, CLASS_IDS, PROP_IDS, IND_IDS)
                self.assertEqual(len(region), 1)
            if card.axioms:       # the card of FALCON does not list them
                self.assertTrue({"subclass_of", "equivalent_classes", "disjoint_classes",
                                 "class_assertion", "object_property_assertion"}
                                <= set(card.axioms), key)

    def test_the_alc_axioms_of_the_cards_have_a_reading(self):
        model = ComposedALC(alc_dataset(), method="dfalc", embed_dim=6)
        model._load_datasets()
        kinds = {type(a).__name__ for a in model._training_datasets}
        for name in ("SubClassOf", "EquivalentClasses", "DisjointClasses", "ClassAssertion",
                     "ObjectPropertyAssertion"):
            self.assertTrue(any(name in k for k in kinds), name)

    def test_the_e2r_card_agrees_with_the_subspace_parts(self):
        card = card_for(E2R)
        self.assertNotIn("existential", " ".join(card.tasks))
        g = METHODS["e2r"](5, 2, 3, embed_dim=4)
        with self.assertRaises(ConstructorUnavailable):
            g.exists(th.tensor([0]), g.classes(th.tensor([0])))
        for task, kind in (("subsumption", "concept_inclusion"),
                           ("disjointness", "concept_disjointness"),
                           ("role-subsumption", "role_inclusion"),
                           ("instance-checking", "class_assertion"),
                           ("role-assertion", "role_assertion")):
            self.assertIn(task, card.tasks)
            self.assertIn(kind, S.AXIOM_KINDS)
        # ⊓ ⊔ ¬ of the card: the parts have them, exactly
        a, b = g.classes(th.tensor([0])), g.classes(th.tensor([1]))
        self.assertEqual(g.parts["conjunction"], "exact")
        self.assertIsInstance(g.conjunction(a, b)[0], S.Subspace)
        self.assertIsInstance(g.disjunction(a, b), S.Subspace)
        self.assertIsInstance(g.negation(a), S.Subspace)
        self.assertEqual(card.axioms, ("class_assertion", "object_property_assertion",
                                       "subclass_of", "equivalent_classes", "disjoint_classes",
                                       "sub_object_property_of", "object_property_range"))


METHOD_CLASS = {"dfalc": DFALCModel, "falcon": FALCONModel, "alcone": ALConeModel}
