"""Octagon, prism, EIKE and Concept2Box are combinations of parts.

``mowl.models.OctagonEL`` and the others build their module from a class model, a relation
model, a measure and an objective (the builders of
:mod:`mowl.representation.library.shapes_methods`) and train with the regimes of their papers.
These tests read the combination back, swap parts, and compare the card of the method with
the parts. (That the combination gives the numbers of the direct implementation is
``tests/baselines``.)
"""
import importlib

import pytest
import torch as th

from mowl.catalog.registry import get_method
from mowl.datasets import Dataset
from mowl.evaluation.normal_form_ranking import NormalFormRankingRegime
from mowl.models import ComposedEL, Concept2Box, EIKE, OctagonEL, PrismEL
from mowl.representation import (METHODS, ComposedGeometry, Geometry, Translation, geometry_of)
from mowl.representation.library import shapes_methods as SM
from mowl.representation.library.shapes_classes import (
    ContextualSoftBoxClasses, EllipsoidClasses, LinkedClasses, OctagonClasses, PrismClasses,
    VectorClasses)
from mowl.representation.library.shapes_measures import (
    ByRegion, CosineOrder, EllipsoidMeasure, GumbelVolumeRatio, LinkedMeasure, OctagonBounds,
    PointDistance, PointToRegion, PrismBounds)
from mowl.representation.library.shapes_objective import ShapeObjective
from mowl.representation.library.shapes_training import ExponentialDecay, SquaredMeanLoss
from mowl.representation.library.structural_methods import StructuralGeometry
from mowl.representation.library.structural_regions import DefinedClasses
from mowl.training import (EitherSideCorruption, NormPenalty, PerColumnCorruption,
                           Regularization, StepDecay, SumLoss)
from mowl.training.regime import ComposedRegime
from tests.datasetFactory import FamilyDataset
from tests.models.kg_region_data import abox_dataset

pytestmark = pytest.mark.filterwarnings("ignore")

D = 8


@pytest.fixture(scope="module")
def family():
    return FamilyDataset()


@pytest.fixture(scope="module")
def abox():
    return abox_dataset()


def _tested(dataset):
    """The dataset with its own ontology also as the test split (for the evaluation)."""
    return Dataset(dataset.ontology, testing=dataset.ontology)


def _finite_losses(model):
    module = model.module
    for form, data in model.training_datasets.items():
        if len(data) == 0:
            continue
        loss = module(data[:], form)
        assert th.isfinite(loss).all(), form
        if form in module.neg_capable_gcis:
            assert th.isfinite(module(data[:], form, neg=True)).all(), (form, "neg")


def _train_and_evaluate(model, dataset):
    """One epoch, then the losses are finite, and an evaluation regime ranks the axioms."""
    th.manual_seed(0)
    model.train(epochs=1)
    _finite_losses(model)
    result = NormalFormRankingRegime(normal_forms=("gci0",), split="test",
                                     scoring="loss", combined=False).evaluate(
        model, _tested(dataset))
    assert result["gci0"]["mrr"] > 0 and th.isfinite(th.tensor(result["gci0"]["mr"]))
    classes = th.arange(min(4, len(model.class_index_dict)))
    cost = model.as_geometry.inside(model.as_geometry.classes(classes),
                                    model.as_geometry.classes(classes.flip(0)))
    assert th.isfinite(cost).all()


# -- (a) the named combination, read back ----------------------------------------------------

def _is_geometry(model):
    assert isinstance(model.module, Geometry) and isinstance(model.module, ComposedGeometry)
    assert model.as_geometry is model.module


def test_octagon_is_products_of_octagons_with_translations(family):
    model = OctagonEL(family, embed_dim=D)
    _is_geometry(model)
    g = model.module
    assert isinstance(g, SM.OctagonELModule)
    assert isinstance(g.class_model, OctagonClasses)
    assert isinstance(g.relation_model, Translation)
    assert isinstance(g.measure, OctagonBounds)
    assert isinstance(g.objective, ShapeObjective)
    assert g.parts["classes"] == "octagon product" and g.parts["relations"] == "translation"
    # the options of the paper's reading: only gci2 has a negative, a role assertion is the
    # distance of the translated point
    assert g.neg_capable_gcis == {"gci2"} and g.objective.assertion == "distance"
    assert g.objective.disjoint_margin is not None
    assert g.regularizers == ()
    assert model.training_regime is None
    assert METHODS["octagon"] is SM.octagon
    # the same combination without the class OctagonEL
    again = ComposedEL(family, method="octagon", embed_dim=D).module
    assert type(again) is type(g) and again.parts == g.parts


def test_octagon_min_width_is_a_regulariser_of_the_combination(family):
    g = OctagonEL(family, embed_dim=D, min_width=0.1).module
    assert [type(r).__name__ for r in g.regularizers] == ["MinWidthPenalty"]
    assert float(g.regularization_loss()) >= 0


def test_prism_is_triangles_times_boxes_with_a_mean_squared_loss(family):
    model = PrismEL(family, embed_dim=D)
    _is_geometry(model)
    g = model.module
    assert isinstance(g, SM.PrismModule) and not isinstance(g, StructuralGeometry)
    assert isinstance(g.class_model, PrismClasses)
    assert isinstance(g.relation_model, Translation)
    assert isinstance(g.measure, PrismBounds)
    assert isinstance(g.objective, ShapeObjective) and g.objective.gci3_reading == "overlap"
    assert g.objective.gci2_negative == "overlap"
    assert g.parts["classes"] == "triangle x box"
    assert g.neg_capable_gcis == {"gci2", "object_property_assertion"}
    regime = model.training_regime
    assert isinstance(regime, SquaredMeanLoss) and regime.square_positive
    assert (regime.pos_target, regime.neg_target) == (0.0, 1.0)
    assert METHODS["prism"] is SM.prism


def test_prism_structural_reads_the_structure_of_the_ontology(family):
    model = PrismEL(family, embed_dim=D, objective="structural")
    g = model.module
    assert isinstance(g, StructuralGeometry) and isinstance(g, SM.PrismModule)
    assert isinstance(g.class_model, DefinedClasses)
    assert isinstance(g.class_model.base, PrismClasses)
    assert type(g.objective).__name__ == "StructuralReading"
    assert {type(r).__name__ for r in g.regularizers} >= {"ExistencePenalty", "SizePenalty"}
    assert not g.wants_structural_tables
    assert isinstance(model.training_regime, SquaredMeanLoss)
    assert not model.training_regime.square_positive
    assert model.objective == "structural"


def test_eike_is_two_linked_spaces_with_a_ranking_loss(abox):
    model = EIKE(abox, embed_dim=D, margins=(0.5, 0.3, 0.7))
    _is_geometry(model)
    g = model.module
    assert isinstance(g, SM.EIKEModule)
    assert isinstance(g.class_model, LinkedClasses)
    assert isinstance(g.class_model.first, EllipsoidClasses)
    assert isinstance(g.class_model.second, VectorClasses)
    assert isinstance(g.measure, LinkedMeasure)
    assert isinstance(g.measure.first, EllipsoidMeasure)
    assert isinstance(g.measure.second, CosineOrder)
    assert g.objective.ranking == {"gci0": 0.5, "class_assertion": 0.3,
                                   "object_property_assertion": 0.7}
    assert set(g.supported_forms) == {"gci0", "class_assertion", "object_property_assertion"}
    # the paper's negatives and its loss are two regimes of one composed regime
    regime = [r for r in model.training_regime.regimes if isinstance(r, PerColumnCorruption)]
    assert len(regime) == 1 and len(model.training_regime.regimes) == 2
    regime = regime[0]
    assert regime.columns == {"gci0": (0, 1), "class_assertion": (0, 1),
                              "object_property_assertion": (0, 2)}
    assert METHODS["eike"] is SM.eike


def test_concept2box_is_soft_boxes_in_context_with_three_measures(abox):
    model = Concept2Box(abox, embed_dim=D)
    _is_geometry(model)
    g = model.module
    assert isinstance(g, SM.Concept2BoxModule)
    assert isinstance(g.class_model, ContextualSoftBoxClasses)
    assert isinstance(g.measure, ByRegion)
    assert [type(m).__name__ for _, _, m in g.measure.rules] == [
        GumbelVolumeRatio.__name__, PointToRegion.__name__, PointDistance.__name__]
    assert g.objective.context_axioms and g.objective.gci2_reading == "axiom"
    assert set(g.supported_forms) == {"gci0", "gci2", "class_assertion",
                                      "object_property_assertion"}
    assert model.training_regime is None
    assert METHODS["concept2box"] is SM.concept2box


def test_the_modules_in_mowl_nn_are_the_part_built_ones():
    import mowl.nn as N
    for name in ("OctagonELModule", "PrismModule", "EIKEModule", "Concept2BoxModule"):
        cls = getattr(N, name)
        assert issubclass(cls, ComposedGeometry), name
        assert cls.__module__ == "mowl.representation.library.shapes_methods", name
    assert isinstance(N.PrismModule(4, 2, 3, embed_dim=6, objective="structural"),
                      StructuralGeometry)
    assert not isinstance(N.PrismModule(4, 2, 3, embed_dim=6), StructuralGeometry)


def test_the_direct_implementations_are_in_the_baselines_and_unchanged_in_name():
    for method, name in (("octagon", "OctagonEL"), ("prism", "PrismEL"), ("eike", "EIKE"),
                         ("concept2box", "Concept2Box")):
        module = importlib.import_module(f"mowl.baselines.models.{method}.model")
        assert getattr(module, name).__module__.startswith("mowl.baselines")


# -- (b) swaps ----------------------------------------------------------------------------------

def test_octagon_swaps(family, abox):
    # an objective option
    _train_and_evaluate(ComposedEL(family, method="octagon", embed_dim=D, margin=0.05,
                                   disjoint_margin=0.2, neg_margin=0.3), family)
    # an added regulariser (the penalty on narrow octagons, and a penalty on the centres) and
    # another training regime (a step decay of the learning rate)
    model = OctagonEL(abox, embed_dim=D, min_width=0.05)
    model.training_regime = ComposedRegime(
        Regularization(NormPenalty(parameters="class_embed")), StepDecay([1], gamma=0.5))
    _train_and_evaluate(model, abox)
    # the parts of the method composed by hand, without the builder: octagons, translations,
    # the exact octagon measure
    by_hand = ComposedEL(abox, classes=lambda n, d: OctagonClasses(n, d), relations="translation",
                         measure=OctagonBounds(), embed_dim=D, margin=0.02)
    by_hand.train(epochs=1)
    # the learning rate decay of OctagonEL is the regime ExponentialDecay
    decayed = OctagonEL(family, embed_dim=D, lr_decay=0.9)
    assert isinstance(decayed.training_regime, ExponentialDecay)
    _train_and_evaluate(decayed, family)


def test_prism_swaps(family, abox):
    # the other reading of the method, by one argument
    _train_and_evaluate(ComposedEL(abox, method="prism", embed_dim=D, reading="structural"),
                        abox)
    # an objective option: the earlier test of the negative existential loss
    _train_and_evaluate(PrismEL(family, embed_dim=D, gci2_negative_triangle="inradius"),
                        family)
    # another training regime: the sum of the mean losses instead of ELBE's squared means
    model = PrismEL(abox, embed_dim=D)
    model.training_regime = SumLoss()
    _train_and_evaluate(model, abox)
    # another measure: the sound emptiness and conjunction tests of the part (the method
    # uses the published ones)
    sound = ComposedEL(family, classes=lambda n, d: PrismClasses(n, d),
                       relations="translation", measure=PrismBounds(), embed_dim=D,
                       margin=0.05)
    _train_and_evaluate(sound, family)
    # an added regulariser
    reg = PrismEL(family, embed_dim=D)
    reg.training_regime = ComposedRegime(reg.training_regime,
                                         Regularization(NormPenalty(parameters="tri_center")))
    _train_and_evaluate(reg, family)


def test_eike_swaps(abox):
    # the sound containment test of the ellipsoid and the hinge on the lengths (the
    # published formulas are the method's defaults)
    _train_and_evaluate(ComposedEL(abox, method="eike", embed_dim=D, containment="bounding",
                                   norm_term="hinge"), abox)
    # a learned instance map
    _train_and_evaluate(EIKE(abox, embed_dim=D, instance_map="matrix", alpha=0.5), abox)
    # another negative sampler: one of the two sides, chosen per axiom
    model = EIKE(abox, embed_dim=D)
    model.training_regime = EitherSideCorruption(columns={
        "gci0": (0, 1), "class_assertion": (0, 1), "object_property_assertion": (0, 2)})
    _train_and_evaluate(model, abox)


def test_concept2box_swaps(abox):
    # the paper's triple for C subclass of some r.D: an objective option
    _train_and_evaluate(Concept2Box(abox, embed_dim=D, gci2_reading="triple"), abox)
    # another training regime: negatives count half
    model = Concept2Box(abox, embed_dim=D)
    model.training_regime = SumLoss(neg_weight=0.5)
    _train_and_evaluate(model, abox)
    # an added regulariser, and the margins and weights of the two other views
    model = ComposedEL(abox, method="concept2box", embed_dim=D, kge_margin=0.5,
                       cross_margin=0.3, instance_weight=0.5,
                       regularizers=(NormPenalty(parameters="class_embed"),))
    assert len(model.module.regularizers) == 1
    _train_and_evaluate(model, abox)


# -- (c) the card and the parts agree ------------------------------------------------------------

_FORMS_OF_TASK = {"subsumption": "gci0", "satisfiability": "gci0_bot",
                  "disjointness": "gci1_bot", "instance-checking": "class_assertion",
                  "role-assertion": "object_property_assertion"}


@pytest.mark.parametrize("key, build", [
    ("octagon", lambda ds: OctagonEL(ds, embed_dim=D)),
    ("prism", lambda ds: PrismEL(ds, embed_dim=D)),
    ("eike", lambda ds: EIKE(ds, embed_dim=D)),
    ("concept2box", lambda ds: Concept2Box(ds, embed_dim=D)),
])
def test_the_card_states_what_the_parts_read(key, build, abox):
    card = get_method(key)
    model = build(abox)
    g = model.module
    # the axiom forms of the card are the forms the geometry reads
    assert set(card.axioms) == set(g.supported_forms), key
    # every task of the card has the axiom form it needs among them
    for task in card.tasks:
        assert _FORMS_OF_TASK[task] in g.supported_forms, (key, task)
    # the card points at the classes users get
    cls = importlib.import_module(card.implementation.rsplit(".", 1)[0])
    assert getattr(cls, card.implementation.rsplit(".", 1)[1]) is type(model)
    module = importlib.import_module(card.module.rsplit(".", 1)[0])
    assert getattr(module, card.module.rsplit(".", 1)[1]) is type(g)
    # the geometry of the model is the module itself
    assert geometry_of(model) is g


def test_the_card_of_octagon_states_no_role_axioms(abox):
    g = OctagonEL(abox, embed_dim=D).module
    assert not g.role_axiom_capable or "role_inclusion" not in g.supported_forms
    assert "role_inclusion" not in g.supported_forms
