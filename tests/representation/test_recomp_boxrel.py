"""ELBE, TransBox, BoxEL, BoxSEL and ELK-Em are combinations of parts.

For each method: the named combination can be read back (``model.module.parts``, the builder
of the method), the parts can be swapped (another relation model, another measure, another
objective option, an added regulariser, another training regime, another evaluation regime)
and the result still trains for one epoch and evaluates to finite numbers, and the catalogue
card of the method agrees with the parts.
"""
import logging

import pytest
import torch as th

from mowl.catalog import get_method
from mowl.evaluation import InclusionAccuracyRegime
from mowl.models import BoxEL, BoxSEL, ELBE, ELKEm, TransBox
from mowl.nn import BoxELModule, BoxSELModule, ELBEModule, ELKEmModule, TransBoxModule
from mowl.representation import (METHODS, REPRESENTATIONS, Bounds, BoxClasses, ComposedGeometry,
                                 FORMS, Geometry, Translation)
from mowl.representation.library.boxrel_boxes import FixableBoxes, MinDeltaBoxes
from mowl.representation.library.boxrel_measures import (AveragedBounds, MaskedBounds,
                                                         VolumeShare)
from mowl.representation.library.boxrel_objectives import (BoxSELObjective, ELKEmObjective,
                                                           TransBoxObjective, boxel_objective)
from mowl.representation.library.boxrel_regularizers import (LocationPenalty, SizeCapPenalty,
                                                             VolumePenalty)
from mowl.representation.library.boxrel_relations import Affine, AffineBoxes, TranslationBoxes
from mowl.representation.objectives import Objective
from mowl.training import NormPenalty, SumLoss
from tests.datasetFactory import FamilyDataset

logging.disable(logging.CRITICAL)

EMBED = 4

#: key -> (class model, relation model, measure, objective) the named combination is made of
COMBINATIONS = {
    "elbe": (BoxClasses, Translation, Bounds, Objective),
    "transbox": (BoxClasses, TranslationBoxes, MaskedBounds, TransBoxObjective),
    "boxel": (MinDeltaBoxes, Affine, VolumeShare, Objective),
    "boxsel": (MinDeltaBoxes, Affine, VolumeShare, BoxSELObjective),
    "elkem": (FixableBoxes, AffineBoxes, AveragedBounds, ELKEmObjective),
}
MODELS = {"elbe": ELBE, "transbox": TransBox, "boxel": BoxEL, "boxsel": BoxSEL,
          "elkem": ELKEm}
MODULES = {"elbe": ELBEModule, "transbox": TransBoxModule, "boxel": BoxELModule,
           "boxsel": BoxSELModule, "elkem": ELKEmModule}


@pytest.fixture(scope="module")
def dataset():
    return FamilyDataset()


def build(key, dataset, **options):
    cls = MODELS[key]
    kwargs = dict(embed_dim=EMBED, learning_rate=0.01, batch_size=64)
    if key == "boxsel":
        kwargs["conditionals"] = [("http://Mother", "http://Female", 0.6)]
    if key == "elkem":
        kwargs["lr_decay_epoch"] = None
    kwargs.update(options)
    return cls(dataset, **kwargs)


def swapped(model, classes=None, relations=None, measure=None, objective=None,
            regularizers=None):
    """The module of ``model`` with some of its parts replaced (the part objects are the
    attributes of :class:`~mowl.representation.ComposedGeometry`), so that the model keeps its
    class and whatever the model asks of its module."""
    m = model.module
    if classes is not None:
        m.class_model = classes
        m.class_embed = classes.centres
    if relations is not None:
        m.relation_model = relations
        m.rel_embed = relations.vectors
    if measure is not None:
        m.measure = measure
    m.measure.bind(m.relation_model)
    if objective is not None:
        m.objective = objective
        m.neg_capable_gcis = frozenset(objective.neg_forms)
    if regularizers is not None:
        m.regularizers = tuple(regularizers)
    return model


def train_and_evaluate(model, dataset):
    """One epoch of training, then the regions of the classes and the inclusion accuracy of
    the evaluation regime: finite numbers."""
    model.train(epochs=1)
    assert all(th.isfinite(p).all() for p in model.module.parameters())
    geometry = model.module
    ids = th.arange(min(5, geometry.num_classes))
    cost = geometry.inside(geometry.classes(ids), geometry.classes(ids.flip(0)))
    assert th.isfinite(cost).all()
    result = InclusionAccuracyRegime(splits=("train",)).evaluate(model, dataset)
    assert 0.0 <= result["train"]["accuracy"] <= 1.0


# ==========================================================================================
# (a) the named combination can be read back
@pytest.mark.parametrize("key", sorted(COMBINATIONS))
def test_the_class_users_get_is_the_named_combination(key, dataset):
    model = build(key, dataset)
    module = model.module
    assert isinstance(module, MODULES[key]) and isinstance(module, ComposedGeometry)
    assert isinstance(module, Geometry)
    classes, relations, measure, objective = COMBINATIONS[key]
    assert type(module.class_model) is classes
    assert type(module.relation_model) is relations
    assert type(module.measure) is measure
    assert isinstance(module.objective, objective)
    parts = module.parts
    assert parts["classes"] == "box"
    assert parts["relations"] == relations.part
    assert parts["measure"] == module.measure.part


@pytest.mark.parametrize("key", sorted(COMBINATIONS))
def test_the_builder_of_the_method_defines_the_same_combination(key, dataset):
    """The module class takes its parts from the builder registered under the key of the
    method: the builder gives the same kinds of parts and the same options."""
    model = build(key, dataset)
    module = model.module
    n, r = len(model.class_index_dict), len(model.object_property_index_dict)
    built = METHODS[key](n, r, 0, embed_dim=EMBED)
    assert type(built.class_model) is type(module.class_model)
    assert type(built.relation_model) is type(module.relation_model)
    assert type(built.measure) is type(module.measure)
    assert type(built.objective) is type(module.objective)
    # (a module of a model that was told the number of individuals has a table for them)
    assert {k: v for k, v in built.parts.items() if k != "individuals"} == {
        k: v for k, v in module.parts.items() if k != "individuals"}
    if key != "elkem":      # the size penalty of ELK-Em is the model's own term (reg_factor)
        assert [type(x) for x in built.regularizers] == [type(x) for x in module.regularizers]


def test_the_regularisers_of_the_boxes_are_parts():
    ds = FamilyDataset()
    assert [type(x) for x in build("boxel", ds).module.regularizers] == [LocationPenalty]
    assert {type(x) for x in build("boxsel", ds).module.regularizers} == {LocationPenalty}
    assert {type(x).__name__ for x in build("transbox", ds).module.regularizers} == {
        "NormPenalty", "MaxNormPenalty"}


def test_elbe_trains_with_a_regime_and_the_other_models_with_their_own_loop(dataset):
    from mowl.training import MSETargetLoss
    regime = build("elbe", dataset).training_regime
    assert isinstance(regime, MSETargetLoss) and regime.reduce == "mean_first"


# ==========================================================================================
# (b) swaps
def test_elbe_with_another_relation_model(dataset):
    model = build("elbe", dataset)
    n = model.module.num_classes
    swapped(model, relations=TranslationBoxes(len(model.object_property_index_dict), EMBED))
    assert model.module.parts["relations"] == "box of translations"
    train_and_evaluate(model, dataset)
    assert n == model.module.num_classes


def test_elbe_with_another_measure(dataset):
    model = swapped(build("elbe", dataset), measure=VolumeShare(temperature=1.0))
    assert model.module.parts["measure"] == "smoothed volume ratio"
    train_and_evaluate(model, dataset)


def test_elbe_with_another_objective_option(dataset):
    model = swapped(build("elbe", dataset), objective=Objective.elbe(0.05, gci3="inclusion"))
    train_and_evaluate(model, dataset)


def test_elbe_with_a_regulariser_and_another_loss(dataset):
    model = build("elbe", dataset)
    model.module.regularizers = model.module.regularizers + (NormPenalty("class_embed"),)
    model.training_regime = SumLoss(neg_weight=0.5)
    train_and_evaluate(model, dataset)
    assert float(model.module.regularization_loss()) > 0


def test_transbox_with_another_measure(dataset):
    plain = Bounds(separation="every", extent="norm", conjunction="nonempty")
    model = swapped(build("transbox", dataset), measure=plain)
    assert type(model.module.measure) is Bounds
    train_and_evaluate(model, dataset)


def test_transbox_with_other_classes_and_a_regulariser(dataset):
    model = build("transbox", dataset)
    classes = BoxClasses(model.module.num_classes, EMBED, init="joint")
    swapped(model, classes=classes,
            regularizers=model.module.regularizers + (NormPenalty("class_embed"),))
    train_and_evaluate(model, dataset)


def test_transbox_with_another_loss_and_the_formulation_of_the_code(dataset):
    model = build("transbox", dataset, formulation="reference", margin=0.0)
    model.training_regime = SumLoss()
    assert model.module.objective.reference
    train_and_evaluate(model, dataset)


def test_boxel_with_another_relation_model(dataset):
    model = build("boxel", dataset)
    swapped(model, relations=TranslationBoxes(len(model.object_property_index_dict), EMBED))
    assert model.module.parts["relations"] == "box of translations"
    train_and_evaluate(model, dataset)


def test_boxel_with_another_measure(dataset):
    # the volume product of C and D (the reading of C and D being disjoint) is a volume
    # measure's; bounds read disjointness as the two boxes being apart
    model = swapped(build("boxel", dataset), measure=Bounds(),
                    objective=boxel_objective(0.0, gci1_bot="apart"))
    assert model.module.parts["measure"] == "bounds per coordinate"
    train_and_evaluate(model, dataset)


def test_boxel_with_another_objective_option_and_a_regulariser(dataset):
    model = build("boxel", dataset)
    swapped(model, objective=boxel_objective(0.0, gci1_bot="apart"),
            regularizers=model.module.regularizers + (VolumePenalty(1.0),))
    train_and_evaluate(model, dataset)


def test_boxsel_with_another_measure_and_the_penalty_weights(dataset):
    model = build("boxsel", dataset)
    model.module.volume_weight = 0.5
    assert {type(x) for x in model.module.regularizers} == {LocationPenalty, VolumePenalty}
    swapped(model, measure=VolumeShare(temperature=None, strict_margin="ignore"))
    train_and_evaluate(model, dataset)
    assert model.mean_absolute_error() >= 0


def test_boxsel_conditionals_are_read_by_the_geometry(dataset):
    model = build("boxsel", dataset)
    assert isinstance(model.module.objective, BoxSELObjective)
    model.train(epochs=1)
    assert isinstance(model.probability("http://Mother", "http://Female"), float)
    loss = model.training_regime.additional_loss(0)     # the regime ConditionalStatements
    assert loss is not None and th.isfinite(loss)


def test_elkem_with_another_measure(dataset):
    model = swapped(build("elkem", dataset), measure=AveragedBounds(extent="norm"))
    train_and_evaluate(model, dataset)


def test_elkem_with_another_objective_option_and_a_regulariser(dataset):
    model = build("elkem", dataset)
    swapped(model, objective=ELKEmObjective(0.05, gci2_overlap=True),
            regularizers=(SizeCapPenalty(weight=0.5),))
    train_and_evaluate(model, dataset)


def test_elkem_with_the_training_regime_of_the_authors_code(dataset):
    from mowl.training import ELKEmCodeTraining
    model = build("elkem", dataset, variant="reference")
    model.training_regime = ELKEmCodeTraining(batch_size=8, k_neg=2, select=False, seed=0)
    train_and_evaluate(model, dataset)


# ==========================================================================================
# (c) the catalogue card and the parts agree
@pytest.mark.parametrize("key", sorted(COMBINATIONS))
def test_the_card_and_the_parts_agree(key, dataset):
    card = get_method(key)
    geometry = build(key, dataset).module
    # the axiom forms the card states are the forms the geometry reads
    assert set(card.axioms) <= set(geometry.supported_forms), key
    assert set(card.axioms) <= set(FORMS) | {"role_inclusion", "role_chain"}
    # the parts the representation catalogue names
    rep = REPRESENTATIONS[key]
    parts = geometry.parts
    assert parts["classes"] in {c.value for c in rep.classes}
    assert parts["relations"] in {r.value for r in rep.relations}
    # the class of the card is the class users get and the module is the module of the card
    assert card.region == "box"
    assert card.implementation == f"mowl.models.{key}.model.{MODELS[key].__name__}"
    module_path = card.module
    assert module_path == f"mowl.nn.el.{key}.module.{MODULES[key].__name__}"


@pytest.mark.parametrize("key", sorted(COMBINATIONS))
def test_the_tasks_of_the_card_are_answered_by_the_geometry(key, dataset):
    card = get_method(key)
    geometry = build(key, dataset).module
    ids = th.arange(3)
    a, b = geometry.classes(ids), geometry.classes(ids.flip(0))
    if "subsumption" in card.tasks:
        assert th.isfinite(geometry.inside(a, b)).all()
    if "disjointness" in card.tasks:
        assert th.isfinite(geometry.apart(a, b)).all()
    if "instance-checking" in card.tasks:
        assert "class_assertion" in geometry.supported_forms
    if "role-assertion" in card.tasks:
        assert "object_property_assertion" in geometry.supported_forms
