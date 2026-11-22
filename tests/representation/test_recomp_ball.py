"""The ball family (ELEmbeddings, EmEL++, EmEL(var), RegD on boxes and on balls) is its parts.

The classes users get (``mowl.models``, ``mowl.nn``) are combinations of a class model, a
relation model, a measure, an objective and regularisers; the direct implementations are in
``mowl.baselines`` and are only the reference. These tests read the combinations back, swap
their parts, and check that the catalogue card and the parts say the same thing.
"""
import pytest
import torch as th

from mowl.catalog import get_method
from mowl.logic import Construct
from mowl.models import ELEmbeddings, EmELPP, EmELVar, RegDELBE, RegDELEm
from mowl.nn import ELEmModule, EmELPPModule, EmELVarModule, RegDELBEModule, RegDELEmModule
from mowl.representation import (
    METHODS, REPRESENTATIONS, Ball, Box, BallClasses, BoxClasses, Bounds, CentreDistance,
    ComposedGeometry, Geometry, Objective, ROLE_FORMS, Translation)
from mowl.representation.library.ballfam_parts import (
    BoundaryDepth, RegDObjective, ToleranceTranslation)
from mowl.training import (MaxNormPenalty, NormPenalty, PerAxiomNormPenalty,
                           UnboundedNegativeLoss)
from tests.datasetFactory import FamilyDataset

DIM = 4

#: key -> (model class, options of the tiny model)
MODELS = {
    "elem": (ELEmbeddings, dict(margin=0.1)),
    "emelpp": (EmELPP, dict(margin=0.1, load_role_axioms=True)),
    "emelvar": (EmELVar, dict(margin=0.1, load_role_axioms=True, sigma_reg_weight=0.5)),
    "regd": (RegDELBE, dict(margin=0.05, num_neg=2)),
    "regd_elem": (RegDELEm, dict(margin=0.05, num_neg=2)),
}

#: key -> (class model, relation model, measure, region) as the method defines them
COMBINATIONS = {
    "elem": ("ball", "translation", "centre distance", Ball),
    "emelpp": ("ball", "translation", "centre distance", Ball),
    "emelvar": ("ball", "translation with tolerance", "centre distance", Ball),
    "regd": ("box", "translation", "boundary and depth", Box),
    "regd_elem": ("ball", "translation", "boundary and depth", Ball),
}

MODULES = {"elem": ELEmModule, "emelpp": EmELPPModule, "emelvar": EmELVarModule,
           "regd": RegDELBEModule, "regd_elem": RegDELEmModule}


@pytest.fixture(scope="module")
def dataset():
    return FamilyDataset()


def build(key, dataset, **extra):
    cls, options = MODELS[key]
    return cls(dataset, embed_dim=DIM, batch_size=64, **{**options, **extra})


def with_parts(model, **replace):
    """The model with some parts of its combination replaced: the same parts, a new
    :class:`~mowl.representation.ComposedGeometry`."""
    g = model.module
    parts = dict(classes=g.class_model, relations=g.relation_model, measure=g.measure,
                 objective=g.objective, individuals=g.individual_model,
                 regularizers=g.regularizers)
    parts.update(replace)
    model.module = ComposedGeometry(
        parts["classes"], parts["relations"], parts["measure"], parts["objective"],
        individuals=parts["individuals"], regularizers=parts["regularizers"])
    return model


def trains_and_evaluates(model, epochs=1):
    """One epoch on the tiny data, then the loss of every form and an evaluation score, all
    finite."""
    model.train(epochs=epochs)
    module = model.module
    for form, data in model.training_datasets.items():
        if len(data):
            assert th.isfinite(module(data[:], form)).all(), form
    model.eval_gci_name = "gci2"
    rows = th.tensor([[0, 0, 1], [1, 0, 2], [2, 0, 3]])
    with th.no_grad():
        assert th.isfinite(model.eval_method(rows)).all()
    return model


# -- (a) the combination, read back -----------------------------------------------------------


@pytest.mark.parametrize("key", MODELS)
def test_the_model_is_the_named_combination(key, dataset):
    model = build(key, dataset)
    module = model.module
    classes, relations, measure, region = COMBINATIONS[key]
    assert isinstance(module, Geometry) and isinstance(module, ComposedGeometry)
    assert isinstance(module, MODULES[key])
    assert module.parts["classes"] == classes
    assert module.parts["relations"] == relations
    assert module.parts["measure"] == measure
    assert module.class_model.region is region
    # the same combination as the builder of the method
    built = METHODS[key](len(model.class_index_dict), len(model.object_property_index_dict),
                         len(model.individual_index_dict), embed_dim=DIM)
    assert built.parts == module.parts
    assert type(built.objective) is type(module.objective)
    assert type(built.class_model) is type(module.class_model)
    assert type(built.relation_model) is type(module.relation_model)
    assert type(built.measure) is type(module.measure)
    assert [type(r) for r in built.regularizers] == [type(r) for r in module.regularizers]


def test_elem_parts():
    m = ELEmModule(6, 2, 3, embed_dim=DIM, margin=0.1)
    assert isinstance(m.class_model, BallClasses) and isinstance(m.relation_model, Translation)
    assert isinstance(m.measure, CentreDistance) and m.measure.conjunction == "published"
    assert type(m.objective) is Objective and m.objective.slack
    assert [type(r) for r in m.regularizers] == [NormPenalty]
    # the tables of the parts are the tables under the names of the earlier module
    assert m.class_embed is m.class_model.centres and m.class_rad is m.class_model.radii
    assert m.rel_embed is m.relation_model.vectors
    assert m.ind_embed is m.individual_model.centres and m.ind_rad is m.individual_model.radii


def test_emelpp_and_emelvar_parts():
    m = EmELPPModule(6, 3, 2, embed_dim=DIM)
    assert isinstance(m.measure, CentreDistance) and not m.regularizers
    assert m.objective.unit_norm and m.objective.gci0_shift is not None
    assert m.role_axiom_capable and m.relation_model.direction == "row"
    v = EmELVarModule(6, 3, 2, embed_dim=DIM, sigma_init=0.3, sigma_reg_weight=0.5)
    assert isinstance(v.relation_model, ToleranceTranslation)
    assert v.rel_sigma is v.relation_model.tolerances
    assert float(v.rel_sigma.weight[0]) == pytest.approx(0.3)
    assert v.objective.tolerance_weight == 0.5 == v.sigma_reg_weight


def test_regd_parts():
    box = RegDELBEModule(6, 2, 3, embed_dim=DIM, lamda=0.5, p=2)
    assert isinstance(box.class_model, BoxClasses) and isinstance(box.measure, BoundaryDepth)
    assert isinstance(box.objective, RegDObjective)
    assert box.objective.regd_forms == {"gci0", "gci1", "gci2", "gci3"}
    assert (box.lamda, box.p) == (0.5, 2) == (box.measure.lamda, box.measure.p)
    assert box.class_offset is box.class_model.offsets
    ball = RegDELEmModule(6, 2, 3, embed_dim=DIM, radius="softplus")
    assert isinstance(ball.class_model, BallClasses)
    assert ball.class_model.radius_form == "softplus" == ball.radius
    assert ball.objective.regd_forms == {"gci0", "gci2", "gci3"}
    assert [type(r) for r in ball.regularizers] == [NormPenalty]


@pytest.mark.parametrize("key", MODELS)
def test_the_options_are_settings_of_the_parts(key, dataset):
    m = build(key, dataset).module
    m.margin = 0.3
    assert m.objective.margin == 0.3
    if hasattr(m, "gci1_radius_term"):
        m.gci1_radius_term = False
        assert not m._ball_measure().radius_term
    if hasattr(m, "negative_loss_form"):
        m.negative_loss_form = "linear"
        assert m.objective.gci2_negative == "linear"
    if hasattr(m, "energy_mode"):
        with m.scoring():
            assert m.measure.raw
        assert not m.measure.raw


# -- (b) swaps --------------------------------------------------------------------------------


def test_elem_with_another_relation_model_and_a_regulariser(dataset):
    model = build("elem", dataset)
    g = model.module
    relations = ToleranceTranslation(len(g.relation_model), DIM)
    with_parts(model, relations=relations)
    assert model.module.parts["relations"] == "translation with tolerance"
    trains_and_evaluates(model)


def test_elem_with_another_measure(dataset):
    model = build("elem", dataset)
    with_parts(model, measure=BoundaryDepth(host=CentreDistance(True, conjunction="published")))
    assert model.module.parts["measure"] == "boundary and depth"
    trains_and_evaluates(model)


def test_elem_with_another_objective_option_and_regulariser(dataset):
    model = build("elem", dataset)
    objective = Objective.elem(0.1, gci3="inclusion", all_negatives=True)
    with_parts(model, objective=objective,
               regularizers=model.module.regularizers + (MaxNormPenalty("class_embed", 1.5),))
    assert len(model.module.regularizers) == 2
    trains_and_evaluates(model)


def test_elem_with_a_training_regime(dataset):
    from mowl.training import ComposedRegime
    model = build("elem", dataset)
    model.training_regime = ComposedRegime(UnboundedNegativeLoss(), PerAxiomNormPenalty())
    trains_and_evaluates(model)
    assert model.module.negative_loss_form == "linear"
    assert model.module.norm_penalty == "per_axiom" and not model.module.regularizers
    assert model.module.objective.gci2_negative == "linear"


def test_emelpp_with_another_relation_model_and_class_model(dataset):
    model = build("emelpp", dataset)
    g = model.module
    with_parts(model, relations=ToleranceTranslation(len(g.relation_model), DIM,
                                                     direction="row"))
    trains_and_evaluates(model)
    model = build("emelpp", dataset)
    with_parts(model, classes=BallClasses(len(model.module.class_model), DIM,
                                          radius="softplus"))
    assert model.module.class_model.radius_form == "softplus"
    trains_and_evaluates(model)


def test_emelpp_with_another_measure_and_objective(dataset):
    model = build("emelpp", dataset)
    with_parts(model, measure=BoundaryDepth(host=CentreDistance(True, "published")))
    trains_and_evaluates(model)
    model = build("emelpp", dataset)
    objective = model.module.objective
    objective.unit_norm = ()
    with_parts(model, objective=objective,
               regularizers=(NormPenalty(parameters=("class_embed",), target=1.0),))
    trains_and_evaluates(model)


def test_emelvar_with_boxes_and_with_another_objective_option(dataset):
    model = build("emelvar", dataset)
    g = model.module
    with_parts(model, classes=BoxClasses(len(g.class_model), DIM), measure=Bounds(),
               individuals=None)
    assert model.module.parts["classes"] == "box"
    assert model.module.parts["relations"] == "translation with tolerance"
    trains_and_evaluates(model)
    model = build("emelvar", dataset)
    model.module.objective.tolerance_weight = 0.0
    trains_and_evaluates(model)


def test_emelvar_without_the_tolerance(dataset):
    model = build("emelvar", dataset)
    g = model.module
    with_parts(model, relations=Translation(len(g.relation_model), DIM, direction="row"))
    assert model.module.parts["relations"] == "translation"
    trains_and_evaluates(model)


def test_regd_with_another_host_measure_and_relation_model(dataset):
    model = build("regd", dataset)
    g = model.module
    measure = BoundaryDepth(1.0, 1, 0.0, 0.0, 1e-4, True, host=Bounds(separation="any"))
    measure.empty_boxes = "formula"
    with_parts(model, measure=measure)
    trains_and_evaluates(model)
    model = build("regd", dataset)
    with_parts(model, relations=ToleranceTranslation(len(g.relation_model), DIM))
    assert model.module.parts["relations"] == "translation with tolerance"
    trains_and_evaluates(model)


def test_regd_with_a_regulariser_and_the_plain_bounds_measure(dataset):
    model = build("regd", dataset)
    g = model.module
    with_parts(model, regularizers=(NormPenalty(parameters=("class_embed",), target=1.0),))
    trains_and_evaluates(model)
    model = build("regd", dataset)
    with_parts(model, measure=Bounds(separation="every", extent="norm",
                                     conjunction="nonempty"), objective=Objective.elbe(0.05))
    assert model.module.parts["measure"] == "bounds per coordinate"
    trains_and_evaluates(model)


def test_regd_elem_with_another_objective_and_measure(dataset):
    model = build("regd_elem", dataset)
    g = model.module
    objective = RegDObjective(Objective.elem(0.05), regd_forms=("gci0", "gci2", "gci3"),
                              flip_gci2=True)
    with_parts(model, objective=objective)
    trains_and_evaluates(model)
    model = build("regd_elem", dataset)
    with_parts(model, measure=CentreDistance(True, "published"), objective=Objective.elem(0.05))
    assert model.module.parts["measure"] == "centre distance"
    trains_and_evaluates(model)


def test_regd_elem_with_another_relation_model(dataset):
    model = build("regd_elem", dataset)
    g = model.module
    with_parts(model, relations=ToleranceTranslation(len(g.relation_model), DIM))
    trains_and_evaluates(model)


# -- the modules equal their direct implementations, with individuals ----------------------------


def _direct(key):
    from mowl.baselines.nn.el.elem.module import ELEmModule as OldELEm
    from mowl.baselines.nn.el.emelpp.module import (EmELPPModule as OldPP,
                                                    EmELVarModule as OldVar)
    from mowl.baselines.nn.el.regd.module import (RegDELBEModule as OldBox,
                                                  RegDELEmModule as OldBall)
    return {"elem": OldELEm, "emelpp": OldPP, "emelvar": OldVar, "regd": OldBox,
            "regd_elem": OldBall}[key]


FORM_COLUMNS = {"gci0": 2, "gci1": 3, "gci2": 3, "gci3": 3, "gci0_bot": 2, "gci1_bot": 2,
                "gci3_bot": 3, "class_assertion": 2, "object_property_assertion": 3,
                "role_inclusion": 2, "role_chain": 3}


def _rows(form, n, r, i, size=20, seed=1):
    g = th.Generator().manual_seed(seed)
    highs = {"gci0": [n, n], "gci1": [n, n, n], "gci2": [n, r, n], "gci3": [r, n, n],
             "gci0_bot": [n, n], "gci1_bot": [n, n], "gci3_bot": [r, n, n],
             "class_assertion": [i, n], "object_property_assertion": [i, r, i],
             "role_inclusion": [r, r], "role_chain": [r, r, r]}[form]
    return th.stack([th.randint(0, h, (size,), generator=g) for h in highs], dim=1)


def test_emelpp_has_no_dele_negatives_to_enable():
    for cls in (EmELPPModule, EmELVarModule):
        with pytest.raises(NotImplementedError):
            cls(4, 2, embed_dim=3).enable_all_negative_losses()


@pytest.mark.parametrize("key", MODELS)
@pytest.mark.parametrize("negatives", [False, True])
def test_every_form_equals_the_direct_module_with_individuals(key, negatives):
    n, r, i = 7, 3, 4
    kwargs = {"embed_dim": 5, "margin": 0.07}
    th.manual_seed(0)
    old = _direct(key)(n, r, i, **kwargs)
    th.manual_seed(0)
    new = MODULES[key](n, r, i, **kwargs)
    if negatives and key in ("emelpp", "emelvar"):
        pytest.skip("EmEL++ has no DELE negatives to enable (see the test above)")
    if negatives:
        old.enable_all_negative_losses()
        new.enable_all_negative_losses()
        assert new.neg_capable_gcis == old.neg_capable_gcis
    new.load_state_dict(old.state_dict())
    forms = [f for f in FORM_COLUMNS if f not in ROLE_FORMS or old.role_axiom_capable]
    assert new.role_axiom_capable == old.role_axiom_capable
    for form in forms:
        rows = _rows(form, n, r, i)
        for neg in ([False, True] if negatives else [False]) + (
                [True] if form in old.neg_capable_gcis and not negatives else []):
            if neg and form not in old.neg_capable_gcis:
                with pytest.raises(NotImplementedError):
                    new(rows, form, neg=True)
                continue
            expected, got = old(rows, form, neg=neg), new(rows, form, neg=neg)
            assert expected.shape == got.shape, (key, form, neg)
            assert th.allclose(expected, got, atol=1e-5, rtol=1e-4), (key, form, neg)
    if hasattr(old, "regularization_loss"):
        assert th.allclose(old.regularization_loss(), new.regularization_loss(), atol=1e-6)


@pytest.mark.parametrize("key", MODELS)
def test_a_state_dict_of_the_direct_module_loads_and_round_trips(key):
    old = _direct(key)(6, 2, 3, embed_dim=4)
    new = MODULES[key](6, 2, 3, embed_dim=4)
    new.load_state_dict(old.state_dict())          # strict: every table of the direct module
    again = MODULES[key](6, 2, 3, embed_dim=4)
    again.load_state_dict(new.state_dict())
    for name, tensor in old.state_dict().items():
        assert th.equal(new.state_dict()[name], tensor), name
        assert th.equal(again.state_dict()[name], tensor), name


# -- (c) the catalogue card and the parts agree -----------------------------------------------


@pytest.mark.parametrize("key", MODELS)
def test_the_card_and_the_parts_agree(key, dataset):
    card = get_method(key)
    module = build(key, dataset).module
    rep = REPRESENTATIONS[key]
    classes, relations, measure, region = COMBINATIONS[key]
    # the representation catalogue names the parts the module is made of
    assert [v.value for v in rep.classes] == [module.parts["classes"]]
    assert [v.value for v in rep.relations] == [module.parts["relations"]]
    assert [v.value for v in rep.measure] == [module.parts["measure"]]
    assert rep.conjunction.value == module.parts["conjunction"]
    assert classes in card.region
    # axiom forms: the card lists what the geometry reads
    assert set(card.axioms) <= set(module.supported_forms)
    # logic: what the expressivity states is what the module can do
    logic = card.supported_expressivity
    assert logic.has_abox == ({"class_assertion", "object_property_assertion"}
                              <= set(module.supported_forms))
    assert (Construct.BOTTOM in logic) == ({"gci0_bot", "gci1_bot", "gci3_bot"}
                                           <= set(module.supported_forms))
    assert (Construct.ROLE_CHAIN in logic) == (module.role_axiom_capable
                                               and "role_chain" in module.supported_forms)
    # tasks: those that need individuals or the bottom concept are supported by the forms
    for task in card.tasks:
        if task == "instance-checking":
            assert "class_assertion" in module.supported_forms
        if task == "role-assertion":
            assert "object_property_assertion" in module.supported_forms
        if task in ("satisfiability", "disjointness"):
            assert "gci0_bot" in module.supported_forms
    assert type(module).__name__ == card.module.rsplit(".", 1)[1]
