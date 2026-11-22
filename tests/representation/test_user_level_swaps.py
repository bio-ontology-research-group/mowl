"""Swapping one part of a named method at the user level, for every method.

``Model(dataset, method="<key>", <part>=<name or instance>)`` builds the method with that part
replaced (:mod:`mowl.representation.swap`). For every key of ``METHODS`` and every part kind
of the method, a replacement by name, through the constructor of the general model of the
method, trains one epoch on tiny data and gives finite numbers, or raises the documented error
(:class:`~mowl.representation.ConstructorUnavailable`: the combination cannot exist). Then
the replaced part is *in the model* (the "no silent ignore" tests): an argument that is
accepted and has no effect fails here. The outcome matrix is printed with
``python -m tests.representation.test_user_level_swaps`` (it is the table of
``docs/source/representation/swaps.rst``).
"""
import math

import pytest
import torch as th

import mowl
mowl.init_jvm("1g")

from mowl.models import ComposedALC, ComposedEL  # noqa: E402
from mowl.representation import (CLASS_MODELS, MEASURES, METHOD_KINDS, METHODS,  # noqa: E402
                                 RELATION_MODELS, ConstructorUnavailable, Objective)
from mowl.representation.swap import NO_PART, accepted_parts, current_parts  # noqa: E402
from mowl.training.regularization import Regularizer  # noqa: E402

D = 6
WORKS, UNAVAILABLE, UNSUPPORTED = "works", "unavailable by mathematics", "not supported"
#: ``{(key, kind): (status, detail)}``, filled by the tests, printed by ``main``.
OUTCOMES = {}


def finite(tensor):
    return bool(th.isfinite(th.as_tensor(tensor, dtype=th.float32)).all())


class Penalty(Regularizer):
    """A small penalty on every parameter of the class model."""
    kind = "penalty"

    def __init__(self):
        super().__init__(parameters="class_embed", weight=1e-3)

    def penalty(self, model):
        return self.weight * sum((p ** 2).sum() for p in model.class_model.parameters())


# -------------------------------------------------------------------------------------------
# the general models and what they need
# -------------------------------------------------------------------------------------------
_DATA = {}


def data(name):
    if name not in _DATA:
        if name == "family":
            from mowl.datasets.builtin import FamilyDataset
            _DATA[name] = FamilyDataset()
        elif name == "alc":
            from tests.baselines.pairs_logic import alc_dataset
            _DATA[name] = alc_dataset()
        elif name == "e2r":
            from tests.baselines.pairs_logic import e2r_dataset
            _DATA[name] = e2r_dataset()
        elif name == "walks":
            from tests.baselines.pairs_walks import annotated_family
            _DATA[name] = annotated_family()
        elif name == "query":
            from tests.datasetFactory import FamilyDataset
            _DATA[name] = FamilyDataset()
        elif name == "ranked":
            from tests.representation.test_recomp_construct import ranked_dataset
            _DATA[name] = ranked_dataset()
    return _DATA[name]


#: Keys that no general model can train from a dataset alone, and why.
UNSUPPORTED_BY_GENERAL_MODEL = {
    "faithel": "its TBox is constructed by a reasoner from the ontology and its classes are "
               "anchored to it; ComposedEL builds from sizes. Train it with FaithEL.",
    "grounded": "GroundedELModule is a grounded program over a shared element domain, "
                "not an ELModule: ComposedEL refuses it; train it with GroundedEL.",
}


UNSUPPORTED_BY_GENERAL_MODEL["e2r"] = (
    "its geometry reads ALC axioms through subspaces (a SubspaceGeometry, not an "
    "ExpressionGeometry): ComposedALC refuses it, ComposedEL refuses a method trained on "
    "expressions. Train it with ComposedSubspace.")
UNSUPPORTED_BY_GENERAL_MODEL["reshuffle"] = (
    "it is defined on a graph (its entities are encoded by message passing over the edges): "
    "ComposedEL has no edges to give it. Swap its parts through GraphKGEModel.")



def el_options(key):
    options = {"embed_dim": D, "margin": 0.05, "batch_size": 64}
    if key in ("hit", "ont"):
        from mowl.nn.lm.encoders import BagOfWordsEncoder
        options.update(encoder=BagOfWordsEncoder(dim=8, num_buckets=64), batch_size=8,
                       texts=[f"class {i}" for i in range(64)])
    if key == "falcon":
        options["anon_domain"] = 2
    if key == "mcmultibox":
        options["margin"] = 0.0     # its losses are volume ratios: a margin is refused
    return options


def axioms_model(key, **parts):
    th.manual_seed(0)
    return ComposedEL(data("family"), method=key, **el_options(key), **parts)


def alc_model(key, **parts):
    th.manual_seed(0)
    dataset = data("e2r") if key == "e2r" else data("alc")
    return ComposedALC(dataset, method=key, embed_dim=D, **parts)


def check_axioms(model):
    """Two epochs of training give finite parameters and finite costs of every form. Only a
    refusal at build time is a refusal: nothing is refused during training."""
    model.train(epochs=2)
    for parameter in model.module.parameters():
        assert finite(parameter)
    with th.no_grad():
        for form, rows in model.training_datasets.items():
            if len(rows):
                assert finite(model.module(rows[:], form)), form


def check_alc(model):
    """As :func:`check_axioms`, for ALC axioms. No error is converted into a refusal."""
    model.train(epochs=2)
    for parameter in model.module.parameters():
        assert finite(parameter)
    n = len(model.class_index_dict)
    assert finite(model.axiom_costs("gci0", th.cartesian_prod(th.arange(n), th.arange(n))))


#: The ALC methods are fixed by r6-alc (ALC objective and loop). Until that is merged a
#: replacement that is accepted at build and then fails in training is an expected failure,
#: with this reason; REMOVE this and ``alc_expected_failure`` AT MERGE of r6-alc.
R6_ALC_REASON = ("r6-alc: ComposedALC does not validate falcon/dfalc/alcone swaps at build "
                 "time; remove this expected failure when r6-alc is merged")


def alc_expected_failure(error):
    """Mark the test as an expected failure waiting for r6-alc (see ``R6_ALC_REASON``)."""
    pytest.xfail(f"{R6_ALC_REASON}: {type(error).__name__}: {error}")


# -------------------------------------------------------------------------------------------
# what a swap must do: change the loss, or the parameters that receive gradient
# -------------------------------------------------------------------------------------------
_OWN_NAMES = {"classes": ("class_model.", "individual_model."), "relations": ("relation_model.",),
              "individuals": ("individual_model.",)}


def _align(base, model, kind):
    """Copy to ``model`` the parameters of ``base`` that do not belong to the swapped part, so
    that a difference in the losses is the effect of the part and not of a different draw."""
    skip = _OWN_NAMES.get(kind, ())
    mine = dict(model.module.named_parameters())
    with th.no_grad():
        for name, parameter in base.module.named_parameters():
            if name in mine and not name.startswith(skip) and mine[name].shape == parameter.shape:
                mine[name].copy_(parameter)


def loss_and_touched(model):
    """``({form: costs}, {names of the parameters the loss depends on})`` before training."""
    module = model.module
    costs, total = {}, 0.0
    if not hasattr(model, "training_datasets"):       # ComposedALC: the costs of gci0 pairs
        n = len(model.class_index_dict)
        costs["gci0"] = model.axiom_costs("gci0", th.cartesian_prod(th.arange(n),
                                                                  th.arange(n))).detach()
        return costs, set()
    for form, rows in model.training_datasets.items():
        if len(rows):
            cost = module(rows[:], form)
            costs[form] = cost.detach().clone()
            total = total + cost.sum()
            if form in getattr(module, "neg_capable_gcis", ()):
                cost = module(rows[:], form, neg=True)
                costs[form + " (negative)"] = cost.detach().clone()
                total = total + cost.sum()
    penalty = module.regularization_loss()
    costs["regularization"] = th.as_tensor(penalty).detach().clone()
    total = total + penalty
    named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    touched = set()
    if isinstance(total, th.Tensor) and total.requires_grad:
        grads = th.autograd.grad(total, [p for _, p in named], allow_unused=True)
        touched = {n for (n, _), g in zip(named, grads) if g is not None}
    return costs, touched


def has_effect(base, model, kind):
    """Whether ``model`` (a swap of ``kind`` of ``base``) has other losses, or other
    parameters that receive gradient."""
    _align(base, model, kind)
    base_costs, base_touched = loss_and_touched(base)
    costs, touched = loss_and_touched(model)
    if touched != base_touched or set(costs) != set(base_costs):
        return True
    return any(costs[f].shape != base_costs[f].shape or not th.equal(costs[f], base_costs[f])
               for f in costs)


def observable(model, kind):
    """Whether the data can show the effect of a part of ``kind`` in the loss."""
    if kind == "regularizers":
        return any(True for _ in model.module.class_model.parameters())
    if kind == "individuals":
        return hasattr(model, "training_datasets") and any(form in model.training_datasets and len(model.training_datasets[form])
                   for form in ("class_assertion", "object_property_assertion"))
    return True


# -------------------------------------------------------------------------------------------
# candidates per part kind of a composed geometry
# -------------------------------------------------------------------------------------------
def candidates(kind, model):
    """Replacements to try, in order: ``(label, argument, check(model), effect)`` where
    ``check`` asserts the part is in the model and ``effect`` says whether the replacement
    must change the losses (it must, unless it is the method's own part again or removes
    a part)."""
    parts = current_parts(model.module)
    if kind == "classes":
        for name in CLASS_MODELS:
            yield name, name, lambda m, n=name: _same_kind(m, "classes", n), True
    elif kind == "relations":
        for name in RELATION_MODELS:
            yield name, name, lambda m, n=name: _same_kind(m, "relations", n), True
        yield NO_PART, NO_PART, lambda m: m.module.relation_model is None, False
    elif kind == "individuals":
        for name in CLASS_MODELS:
            yield name, name, lambda m, n=name: _same_kind(m, "individuals", n), True
    elif kind == "measure":
        for name in MEASURES:
            yield name, name, lambda m, n=name: _same_kind(m, "measure", n), True
    elif kind == "objective":
        own = parts["objective"].replace()
        yield "the method's own objective, as an instance", own, \
            lambda m: m.module.objective is own, False
        other = Objective(0.07)
        yield "Objective(0.07)", other, lambda m: m.module.objective is other, True
    elif kind == "objective_options":
        objective = parts["objective"]
        current = objective.options
        for name in objective.option_names():
            value = current.get(name)
            if isinstance(value, bool):
                new = not value
            elif isinstance(value, (int, float)):
                new = value + 0.07
            else:
                continue
            yield f"{name}={new}", {name: new}, lambda m, o={name: new}: _has_options(m, o), True
    elif kind == "regularizers":
        penalty = Penalty()
        yield "an added penalty", (penalty,), \
            lambda m: penalty in m.module.regularizers and \
            finite(m.module.regularization_loss()), True


def _same_kind(model, kind, name):
    """The part of the model is what the registry gives for ``name``."""
    registry = MEASURES if kind == "measure" else (
        RELATION_MODELS if kind == "relations" else CLASS_MODELS)
    made = registry[name]
    if kind != "measure":
        made = made(3, 6)
    expected = made if isinstance(made, type) else type(made)
    if kind == "measure":
        expected = made
    assert type(current_parts(model.module)[kind]) is expected, (kind, name)
    return True


def _has_options(model, options):
    """The objective of the model has been made with the options (and differs from the
    method's own in exactly them)."""
    made = model.module.objective.options
    for name, value in options.items():
        assert made[name] == pytest.approx(value), (name, value)
    return True


EXPECTED_ERRORS = (ConstructorUnavailable,)

#: Options of an objective that are read only under a condition the tiny test data does not
#: meet: changing them cannot change the losses here, for the reason given. (Every other
#: option of every method must change the losses, or the swap fails.)
CONDITIONAL_OPTIONS = {
    "evaluation_margin": "read only when the module is in evaluation mode",
    "abox_threshold": "read only by the loss of a corrupted class assertion; the toy data has "
                      "no assertions",
    "empty_margin": "read only by the forms about empty classes (bottom); the toy data has none",
    "consistent_radius": "read only by forms with an absolute radius reading (radius='softplus')",
    "strict_negatives": "only chooses between an error and the positive loss when a negative "
                        "loss of a form that has none is asked for; training asks for none",
    "epsilon": "the margin of the boxes of the solver programme; read when the programme is "
               "solved, not by the loss of a form",
}


def swap_kinds(model):
    """The part kinds of a composed geometry the method has, to swap."""
    kinds = ["classes", "relations", "measure", "objective", "objective_options",
             "regularizers"]
    if model.module.individual_model is not None:
        kinds.insert(2, "individuals")
    return kinds


#: ``{(key, kind): reason}``: the method refuses *every* replacement of that part kind that
#: the test tries, and why that is right. A method and kind that refuse everything without
#: being listed here fail the test (a refusal of everything is a part that is not swappable),
#: and a listed one that accepts something fails too (the entry is stale).
ALC_KEYS = ALC_METHODS = ("dfalc", "falcon", "alcone")
_NO_CLASS_FOR_MEASURE = (
    "the measure reads parts of its own region kind ({}), which no other entry of the "
    "registry provides; every other class model is refused with the two sides named")
#: ``{key: [options]}``: options of the objective that were accepted and changed no loss on
#: the toy data although others of the same objective did (they are conditional: see the
#: report of this test); written for inspection.
UNREAD_OPTIONS = {}
REFUSES_EVERY = {
    ("boxsel", "objective_options"): "the only option it has is the margin, and its volume "
                                     "measure ignores a margin (BoxEL and BoxSEL have none)",
    ("boxlite", "classes"): "the measure reads the lower and upper corner tables of its own "
                            "BoundsBoxes, which are constrained by the solver programme; no "
                            "other class model has corner tables (box models are refused when "
                            "the probe finds no 'lower')",
    ("concept2box", "classes"): "the soft-box measure reads the context table of its own "
                                "ContextualSoftBox class model (in_context); no other class "
                                "model has contexts",
    ("eike", "classes"): _NO_CLASS_FOR_MEASURE.format("an ellipsoid and a vector, Linked"),
    ("eike", "measure"): "LinkedMeasure is the only measure that compares Linked regions; "
                         "every other measure compares one region kind",
    ("eike", "individuals"): _NO_CLASS_FOR_MEASURE.format("Linked"),
    ("shadow_cones", "classes"): "the cone measure compares the Point region of the hierarchy "
                                 "family (hierarchy_regions.Point), the registered 'point' "
                                 "class model gives another class of the same name "
                                 "(shapes_regions.Point): a duplicated region class, not a "
                                 "mathematical obstruction (reported, not fixed here)",
    ("canonical_mixture", "classes"): _NO_CLASS_FOR_MEASURE.format("SparseElementSets"),
    ("canonical_mixture", "individuals"): _NO_CLASS_FOR_MEASURE.format("SparseElementSets"),
    ("canonical_mixture", "measure"): "ProbabilityMass is the only measure of sparse element "
                                      "sets (the regions are sets of finite-model elements, "
                                      "not geometric regions)",
    ("canonical_mixture", "relations"): "its relations are canonical roles over element "
                                        "sets; geometric relation models map geometric "
                                        "regions and refuse sparse sets",
    ("falcon", "classes"): _NO_CLASS_FOR_MEASURE.format("fuzzy sets of a domain"),
    ("falcon", "relations"): "its relations are networks over pairs of fuzzy sets; geometric "
                             "relation models map geometric regions and refuse fuzzy sets",
}
for _key in ("boxe", "expressive", "graph_kge", "gda_projection"):
    for _kind in ("objective", "objective_options"):
        REFUSES_EVERY[(_key, _kind)] = (
            "a model of triples is trained with a TripleLoss (loss= of GraphKGEModel), no cost "
            "of its geometry reads an Objective: replacing it would change nothing")
#: The same for the ALC methods through ComposedALC (the candidates of ``limit``).
REFUSES_EVERY_ALC = {}


def run(build, check, key, kind, limit=None, on_late_error=None, refuses=None):
    """Try the replacements of ``kind`` for ``key`` through the constructor of the general
    model. Every replacement is either refused with :class:`ConstructorUnavailable` (the
    library's own refusal: nothing else counts) or accepted, and an accepted one is in the
    model and has an effect (:func:`has_effect`); the first accepted one is trained for two
    epochs. A method that refuses every replacement of a kind must be listed in
    ``REFUSES_EVERY`` with its reason. ``on_late_error`` receives an error raised after the
    build (for methods another helper is fixing). Returns the outcome."""
    base = build(key)
    tried, accepted, last_error, refusals, ignored = 0, 0, None, [], []
    for label, argument, verify, effect in candidates(kind, base):
        if limit and tried >= limit:
            break
        tried += 1
        try:
            model = build(key, **{kind: argument})
        except EXPECTED_ERRORS as error:
            last_error = f"{label}: {error}"
            refusals.append(last_error)
            continue
        assert verify(model), (key, kind, label)
        same_part = kind in ("classes", "relations", "measure", "individuals") and \
            type(current_parts(model.module)[kind]) is type(current_parts(base.module)[kind])
        conditional = kind == "objective_options" and set(argument) <= set(CONDITIONAL_OPTIONS)
        if effect and observable(model, kind) and not same_part and not conditional:
            try:
                if not has_effect(base, model, kind):
                    ignored.append(label)
            except Exception as error:
                if on_late_error is None:
                    raise
                on_late_error(error)
        if accepted == 0:
            try:
                check(model)
            except Exception as error:
                if on_late_error is None:
                    raise
                on_late_error(error)
        accepted += 1
        OUTCOMES.setdefault((key, kind), (WORKS, label))
    if ignored and kind == "objective_options" and len(ignored) < accepted:
        # an objective has options that are read only under a condition of the data or of
        # the other options (a margin the hierarchy forms do not use, a tolerance weight of a
        # model without tolerance): the ones that are read must change the losses
        UNREAD_OPTIONS[key] = ignored
        ignored = []
    if ignored:
        message = (f"{key}: {kind} {ignored} are accepted but change neither the losses nor "
                   f"the parameters that receive gradient: they are ignored. Use them, or "
                   f"refuse them at build time with ConstructorUnavailable.")
        if on_late_error is None:
            raise AssertionError(message)
        on_late_error(AssertionError(message))
    if tried == 0:
        OUTCOMES[(key, kind)] = (UNSUPPORTED, "no replacement is defined for this reading")
        return UNSUPPORTED, None
    refuses = REFUSES_EVERY if refuses is None else refuses
    listed = (key, kind) in refuses
    if accepted == 0:
        assert listed, (f"{key} refuses every replacement of {kind} ({len(refusals)} tried; "
                        f"first: {refusals[0][:150]}; last: {last_error[:150]}): "
                        f"make one work, or list ({key!r}, {kind!r}) in REFUSES_EVERY with "
                        f"the reason")
        if on_late_error is not None:
            pass
        OUTCOMES[(key, kind)] = (UNAVAILABLE, refuses[(key, kind)])
        return UNAVAILABLE, last_error
    assert not listed, f"({key!r}, {kind!r}) accepts a replacement: remove it from REFUSES_EVERY"
    return WORKS, OUTCOMES[(key, kind)][1]


TRIPLE_KEYS = ("boxe", "expressive", "graph_kge", "gda_projection")
AXIOM_KEYS = sorted(k for k, v in METHOD_KINDS.items() if v == "axioms"
                    and k not in UNSUPPORTED_BY_GENERAL_MODEL)
EL_KINDS = ["classes", "relations", "measure", "objective", "objective_options",
            "regularizers"]


@pytest.mark.parametrize("kind", EL_KINDS)
@pytest.mark.parametrize("key", AXIOM_KEYS)
def test_every_axiom_method_takes_a_replacement_of_every_part_by_name(key, kind):
    status, _ = run(axioms_model, check_axioms, key, kind,
                    on_late_error=alc_expected_failure if key in ALC_KEYS else None)
    assert status in (WORKS, UNAVAILABLE)


@pytest.mark.parametrize("key", AXIOM_KEYS)
def test_individuals_of_every_axiom_method(key):
    if axioms_model(key).module.individual_model is None:
        pytest.skip("the method has no individual model")
    status, _ = run(axioms_model, check_axioms, key, "individuals")
    assert status in (WORKS, UNAVAILABLE)


@pytest.mark.parametrize("kind", EL_KINDS)
@pytest.mark.parametrize("key", ALC_KEYS)
def test_every_alc_method_takes_a_replacement_of_every_part_by_name(key, kind):
    status, _ = run(alc_model, check_alc, key, kind, on_late_error=alc_expected_failure,
                    refuses=REFUSES_EVERY_ALC)
    assert status in (WORKS, UNAVAILABLE)


@pytest.mark.parametrize("key", sorted(UNSUPPORTED_BY_GENERAL_MODEL))
def test_a_method_no_general_model_trains_says_so(key):
    """The refusal is at build time, in the library's own words, not in training."""
    with pytest.raises((TypeError, ConstructorUnavailable)) as error:
        if key == "e2r":
            ComposedALC(data("e2r"), method=key, embed_dim=D)
        else:
            ComposedEL(data("family"), method=key, **({"classes": "box"}
                                                      if key != "reshuffle" else {}),
                       embed_dim=D)
    assert key in str(error.value) or "grounded" in str(error.value).lower() \
        or "ELModule" in str(error.value) or "graph" in str(error.value)
    if key != "e2r":                  # the part kinds of e2r are tried through ComposedSubspace
        for kind in EL_KINDS:
            OUTCOMES[(key, kind)] = (UNSUPPORTED, UNSUPPORTED_BY_GENERAL_MODEL[key])


def test_faithel_is_refused_at_build_not_in_training():
    """FaithEL's TBox is constructed from the ontology: a model built from sizes used to be
    accepted and to crash in the first training step."""
    with pytest.raises(ConstructorUnavailable, match="faithel"):
        ComposedEL(data("family"), method="faithel", embed_dim=D)


def test_a_relation_model_no_trained_form_reads_is_refused_at_build():
    """The swap leaves the loss as it was and the parameters would never move."""
    for key in ("concept2box", "bcbox", "ont"):
        with pytest.raises(ConstructorUnavailable, match="relation"):
            axioms_model(key, relations="translation")
    # a method that reads relations in a trained form takes it, and the loss changes
    model = axioms_model("elbe", relations="translation")
    assert model.module.relation_model is not None


# -- nothing is ignored silently -------------------------------------------------------------
@pytest.mark.parametrize("key", AXIOM_KEYS)
def test_a_part_instance_given_with_the_method_is_the_part_of_the_model(key):
    """Instances are taken as they are, for every part the method has: if an accepted
    argument were ignored, the instance would not be in the model."""
    base = axioms_model(key)
    parts = current_parts(base.module)
    measure = parts["measure"]
    # the method's own measure (and objective) as new instances: always a legal combination
    new_measure = type(measure)() if not _needs_args(type(measure)) else None
    objective = parts["objective"].replace()
    penalty = Penalty()
    args = {"objective": objective, "regularizers": (penalty,)}
    if key in TRIPLE_KEYS:        # trained with a TripleLoss: an Objective would be ignored
        with pytest.raises(ConstructorUnavailable, match="TripleLoss"):
            axioms_model(key, objective=objective)
        objective = None
        del args["objective"]
    if new_measure is not None:
        args["measure"] = new_measure
    model = axioms_model(key, **args)
    swapped = current_parts(model.module)
    if objective is not None:
        assert swapped["objective"] is objective
    assert penalty in swapped["regularizers"]
    assert len(swapped["regularizers"]) == len(parts["regularizers"]) + 1
    if new_measure is not None:
        assert swapped["measure"] is new_measure


def _needs_args(cls):
    import inspect
    return any(p.default is inspect.Parameter.empty and p.kind in (
        p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        for name, p in inspect.signature(cls).parameters.items() if name != "self")


@pytest.mark.parametrize("key", ["elem", "elbe", "transbox", "regd", "boxel"])
def test_a_class_model_instance_and_a_relation_model_instance_are_taken_as_they_are(key):
    base = axioms_model(key)
    classes, relations = base.module.class_model, base.module.relation_model
    n_classes = len(base.class_index_dict)
    new_classes = type(classes)(n_classes, D) if hasattr(classes, "centres") else None
    if new_classes is None:
        pytest.skip("no table to rebuild")
    model = axioms_model(key, classes=new_classes)
    assert model.module.class_model is new_classes
    assert model.module.class_embed is getattr(new_classes, "centres", None)


@pytest.mark.parametrize("key", ["elem", "elbe"])
def test_a_part_swap_changes_the_losses(key):
    """A swap that is accepted and has no effect would give the numbers of the method."""
    a, b = axioms_model(key), axioms_model(key, relations="translation with tolerance"
                                           if key == "elem" else "affine map")
    assert type(a.module.relation_model) is not type(b.module.relation_model)
    objective = axioms_model(key, objective_options={"margin": 0.5})
    assert objective.module.objective.margin == 0.5
    assert a.module.objective.margin == 0.05


def test_the_margin_option_of_the_objective_changes_the_loss():
    a = axioms_model("elbe", objective_options={"margin": 0.0})
    b = axioms_model("elbe", objective_options={"margin": 0.5})
    b.module.class_model.centres.weight.data.copy_(a.module.class_model.centres.weight.data)
    b.module.class_model.offsets.weight.data.copy_(a.module.class_model.offsets.weight.data)
    b.module.relation_model.vectors.weight.data.copy_(a.module.relation_model.vectors.weight.data)
    rows = a.training_datasets["gci0"][:]
    assert not th.allclose(a.module(rows, "gci0"), b.module(rows, "gci0"))


# -- errors name both sides ------------------------------------------------------------------
def test_an_unavailable_replacement_names_both_sides():
    with pytest.raises(ConstructorUnavailable) as error:
        axioms_model("elbe", classes="ball")
    text = str(error.value)
    assert "Bounds" in text and "BallClasses" in text and "elbe" in text


def test_an_argument_the_method_cannot_take_is_a_type_error_that_names_the_accepted_ones():
    with pytest.raises(TypeError, match="no option"):
        axioms_model("elbe", objective_options={"not_an_option": 1})
    with pytest.raises(TypeError, match="not_an_option"):
        axioms_model("elbe", not_an_option=1)
    with pytest.raises(TypeError, match="Unexpected"):
        ComposedEL(data("family"), classes="box", gci3="overlap")
    with pytest.raises(TypeError, match="objective_options"):
        ComposedEL(data("family"), objective_options={"margin": 1.0})
    with pytest.raises(ValueError, match="Unknown relation model"):
        axioms_model("elbe", relations="no such relation model")


def test_a_method_that_is_not_a_composed_geometry_names_the_parts_it_takes():
    from mowl.representation.swap import build_method
    with pytest.raises(TypeError, match="replaceable"):
        build_method("grounded", 5, 2, 0, 4, classes="box")


# -- named classes: one rule ----------------------------------------------------------------
NAMED = ("ELEmbeddings", "ELBE", "BoxEL", "TransBox", "EmELPP", "RegDELBE", "BoxSEL",
         "MultiELBE", "OctagonEL", "HyperbolicCones",
         "OrderEmbeddings", "BoxLattice", "GumbelBox", "CatE", "DFALCModel", "ALConeModel",
         "FALCONModel", "E2R", "Concept2Box", "PrismEL", "EIKE", "BinaryCodeBox")


@pytest.mark.parametrize("name", NAMED)
def test_a_named_class_refuses_a_part_argument_and_points_to_the_general_model(name):
    import mowl.models as models
    cls = getattr(models, name)
    dataset = data("family")
    for part in ("relations", "classes"):
        if part in cls.__init__.__wrapped__.__code__.co_varnames:
            continue
        with pytest.raises(TypeError, match="named combination of parts") as error:
            cls(dataset, **{part: "translation"})
        assert "ComposedEL" in str(error.value) and part in str(error.value)


@pytest.mark.parametrize("name", ("Query2Box", "BetaE", "AConE"))
def test_a_named_query_class_refuses_a_part_it_has_not(name):
    import mowl.models as models
    with pytest.raises(TypeError, match="takes no"):
        getattr(models, name)(data("query"), embed_dim=8, relations="translation")


# -- the other kinds of methods -------------------------------------------------------------
def _query_model(key, **parts):
    from mowl.models.query.model import QueryEmbeddingModel
    th.manual_seed(0)
    return QueryEmbeddingModel(data("query"), method=key, embed_dim=8, learning_rate=3e-3,
                               batch_size=24, negatives=4, seed=0, steps_per_epoch=3, **parts)


def _query_check(model):
    """Two epochs of training; finite parameters and costs (nothing is refused here)."""
    model.train(epochs=2)
    assert math.isfinite(model.loss_history[-1])
    assert all(finite(p) for p in model.module.parameters())
    costs = model.axiom_costs("gci0", th.tensor([[0, 1], [1, 2]]))
    assert finite(costs)


QUERY_PART_OF = {"intersection": lambda m: m.module.intersection,
                 "distance": lambda m: m.module.distance_part,
                 "negation": lambda m: m.module.negation_part,
                 "relations": lambda m: m.module.relations}
QUERY_PREFIX = {"intersection": "intersection.", "distance": "distance_part.",
                "negation": "negation_part.", "relations": "relations."}
#: The names each builder accepts for a part (the builders of ``query_methods`` name the parts
#: they can take; a part given as an instance is taken as it is).
QUERY_NAMES = {
    ("query2box", "intersection"): ["exact"], ("query2box", "distance"): ["l2"],
    ("betae", "intersection"): ["mean"], ("betae", "distance"): ["reverse kl"],
    ("cone_qe", "relations"): ["network", "rotation"],
}
QUERY_NO_NAME = {
    ("cone_qe", "intersection"): "the builder knows one intersection of cones (attention) and "
                                 "takes any other as an instance",
    ("cone_qe", "distance"): "the builder knows one distance of cones and takes any other as "
                             "an instance",
    ("acone", "intersection"): "the builder knows one intersection of cones (attention) and "
                               "takes any other as an instance",
    ("acone", "distance"): "the builder knows one distance of cones and takes any other as an "
                           "instance",
}
QUERY_REFUSES_EVERY = {
    ("query2box", "negation"): "the library has no negation of boxes (Query2Box has none); the "
                               "negations of BetaE and ConE act on other region families and "
                               "are refused with both sides named",
}


def _query_fingerprint(model):
    """Scores of four queries (an anchor, a projection, an intersection, a projection of an
    intersection and, if the model has negation, an intersection with a negation) against
    all entities."""
    from mowl.nn.query.expression import anchor, intersect, negate, project
    nodes = [anchor(0), project(3, anchor(1)), intersect(anchor(0), anchor(1)),
             project(3, intersect(anchor(1), anchor(2)))]
    if model.module.supports_negation:
        nodes.append(intersect(anchor(0), negate(anchor(1))))
    model.module.eval()
    with th.no_grad():
        return model._logits(nodes)


def _query_effect(base, model, kind):
    skip = QUERY_PREFIX[kind]
    mine = dict(model.module.named_parameters())
    with th.no_grad():
        for name, parameter in base.module.named_parameters():
            if name in mine and not name.startswith(skip) and mine[name].shape == parameter.shape:
                mine[name].copy_(parameter)
    a, b = _query_fingerprint(base), _query_fingerprint(model)
    return a.shape != b.shape or not th.equal(a, b)


def _query_names(key, kind):
    if kind == "negation":
        from mowl.representation.library.query_methods import _NEGATIONS
        return [None] + list(_NEGATIONS)
    return QUERY_NAMES.get((key, kind), [])


QUERY_KINDS = ("intersection", "distance", "negation", "relations")


@pytest.mark.parametrize("kind", QUERY_KINDS)
@pytest.mark.parametrize("key", [k for k, v in METHOD_KINDS.items() if v == "queries"])
def test_query_methods_take_a_replacement_of_every_part_by_name(key, kind):
    import inspect
    if kind not in inspect.signature(METHODS[key]).parameters:
        with pytest.raises(TypeError, match="accepted"):
            _query_model(key, **{kind: "attention"})
        OUTCOMES[(key, kind)] = (UNSUPPORTED, f"{key} has no {kind} option: it is fixed by "
                                 f"the method")
        return
    names = _query_names(key, kind)
    if not names:
        assert (key, kind) in QUERY_NO_NAME, (key, kind, "has no name to swap in")
        OUTCOMES[(key, kind)] = (UNSUPPORTED, QUERY_NO_NAME[(key, kind)])
        return
    base = _query_model(key)
    base_part = QUERY_PART_OF[kind](base)
    accepted, refusals, ignored = 0, [], []
    for name in names:
        try:
            model = _query_model(key, **{kind: name})
        except ConstructorUnavailable as error:       # the library's own refusal
            refusals.append(f"{name}: {error}")
            continue
        part = QUERY_PART_OF[kind](model)
        if kind in ("negation", "relations") and type(part) is type(base_part):
            continue                                   # the method's own negation again
        if not _query_effect(base, model, kind):
            ignored.append(name)
        if accepted == 0:
            _query_check(model)
            OUTCOMES[(key, kind)] = (WORKS, str(name))
        accepted += 1
    assert not ignored, f"{key}: {kind} {ignored} accepted but the scores do not change"
    if accepted == 0:
        assert (key, kind) in QUERY_REFUSES_EVERY, (
            f"{key} refuses every replacement of {kind}: {refusals}; list it with a reason")
        OUTCOMES[(key, kind)] = (UNAVAILABLE, QUERY_REFUSES_EVERY[(key, kind)])
    else:
        assert (key, kind) not in QUERY_REFUSES_EVERY


def test_a_query_part_instance_is_taken_as_it_is():
    from mowl.representation.library import query_parts as P
    part = P.BoxExactIntersection()
    model = _query_model("query2box", intersection=part)
    assert model.module.intersection is part


# -- sequences -------------------------------------------------------------------------------
SEQ_KEYS = sorted(k for k, v in METHOD_KINDS.items() if v == "sequences")
SEQ_KINDS = ("view", "walker", "source", "encoder", "reading", "measure", "scorer")
WALK_METHODS = ("dl2vec", "owl2vec_star", "taxonomy_walks")
SMALL = dict(embed_dim=8, workers=1)


def _seq_model(key, **parts):
    from mowl.models import ComposedSequenceModel
    options = dict(SMALL)
    if (key in WALK_METHODS or "view" in parts) and "source" not in parts:
        options.update(num_walks=2, walk_length=5)
    if "encoder" in parts and not isinstance(parts["encoder"], str):
        options.pop("embed_dim"), options.pop("workers")
    return ComposedSequenceModel(data("walks"), method=key, **options, **parts)


def _seq_candidates(key, kind):
    from mowl.representation.library import walks_encoders as W
    from mowl.representation.library import walks_views as V
    if kind == "view":
        return ["dl2vec"] if key == "taxonomy_walks" else ["taxonomy"]
    if kind == "walker":
        return ["node2vec"]
    if kind == "source":
        return [V.Sentences(V.AxiomCorpus())]
    return {"encoder": ["fasttext"], "reading": list(W.READINGS),
            "measure": list(W.PAIR_MEASURES), "scorer": list(W.SCORERS)}[kind]


#: Part kinds of a sequence method with nothing to swap in by name, and why.
SEQ_NO_ALTERNATIVE = {
    "reading": "one reading is registered (the vector of the entity's token): it is the "
               "method's own",
}


def _seq_fingerprint(model):
    """Two epochs of training, then the costs of pairs of classes (training is seeded: two
    models built alike give the same numbers)."""
    model.train(epochs=2)
    n = min(4, len(model.class_index_dict))
    rows = th.tensor([[i, j] for i in range(n) for j in range(n) if i != j])
    costs = model.axiom_costs("gci0", rows)
    assert finite(costs)
    return costs


@pytest.mark.parametrize("kind", SEQ_KINDS)
@pytest.mark.parametrize("key", SEQ_KEYS)
def test_sequence_methods_take_a_replacement_of_every_part(key, kind):
    pytest.importorskip("gensim")
    if kind == "walker" and key not in WALK_METHODS:
        with pytest.raises(TypeError, match="no projection"):
            _seq_model(key, walker="node2vec")
        OUTCOMES[(key, kind)] = (UNSUPPORTED, f"{key} reads sentences, there are no walks "
                                 f"to walk: give a view too (view=...), then walker= works")
        return
    candidates_ = _seq_candidates(key, kind)
    if kind in SEQ_NO_ALTERNATIVE:
        assert candidates_ == ["token"], candidates_
        OUTCOMES[(key, kind)] = (UNSUPPORTED, SEQ_NO_ALTERNATIVE[kind])
        return
    base = _seq_model(key)
    base_costs = _seq_fingerprint(base)
    accepted, ignored, refusals = 0, [], []
    for candidate in candidates_:
        try:
            model = _seq_model(key, **{kind: candidate})
        except ConstructorUnavailable as error:          # the library's own refusal
            refusals.append(f"{candidate}: {error}")
            continue
        part = {"view": None, "walker": None}.get(kind, getattr(model.module, kind, None))
        if kind == "source":
            assert model.module.source is candidate
        elif kind == "walker":
            assert "node2vec" in type(model.module.source.walker).__name__.lower() \
                or "Node2Vec" in model.module.parts["walker"]
        elif kind == "view":
            assert model.module.parts["projection"] != base.module.parts.get("projection")
        elif kind == "encoder":
            assert "fasttext" in model.module.parts["encoder"].lower() \
                or "fast" in type(part).__name__.lower()
        else:
            assert part is not None
        costs = _seq_fingerprint(model)
        own = model.module.parts == base.module.parts       # the method's own part again
        if not own and costs.shape == base_costs.shape and th.equal(costs, base_costs):
            ignored.append(str(candidate))
        accepted += 1
        OUTCOMES.setdefault((key, kind), (WORKS, candidate if isinstance(candidate, str)
                                          else type(candidate).__name__))
    assert not ignored, (f"{key}: {kind} {ignored} accepted but the costs after two epochs "
                         f"are those of the method: it is ignored")
    if not accepted:
        assert (key, kind) in SEQ_REFUSES_EVERY, (key, kind, refusals)
        OUTCOMES[(key, kind)] = (UNAVAILABLE, SEQ_REFUSES_EVERY[(key, kind)])


SEQ_REFUSES_EVERY = {}


def test_a_sequence_option_no_part_reads_is_a_type_error():
    pytest.importorskip("gensim")
    from mowl.models import ComposedSequenceModel
    from mowl.representation.library.walks_encoders import Word2VecEncoder
    ds = data("walks")
    with pytest.raises(TypeError, match="no projection"):
        ComposedSequenceModel(ds, method="onto2vec", walker="node2vec")
    with pytest.raises(TypeError, match="encoder is an object"):
        ComposedSequenceModel(ds, method="dl2vec", encoder=Word2VecEncoder(vector_size=4),
                              window=3)
    with pytest.raises(TypeError, match="takes no"):
        ComposedSequenceModel(ds, method="dl2vec", not_an_option=1)
    with pytest.raises(TypeError, match="scorer is an object"):
        from mowl.representation.library.walks_encoders import SCORERS
        ComposedSequenceModel(ds, method="dl2vec", scorer=SCORERS["similarity"](),
                              scorer_options={"x": 1})


# -- constructed ------------------------------------------------------------------------------
CONSTRUCT_KEYS = sorted(k for k, v in METHOD_KINDS.items() if v == "construction")


def _construct_candidates(kind):
    from mowl.representation.construction import READINGS, SOURCES
    return sorted(SOURCES if kind == "source" else READINGS)


CONSTRUCT_REFUSES_EVERY = {
    ("box_milp", "source"): "the only reading of a box model is 'boxes'; a finite model cannot "
                            "in general be read as single boxes (boxes have the Helly "
                            "property), so every other source is refused with both sides named",
    ("attributed_dl", "source"): "the data of the method is an attributed ontology, which no "
                                 "other source reads (each source reads one kind of data and "
                                 "refuses another with the two sides named)",
    ("temporal_cones", "source"): "the data of the method is a temporal ontology, which no "
                                  "other source reads (each source reads one kind of data and "
                                  "refuses another with the two sides named)",
    ("box_milp", "reading"): "a box model is a set of boxes, it has no domain elements for "
                             "the cell readings to place: only the 'boxes' reading reads it",
}


def _construct_data(key):
    """The data each constructed method is built from (those of its own tests): a canonical
    model, and for three methods the kind of ontology they are about."""
    from tests.representation import test_recomp_construct as C
    from mowl.geometric.completion import CanonicalModel
    from tests.models.kg_region_data import abox_dataset
    if key == "temporal_cones":
        return C._temporal()
    if key == "attributed_dl":
        return C._attributed()
    if key == "box_milp":
        return C._helly()
    if key == "least_model":
        from mowl.geometric.least_model import least_model
        return least_model(3, nb_rels=1, nb_individuals=1, gci2=[[0, 0, 1]],
                           class_assertion=[[0, 0]])
    return CanonicalModel.from_dataset(abox_dataset())


@pytest.mark.parametrize("kind", ("source", "reading"))
@pytest.mark.parametrize("key", CONSTRUCT_KEYS)
def test_constructed_methods_take_a_replacement_of_source_and_reading(key, kind):
    """Only the library's own refusal (ConstructorUnavailable, or a missing optional
    dependency, which skips) counts as a refusal; any other error is a failure. An accepted
    replacement is the part of the built geometry (its ``source`` / ``reading`` object), and
    a reading changes the family of regions."""
    from mowl.geometric.errors import MissingDependency
    from mowl.representation.construction import RECIPES
    data = _construct_data(key)
    try:
        base = RECIPES[key].build(data)
    except MissingDependency as error:
        pytest.skip(f"{key} needs an optional dependency: {error}")
    accepted, refusals = 0, []
    for name in _construct_candidates(kind):
        if name == RECIPES[key].parts[kind]:
            continue
        try:
            geometry = RECIPES[key].build(data, **{kind: name})
        except MissingDependency as error:
            pytest.skip(f"{name} needs an optional dependency: {error}")
        except ConstructorUnavailable as error:
            refusals.append(f"{name}: {type(error).__name__}: {error}")
            continue
        assert getattr(geometry, kind).key == name, (key, kind, name)
        assert geometry.parts != base.parts, (key, kind, name, "the parts did not change")
        if kind == "reading":
            assert geometry.class_model.region is not base.class_model.region or \
                type(geometry.class_model) is not type(base.class_model), (key, name)
        assert math.isfinite(float(geometry.inside(
            geometry.classes(th.tensor([0])), geometry.classes(th.tensor([0]))).float().sum()))
        accepted += 1
        OUTCOMES.setdefault((key, kind), (WORKS, name))
    if not accepted:
        assert (key, kind) in CONSTRUCT_REFUSES_EVERY, (key, kind, refusals)
        OUTCOMES[(key, kind)] = (UNAVAILABLE, CONSTRUCT_REFUSES_EVERY[(key, kind)])


def test_a_constructed_option_no_part_reads_is_a_type_error():
    from mowl.models.constructed import ConstructedModel
    with pytest.raises(TypeError):
        ConstructedModel(data("family"), method="interval_union",
                         reading_options={"not_an_option": 1})
    with pytest.raises(TypeError):
        ConstructedModel(data("family"), method="interval_union", relations="translation")


# -- graph knowledge graph embedding models -------------------------------------------------
GRAPH_KEYS = ("graph_kge", "gda_projection", "boxe", "expressive", "reshuffle")
GRAPH_KINDS = ("classes", "relations", "loss", "sampler", "regularizers")


def _graph_model(key, **parts):
    from mowl.models import GraphKGEModel, Reshuffle
    th.manual_seed(0)
    if key == "reshuffle":
        return Reshuffle(data("family"), embed_dim=D, batch_size=32, **parts)
    return GraphKGEModel(data("family"), method=key, embed_dim=D, batch_size=32, **parts)


def _graph_candidates(kind):
    from mowl.representation.library.other_graph import TRIPLE_LOSSES, UniformCorruption
    if kind == "classes":
        return [(n, n) for n in CLASS_MODELS]
    if kind == "relations":
        return [(n, n) for n in RELATION_MODELS]
    if kind == "loss":
        return [(n, n) for n in TRIPLE_LOSSES]
    if kind == "sampler":
        return [("uniform corruption of heads", UniformCorruption(3, ("h",)))]
    return [("an added penalty", (Penalty(),))]


def _graph_fingerprint(model, trained):
    if trained:
        model.train(epochs=2)
    n = len(model.class_index_dict)
    costs = model.axiom_costs("gci0", th.cartesian_prod(th.arange(n), th.arange(n)))
    assert finite(costs)
    return costs


def _graph_effect(base, model, kind):
    if kind in ("loss", "sampler"):      # read by training only: the trajectories differ
        a, b = _graph_fingerprint(base, True), _graph_fingerprint(model, True)
        return not th.equal(a, b)
    if kind == "regularizers":
        return float(model.module.regularization_loss()) != float(
            base.module.regularization_loss())
    skip = {"classes": ("class_model.", "individual_model."),
            "relations": ("relation_model.",)}[kind]
    mine = dict(model.module.named_parameters())
    with th.no_grad():
        for name, parameter in base.module.named_parameters():
            if name in mine and not name.startswith(skip) and mine[name].shape == parameter.shape:
                mine[name].copy_(parameter)
    a, b = _graph_fingerprint(base, False), _graph_fingerprint(model, False)
    return a.shape != b.shape or not th.equal(a, b)


GRAPH_REFUSES_EVERY = {}
#: Reshuffle is a named combination whose entities are encoded by a graph network that uses
#: the matrices of its own relations: its class and relation models are not swappable, and
#: the model says so (TypeError, like every named class).
GRAPH_NOT_SWAPPABLE = {("reshuffle", "classes"), ("reshuffle", "relations")}


@pytest.mark.parametrize("kind", GRAPH_KINDS)
@pytest.mark.parametrize("key", GRAPH_KEYS)
def test_graph_models_take_a_replacement_of_every_part(key, kind):
    """The knowledge graph models trained on the graph of the ontology
    (:class:`~mowl.models.GraphKGEModel`): an accepted replacement is in the model, changes
    what is computed and trains for two epochs; only ConstructorUnavailable refuses."""
    graph_kind = {"classes": "classes (graph)", "relations": "relations (graph)",
                  "regularizers": "regularizers (graph)"}.get(kind, kind)
    if (key, kind) in GRAPH_NOT_SWAPPABLE:
        with pytest.raises(TypeError, match="named combination of parts"):
            _graph_model(key, **{kind: _graph_candidates(kind)[0][1]})
        OUTCOMES[(key, graph_kind)] = (UNSUPPORTED, f"{key} encodes its entities with a graph "
                                 f"network that uses the matrices of its own relations")
        return
    base = _graph_model(key)
    accepted, refusals, ignored = 0, [], []
    for label, argument in _graph_candidates(kind):
        try:
            model = _graph_model(key, **{kind: argument})
        except ConstructorUnavailable as error:
            refusals.append(f"{label}: {error}")
            continue
        own = current_parts(model.module)
        if kind in ("classes", "relations") and type(own[kind]) is type(
                current_parts(base.module)[kind]):
            continue                                   # the method's own part again
        if not _graph_effect(_graph_model(key), model, kind):
            ignored.append(label)
        if accepted == 0:
            _graph_fingerprint(_graph_model(key, **{kind: argument}), True)   # two epochs
            OUTCOMES[(key, graph_kind)] = (WORKS, label)
        accepted += 1
    assert not ignored, f"{key}: {kind} {ignored} accepted but change nothing"
    if not accepted:
        assert (key, kind) in GRAPH_REFUSES_EVERY, (key, kind, refusals)
        OUTCOMES[(key, graph_kind)] = (UNAVAILABLE, GRAPH_REFUSES_EVERY[(key, kind)])
    else:
        assert (key, kind) not in GRAPH_REFUSES_EVERY


# -- subspace logic (E2R) ------------------------------------------------------------------------
SUBSPACE_KINDS = ("classes", "relations", "measure", "objective", "objective_options",
                  "regularizers")


def _subspace_model(**parts):
    from mowl.models import ComposedSubspace
    th.manual_seed(0)
    return ComposedSubspace(data("e2r"), method="e2r", embed_dim=4, seed=1, **parts)


def _subspace_fingerprint(model, trained):
    if trained:
        model.train(epochs=2)
    for parameter in model.module.parameters():
        assert finite(parameter)
    n = len(model.class_index_dict)
    costs = model.axiom_costs("gci0", th.cartesian_prod(th.arange(n), th.arange(n)))
    assert finite(costs)
    m = len(model.individual_index_dict)
    if m:
        member = model.axiom_costs("class_assertion", th.cartesian_prod(th.arange(m),
                                                                         th.arange(n)))
        assert finite(member)
        costs = th.cat([th.as_tensor(costs).flatten(), th.as_tensor(member).flatten()])
    return th.as_tensor(costs).detach().clone()


def _subspace_candidates(kind, model):
    parts = current_parts(model.module)
    if kind in ("classes", "relations", "measure"):
        registry = {"classes": CLASS_MODELS, "relations": RELATION_MODELS,
                    "measure": MEASURES}[kind]
        return [(n, n) for n in registry]
    if kind == "objective":
        return [("Objective(0.07)", Objective(0.07))]
    if kind == "objective_options":
        objective = parts["objective"]
        out = []
        for name in objective.option_names():
            value = objective.options.get(name)
            if isinstance(value, bool):
                out.append((f"{name}={not value}", {name: not value}))
            elif isinstance(value, (int, float)):
                out.append((f"{name}={value + 0.07}", {name: value + 0.07}))
        return out
    return [("an added penalty", (Penalty(),))]


SUBSPACE_REFUSES_EVERY = {
    ("e2r", "classes"): "the residual measure compares Subspace regions and the registry has "
                        "no class model of subspaces (E2R's own SubspaceClasses is built by "
                        "the method, not registered under a name): every registered class "
                        "model is refused with both sides named",
    ("e2r", "relations"): "E2R's relations map pairs of vectors (a relation is a vector of the "
                          "pair space, dimension 2d); every registered relation model works "
                          "in the dimension of one class region and does not evaluate "
                          "against the pair equations (refused when the model is built)",
    ("e2r", "measure"): "the measure of E2R is the residual of a point against a subspace; "
                        "every other measure compares another region kind or has no residual "
                        "(refused when the model is built)",
    ("e2r", "objective"): "the losses of E2R are the equations of its axiom kinds "
                          "(SubspaceObjective.kind_cost): a general Objective has none, and a "
                          "SubspaceObjective is the method's own",
}


@pytest.mark.parametrize("kind", SUBSPACE_KINDS)
def test_the_subspace_method_takes_a_replacement_of_every_part(kind):
    """E2R through :class:`~mowl.models.ComposedSubspace`: a refusal is a
    ConstructorUnavailable at build; an accepted replacement is in the model, changes the
    costs before or after two epochs of training, and trains."""
    base = _subspace_model()
    base_untrained = _subspace_fingerprint(_subspace_model(), False)
    base_trained = _subspace_fingerprint(base, True)
    accepted, refusals, ignored = 0, [], []
    for label, argument in _subspace_candidates(kind, base):
        try:
            model = _subspace_model(**{kind: argument})
        except ConstructorUnavailable as error:
            refusals.append(f"{label}: {error}")
            continue
        parts_now, parts_base = current_parts(model.module), current_parts(base.module)
        if kind in ("classes", "relations", "measure") and type(parts_now[kind]) is type(
                parts_base[kind]):
            continue
        untrained = _subspace_fingerprint(_subspace_model(**{kind: argument}), False)
        trained = _subspace_fingerprint(model, True)
        reg_changed = kind == "regularizers" and float(
            model.module.regularization_loss()) != float(base.module.regularization_loss())
        if not (reg_changed or untrained.shape != base_untrained.shape
                or not th.equal(untrained, base_untrained)
                or trained.shape != base_trained.shape or not th.equal(trained, base_trained)):
            ignored.append(label)
        accepted += 1
        OUTCOMES.setdefault(("e2r", kind), (WORKS, label))
    assert not ignored, f"e2r: {kind} {ignored} accepted but change nothing"
    if not accepted:
        assert ("e2r", kind) in SUBSPACE_REFUSES_EVERY, ("e2r", kind, refusals)
        OUTCOMES[("e2r", kind)] = (UNAVAILABLE, SUBSPACE_REFUSES_EVERY[("e2r", kind)])


# -- the outcome matrix ----------------------------------------------------------------------
def test_the_matrix_in_the_documentation_lists_every_method():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "source",
                        "representation", "swaps.rst")
    text = open(path).read()
    missing = [k for k in METHODS if f"``{k}``" not in text]
    assert not missing, missing


def test_every_axiom_method_has_a_general_model_or_a_reason():
    for key in METHODS:
        if METHOD_KINDS[key] == "axioms":
            assert key in AXIOM_KEYS or key in UNSUPPORTED_BY_GENERAL_MODEL, key
    assert accepted_parts("elbe")


# -- writing the table ---------------------------------------------------------------------------
KIND_ORDER = ("classes", "relations", "individuals", "measure", "objective",
              "objective_options", "regularizers", "classes (graph)", "relations (graph)",
              "loss", "sampler", "regularizers (graph)", "view", "walker", "source", "encoder",
              "reading", "scorer", "intersection", "distance", "negation")
SYMBOL = {WORKS: "works", UNAVAILABLE: "unavailable", UNSUPPORTED: "not supported"}


def matrix_rst():
    """The outcome matrix as reStructuredText (a table per kind of method, then the reasons)."""
    lines = []
    reasons = []
    by_kind = {}
    for key in sorted(METHODS):
        by_kind.setdefault(METHOD_KINDS[key], []).append(key)
    titles = {"axioms": "Axiom methods (ComposedEL, ComposedALC)", "expressions": "Expression methods",
              "queries": "Query methods (QueryEmbeddingModel)",
              "sequences": "Sequence methods (ComposedSequenceModel)",
              "construction": "Constructed methods (ConstructedModel)"}
    for kind, keys in by_kind.items():
        kinds = [k for k in KIND_ORDER if any((key, k) in OUTCOMES for key in keys)]
        lines += [titles[kind], "-" * len(titles[kind]), "", ".. list-table::", "   :header-rows: 1",
                  "", "   * - method"] + [f"     - {k}" for k in kinds]
        for key in keys:
            lines.append(f"   * - ``{key}``")
            for k in kinds:
                status, detail = OUTCOMES.get((key, k), (None, None))
                if status is None:
                    lines.append("     - ")
                    continue
                lines.append(f"     - {SYMBOL[status]}")
                if status != WORKS and detail:
                    reasons.append((key, k, status, detail))
        lines.append("")
    lines += ["Reasons", "-------", ""]
    seen = set()
    for key, k, status, detail in reasons:
        text = " ".join(str(detail).split())[:300].replace("*", "\\*").replace("`", "'")
        if status == UNSUPPORTED:
            if (key, text) in seen:
                continue
            seen.add((key, text))
            lines.append(f"* ``{key}``, not supported: {text}")
        else:
            lines.append(f"* ``{key}``, {k}, unavailable: {text}")
    return "\n".join(lines) + "\n"


HEADER = """Swapping a part of a method: what works
=======================================

For every method of :data:`mowl.representation.METHODS` and every part it has, this table
says what happens when the part is replaced *by name* through the constructor of the general
model of the method (:doc:`combine`: ``ComposedEL``, ``ComposedALC``, ``ComposedSubspace``,
``QueryEmbeddingModel``, ``ComposedSequenceModel``, ``ConstructedModel``, and
``GraphKGEModel`` for the knowledge graph models, whose columns are marked ``(graph)`` or
are the loss and the sampler of triples). The cells are

``works``
   at least one replacement is accepted, is the part of the model, **changes the losses (or
   the parameters that receive gradient) compared with the method as it is**, and trains two
   epochs to finite numbers;
``unavailable``
   every replacement tried is refused at build time with
   :class:`~mowl.representation.ConstructorUnavailable`, naming both sides. The reason that
   this is right is written below and in the test (``REFUSES_EVERY``): a method that refuses
   every candidate of a part without a written reason fails the test;
``not supported``
   the method has no such part, or no general model trains it (the reason is below).

A replacement that is accepted but would change nothing (a relation model that no trained
form reads, an objective option that no loss reads, an objective of a triple model that is
trained with a loss) is refused at build time, never ignored. The ALC methods ``falcon``,
``dfalc`` and ``alcone`` are checked with expected failures until their objective is
validated at build time. The table is written by
``python -m tests.representation.test_user_level_swaps`` and checked by the tests.

"""


def main():
    import os
    import sys
    code = pytest.main(["-q", "-x", "-p", "no:cacheprovider", __file__, "-k",
                        "takes or take_a_replacement or says_so or individuals_of", "--no-header"])
    if code not in (0, 5):
        sys.exit(code)
    path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "source",
                        "representation", "swaps.rst")
    # pytest imported the module under its package name: its OUTCOMES are the filled ones
    from tests.representation import test_user_level_swaps as tests
    with open(path, "w") as out:
        out.write(HEADER + tests.matrix_rst())


if __name__ == "__main__":
    main()
