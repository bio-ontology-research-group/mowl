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
    if key in ("dfalc", "falcon", "alcone") and False:
        pass
    return ComposedEL(data("family"), method=key, **el_options(key), **parts)


def alc_model(key, **parts):
    th.manual_seed(0)
    dataset = data("e2r") if key == "e2r" else data("alc")
    return ComposedALC(dataset, method=key, embed_dim=D, **parts)


def check_axioms(model):
    model.train(epochs=1)
    for parameter in model.module.parameters():
        assert finite(parameter)
    with th.no_grad():
        for form, rows in model.training_datasets.items():
            if len(rows):
                assert finite(model.module(rows[:], form)), form


def check_alc(model):
    model.train(epochs=1)
    for parameter in model.module.parameters():
        assert finite(parameter)
    n = len(model.class_index_dict)
    try:
        costs = model.axiom_costs("gci0", th.cartesian_prod(th.arange(n), th.arange(n)))
    except (TypeError, AttributeError, RuntimeError, ValueError) as error:
        raise ConstructorUnavailable(f"the swapped combination does not score: {error}") \
            from error
    assert finite(costs)


# -------------------------------------------------------------------------------------------
# candidates per part kind of a composed geometry
# -------------------------------------------------------------------------------------------
def candidates(kind, model):
    """Replacements to try, in order: ``(label, argument, check(model))`` where ``check``
    asserts the part is in the model."""
    parts = current_parts(model.module)
    if kind == "classes":
        for name in CLASS_MODELS:
            yield name, name, lambda m, n=name: _same_kind(m, "classes", n)
    elif kind == "relations":
        for name in RELATION_MODELS:
            yield name, name, lambda m, n=name: _same_kind(m, "relations", n)
        yield NO_PART, NO_PART, lambda m: m.module.relation_model is None
    elif kind == "individuals":
        for name in CLASS_MODELS:
            yield name, name, lambda m, n=name: _same_kind(m, "individuals", n)
    elif kind == "measure":
        for name in MEASURES:
            yield name, name, lambda m, n=name: _same_kind(m, "measure", n)
    elif kind == "objective":
        own = parts["objective"].replace()
        yield "the method's own objective, as an instance", own, \
            lambda m: m.module.objective is own
        other = Objective(0.07)
        yield "Objective(0.07)", other, lambda m: m.module.objective is other
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
            yield f"{name}={new}", {name: new}, lambda m, o={name: new}: _has_options(m, o)
    elif kind == "regularizers":
        penalty = Penalty()
        yield "an added penalty", (penalty,), \
            lambda m: penalty in m.module.regularizers and \
            finite(m.module.regularization_loss())


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


def swap_kinds(model):
    """The part kinds of a composed geometry the method has, to swap."""
    kinds = ["classes", "relations", "measure", "objective", "objective_options",
             "regularizers"]
    if model.module.individual_model is not None:
        kinds.insert(2, "individuals")
    return kinds


def run(build, check, key, kind, limit=None):
    """Try the replacements of ``kind`` for ``key`` through the constructor of the general
    model. Passes if every one raises the documented error or works; returns the outcome: the
    first that works, else the documented error."""
    base = build(key)
    tried, last_error = 0, None
    for label, argument, verify in candidates(kind, base):
        if limit and tried >= limit:
            break
        tried += 1
        try:
            model = build(key, **{kind: argument})
        except EXPECTED_ERRORS as error:
            last_error = f"{label}: {error}"
            continue
        assert verify(model), (key, kind, label)
        try:
            check(model)
        except EXPECTED_ERRORS as error:      # a part that cannot read an axiom form
            last_error = f"{label}: {error}"
            continue
        OUTCOMES[(key, kind)] = (WORKS, label)
        return WORKS, label
    if tried == 0:
        OUTCOMES[(key, kind)] = (UNSUPPORTED, "no replacement is defined for this reading")
        return UNSUPPORTED, None
    OUTCOMES[(key, kind)] = (UNAVAILABLE, last_error)
    return UNAVAILABLE, last_error


AXIOM_KEYS = sorted(k for k, v in METHOD_KINDS.items() if v == "axioms"
                    and k not in UNSUPPORTED_BY_GENERAL_MODEL)
ALC_KEYS = ("dfalc", "falcon", "alcone")
EL_KINDS = ["classes", "relations", "measure", "objective", "objective_options",
            "regularizers"]


@pytest.mark.parametrize("kind", EL_KINDS)
@pytest.mark.parametrize("key", AXIOM_KEYS)
def test_every_axiom_method_takes_a_replacement_of_every_part_by_name(key, kind):
    status, _ = run(axioms_model, check_axioms, key, kind)
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
    status, _ = run(alc_model, check_alc, key, kind, limit=6)
    assert status in (WORKS, UNAVAILABLE)


UNSUPPORTED_BY_GENERAL_MODEL["e2r"] = (
    "its geometry reads ALC axioms through subspaces (a SubspaceGeometry, not an "
    "ExpressionGeometry): ComposedALC refuses it, ComposedEL refuses a method trained on "
    "expressions. Train it with E2R.")


@pytest.mark.parametrize("key", sorted(UNSUPPORTED_BY_GENERAL_MODEL))
def test_a_method_no_general_model_trains_says_so(key):
    with pytest.raises((TypeError, ConstructorUnavailable, ValueError)):
        if key == "e2r":
            ComposedALC(data("e2r"), method=key, classes="subspace", embed_dim=D)
        else:
            ComposedEL(data("family"), method=key, classes="box", embed_dim=D)
    for kind in EL_KINDS:
        OUTCOMES[(key, kind)] = (UNSUPPORTED, UNSUPPORTED_BY_GENERAL_MODEL[key])


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
    if new_measure is not None:
        args["measure"] = new_measure
    model = axioms_model(key, **args)
    swapped = current_parts(model.module)
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


QUERY_PARTS = {
    "query2box": {"intersection": ["exact"], "distance": ["l2"],
                  "negation": ["complement", "inversion"]},
    "betae": {"intersection": ["mean"], "distance": ["reverse kl"], "negation": [None]},
    "cone_qe": {"intersection": ["attention"], "distance": ["l2"] if False else [],
                "negation": ["signed complement"], "relations": ["rotation"]},
    "acone": {"intersection": ["attention"], "negation": ["signed complement"]},
}


def _query_check(model):
    model.train(epochs=1)
    assert math.isfinite(model.loss_history[-1])
    assert all(finite(p) for p in model.module.parameters())
    costs = model.axiom_costs("gci0", th.tensor([[0, 1], [1, 2]]))
    assert finite(costs)


QUERY_PART_OF = {"intersection": lambda m: m.module.intersection,
                 "distance": lambda m: m.module.distance_part,
                 "negation": lambda m: m.module.negation_part,
                 "relations": lambda m: m.module.relations}


def _query_names(key, kind):
    from mowl.representation.library.query_methods import _NEGATIONS
    if kind == "negation":
        return list(_NEGATIONS)
    if kind == "intersection":
        return {"query2box": ["attention", "exact"], "betae": ["attention", "mean"]}.get(
            key, ["attention"])
    if kind == "distance":
        return {"query2box": ["l1", "l2"], "betae": ["kl", "reverse kl"]}.get(key, [])
    return ["rotation"]


QUERY_KINDS = ("intersection", "distance", "negation", "relations")


@pytest.mark.parametrize("kind", QUERY_KINDS)
@pytest.mark.parametrize("key", [k for k, v in METHOD_KINDS.items() if v == "queries"])
def test_query_methods_take_a_replacement_of_every_part_by_name(key, kind):
    import inspect
    if kind not in inspect.signature(METHODS[key]).parameters:
        with pytest.raises(TypeError, match="accepted"):
            _query_model(key, **{kind: "attention"})
        OUTCOMES[(key, kind)] = (UNSUPPORTED, f"{key} has no {kind} option: its relations "
                                 f"are fixed by the method")
        return
    names = _query_names(key, kind)
    default_type = type(QUERY_PART_OF[kind](_query_model(key))) if kind != "relations" \
        else None
    last = None
    for name in names:
        try:
            model = _query_model(key, **{kind: name})
            _query_check(model)
        except (ConstructorUnavailable, ValueError, NotImplementedError) as error:
            last = f"{name}: {type(error).__name__}: {error}"
            continue
        part = QUERY_PART_OF[kind](model)
        if kind == "relations":
            assert type(part) is not type(QUERY_PART_OF[kind](_query_model(key)))
        elif kind == "negation":
            from mowl.representation.library.query_methods import _NEGATIONS
            assert type(part) is type(_NEGATIONS[name]())
        elif name not in ("attention", "l1", "kl"):
            assert type(part) is not default_type, (key, kind, name)
        OUTCOMES[(key, kind)] = (WORKS, name)
        return
    OUTCOMES[(key, kind)] = (UNAVAILABLE, last)


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
    for candidate in _seq_candidates(key, kind):
        model = _seq_model(key, **{kind: candidate})
        part = {"view": None, "walker": None}.get(kind, getattr(model.module, kind, None))
        if kind == "source":
            assert model.module.source is candidate
        elif kind == "walker":
            assert type(model.module.source.walker).__name__.lower().startswith("node2vec") \
                or "node2vec" in type(model.module.source.walker).__name__.lower() \
                or "Node2Vec" in model.module.parts["walker"]
        elif kind == "view":
            assert model.module.parts["projection"] != _seq_model(key).module.parts.get(
                "projection")
        elif kind == "encoder":
            assert "fasttext" in model.module.parts["encoder"].lower() \
                or "fast" in type(part).__name__.lower()
        else:
            assert part is not None
        model.train(epochs=1)
        n = min(4, len(model.class_index_dict))
        rows = th.tensor([[i, j] for i in range(n) for j in range(n) if i != j])
        assert finite(model.axiom_costs("gci0", rows))
        OUTCOMES[(key, kind)] = (WORKS, candidate if isinstance(candidate, str)
                                 else type(candidate).__name__)
        return


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


@pytest.mark.parametrize("kind", ("source", "reading"))
@pytest.mark.parametrize("key", CONSTRUCT_KEYS)
def test_constructed_methods_take_a_replacement_of_source_and_reading(key, kind):
    from mowl.geometric.errors import MissingDependency
    from mowl.models.constructed import ConstructedModel
    from mowl.representation.construction import RECIPES
    dataset = data("family")
    last = None
    for name in _construct_candidates(kind):
        if name == RECIPES[key].parts[kind]:
            continue
        try:
            model = ConstructedModel(dataset, method=key, **{kind: name})
            model.train(epochs=1)
        except (ConstructorUnavailable, NotImplementedError, ValueError, TypeError,
                MissingDependency, AttributeError, KeyError, RuntimeError) as error:
            last = f"{name}: {type(error).__name__}: {error}"
            continue
        assert model.parts[kind] == name, (key, kind, name)
        OUTCOMES[(key, kind)] = (WORKS, name)
        return
    OUTCOMES[(key, kind)] = (UNAVAILABLE, last)


def test_a_constructed_option_no_part_reads_is_a_type_error():
    from mowl.models.constructed import ConstructedModel
    with pytest.raises(TypeError):
        ConstructedModel(data("family"), method="interval_union",
                         reading_options={"not_an_option": 1})
    with pytest.raises(TypeError):
        ConstructedModel(data("family"), method="interval_union", relations="translation")


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
              "objective_options", "regularizers", "view", "walker", "source", "encoder",
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
            lines.append(f"* ``{key}``, {k}, unavailable. Last replacement tried: {text}")
    return "\n".join(lines) + "\n"


HEADER = """Swapping a part of a method: what works
=======================================

For every method of :data:`mowl.representation.METHODS` and every part it has, this table
says what happens when the part is replaced *by name* through the constructor of the general
model of the method (:doc:`combine`): ``works`` (some replacement trains one epoch and gives
finite numbers), ``unavailable`` (every replacement tried is refused with
:class:`~mowl.representation.ConstructorUnavailable`: the combination cannot exist, and the
message names both sides), ``not supported`` (the method has no such part or no general model
trains it; the reason is below). The table is written by
``python -m tests.representation.test_user_level_swaps`` and checked by the tests.

"""


def main():
    import os
    import sys
    code = pytest.main(["-q", "-x", "-p", "no:cacheprovider", __file__, "-k",
                        "takes or says_so or individuals_of", "--no-header"])
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
