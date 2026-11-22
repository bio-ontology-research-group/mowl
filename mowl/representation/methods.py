"""Published methods as combinations of parts.

Each function returns the :class:`~mowl.representation.ComposedGeometry` of one method:
with new tables, to train the method from its parts, or with the tables of an existing
module (``tables``), which is how :func:`mowl.representation.geometry_of` describes that
module. :data:`METHODS` lists them by catalogue key; :data:`CLASS_MODELS`,
:data:`RELATION_MODELS` and :data:`MEASURES` list the parts that can be combined freely.
"""

from mowl.representation.class_models import BallClasses, BoxClasses
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.measures import CentreDistance, Bounds
from mowl.representation.objectives import Objective
from mowl.representation.regions import Ball, Box
from mowl.representation.relation_models import Translation

__all__ = ["METHODS", "METHOD_KINDS", "CLASS_MODELS", "RELATION_MODELS", "MEASURES",
           "DEFAULT_MEASURE",
           "compose", "elem", "elbe", "register_class_model", "register_relation_model",
           "register_measure", "register_method"]

#: Class models by name.
CLASS_MODELS = {"ball": BallClasses, "box": BoxClasses}
#: Relation models by name.
RELATION_MODELS = {"translation": Translation}
#: Measures by name.
MEASURES = {"centre distance": CentreDistance, "bounds per coordinate": Bounds}
#: The measure used for a region class when none is named.
DEFAULT_MEASURE = {Ball: CentreDistance, Box: Bounds}
#: Catalogue key -> function building the method from parts.
METHODS = {}
#: Catalogue key -> what the method is trained on: ``"axioms"`` (normal-form axioms, what
#: :class:`~mowl.models.ComposedEL` trains) or ``"queries"`` (query answering).
METHOD_KINDS = {}


def _unique(registry, name, value):
    """Names are unique: a second part under a taken name would silently replace the
    first for everyone who asks for it by name."""
    if name in registry and registry[name] is not value:
        raise ValueError(f"{name!r} is already registered ({registry[name]!r}).")


def register_class_model(name):
    """Decorator: make a class model available under ``name`` (a function or class
    ``(size, embed_dim) -> Classes``)."""
    def decorate(factory):
        _unique(CLASS_MODELS, name, factory)
        CLASS_MODELS[name] = factory
        return factory
    return decorate


def register_relation_model(name):
    """Decorator: make a relation model available under ``name``."""
    def decorate(factory):
        _unique(RELATION_MODELS, name, factory)
        RELATION_MODELS[name] = factory
        return factory
    return decorate


def register_measure(name, default_for=()):
    """Decorator: make a measure available under ``name``; ``default_for`` lists region
    classes it becomes the default measure of."""
    def decorate(factory):
        _unique(MEASURES, name, factory)
        MEASURES[name] = factory
        for region in default_for:
            DEFAULT_MEASURE[region] = factory
        return factory
    return decorate


def register_method(key, trained_on="axioms"):
    """Decorator: register a function that builds the method with catalogue key ``key``
    from parts. Signature: ``(nb_classes, nb_relations, nb_individuals=0, embed_dim=50,
    ..., tables=None) -> Geometry``.

    :param trained_on: ``"axioms"`` (default): the geometry is an
        :class:`~mowl.nn.ELModule` that
        :class:`~mowl.models.ComposedEL` trains on the normal-form axioms (a geometry that
        reads only some forms says which: ``supported_forms``). ``"queries"``: the method is
        trained on query structures (:mod:`mowl.models.query`) and ``ComposedEL`` refuses it.
    """
    if trained_on not in ("axioms", "queries"):
        raise ValueError("trained_on must be 'axioms' or 'queries'.")

    def decorate(function):
        _unique(METHODS, key, function)
        METHODS[key] = function
        METHOD_KINDS[key] = trained_on
        return function
    return decorate


def _part(value, registry, kind):
    if isinstance(value, str):
        if value not in registry:
            raise ValueError(f"Unknown {kind} {value!r}; available: {sorted(registry)}.")
        return registry[value]
    return value


def compose(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, classes="box",
            relations="translation", measure=None, objective=None, individuals=True,
            regularizers=()):
    """A geometry from freely chosen parts.

    :param classes: Name in :data:`CLASS_MODELS`, or a function ``(size, embed_dim)`` that
        returns a class model.
    :param relations: Name in :data:`RELATION_MODELS`, such a function, or ``None``.
    :param measure: Name in :data:`MEASURES`, a measure object, or ``None`` for the
        default measure of the class model.
    :param objective: An :class:`~mowl.representation.Objective` (default: semantic,
        strict, margin 0).
    :param individuals: Whether individuals get regions of the same kind as classes (only
        if there are any).
    """
    class_factory = _part(classes, CLASS_MODELS, "class model")
    class_model = class_factory(nb_classes, embed_dim)
    relation_model = None
    if relations is not None and nb_relations:
        relation_model = _part(relations, RELATION_MODELS, "relation model")(
            nb_relations, embed_dim)
    if measure is None:
        measure = DEFAULT_MEASURE[class_model.region]()
    elif isinstance(measure, str):
        measure = _part(measure, MEASURES, "measure")()
    individual_model = class_factory(nb_individuals, embed_dim) \
        if individuals and nb_individuals else None
    return ComposedGeometry(class_model, relation_model, measure, objective,
                            individuals=individual_model, regularizers=regularizers)


def _elbe_bounds():
    """The box measure of ELBE's losses: separated in every coordinate, the size of a box
    is the norm of its half-widths, a conjunction must not be empty."""
    return Bounds(separation="every", extent="norm", conjunction="nonempty")


def _norm_penalty(reg_norm, individuals):
    from mowl.training.regularization import NormPenalty
    names = ("class_embed", "ind_embed") if individuals else ("class_embed",)
    return (NormPenalty(parameters=names, target=reg_norm),)


def elem(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.1, reg_norm=1,
         all_negatives=False, linear_negative=False, epsilon=0.01, gci1_radius_term=True,
         gci3="overlap", tables=None):
    """ELEmbeddings (Kulmanov et al. 2019): balls, translations, distance of centres, a
    slack margin and the unit-norm penalty on the centres.

    :param gci3: ``"overlap"`` as published; ``"inclusion"`` is the corrected reading.
    :param tables: A module with ``class_embed``, ``class_rad``, ``rel_embed`` (and
        ``ind_embed``, ``ind_rad``) to share parameters with.
    """
    t = tables
    classes = BallClasses(nb_classes, embed_dim) if t is None else BallClasses(
        centres=t.class_embed, radii=t.class_rad)
    relations = Translation(nb_relations, embed_dim) if t is None else Translation(
        vectors=t.rel_embed)
    individuals = None
    if t is None and nb_individuals:
        individuals = BallClasses(nb_individuals, embed_dim)
    elif t is not None and getattr(t, "ind_embed", None) is not None:
        individuals = BallClasses(centres=t.ind_embed, radii=t.ind_rad)
    objective = Objective.elem(margin, all_negatives=all_negatives,
                               linear_negative=linear_negative, epsilon=epsilon, gci3=gci3)
    return ComposedGeometry(classes, relations,
                            CentreDistance(gci1_radius_term, conjunction="published"),
                            objective,
                            individuals=individuals,
                            regularizers=_norm_penalty(reg_norm, individuals is not None))


def elbe(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.1,
         all_negatives=False, epsilon=0.01, gci3="overlap", init="separate", tables=None):
    """ELBE (Peng et al. 2022): ELEmbeddings with boxes for balls, compared coordinate by
    coordinate, with a strict margin and no penalty.

    :param init: ``"separate"`` or ``"joint"``, see :class:`~mowl.representation.BoxClasses`.
    :param tables: A module with ``class_embed``, ``class_offset``, ``rel_embed`` (and
        ``ind_embed``, ``ind_offset``) to share parameters with.
    """
    t = tables
    classes = BoxClasses(nb_classes, embed_dim, init=init) if t is None else BoxClasses(
        centres=t.class_embed, offsets=t.class_offset)
    relations = Translation(nb_relations, embed_dim) if t is None else Translation(
        vectors=t.rel_embed)
    individuals = None
    if t is None and nb_individuals:
        individuals = BoxClasses(nb_individuals, embed_dim)
    elif t is not None and getattr(t, "ind_embed", None) is not None:
        individuals = BoxClasses(centres=t.ind_embed, offsets=t.ind_offset)
    objective = Objective.elbe(margin, all_negatives=all_negatives, epsilon=epsilon,
                               gci3=gci3)
    return ComposedGeometry(classes, relations, _elbe_bounds(), objective,
                            individuals=individuals)


METHODS.update({"elem": elem, "elbe": elbe})
METHOD_KINDS.update({"elem": "axioms", "elbe": "axioms"})
