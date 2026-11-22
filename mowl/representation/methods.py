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

__all__ = ["METHODS", "CLASS_MODELS", "RELATION_MODELS", "MEASURES", "DEFAULT_MEASURE",
           "compose", "elem", "elbe"]

#: Class models by name.
CLASS_MODELS = {"ball": BallClasses, "box": BoxClasses}
#: Relation models by name.
RELATION_MODELS = {"translation": Translation}
#: Measures by name.
MEASURES = {"centre distance": CentreDistance, "bounds per coordinate": Bounds}
#: The measure used for a region class when none is named.
DEFAULT_MEASURE = {Ball: CentreDistance, Box: Bounds}


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
    return ComposedGeometry(classes, relations, CentreDistance(gci1_radius_term), objective,
                            individuals=individuals,
                            regularizers=_norm_penalty(reg_norm, individuals is not None))


def elbe(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.1,
         all_negatives=False, epsilon=0.01, gci3="overlap", tables=None):
    """ELBE (Peng et al. 2022): ELEmbeddings with boxes for balls, compared coordinate by
    coordinate, with a strict margin and no penalty.

    :param tables: A module with ``class_embed``, ``class_offset``, ``rel_embed`` (and
        ``ind_embed``, ``ind_offset``) to share parameters with.
    """
    t = tables
    classes = BoxClasses(nb_classes, embed_dim) if t is None else BoxClasses(
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
    return ComposedGeometry(classes, relations, Bounds(separation="every"), objective,
                            individuals=individuals)


#: Catalogue key -> function building the method from parts.
METHODS = {"elem": elem, "elbe": elbe}
