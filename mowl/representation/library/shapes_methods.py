"""The methods of the shapes group built from parts, and their adapters.

======================  =====================================================================
Catalogue key           Parts
======================  =====================================================================
``octagon``             octagon products, translation, points, octagon bounds
``prism``               prisms, translation, points, prism bounds
``eike``                (ellipsoids, vectors) linked, translation, (points, mapped vectors)
                        linked, ellipsoid measure linked to cosine order, ranking objective
``concept2box``         contextual soft boxes, (context transform, translation) by region,
                        points, (volume ratio, point-to-box, point distance) by region
======================  =====================================================================

Each function builds the geometry with new tables, or with the tables of an existing module
(``tables``), which is what :func:`~mowl.representation.geometry_of` does. The options the
published method sets (and a good default does not) are explicit arguments.
"""

import torch as th
import torch.nn as nn

from mowl.nn.el.octagon import geometry as G
from mowl.representation.adapters import register_adapter
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.methods import register_method
from mowl.representation.relation_models import Translation
from mowl.training.regularization import Regularizer
from mowl.representation.library.shapes_classes import (
    Points, OctagonClasses, PrismClasses, EllipsoidClasses, VectorClasses, MappedVectors,
    LinkedClasses, ContextualSoftBoxClasses, ContextTransform, RelationsByRegion, table)
from mowl.representation.library.shapes_measures import (
    OctagonBounds, PrismBounds, EllipsoidMeasure, CosineOrder, LinkedMeasure, GumbelVolumeRatio,
    PointToRegion, PointDistance, ByRegion)
from mowl.representation.library.shapes_objective import ShapeObjective
from mowl.representation.library.shapes_regions import Point, SoftBox
from mowl.representation.regions import Box

__all__ = ["MinWidthPenalty", "octagon", "prism", "eike", "concept2box"]


class MinWidthPenalty(Regularizer):
    r""":math:`\operatorname{mean}_C \sum [w_{\min} - \mathrm{width}]_+` over all classes,
    factors and facet axes of an octagon-product class model: keeps octagons from
    collapsing to points.

    :param min_width: :math:`w_{\min}`.
    """

    kind = "penalty"

    def __init__(self, min_width=0.1, weight=1.0):
        super().__init__(parameters="class_embed", weight=weight)
        self.min_width = float(min_width)

    def penalty(self, model):
        module = getattr(model, "module", model)
        n = len(module.class_model)
        if n == 0 or self.min_width <= 0.0:
            return None
        h = module.class_model.all().bounds
        short = th.relu(self.min_width - G.widths(h))
        return self.weight * short.reshape(n, -1).sum(dim=-1).mean()


def _translations(t, nb_relations, embed_dim, init, scale):
    if t is not None:
        return Translation(vectors=t.rel_embed)
    return Translation(vectors=table(max(nb_relations, 1), embed_dim, init, scale))


# -- octagon ----------------------------------------------------------------------------------

@register_method("octagon")
def octagon(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.0,
            disjoint_margin=None, neg_margin=0.1, init_radius=0.5, min_width=0.0,
            tables=None):
    """The octagon-product method of mOWL (an adaptation of Charpenay and Schockaert 2024):
    classes are products of 2-D octagons, relations translations, individuals points, bounds
    compared per direction.

    Published options set here: the negative loss of :math:`C \\sqsubseteq \\exists r.D` is
    "the translated :math:`C` sticks out of :math:`D`" (``gci2_negative="outside"`` with
    ``neg_margin``), a role assertion is the Euclidean distance (``assertion="distance"``,
    no negative), and only :math:`C \\sqsubseteq \\exists r.D` has a negative loss.

    :param tables: A module with ``class_embed``, ``class_radius``, ``rel_embed`` (and
        ``ind_embed``) to share parameters with.
    """
    t = tables
    classes = OctagonClasses(nb_classes, embed_dim, init_radius=init_radius) if t is None \
        else OctagonClasses(centres=t.class_embed, radii=t.class_radius)
    relations = _translations(t, nb_relations, embed_dim, "uniform", 0.1)
    individuals = None
    if t is None and nb_individuals:
        individuals = Points(nb_individuals, embed_dim, family="octagon", init="uniform")
    elif t is not None and getattr(t, "ind_embed", None) is not None:
        individuals = Points(family="octagon", centres=t.ind_embed)
    objective = ShapeObjective(
        margin, gci3="inclusion", gci2_negative="outside", negatives=("gci2",),
        disjoint_margin=disjoint_margin, neg_margins={"gci2": neg_margin},
        assertion="distance", strict_negatives=False)
    regularizers = (MinWidthPenalty(min_width),) if min_width and min_width > 0 else ()
    return ComposedGeometry(classes, relations, OctagonBounds(), objective,
                            individuals=individuals, regularizers=regularizers,
                            neg_capable_gcis={"gci2"})


@register_adapter("OctagonELModule")
def _octagon(module):
    own = module.disjoint_margin != module.margin
    return octagon(None, None, margin=module.margin,
                   disjoint_margin=module.disjoint_margin if own else None,
                   neg_margin=module.neg_margin, min_width=module.min_width, tables=module)


# -- prism ------------------------------------------------------------------------------------

@register_method("prism")
def prism(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.1,
          gci2_negative_triangle="overlap", reading="axiom", existence_weight=1.0,
          size_reg=1e-3, empty_box_reading="signed", tables=None):
    r"""The prism method (Federico, Leemhuis and Kutz 2026), default objective of the module
    (``objective="axiom"``): classes are triangle :math:`\times` box, relations translations,
    individuals points.

    Published options set here: :math:`\exists r.C \sqsubseteq D` only has to overlap
    :math:`D` (``gci3="overlap"``), the loss of :math:`C \sqsubseteq \bot` is the support
    sum of the triangle plus ``margin`` and ignores the box (``PrismBounds(empty="triangle")``),
    :math:`A \sqcap B \sqsubseteq C` also needs the intersection to be non-empty
    (``PrismBounds(conjunction="nonempty")``, with the absolute half-widths of the box part:
    ``Prism.intersect(signed=False)``), and the negative loss of
    :math:`C \sqsubseteq \exists r.D` is the exact overlap test of
    the translated triangle and the triangle of :math:`D` (``gci2_negative_triangle="overlap"``,
    the measure's ``meets``); ``"inradius"`` is the earlier test on the inscribed discs
    (``meets_shrunk``), which is not an overlap test: a small triangle in a corner of a large
    one overlaps it and still has a positive cost.

    ``reading="structural"`` (the module's ``objective="structural"``): the structural reading
    (:class:`~mowl.representation.library.structural_reading.StructuralReading`: squared hinges
    with non-emptiness gates for the concept forms, rows that state a definition zero), the
    class model wrapped in
    :class:`~mowl.representation.library.structural_regions.DefinedClasses` (a defined class is
    the intersection of its definers, with signed box half-widths:
    :meth:`~mowl.representation.library.structural_methods.StructuralGeometry.set_structural_tables`
    attaches the table) and the existence and size penalties (``existence_weight``,
    ``size_reg``); ``margin`` is the tightening margin :math:`m`. The forms about relations
    and assertions read a defined class that is empty only in its box part (negative
    half-width) as empty (``empty_box_reading="signed"``, the default: an empty source or
    filler makes the row vacuous, an empty target contains nothing). ``"absolute"`` is the
    earlier reading of the module (``empty_regions="abs"``): the absolute value of the
    half-widths, a phantom non-empty box.

    :param tables: A module with ``tri_center``, ``tri_inradius``, ``box_center``,
        ``box_offset``, ``rel_embed`` (and ``ind_embed``) to share parameters with.
    """
    if gci2_negative_triangle not in ("overlap", "inradius"):
        raise ValueError("gci2_negative_triangle must be 'overlap' or 'inradius'.")
    if reading not in ("axiom", "structural"):
        raise ValueError("reading must be 'axiom' or 'structural'.")
    if empty_box_reading not in ("signed", "absolute"):
        raise ValueError("empty_box_reading must be 'signed' or 'absolute'.")
    t = tables
    if t is not None:
        empty_box_reading = getattr(t, "empty_box_reading", empty_box_reading)
    if t is not None:
        gci2_negative_triangle = getattr(t, "gci2_negative_triangle", gci2_negative_triangle)
    if t is None:
        classes = PrismClasses(nb_classes, embed_dim)
    else:
        classes = PrismClasses(tri_centres=t.tri_center, tri_inradii=t.tri_inradius,
                               box_centres=t.box_center, box_offsets=t.box_offset)
    relations = _translations(t, nb_relations, embed_dim, "unit", 1.0)
    individuals = None
    if t is None and nb_individuals:
        individuals = Points(nb_individuals, embed_dim, family="prism")
    elif t is not None and getattr(t, "ind_embed", None) is not None:
        individuals = Points(family="prism", centres=t.ind_embed)
    measure = PrismBounds(empty="triangle", conjunction="nonempty", empty_prisms="formula")
    options = dict(
        gci3="overlap",
        gci2_negative="overlap" if gci2_negative_triangle == "overlap" else "shrunk",
        negatives="gci2", empty_margin=True, strict_negatives=False)
    if reading == "structural":
        from mowl.representation.library.structural_methods import StructuralGeometry
        from mowl.representation.library.structural_reading import StructuralReading
        geometry = StructuralGeometry(
            classes, relations, measure,
            StructuralReading(margin, empty_regions="abs" if empty_box_reading == "absolute"
                              else "keep", gci3_bot="filler", **options),
            individuals=individuals, existence_weight=existence_weight, size_reg=size_reg,
            size_reg_without_tables=True,  # as the prism module (which shrinks owl:Thing too)
            size_tables=lambda m: [m._slot_classes.inradii.weight,
                                   m._slot_classes.box_offsets.weight],
            size_scales=(1.0, 2.0), neg_capable_gcis={"gci2", "object_property_assertion"})
        if t is not None and getattr(t, "definition_table", None) is not None:
            geometry.set_structural_tables(t.definition_table,
                                           getattr(t, "existence_targets", None),
                                           getattr(t, "top_idx", None))
        return geometry
    objective = ShapeObjective(margin, **options)
    return ComposedGeometry(classes, relations,
                            measure, objective,
                            individuals=individuals,
                            neg_capable_gcis={"gci2", "object_property_assertion"})


@register_adapter("PrismModule")
def _prism(module):
    return prism(None, None, margin=module.margin,
                 reading=getattr(module, "objective", "axiom"),
                 existence_weight=getattr(module, "existence_weight", 1.0),
                 size_reg=getattr(module, "size_reg", 1e-3),
                 empty_box_reading=getattr(module, "empty_box_reading", "signed"),
                 tables=module)


# -- eike -------------------------------------------------------------------------------------

@register_method("eike")
def eike(nb_classes, nb_relations, nb_individuals=0, embed_dim=100, margin=0.0, alpha=1.0,
         margins=(1.0, 1.0, 1.0), instance_map="eye", intensional_init=None,
         freeze_intensional=False, containment="published", norm_term="signed",
         tables=None):
    r"""EIKE (Wang et al. 2024): every name has an ellipsoid (what it contains) and a vector
    (what it means), individuals are points and the vector :math:`M i`, relations are
    translations of individuals. The cost of an axiom is the cost in the first space plus
    ``alpha`` times the cost in the second; a negative axiom costs its positive cost minus
    the margin of its form, so that the ranking loss of the model is
    ``relu(positive - negative)``.

    Published options set here: ``containment="published"`` (the formula as printed, which
    is not containment, see :class:`~mowl.representation.library.shapes_measures.EllipsoidMeasure`)
    and ``norm_term="signed"`` (a signed length difference, which makes costs negative).
    The defaults of the measures are the sound ones.

    ``margin`` is not used (the margins of EIKE are ``margins``, one per form).

    :param margins: ``(gamma_sub, gamma_ins, gamma_rel)``.
    :param instance_map: ``"eye"`` (fixed identity) or ``"matrix"`` (learned).
    :param intensional_init: Optional ``(nb_classes, embed_dim)`` array of vectors for the
        second space (rows with NaN keep their random initialisation).
    :param tables: An :class:`~mowl.nn.el.eike.module.EIKEModule` to share parameters with.
    """
    t = tables
    gamma_sub, gamma_ins, gamma_rel = margins if t is None else (
        t.gamma_sub, t.gamma_ins, t.gamma_rel)
    if t is None:
        first = EllipsoidClasses(nb_classes, embed_dim)
        second = VectorClasses(nb_classes, embed_dim)
        if intensional_init is not None:
            import numpy as np
            init = th.as_tensor(np.asarray(intensional_init, dtype="float32"))
            keep = ~th.isnan(init).any(dim=1)
            with th.no_grad():
                second.centres.weight[keep] = init[keep]
        if freeze_intensional:
            second.centres.weight.requires_grad_(False)
        if instance_map == "matrix":
            matrix = nn.Parameter(th.eye(embed_dim) + 0.01 * th.randn(embed_dim, embed_dim))
        elif instance_map == "eye":
            matrix = th.eye(embed_dim)
        else:
            raise ValueError("instance_map must be 'eye' or 'matrix'.")
        points = Points(max(nb_individuals, 1), embed_dim, init="uniform", scale=0.5)
        relations = Translation(vectors=table(max(nb_relations, 1), embed_dim, "uniform",
                                              0.5))
    else:
        first = EllipsoidClasses(centres=t.class_center, axes_raw=t.class_axes_raw,
                                 radius_raw=t.class_radius_raw)
        second = VectorClasses(centres=t.class_intensional)
        points = Points(centres=t.ind_embed)
        matrix = None
        relations = Translation(vectors=t.rel_embed)
        alpha = t.alpha
    classes = LinkedClasses(first, second)
    individuals = LinkedClasses(points, MappedVectors(
        points.centres, matrix, owner=None if t is None else (t, "M")))
    measure = LinkedMeasure(EllipsoidMeasure(containment), CosineOrder(norm_term), alpha)
    objective = ShapeObjective(
        0.0, negatives="gci2", strict_negatives=False,
        ranking={"gci0": gamma_sub, "class_assertion": gamma_ins,
                 "object_property_assertion": gamma_rel})
    return ComposedGeometry(classes, relations, measure, objective, individuals=individuals,
                            neg_capable_gcis={"gci0", "class_assertion",
                                              "object_property_assertion"})


@register_adapter("EIKEModule")
def _eike(module):
    return eike(None, None, tables=module)


# -- concept2box --------------------------------------------------------------------------------

@register_method("concept2box")
def concept2box(nb_classes, nb_relations, nb_individuals=0, embed_dim=32, margin=0.0,
                beta=1.0, alpha=0.5, kge_margin=0.3, cross_margin=0.15, hidden_dim=None,
                instance_weight=1.0, cross_weight=1.0, gci2_reading="axiom", tables=None):
    r"""Concept2Box (Huang et al. 2023): classes are Gumbel boxes that depend on a context
    (a relation), individuals are points translated by relations, a class assertion links
    the two by the distance of a point to a box.

    Published options set here: the axioms are read as triples with boxes in the context of
    the relation (``context_axioms=True``); for :math:`C \sqsubseteq \exists r.D` the box of
    :math:`C` inside that of :math:`D` (``gci2_reading="axiom"``, the default) or the paper's
    triple :math:`(C, r, D)` with the box of :math:`D` inside that of :math:`C`
    (``"triple"``). The
    deviation of the volume ratio is squared (``power=2``), the distance of a point is
    weighted by the volume and squashed by a sigmoid, and the negative losses have their own
    margins (``kge_margin``, ``cross_margin``) and weights.

    :param gci2_reading: ``"axiom"`` or ``"triple"``, see above.
    :param tables: A :class:`~mowl.nn.el.concept2box.module.Concept2BoxModule` to share
        parameters with (its ``gci2_reading`` is used).
    """
    t = tables
    if t is None:
        classes = ContextualSoftBoxClasses(nb_classes, embed_dim, nb_relations, beta,
                                           hidden_dim)
        translations = Translation(vectors=table(max(nb_relations, 1), embed_dim, "normal",
                                                 0.5))
        points = Points(nb_individuals, embed_dim, init="normal", scale=0.5) \
            if nb_individuals else None
    else:
        beta, alpha = t.beta, t.alpha
        kge_margin, cross_margin = t.kge_margin, t.cross_margin
        instance_weight, cross_weight = t.instance_weight, t.cross_weight
        gci2_reading = t.gci2_reading
        classes = ContextualSoftBoxClasses(beta=beta, centres=t.class_embed,
                                           contexts=t.ontology_rel_embed, project=t.project)
        translations = Translation(vectors=t.rel_embed)
        points = Points(centres=t.ind_embed) if t.ind_embed is not None else None
    relations = RelationsByRegion([(SoftBox, ContextTransform(classes)),
                                   (Point, translations)])
    measure = ByRegion([
        (Box, Box, GumbelVolumeRatio(beta, power=2)),
        (Point, Box, PointToRegion(alpha, beta, weighted=True, squash=True)),
        (Point, Point, PointDistance(1))])
    objective = ShapeObjective(
        margin, negatives="all", context_axioms=True, gci2_reading=gci2_reading,
        neg_margins={"class_assertion": cross_margin,
                     "object_property_assertion": kge_margin},
        weights={"class_assertion": cross_weight,
                 "object_property_assertion": instance_weight})
    return ComposedGeometry(classes, relations, measure, objective, individuals=points,
                            neg_capable_gcis={"gci0", "gci2", "class_assertion",
                                              "object_property_assertion"})


@register_adapter("Concept2BoxModule")
def _concept2box(module):
    return concept2box(None, None, margin=0.0, tables=module)
