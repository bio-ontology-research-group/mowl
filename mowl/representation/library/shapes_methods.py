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

from collections import namedtuple
from functools import partial

import torch as th
import torch.nn as nn

from mowl.nn.el.octagon import geometry as G
from mowl.representation.adapters import register_adapter
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.methods import register_method
from mowl.representation.objectives import FORMS
from mowl.representation.relation_models import Translation
from mowl.training.negatives import PerColumnCorruption
from mowl.training.regularization import Regularizer
from mowl.representation.library.shapes_classes import (
    Points, OctagonClasses, PrismClasses, EllipsoidClasses, VectorClasses, MappedVectors,
    LinkedClasses, ContextualSoftBoxClasses, ContextTransform, RelationsByRegion, table)
from mowl.representation.library.shapes_measures import (
    OctagonBounds, PrismBounds, EllipsoidMeasure, CosineOrder, LinkedMeasure, GumbelVolumeRatio,
    PointToRegion, PointDistance, ByRegion)
from mowl.representation.library.shapes_objective import ShapeObjective
from mowl.representation.library.structural_methods import StructuralGeometry
from mowl.representation.library.shapes_regions import Point, SoftBox
from mowl.representation.library.shapes_training import SquaredMeanLoss
from mowl.representation.regions import Box

__all__ = ["MinWidthPenalty", "octagon", "prism", "eike", "concept2box", "OctagonELModule",
           "PrismModule", "EIKEModule", "Concept2BoxModule"]


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


def _instance(cls, into, base_init, *parts, **options):
    """The geometry of the parts: a new ``cls``, or ``into`` (a module class that is being
    initialised with the arguments of its constructor) initialised with them."""
    target = cls.__new__(cls) if into is None else into
    base_init(target, *parts, **options)
    return target


def _translations(t, nb_relations, embed_dim, init, scale):
    if t is not None:
        return Translation(vectors=t.rel_embed)
    return Translation(vectors=table(max(nb_relations, 1), embed_dim, init, scale))


# -- octagon ----------------------------------------------------------------------------------

@register_method("octagon")
def octagon(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.0,
            disjoint_margin=None, neg_margin=0.1, init_radius=0.5, min_width=0.0,
            tables=None, _into=None):
    """The octagon-product method of mOWL (an adaptation of Charpenay and Schockaert 2024):
    classes are products of 2-D octagons, relations translations, individuals points, bounds
    compared per direction.

    Published options set here: the negative loss of :math:`C \\sqsubseteq \\exists r.D` is
    "the translated :math:`C` sticks out of :math:`D`" (``gci2_negative="outside"`` with
    ``neg_margin``), a role assertion is the Euclidean distance (``assertion="distance"``,
    no negative), and only :math:`C \\sqsubseteq \\exists r.D` has a negative loss.

    Role axioms are not read (the translation of a role would have to equal that of its
    super-role), so the geometry is trained on the concept and assertion forms only.

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
    return _instance(OctagonELModule if t is None else ComposedGeometry, _into,
                     ComposedGeometry.__init__, classes, relations, OctagonBounds(), objective,
                     individuals=individuals, regularizers=regularizers,
                     supported_forms=FORMS, neg_capable_gcis={"gci2"})


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
          size_reg=1e-3, empty_box_reading="signed", size_reg_without_tables=False,
          max_depth=8, structural_definitions=True, tables=None, _into=None):
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
    half-widths, a phantom non-empty box. The size penalty is zero until the definition table
    is attached (it has to know ``owl:Thing``); ``size_reg_without_tables=True`` applies it to
    every class before that too, which shrinks ``owl:Thing`` (the module's earlier behaviour).

    :param max_depth: The deepest definition level of the structural reading (the definition
        table compiles acyclic conjunctions up to that depth).
    :param structural_definitions: Whether the structural reading compiles the definitions
        :math:`A \\equiv X \\sqcap Y` (the rows that state one cost nothing).
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
        size_reg_without_tables = getattr(t, "size_reg_without_tables", size_reg_without_tables)
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
        from mowl.representation.library.structural_reading import StructuralReading
        geometry = _instance(
            StructuralPrismModule if t is None else StructuralGeometry, _into,
            StructuralGeometry.__init__,
            classes, relations, measure,
            StructuralReading(margin, empty_regions="abs" if empty_box_reading == "absolute"
                              else "keep", gci3_bot="filler",
                              definitions=structural_definitions, **options),
            individuals=individuals, existence_weight=existence_weight, size_reg=size_reg,
            size_reg_without_tables=size_reg_without_tables,
            size_tables=lambda m: [m._slot_classes.inradii.weight,
                                   m._slot_classes.box_offsets.weight],
            size_scales=(1.0, 2.0), supported_forms=FORMS,
            neg_capable_gcis={"gci2", "object_property_assertion"})
        geometry.max_depth = max_depth
        geometry.default_regimes = (partial(SquaredMeanLoss, 0.0, 1.0, False),)
        if t is not None and getattr(t, "definition_table", None) is not None:
            geometry.set_structural_tables(t.definition_table,
                                           getattr(t, "existence_targets", None),
                                           getattr(t, "top_idx", None))
        return geometry
    objective = ShapeObjective(margin, **options)
    geometry = _instance(PrismModule if t is None else ComposedGeometry, _into,
                         ComposedGeometry.__init__, classes, relations, measure, objective,
                         individuals=individuals, supported_forms=FORMS,
                         neg_capable_gcis={"gci2", "object_property_assertion"})
    geometry.default_regimes = (partial(SquaredMeanLoss, 0.0, 1.0, True),)
    return geometry


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
         tables=None, _into=None):
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
    forms = ("gci0", "class_assertion", "object_property_assertion")
    geometry = _instance(EIKEModule if t is None else ComposedGeometry, _into,
                         ComposedGeometry.__init__, classes, relations, measure, objective,
                         individuals=individuals, supported_forms=forms,
                         neg_capable_gcis=forms)
    # the paper's negatives: head or tail corrupted, one negative of each kind per positive
    geometry.default_regimes = (partial(PerColumnCorruption, columns={
        "gci0": (0, 1), "class_assertion": (0, 1), "object_property_assertion": (0, 2)}),)
    return geometry


@register_adapter("EIKEModule")
def _eike(module):
    return eike(None, None, tables=module)


# -- concept2box --------------------------------------------------------------------------------

@register_method("concept2box")
def concept2box(nb_classes, nb_relations, nb_individuals=0, embed_dim=32, margin=0.0,
                beta=1.0, alpha=0.5, kge_margin=0.3, cross_margin=0.15, hidden_dim=None,
                instance_weight=1.0, cross_weight=1.0, gci2_reading="axiom", tables=None,
                _into=None):
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
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
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
    forms = ("gci0", "gci2", "class_assertion", "object_property_assertion")
    return _instance(Concept2BoxModule if t is None else ComposedGeometry, _into,
                     ComposedGeometry.__init__, classes, relations, measure, objective,
                     individuals=points, supported_forms=forms, neg_capable_gcis=forms)


@register_adapter("Concept2BoxModule")
def _concept2box(module):
    return concept2box(None, None, margin=0.0, tables=module)


# -- the modules of the four methods --------------------------------------------------------------
#
# ``mowl.nn.OctagonELModule`` and the others are the geometries of the builders above, with the
# constructor arguments the modules have always had. The earlier hand-written loss modules are
# in ``mowl.baselines.nn.el``.

class OctagonELModule(ComposedGeometry):
    r"""The octagon method (:func:`octagon`) as a module: classes are products of 2-D octagons
    (``class_embed``: centre, ``class_radius``: eight radii per pair of coordinates),
    relations translations (``rel_embed``), individuals points (``ind_embed``); inclusion,
    emptiness and disjointness are the exact tests of
    :class:`~mowl.representation.library.shapes_measures.OctagonBounds`.

    :param nb_ont_classes: Number of classes.
    :param nb_rels: Number of object properties.
    :param nb_inds: Number of individuals, or ``None``.
    :param embed_dim: Even dimension :math:`d \ge 2`.
    :param margin: Margin of the inclusion and membership losses.
    :param disjoint_margin: Margin of ``gci1_bot``; ``None`` uses ``margin``.
    :param neg_margin: Margin of the negative ``gci2`` loss.
    :param init_radius: Radii are initialised uniformly in ``[0, init_radius]``.
    :param min_width: If positive, a penalty on octagons narrower than this
        (:class:`MinWidthPenalty`).
    """

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, margin=0.0,
                 disjoint_margin=None, neg_margin=0.1, init_radius=0.5, min_width=0.0):
        octagon(nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim, margin=margin,
                disjoint_margin=disjoint_margin, neg_margin=neg_margin,
                init_radius=init_radius, min_width=min_width, _into=self)
        self.min_width = float(min_width)

    @property
    def class_radius(self):
        return self.class_model.radii

    @property
    def embed_dim(self):
        return self.class_model.embed_dim

    @property
    def nb_pairs(self):
        return self.embed_dim // 2

    @property
    def nb_ont_classes(self):
        return len(self.class_model)

    @property
    def disjoint_margin(self):
        return self.objective.disjoint_margin

    @property
    def neg_margin(self):
        return self.objective.neg_margins["gci2"]

    def class_bounds(self, idx):
        """Octagon bounds of the classes ``idx``, shape ``(B, P, 8)``."""
        return self.class_model(idx).bounds

    def set_class_bounds(self, idx, h):
        """Overwrite the parameters of class ``idx`` so that its bounds are ``h``: the witness
        point of each (non-empty) factor as centre and the radii :math:`h_k - n_k \\cdot c
        \\ge 0`. For tests and exact initialisation.

        :param h: Bounds, shape ``(P, 8)``; every factor must be non-empty.
        """
        h = h.to(self.class_embed.weight)
        if bool((G.emptiness_slack(h) < -1e-9).any()):
            raise ValueError("Every factor must be non-empty.")
        centre = G.witness_point(G.tighten(h))
        radius = h - centre @ G.directions_like(h).T
        with th.no_grad():
            self.class_embed.weight[idx] = centre.reshape(-1)
            self.class_radius.weight[idx] = th.clamp_min(radius, 0.0).reshape(-1)

    def all_class_bounds(self):
        """Bounds of all classes, detached, shape ``(nb_ont_classes, P, 8)``."""
        with th.no_grad():
            ids = th.arange(len(self.class_model), device=self.device)
            return self.class_model(ids).bounds.detach()


class PrismModule(ComposedGeometry):
    r"""The prism method (:func:`prism`, ``objective="axiom"``) as a module: classes are
    triangle :math:`\times` box (``tri_center``, ``tri_inradius``, ``box_center``,
    ``box_offset``: :math:`2d - 1` numbers per class), relations translations
    (``rel_embed``), individuals points (``ind_embed``). With ``objective="structural"`` the
    module is also a :class:`~mowl.representation.library.structural_methods.StructuralGeometry`
    (:class:`StructuralPrismModule`): it reads the structure of the ontology once
    :meth:`set_structural_tables` has been called.

    The arguments are those of :func:`prism`: ``margin``, ``objective`` (``"axiom"`` or
    ``"structural"``, the ``reading`` of the builder), ``max_depth`` and
    ``structural_definitions`` (the definition table of the structural reading),
    ``existence_weight``, ``size_reg``, ``gci2_negative_triangle``, ``empty_box_reading``
    and ``size_reg_without_tables``.
    """

    def __new__(cls, nb_ont_classes=None, nb_rels=None, nb_inds=None, embed_dim=50, margin=0.1,
                objective="axiom", *args, **kwargs):
        if cls is PrismModule and objective == "structural":
            cls = StructuralPrismModule
        return super().__new__(cls)

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, margin=0.1,
                 objective="axiom", max_depth=8, structural_definitions=True,
                 existence_weight=1.0, size_reg=1e-3, gci2_negative_triangle="overlap",
                 empty_box_reading="signed", size_reg_without_tables=False):
        if objective not in ("axiom", "structural"):
            raise ValueError(f"objective must be 'axiom' or 'structural', got {objective!r}.")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError(f"max_depth must be an int >= 0, got {max_depth!r}")
        if not isinstance(structural_definitions, bool):
            raise ValueError(
                f"structural_definitions must be a bool, got {structural_definitions!r}")
        prism(nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim, margin=margin,
              gci2_negative_triangle=gci2_negative_triangle, reading=objective,
              existence_weight=existence_weight, size_reg=size_reg,
              empty_box_reading=empty_box_reading, max_depth=max_depth,
              structural_definitions=structural_definitions,
              size_reg_without_tables=size_reg_without_tables, _into=self)

    @property
    def tri_center(self):
        return self.class_model.centres

    @property
    def tri_inradius(self):
        return self.class_model.inradii

    @property
    def box_center(self):
        return self.class_model.box_centres

    @property
    def box_offset(self):
        return self.class_model.box_offsets

    @property
    def embed_dim(self):
        return self.class_model.embed_dim


#: Support values ``h`` of the triangle, centres ``cen`` and signed half-widths ``w`` of the box
#: of prisms (the form the structural tests of the prism method read).
PrismRegions = namedtuple("PrismRegions", ["h", "cen", "w"])


class StructuralPrismModule(PrismModule, StructuralGeometry):
    """:class:`PrismModule` with ``objective="structural"``: a
    :class:`~mowl.representation.library.structural_methods.StructuralGeometry`."""

    @property
    def tri_center(self):
        return self._slot_classes.centres

    @property
    def tri_inradius(self):
        return self._slot_classes.inradii

    @property
    def box_center(self):
        return self._slot_classes.box_centres

    @property
    def box_offset(self):
        return self._slot_classes.box_offsets

    def class_regions(self, idx):
        """The prisms of the classes ``idx`` (defined classes are intersections, with signed
        half-widths) as ``(h, cen, w)``."""
        prism_ = self.class_model(idx)
        return PrismRegions(prism_.supports, prism_.centre, prism_.offset)

    def all_class_regions(self):
        """The prisms of all classes, detached."""
        with th.no_grad():
            return self.class_regions(th.arange(len(self.class_model), device=self.device))

    def structural_extra_loss(self):
        """The existence and size penalties (the regularisation of the structural reading)."""
        return self.regularization_loss()

    @property
    def structural_table_options(self):
        """Those of the structural geometry, and the deepest definition level
        (``max_depth``)."""
        return dict(super().structural_table_options, max_level=self.max_depth)


class EIKEModule(ComposedGeometry):
    r"""EIKE (:func:`eike`) as a module: every name has an ellipsoid (``class_center``,
    ``class_axes_raw``, ``class_radius_raw``) and an intensional vector
    (``class_intensional``), individuals are points (``ind_embed``) and the vectors
    ``M i`` (``M``), relations translations (``rel_embed``).

    :param nb_classes: Number of classes.
    :param nb_rels: Number of object properties.
    :param nb_inds: Number of individuals (at least one row is made).
    :param embed_dim: Dimension of both spaces.
    :param alpha: Weight of the intensional costs.
    :param margins: ``(gamma_sub, gamma_ins, gamma_rel)``.
    :param instance_map: ``"eye"`` or ``"matrix"``.
    :param intensional_init: Optional ``(nb_classes, embed_dim)`` array of vectors for the
        intensional space (rows with NaN keep their random initialisation).
    :param freeze_intensional: Keep the intensional vectors fixed.
    """

    def __init__(self, nb_classes, nb_rels, nb_inds, embed_dim=100, alpha=1.0,
                 margins=(1.0, 1.0, 1.0), instance_map="eye", intensional_init=None,
                 freeze_intensional=False):
        if intensional_init is not None:
            shape = tuple(getattr(intensional_init, "shape", (len(intensional_init),
                                                              embed_dim)))
            if shape != (nb_classes, embed_dim):
                raise ValueError(f"intensional_init must have shape ({nb_classes}, "
                                 f"{embed_dim}), got {shape}.")
        eike(nb_classes, nb_rels, nb_inds, embed_dim=embed_dim, alpha=alpha, margins=margins,
             instance_map=instance_map, intensional_init=intensional_init,
             freeze_intensional=freeze_intensional, _into=self)

    @property
    def embed_dim(self):
        return self.class_model.first.centres.embedding_dim

    @property
    def class_center(self):
        return self.class_model.first.centres

    @property
    def class_axes_raw(self):
        return self.class_model.first.axes_raw

    @property
    def class_radius_raw(self):
        return self.class_model.first.radius_raw

    @property
    def class_intensional(self):
        return self.class_model.second.centres

    @property
    def M(self):
        return self.individual_model.second._matrix(self.ind_embed.weight)

    @property
    def alpha(self):
        return self.measure.weight

    @property
    def gamma_sub(self):
        return self.objective.ranking["gci0"]

    @property
    def gamma_ins(self):
        return self.objective.ranking["class_assertion"]

    @property
    def gamma_rel(self):
        return self.objective.ranking["object_property_assertion"]


class Concept2BoxModule(ComposedGeometry):
    r"""Concept2Box (:func:`concept2box`) as a module: classes are Gumbel boxes in the context
    of a relation (``class_embed``, ``ontology_rel_embed``, ``box_network``), individuals points
    (``ind_embed``), relations translations of individuals (``rel_embed``).

    :param nb_classes: Number of classes.
    :param nb_rels: Number of object properties.
    :param nb_inds: Number of individuals (``None``: no ABox).
    :param embed_dim: Dimension of boxes and vectors.
    :param beta: Gumbel scale.
    :param alpha: Weight of the distance to the centre of a point inside a box
        (``0 < alpha < 1``).
    :param kge_margin: Margin of the instance view.
    :param cross_margin: Margin of the cross view.
    :param hidden_dim: Hidden size of the MLP (default ``2 * embed_dim``).
    :param instance_weight: :math:`\lambda_1`.
    :param cross_weight: :math:`\lambda_2`.
    :param gci2_reading: ``"axiom"`` or ``"triple"`` (see :func:`concept2box`).
    """

    def __init__(self, nb_classes, nb_rels, nb_inds=None, embed_dim=32, beta=1.0, alpha=0.5,
                 kge_margin=0.3, cross_margin=0.15, hidden_dim=None, instance_weight=1.0,
                 cross_weight=1.0, gci2_reading="axiom"):
        concept2box(nb_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim, beta=beta,
                    alpha=alpha, kge_margin=kge_margin, cross_margin=cross_margin,
                    hidden_dim=hidden_dim, instance_weight=instance_weight,
                    cross_weight=cross_weight, gci2_reading=gci2_reading, _into=self)

    @property
    def embed_dim(self):
        return self.class_model.embed_dim

    @property
    def ontology_rel_embed(self):
        return self.class_model.contexts

    @property
    def box_network(self):
        """The two-layer network that maps a class and a relation to a box (the baseline
        module calls it ``project``; here that name is the projection hook of
        :class:`~mowl.representation.ComposedGeometry`)."""
        return self.class_model.project

    @property
    def beta(self):
        return self.class_model.beta

    @property
    def alpha(self):
        return self.measure.rules[1][2].alpha

    @property
    def subclass_id(self):
        return self.class_model.subclass_id

    @property
    def null_id(self):
        return self.class_model.null_id

    @property
    def gci2_reading(self):
        return self.objective.gci2_reading

    @property
    def kge_margin(self):
        return self.objective.neg_margins["object_property_assertion"]

    @property
    def cross_margin(self):
        return self.objective.neg_margins["class_assertion"]

    @property
    def instance_weight(self):
        return self.objective.weights["object_property_assertion"]

    @property
    def cross_weight(self):
        return self.objective.weights["class_assertion"]
