"""Query2Box, BetaE, ConE and AConE from parts, and their common interface.

:class:`ComposedQueryGeometry` is a :class:`~mowl.nn.query.base.QueryGeometry` (so a
query embedding model can train it) and a :class:`~mowl.representation.Geometry`. **The
methods are defined here, each as one named combination of parts**: the builders
:func:`query2box`, :func:`betae`, :func:`cone_qe` and :func:`acone` (registered in
:data:`~mowl.representation.METHODS` with ``trained_on="queries"``) say which entity model,
relation model, intersection operator, negation operator and distance make up the method,
and every one of those can be swapped by an argument (``intersection="exact"``,
``distance="l2"``, an instance of another part, ...). Parts of different region families
(a cone negation on boxes) are refused with an error that says so. The classes
:class:`Query2BoxGeometry`, :class:`BetaEGeometry`, :class:`ConEGeometry` and
:class:`AConEGeometry` are the modules the builders return, with the constructors of
:mod:`mowl.nn.query.geometry`. With ``tables=`` a builder describes another module that
holds the tables and networks of a method (for example the direct implementation in
``mowl.baselines``) and shares its parameters, which is what
:func:`~mowl.representation.geometry_of` returns for it.

Relation ids follow the query models: relation ``0`` is ``isa``, ``1`` its inverse, role
``k`` is ``2 + 2k`` and its inverse ``3 + 2k``. The interface takes *role* ids:
``exists(k, C)`` projects the region of ``C`` along the inverse of role ``k``, which is
how :mod:`mowl.nn.query.expression` reads :math:`\\exists r.C`.

A named class is "everything below it": ``classes(ids)`` is the projection of the anchor
along ``isa`` inverse (``extension=True``, the default of the query models), or the anchor
itself with ``extension=False``. ``individuals(ids)`` is the anchor of an individual.
"""

import torch as th
import torch.nn as nn

from mowl.nn.query.base import QueryGeometry, UnsupportedOperator
from mowl.representation.adapters import register_adapter
from mowl.representation.class_models import Classes
from mowl.representation.geometry import Geometry
from mowl.representation.library.query_parts import (
    PointBoxes, BetaPoints, ConePoints, BetaNetwork, ConeNetwork, ConeRotation,
    BoxAttentionIntersection, BoxExactIntersection, BetaAttentionIntersection,
    BetaMeanIntersection, ConeAttentionIntersection, BetaInversion, ConeComplement,
    BoxDistance, BetaDistance, ConeDistance, deferred_initialisation, _uniform)
from mowl.representation.library.boxrel_relations import _ACTIVATIONS, TranslationBoxes
from mowl.representation.library.query_regions import (
    BetaKL, ConeBounds, as_state, from_state, lift)
from mowl.representation.measures import Bounds
from mowl.representation.methods import register_method
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["ComposedQueryGeometry", "QueryClasses", "query2box", "betae", "cone_qe", "acone",
           "Query2BoxGeometry", "BetaEGeometry", "ConEGeometry", "AConEGeometry",
           "role_inverse", "ISA_INV"]

ISA_INV = 1


def role_inverse(role_ids):
    """The relation id from the filler to the subject of the roles ``role_ids``."""
    return 3 + 2 * role_ids


class ComposedQueryGeometry(QueryGeometry, Geometry):
    """A query embedding geometry put together from parts.

    :param entities: Entity model (:class:`PointBoxes`, :class:`BetaPoints`,
        :class:`ConePoints`).
    :param relations: Relation model with ``image`` (and ``inverse_of`` for ``exists``).
    :param intersection: Intersection operator (a module of a list of regions).
    :param distance: Distance of an entity from a query region.
    :param measure: Measure that compares two query regions (:meth:`inside`).
    :param negation: Negation operator or ``None``.
    :param num_classes: Number of named classes: entities ``0..num_classes-1`` are
        classes, the rest individuals. ``None`` if unknown (then :meth:`individuals`
        raises).
    :param extension: Whether :meth:`classes` is the extension of the class (default) or
        the anchor.
    """

    def __init__(self, entities, relations, intersection, distance, measure, negation=None,
                 dim=None, gamma=24.0, num_classes=None, extension=True, isa_inverse=ISA_INV):
        gamma = float(gamma)
        super().__init__(len(entities), len(relations), dim, gamma)
        self.entities, self.relations = entities, relations
        self.intersection, self.distance_part = intersection, distance
        self.negation_part, self.measure = negation, measure
        self.supports_negation = negation is not None
        self._num_classes = num_classes
        self.extension = extension
        self.isa_inverse = isa_inverse
        self.region_class = entities.region
        self._check_family()

    def _check_family(self):
        """Refuse parts that cannot read each other's regions (a cone negation on boxes)."""
        region = self.region_class
        self.relations.check_region(region)
        for label, part in (("intersection operator", self.intersection),
                            ("distance", self.distance_part),
                            ("negation operator", self.negation_part)):
            family = getattr(part, "region", None)
            if part is not None and family is not None and not issubclass(region, family):
                raise ConstructorUnavailable(
                    f"The {label} {type(part).__name__} works on {family.__name__} regions, "
                    f"but the entity model {type(self.entities).__name__} gives "
                    f"{region.__name__} regions: choose parts of one region family.")

    # -- options of the parts, as the modules of the methods have them -----------------------
    @property
    def signed_negation(self):
        """Whether the negation is the published ``pi - aperture`` (cones)."""
        return self.negation_part.signed

    @property
    def aperture_mode(self):
        """``"clip"`` or ``"wrap"``: how a rotation keeps the aperture an angle."""
        return self.relations.aperture

    @property
    def clamp_extent(self):
        """Whether a projected box has its extent cut at zero (boxes)."""
        return self.relations.clamp_extent

    @property
    def cen(self):
        """The weight of the in-region distance."""
        return self.distance_part.cen

    # -- the QueryGeometry operators (states are tuples of tensors) --------------------------
    def _region(self, state):
        return from_state(self.region_class, state)

    def anchor(self, entity_ids):
        return as_state(self.entities(entity_ids))

    def project(self, state, relation_ids):
        return as_state(self.relations.image(relation_ids, self._region(state)))

    def intersect(self, states):
        return as_state(self.intersection([self._region(s) for s in states]))

    def negate(self, state):
        if self.negation_part is None:
            raise UnsupportedOperator(f"{type(self).__name__} has no negation operator.")
        return as_state(self.negation_part(self._region(state)))

    def distance(self, state, entity_ids):
        return self.distance_part(lift(self._region(state)), self.entities(entity_ids))

    def entity_vectors(self):
        return self.entities.flat()

    def relation_vectors(self):
        return self.relations.flat()

    # -- the Geometry interface (regions) ----------------------------------------------------
    def anchors(self, ids):
        """The regions of the entities themselves (points, point distributions, rays)."""
        return self.entities(ids)

    def classes(self, ids):
        region = self.entities(ids)
        if not self.extension:
            return region
        return self.relations.image(th.full_like(ids, self.isa_inverse), region)

    def individuals(self, ids):
        if self._num_classes is None:
            raise ConstructorUnavailable(
                "The number of classes is unknown, so individual ids cannot be turned into "
                "entity ids: build the geometry with num_classes=... (query_geometry(model) "
                "does).")
        return self.entities(ids + self._num_classes)

    def exists(self, relation_ids, region):
        """Region of :math:`\\exists r.C` for the *role* ids ``relation_ids``."""
        return self.relations.preimage(relation_ids, region)

    def conjunction(self, a, b):
        regions = [a, b]
        return self.intersection(regions), self.intersection.gap(regions)

    def negation(self, a):
        if self.negation_part is None:
            raise ConstructorUnavailable(f"{type(self).__name__} has no negation.")
        return self.negation_part(a)

    def inside(self, a, b):
        return self.measure.inside(a, b, 0.0)

    def apart(self, a, b):
        return self.measure.apart(a, b, 0.0)

    def member(self, individual_ids, region):
        """The distance of the individuals to the query regions: the model's own distance
        (not the logit), ``(B, 1)``."""
        if self._num_classes is None:
            self.individuals(individual_ids)            # raises with the explanation
        ids = (individual_ids + self._num_classes).unsqueeze(1)
        return self.distance_part(lift(region), self.entities(ids))

    @property
    def num_classes(self):
        if self._num_classes is None:
            raise NotImplementedError("The number of classes is unknown.")
        return self._num_classes

    @property
    def num_individuals(self):
        return 0 if self._num_classes is None else self.num_entities - self._num_classes

    @property
    def parts(self):
        return {"classes": self.entities.part, "relations": self.relations.part,
                "measure": self.measure.part,
                "intersection": type(self.intersection).__name__,
                "distance": type(self.distance_part).__name__,
                "conjunction": "exact" if self.intersection.exact else "approximate",
                "negation": "none" if self.negation_part is None
                else ("exact" if getattr(self.negation_part, "exact", False)
                      else "approximate")}


class QueryClasses(Classes):
    """The classes of a query geometry as a class model: a class is the region
    :meth:`ComposedQueryGeometry.classes` (its extension: the anchor projected along
    ``isa`` inverse). The table is the entity table of the geometry (shared).

    This can serve as the class model of a :class:`~mowl.models.ComposedEL`, but it is a
    poor one for subsumption: a relation model gives every projection of every anchor the
    same size, so all classes are regions of one size (for boxes: one offset), and
    :math:`C \\sqsubseteq D` can only hold by the two being equal. Query models rank
    entities by their distance to a region instead of comparing two regions.
    """

    exact_conjunction = False

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.centres = geometry.entities.table
        self.region = geometry.region_class
        self.part = geometry.entities.part

    def forward(self, ids):
        return self.geometry.classes(ids)


def _sizes(nb_classes, nb_relations, nb_individuals):
    return nb_classes + nb_individuals, 2 + 2 * nb_relations


def _part(value, names, kind):
    """``value`` if it is a module, else the part ``names[value]()``."""
    if isinstance(value, nn.Module):
        return value
    if value not in names:
        raise ValueError(f"{kind} must be a part or one of {sorted(map(str, names))}, "
                         f"not {value!r}.")
    return names[value]()


_NEGATIONS = {"inversion": BetaInversion, "complement": ConeComplement,
              "signed complement": lambda: ConeComplement(signed=True)}


def _negation(value, default):
    if isinstance(value, str) and value == "default":
        value = default
    if value is None:
        return None
    return _part(value, _NEGATIONS, "negation")


def _query2box_parts(n_entities, n_rows, dim, gamma, center_weight, activation, clamp_extent,
                     intersection, distance, negation, tables):
    if tables is not None:
        for name, function in _ACTIVATIONS.items():
            if tables.activation is function:
                activation = name
        clamp_extent = tables.clamp_extent
        center_weight = tables.cen
        entities = PointBoxes(table=tables.entity_embedding)
        relations = TranslationBoxes(vectors=tables.relation_embedding,
                                     offsets=tables.offset_embedding, activation=activation,
                                     inverse_of=role_inverse, clamp_extent=clamp_extent)
        attention = lambda: BoxAttentionIntersection(tables=tables)  # noqa: E731
    else:
        r = (gamma + 2.0) / dim
        entities = PointBoxes(n_entities, dim, init_range=r)
        # the tables are drawn as in the reference code (one uniform tensor each)
        relations = TranslationBoxes(
            vectors=_uniform(n_rows, dim, -r, r), offsets=_uniform(n_rows, dim, 0, r),
            activation=activation, inverse_of=role_inverse, clamp_extent=clamp_extent)
        attention = lambda: BoxAttentionIntersection(dim)  # noqa: E731
    operator = _part(intersection, {"attention": attention, "exact": BoxExactIntersection},
                     "intersection")
    dist = _part(distance if distance is not None else "l1",
                 {"l1": lambda: BoxDistance(center_weight),
                  "l2": lambda: BoxDistance(center_weight, norm=2)}, "distance")
    return entities, relations, operator, dist, Bounds(), _negation(negation, None)


def _betae_parts(n_entities, n_rows, dim, gamma, hidden, layers, intersection, distance,
                 negation, tables):
    if tables is not None:
        entities = BetaPoints(table=tables.entity_embedding)
        relations = BetaNetwork(tables=tables, inverse_of=role_inverse)
        attention = lambda: BetaAttentionIntersection(tables=tables)  # noqa: E731
        operator = _part(intersection, {"attention": attention, "mean": BetaMeanIntersection},
                         "intersection")
    else:
        r = (gamma + 2.0) / dim
        # the layers of all parts are created first and initialised at the end, in the order
        # of the reference code: a seed gives the numbers of the direct implementation
        with deferred_initialisation():
            entities = BetaPoints(n_entities, dim, init_range=r)
            relations = BetaNetwork(n_rows, dim, hidden, layers, inverse_of=role_inverse,
                                    init_range=r)
            attention = lambda: BetaAttentionIntersection(dim)  # noqa: E731
            operator = _part(intersection, {"attention": attention,
                                            "mean": BetaMeanIntersection}, "intersection")
    dist = _part(distance if distance is not None else "kl",
                 {"kl": BetaDistance, "reverse kl": lambda: BetaDistance(reverse=True)},
                 "distance")
    return entities, relations, operator, dist, BetaKL(), _negation(negation, "inversion")


def _cone_parts(n_entities, n_rows, dim, gamma, center_weight, hidden, layers, dropout,
                relations, signed_negation, aperture, intersection, distance, negation,
                tables):
    if relations not in ("network", "rotation"):
        raise ValueError("relations must be 'network' or 'rotation'.")
    if tables is not None:
        if relations == "network" and not hasattr(tables, "proj_in"):
            raise ValueError("These tables have no projection network; use "
                             "relations='rotation'.")
        center_weight = tables.cen
        signed_negation = tables.signed_negation
        aperture = getattr(tables, "aperture_mode", aperture)
        angle_range = tables.embedding_range
        entities = ConePoints(table=tables.entity_embedding, angle_range=angle_range)
        rel = (ConeNetwork(tables=tables, inverse_of=role_inverse) if relations == "network"
               else ConeRotation(tables=tables, inverse_of=role_inverse,
                                 aperture=aperture, aperture_change="signed"))
        attention = lambda: ConeAttentionIntersection(tables=tables)  # noqa: E731
        modulus = tables.modulus
    else:
        angle_range = (gamma + 2.0) / dim
        entities = ConePoints(n_entities, dim, angle_range=angle_range)
        # the tables of the relations, the intersection and then the projection network are
        # created in this order (the reference code of ConE): a seed gives the numbers of the
        # direct implementation
        rel = (ConeNetwork(n_rows, dim, hidden, layers, angle_range=angle_range,
                           inverse_of=role_inverse, build_network=False)
               if relations == "network"
               else ConeRotation(n_rows, dim, angle_range=angle_range,
                                 inverse_of=role_inverse, aperture=aperture,
                                 aperture_change="signed"))
        attention = lambda: ConeAttentionIntersection(dim, dropout)  # noqa: E731
        modulus = None
    operator = _part(intersection, {"attention": attention}, "intersection")
    if tables is None and relations == "network":
        rel.build_network(hidden, layers)
    if distance is None:
        dist = ConeDistance(center_weight, modulus=modulus, angle_range=angle_range)
    else:
        dist = _part(distance, {}, "distance")
    return (entities, rel, operator, dist, ConeBounds(),
            _negation(negation, "signed complement" if signed_negation else "complement"))


class _NamedQueryGeometry(ComposedQueryGeometry):
    """A :class:`ComposedQueryGeometry` with the constructor of a method's module."""

    def _assemble(self, parts, dim, gamma, num_classes, extension):
        entities, relations, intersection, distance, measure, negation = parts
        ComposedQueryGeometry.__init__(
            self, entities, relations, intersection, distance, measure, negation=negation,
            dim=dim, gamma=gamma, num_classes=num_classes, extension=extension)


class Query2BoxGeometry(_NamedQueryGeometry):
    """Query2Box as a combination of parts: point entities (:class:`PointBoxes`), boxes of
    translations (:class:`~mowl.representation.library.boxrel_relations.TranslationBoxes`),
    the attention intersection of boxes, the out-of-box plus in-box distance, no negation.

    :param center_weight: ``cen``, the weight of the in-box distance (0.02 in the paper).
    :param activation: ``"relu"``, ``"softplus"`` or ``"none"``, applied to the relation
        offsets (the reference code offers all three).
    :param clamp_extent: ``False`` (default, as in the reference code): the offset of a
        projected box is the sum of the offsets, which with ``activation="none"`` can become
        negative. ``True``: offsets are clamped at 0 after each projection.
    :param intersection: ``"attention"`` (the paper), ``"exact"`` (the set intersection of
        the boxes, no parameters) or an intersection operator of boxes.
    :param distance: ``"l1"`` (the paper), ``"l2"`` or a distance part of boxes.
    :param negation: ``None`` (the paper has none) or a negation part of boxes (there is
        none in the library: asking for another family's raises).
    """

    def __init__(self, num_entities, num_relations, dim, gamma=24.0, center_weight=0.02,
                 activation="relu", clamp_extent=False, intersection="attention",
                 distance=None, negation=None, num_classes=None, extension=True, tables=None):
        self._assemble(_query2box_parts(num_entities, num_relations, dim, float(gamma),
                                        center_weight, activation, clamp_extent,
                                        intersection, distance, negation, tables),
                       dim, gamma, num_classes, extension)


class BetaEGeometry(_NamedQueryGeometry):
    """BetaE as a combination of parts: Beta distributions (:class:`BetaPoints`), a network
    relation model, the attention intersection, the inversion as negation, the KL divergence.

    :param hidden: Width of the projection network (1600 in the reference code).
    :param layers: Hidden layers of the projection network (2 in the code).
    :param intersection: ``"attention"`` (the paper), ``"mean"`` (the plain average of the
        parameters) or an intersection operator of distributions.
    :param distance: ``"kl"`` (the paper: ``KL(entity || query)``), ``"reverse kl"`` or a
        distance part of distributions.
    :param negation: ``"default"`` (the inversion of the paper), ``None`` (no negation) or a
        negation part.
    """

    def __init__(self, num_entities, num_relations, dim, gamma=24.0, hidden=128, layers=1,
                 intersection="attention", distance=None, negation="default",
                 num_classes=None, extension=True, tables=None):
        self._assemble(_betae_parts(num_entities, num_relations, dim, float(gamma), hidden,
                                    layers, intersection, distance, negation, tables),
                       dim, gamma, num_classes, extension)


class ConEGeometry(_NamedQueryGeometry):
    """ConE as a combination of parts: sector cones (:class:`ConePoints`), a network
    relation model (:class:`ConeNetwork`), the SemanticAverage / CardMin intersection, the
    complement as negation and the angular distance.

    :param center_weight: ``cen``, the weight of the inside distance (0.02 in the code).
    :param hidden: Width of the projection network (1600 in the code, 2 layers).
    :param layers: Layers of the projection network.
    :param dropout: Dropout on the apertures in the intersection (0.2 in the code).
    :param signed_negation: ``False`` (default): the complement of an aperture ``a`` is
        ``pi - |a|``; ``True``: ``pi - a`` (ConE's own code; the same for apertures in
        ``[0, pi]``).
    :param negation: ``"default"`` (the complement, signed as ``signed_negation`` says),
        ``None`` (no negation) or a negation part.
    """

    def __init__(self, num_entities, num_relations, dim, gamma=20.0, center_weight=0.02,
                 hidden=128, layers=1, dropout=0.0, signed_negation=False,
                 intersection="attention", distance=None, negation="default",
                 num_classes=None, extension=True, tables=None):
        self._build("network", num_entities, num_relations, dim, gamma, center_weight, hidden,
                    layers, dropout, signed_negation, "clip", intersection, distance,
                    negation, num_classes, extension, tables)

    def _build(self, relations, num_entities, num_relations, dim, gamma, center_weight, hidden,
               layers, dropout, signed_negation, aperture, intersection, distance, negation,
               num_classes, extension, tables):
        self._assemble(_cone_parts(num_entities, num_relations, dim, float(gamma),
                                   center_weight, hidden, layers, dropout, relations,
                                   signed_negation, aperture, intersection, distance,
                                   negation, tables),
                       dim, gamma, num_classes, extension)


class AConEGeometry(ConEGeometry):
    """AConE as a combination of parts: ConE with the rotation as relation model
    (:class:`ConeRotation`, no projection network).

    :param aperture: ``"wrap"`` (default, the authors' code): the aperture is wrapped to
        ``(-pi, pi]``, so it can be negative; ``"clip"``: kept in ``[0, pi]``.

    The other arguments are those of :class:`ConEGeometry` (without the network).
    """

    def __init__(self, num_entities, num_relations, dim, gamma=20.0, center_weight=0.02,
                 hidden=128, layers=1, dropout=0.0, signed_negation=False, aperture="wrap",
                 intersection="attention", distance=None, negation="default",
                 num_classes=None, extension=True, tables=None):
        if aperture not in ("wrap", "clip"):
            raise ValueError("aperture must be 'wrap' or 'clip'.")
        self._build("rotation", num_entities, num_relations, dim, gamma, center_weight, hidden,
                    layers, dropout, signed_negation, aperture, intersection, distance,
                    negation, num_classes, extension, tables)


@register_method("query2box", trained_on="queries")
def query2box(nb_classes=None, nb_relations=None, nb_individuals=0, embed_dim=32, gamma=24.0,
              center_weight=0.02, activation="relu", intersection="attention",
              clamp_extent=False, distance=None, negation=None, tables=None,
              num_classes=None, extension=True):
    """Query2Box (Ren et al. 2020): point entities, boxes of translations, a learned
    intersection, no negation, the out-of-box plus in-box distance.

    The combination (see :class:`Query2BoxGeometry`): entity model :class:`PointBoxes`;
    relation model ``TranslationBoxes`` (activation ``relu``); intersection
    :class:`BoxAttentionIntersection`; distance :class:`BoxDistance`; measure
    :class:`~mowl.representation.Bounds`; no negation.

    :param nb_relations: Number of roles (the relations are ``isa`` and the roles, each
        with its inverse).
    :param intersection: ``"attention"`` (the paper) or ``"exact"``: the set intersection
        of the boxes, which has no parameters (or a part).
    :param distance: ``"l1"`` (the paper), ``"l2"`` or a part.
    :param negation: ``None`` or a part of boxes.
    :param clamp_extent: ``False`` (default, as in the reference code): the half-width of a
        projected box is the plain sum of the half-widths, which with ``activation="none"`` can
        be negative. ``True``: clamped at 0 (the option of the part; the activations ``relu``
        and ``softplus`` never give a negative one, and for them the two agree).
    :param tables: A module with the tables of Query2Box (the direct implementation) to share
        parameters with (``activation``, ``clamp_extent`` and ``center_weight`` are then read
        from it).
    """
    if tables is not None:
        parts = _query2box_parts(None, None, tables.dim, float(tables.gamma), center_weight,
                                 activation, clamp_extent, intersection, distance, negation,
                                 tables)
        return ComposedQueryGeometry(*parts[:5], negation=parts[5], dim=tables.dim,
                                     gamma=float(tables.gamma), num_classes=num_classes,
                                     extension=extension)
    n_entities, n_rows = _sizes(nb_classes, nb_relations, nb_individuals)
    return Query2BoxGeometry(n_entities, n_rows, embed_dim, gamma, center_weight, activation,
                             clamp_extent, intersection, distance, negation, num_classes,
                             extension)


@register_method("betae", trained_on="queries")
def betae(nb_classes=None, nb_relations=None, nb_individuals=0, embed_dim=32, gamma=24.0,
          hidden=128, layers=1, intersection="attention", distance=None, negation="default",
          tables=None, num_classes=None, extension=True):
    """BetaE (Ren and Leskovec 2020): Beta distributions, a network relation model, an
    attention intersection, the inversion as negation and the KL divergence.

    The combination (see :class:`BetaEGeometry`): entity model :class:`BetaPoints`; relation
    model :class:`BetaNetwork`; intersection :class:`BetaAttentionIntersection`; negation
    :class:`BetaInversion`; distance :class:`BetaDistance`; measure :class:`BetaKL`.

    :param intersection: ``"attention"``, ``"mean"`` or a part.
    :param distance: ``"kl"``, ``"reverse kl"`` or a part.
    :param negation: ``"default"`` (the inversion), ``None`` or a part.
    :param tables: A module with the tables of BetaE (the direct implementation) to share
        parameters with.
    """
    if tables is not None:
        parts = _betae_parts(None, None, tables.dim, float(tables.gamma), hidden, layers,
                             intersection, distance, negation, tables)
        return ComposedQueryGeometry(*parts[:5], negation=parts[5], dim=tables.dim,
                                     gamma=float(tables.gamma), num_classes=num_classes,
                                     extension=extension)
    n_entities, n_rows = _sizes(nb_classes, nb_relations, nb_individuals)
    return BetaEGeometry(n_entities, n_rows, embed_dim, gamma, hidden, layers, intersection,
                         distance, negation, num_classes, extension)


@register_method("cone_qe", trained_on="queries")
def cone_qe(nb_classes=None, nb_relations=None, nb_individuals=0, embed_dim=32, gamma=20.0,
            center_weight=0.02, hidden=128, layers=1, dropout=0.0, relations="network",
            signed_negation=False, aperture="clip", intersection="attention", distance=None,
            negation="default", tables=None, num_classes=None, extension=True):
    """ConE (Zhang et al. 2021): sector cones, a network relation model, the
    SemanticAverage / CardMin intersection, the complement and the angular distance.

    The combination (see :class:`ConEGeometry`): entity model :class:`ConePoints`; relation
    model :class:`ConeNetwork`; intersection :class:`ConeAttentionIntersection`; negation
    :class:`ConeComplement`; distance :class:`ConeDistance`; measure :class:`ConeBounds`.

    :param relations: ``"network"`` (ConE) or ``"rotation"`` (the relation model of AConE;
        :func:`acone` is this call with the authors' readings below).
    :param signed_negation: ``False`` (default): the complement of an aperture :math:`a` is
        :math:`\\pi - |a|`, which never exceeds :math:`\\pi`; ``True``: :math:`\\pi - a`,
        which does for a negative aperture (AConE's code).
    :param aperture: With ``relations="rotation"``: ``"clip"`` (default) keeps the aperture
        in :math:`[0, \\pi]`; ``"wrap"`` wraps it to :math:`(-\\pi, \\pi]`, which gives
        negative apertures (AConE's code).
    :param tables: A module with the tables of ConE (or, with ``relations="rotation"``, of
        AConE; the direct implementation) to share parameters with.
    """
    if relations not in ("network", "rotation"):
        raise ValueError("relations must be 'network' or 'rotation'.")
    if tables is not None:
        dim, g = tables.dim, float(tables.gamma)
        parts = _cone_parts(None, None, dim, g, center_weight, hidden, layers, dropout,
                            relations, signed_negation, aperture, intersection, distance,
                            negation, tables)
        return ComposedQueryGeometry(*parts[:5], negation=parts[5], dim=dim, gamma=g,
                                     num_classes=num_classes, extension=extension)
    n_entities, n_rows = _sizes(nb_classes, nb_relations, nb_individuals)
    if relations == "rotation":
        return AConEGeometry(n_entities, n_rows, embed_dim, gamma, center_weight, hidden,
                             layers, dropout, signed_negation, aperture, intersection,
                             distance, negation, num_classes, extension)
    return ConEGeometry(n_entities, n_rows, embed_dim, gamma, center_weight, hidden, layers,
                        dropout, signed_negation, intersection, distance, negation,
                        num_classes, extension)


@register_method("acone", trained_on="queries")
def acone(nb_classes=None, nb_relations=None, nb_individuals=0, embed_dim=32, gamma=20.0,
          center_weight=0.02, dropout=0.0, signed_negation=True, aperture="wrap",
          intersection="attention", distance=None, negation="default", tables=None,
          num_classes=None, extension=True):
    """AConE (He et al. 2024): ConE with the rotation as relation model (:func:`cone_qe` with
    ``relations="rotation"``) and the authors' two readings, which are their code's and not
    ConE's: the aperture is wrapped to :math:`(-\\pi, \\pi]` (``aperture="wrap"``) and the
    complement is :math:`\\pi - a` (``signed_negation=True``); ``aperture="clip"`` and
    ``signed_negation=False`` give the apertures in :math:`[0, \\pi]` that
    :func:`cone_qe` has by default.

    The combination is that of :func:`cone_qe` with the relation model
    :class:`ConeRotation`."""
    return cone_qe(nb_classes, nb_relations, nb_individuals, embed_dim, gamma, center_weight,
                   dropout=dropout, relations="rotation", signed_negation=signed_negation,
                   aperture=aperture, intersection=intersection, distance=distance,
                   negation=negation, tables=tables, num_classes=num_classes,
                   extension=extension)


# ---------------------------------------------------------------------------------------
# adapters: geometry_of(module) of a module that holds the tables of a method but is not a
# combination of parts (the direct implementation in mowl.baselines)
# ---------------------------------------------------------------------------------------

def _number_of_classes(module):
    return getattr(module, "num_classes", None)


@register_adapter("Query2BoxGeometry")
def _query2box(module):
    return query2box(tables=module, num_classes=_number_of_classes(module))


@register_adapter("BetaEGeometry")
def _betae(module):
    return betae(tables=module, num_classes=_number_of_classes(module))


@register_adapter("ConEGeometry")
def _cone(module):
    return cone_qe(tables=module, num_classes=_number_of_classes(module))


@register_adapter("AConEGeometry")
def _acone(module):
    return acone(tables=module, num_classes=_number_of_classes(module))


def query_geometry(model, extension=None):
    """The :class:`~mowl.representation.Geometry` of a query embedding model, with the
    number of classes of the model, so that :meth:`~Geometry.individuals` and
    :meth:`~Geometry.member` work (``geometry_of(model)`` cannot know it). A model built
    from parts has such a geometry as its module (``extension`` is then its own unless
    given); the module of another implementation is described by an adapter (``extension``
    default ``True``)."""
    from mowl.representation.adapters import ADAPTERS
    if isinstance(model.module, Geometry):
        geometry = model.module
    else:
        geometry = ADAPTERS[type(model.module).__name__](model.module)
    geometry._num_classes = model.num_classes
    if extension is not None:
        geometry.extension = extension
    elif geometry is not model.module:
        geometry.extension = True
    return geometry


__all__ += ["query_geometry"]
