"""Query2Box, BetaE, ConE and AConE from parts, and their common interface.

:class:`ComposedQueryGeometry` is a :class:`~mowl.nn.query.geometry.QueryGeometry` (so a
query embedding model can train it: ``model.module = query2box(None, None, tables=...)``)
and a :class:`~mowl.representation.Geometry`. The methods are functions of parts
(:func:`query2box`, :func:`betae`, :func:`cone_qe`, :func:`acone`); with ``tables=`` they
describe an existing :mod:`mowl.nn.query.geometry` module and share its parameters, which
is what :func:`~mowl.representation.geometry_of` returns for it.

Relation ids follow the query models: relation ``0`` is ``isa``, ``1`` its inverse, role
``k`` is ``2 + 2k`` and its inverse ``3 + 2k``. The interface takes *role* ids:
``exists(k, C)`` projects the region of ``C`` along the inverse of role ``k``, which is
how :mod:`mowl.nn.query.expression` reads :math:`\\exists r.C`.

A named class is "everything below it": ``classes(ids)`` is the projection of the anchor
along ``isa`` inverse (``extension=True``, the default of the query models), or the anchor
itself with ``extension=False``. ``individuals(ids)`` is the anchor of an individual.
"""

import torch as th

from mowl.nn.query.geometry import QueryGeometry, UnsupportedOperator
from mowl.representation.adapters import register_adapter
from mowl.representation.class_models import Classes
from mowl.representation.geometry import Geometry
from mowl.representation.library.query_parts import (
    PointBoxes, BetaPoints, ConePoints, BetaNetwork, ConeNetwork, ConeRotation,
    BoxAttentionIntersection, BoxExactIntersection, BetaAttentionIntersection,
    ConeAttentionIntersection, BetaInversion, ConeComplement, BoxDistance, BetaDistance,
    ConeDistance)
from mowl.representation.library.boxrel_relations import _ACTIVATIONS, TranslationBoxes
from mowl.representation.library.query_regions import (
    BetaKL, ConeBounds, as_state, from_state, lift)
from mowl.representation.measures import Bounds
from mowl.representation.methods import register_method
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["ComposedQueryGeometry", "QueryClasses", "query2box", "betae", "cone_qe", "acone",
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
        super().__init__(len(entities), len(relations), dim, gamma)
        self.entities, self.relations = entities, relations
        self.intersection, self.distance_part = intersection, distance
        self.negation_part, self.measure = negation, measure
        self.supports_negation = negation is not None
        self._num_classes = num_classes
        self.extension = extension
        self.isa_inverse = isa_inverse
        self.region_class = entities.region

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
                "conjunction": "exact" if self.intersection.exact else "approximate",
                "negation": "none" if self.negation_part is None
                else type(self.negation_part).__name__}


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


def query2box(nb_classes=None, nb_relations=None, nb_individuals=0, embed_dim=32, gamma=24.0,
              center_weight=0.02, activation="relu", intersection="attention", tables=None,
              num_classes=None, extension=True):
    """Query2Box (Ren et al. 2020): point entities, boxes of translations, a learned
    intersection, no negation, the out-of-box plus in-box distance.

    :param nb_relations: Number of roles (the relations are ``isa`` and the roles, each
        with its inverse).
    :param intersection: ``"attention"`` (the paper) or ``"exact"``: the set intersection
        of the boxes, which has no parameters.
    :param tables: A :class:`~mowl.nn.query.geometry.Query2BoxGeometry` to share parameters
        with (``activation`` and ``center_weight`` are then read from it).
    """
    if tables is not None:
        for name, function in _ACTIVATIONS.items():
            if tables.activation is function:
                activation = name
        center_weight, dim, gamma = tables.cen, tables.dim, float(tables.gamma)
        entities = PointBoxes(table=tables.entity_embedding)
        relations = TranslationBoxes(vectors=tables.relation_embedding,
                                     offsets=tables.offset_embedding, activation=activation,
                                     inverse_of=role_inverse)
        attention = BoxAttentionIntersection(tables=tables)
    else:
        dim = embed_dim
        n_entities, n_rows = _sizes(nb_classes, nb_relations, nb_individuals)
        r = (gamma + 2.0) / dim
        entities = PointBoxes(n_entities, dim, init_range=r)
        relations = TranslationBoxes(n_rows, dim, activation=activation,
                                     inverse_of=role_inverse, init_range=r)
        attention = BoxAttentionIntersection(dim)
    if intersection == "attention":
        operator = attention
    elif intersection == "exact":
        operator = BoxExactIntersection()
    else:
        raise ValueError("intersection must be 'attention' or 'exact'.")
    return ComposedQueryGeometry(entities, relations, operator, BoxDistance(center_weight),
                                 Bounds(), dim=dim, gamma=gamma, num_classes=num_classes,
                                 extension=extension)


def betae(nb_classes=None, nb_relations=None, nb_individuals=0, embed_dim=32, gamma=24.0,
          hidden=128, layers=1, tables=None, num_classes=None, extension=True):
    """BetaE (Ren and Leskovec 2020): Beta distributions, a network relation model, an
    attention intersection, the inversion as negation and the KL divergence.

    :param tables: A :class:`~mowl.nn.query.geometry.BetaEGeometry` to share parameters
        with.
    """
    if tables is not None:
        dim, gamma = tables.dim, float(tables.gamma)
        entities = BetaPoints(table=tables.entity_embedding)
        relations = BetaNetwork(tables=tables, inverse_of=role_inverse)
        operator = BetaAttentionIntersection(tables=tables)
    else:
        dim = embed_dim
        n_entities, n_rows = _sizes(nb_classes, nb_relations, nb_individuals)
        r = (gamma + 2.0) / dim
        entities = BetaPoints(n_entities, dim, init_range=r)
        relations = BetaNetwork(n_rows, dim, hidden, layers, inverse_of=role_inverse,
                                init_range=r)
        operator = BetaAttentionIntersection(dim)
    return ComposedQueryGeometry(entities, relations, operator, BetaDistance(), BetaKL(),
                                 negation=BetaInversion(), dim=dim, gamma=gamma,
                                 num_classes=num_classes, extension=extension)


@register_method("cone_qe", trained_on="queries")
def cone_qe(nb_classes=None, nb_relations=None, nb_individuals=0, embed_dim=32, gamma=20.0,
            center_weight=0.02, hidden=128, layers=1, dropout=0.0, relations="network",
            tables=None, num_classes=None, extension=True):
    """ConE (Zhang et al. 2021): sector cones, a network relation model, the
    SemanticAverage / CardMin intersection, the complement and the angular distance.

    :param relations: ``"network"`` (ConE) or ``"rotation"`` (the relation model of AConE;
        :func:`acone` is this call).
    :param tables: A :class:`~mowl.nn.query.geometry.ConEGeometry` (or, with
        ``relations="rotation"``, an :class:`~mowl.nn.query.geometry.AConEGeometry`) to
        share parameters with.
    """
    if relations not in ("network", "rotation"):
        raise ValueError("relations must be 'network' or 'rotation'.")
    if tables is not None:
        if relations == "network" and not hasattr(tables, "proj_in"):
            raise ValueError("These tables have no projection network; use "
                             "relations='rotation'.")
        dim, gamma, center_weight = tables.dim, float(tables.gamma), tables.cen
        angle_range = tables.embedding_range
        entities = ConePoints(table=tables.entity_embedding, angle_range=angle_range)
        rel = (ConeNetwork(tables=tables, inverse_of=role_inverse) if relations == "network"
               else ConeRotation(tables=tables, inverse_of=role_inverse,
                                 aperture="wrap"))
        operator = ConeAttentionIntersection(tables=tables)
        distance = ConeDistance(center_weight, modulus=tables.modulus)
    else:
        dim = embed_dim
        n_entities, n_rows = _sizes(nb_classes, nb_relations, nb_individuals)
        angle_range = (gamma + 2.0) / dim
        entities = ConePoints(n_entities, dim, angle_range=angle_range)
        rel = (ConeNetwork(n_rows, dim, hidden, layers, angle_range=angle_range,
                           inverse_of=role_inverse) if relations == "network"
               else ConeRotation(n_rows, dim, angle_range=angle_range,
                                 inverse_of=role_inverse, aperture="wrap"))
        operator = ConeAttentionIntersection(dim, dropout)
        distance = ConeDistance(center_weight, angle_range=angle_range)
    return ComposedQueryGeometry(entities, rel, operator, distance, ConeBounds(),
                                 negation=ConeComplement(signed=True), dim=dim, gamma=gamma,
                                 num_classes=num_classes, extension=extension)


@register_method("acone", trained_on="queries")
def acone(nb_classes=None, nb_relations=None, nb_individuals=0, embed_dim=32, gamma=20.0,
          center_weight=0.02, dropout=0.0, tables=None, num_classes=None, extension=True):
    """AConE (He et al. 2024): ConE with the rotation as relation model, nothing else
    changed (:func:`cone_qe` with ``relations="rotation"``)."""
    return cone_qe(nb_classes, nb_relations, nb_individuals, embed_dim, gamma, center_weight,
                   dropout=dropout, relations="rotation", tables=tables,
                   num_classes=num_classes, extension=extension)


register_method("query2box", trained_on="queries")(query2box)
register_method("betae", trained_on="queries")(betae)


# ---------------------------------------------------------------------------------------
# adapters: geometry_of(module)
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


def query_geometry(model, extension=True):
    """The :class:`~mowl.representation.Geometry` of a query embedding model, with the
    number of classes of the model, so that :meth:`~Geometry.individuals` and
    :meth:`~Geometry.member` work (``geometry_of(model)`` cannot know it)."""
    from mowl.representation.adapters import ADAPTERS
    adapter = ADAPTERS[type(model.module).__name__]
    geometry = adapter(model.module)
    geometry._num_classes = model.num_classes
    geometry.extension = extension
    return geometry


__all__ += ["query_geometry"]
