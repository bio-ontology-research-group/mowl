"""Knowledge graph models on a graph projection (``graph_kge``, ``boxe``, ``expressive``,
``reshuffle``) in the common interface.

In these methods classes and individuals are *entities* (points), the projection decides
how an axiom becomes a triple, and a model is a split that PyKEEN already makes: entity and
relation **representations**, and an **interaction** function that scores a triple from
them. This module exposes that split instead of inventing another one.

Two things are provided.

1. :class:`PyKEENGeometry` is the :class:`~mowl.representation.Geometry` of a trained
   :class:`~mowl.models.graph_kge.graph_pykeen_model.GraphPlusPyKEENModel` (and so of
   ``BoxE``, ``ExpressivE``, ``Reshuffle`` and any other PyKEEN model). It reads the
   model's own tables; its costs are the model's own ``axiom_costs`` (minus the score of
   the triple an axiom is projected to). Only what is a triple is defined: ``C subclassof
   D`` (``inside``), ``C r D`` for :math:`C \\sqsubseteq \\exists r.D` (``inside`` of an
   :meth:`~PyKEENGeometry.exists` region, where the filler is a named class), class
   assertions (``member``, the ``type`` edge) and role assertions. Conjunction, negation,
   disjunction and :math:`\\exists r.C \\sqsubseteq D` are not defined (the triple score
   has no region for them); ``apart`` is not defined.
2. Parts, for the methods that *are* combinations: a class model of points
   (:class:`PointClasses`, optionally with a bump), relation models of
   :class:`BoxERelations` (a head box and a tail box per relation) and
   :class:`ExpressivERelations` (a parallelogram over pairs) and the measure
   :class:`TripleScore`. ``boxe``, ``expressive`` and ``graph_kge`` (point classes with any
   relation model, TransE by default) are built from them by the registered methods. A
   relation here is a set of *pairs*, so there is no image or preimage region of the class
   model: :meth:`~BoxERelations.preimage` is a symbolic region
   (:class:`Related`) that only the triple score can read. They also work on the centres
   of balls and boxes, which is how they are combined with parts of other methods.

``reshuffle`` is not a combination of parts: the entity vectors are computed by a graph
network over the whole projection, so they are not a table of independent rows, and the
relation is a matrix on that encoding. It has the geometry of point 1 (PyKEEN's
representations are then the encoded ones), and no parts.
"""

from dataclasses import dataclass

import torch as th
import torch.nn as nn

from mowl.representation.adapters import register_adapter
from mowl.representation.class_models import Classes
from mowl.representation.geometry import ComposedGeometry, Geometry
from mowl.representation.methods import (register_class_model, register_measure,
                                         register_method, register_relation_model)
from mowl.representation.measures import RegionMeasure
from mowl.representation.objectives import Objective
from mowl.representation.regions import Ball, Box, ConstructorUnavailable, Region
from mowl.representation.relation_models import Relations

__all__ = ["Point", "PointClasses", "Related", "TripleScore", "BoxERelations",
           "ExpressivERelations", "TripleGeometry", "PyKEENGeometry", "Entities",
           "ExistsEntities", "PyKEENTable", "graph_kge", "boxe", "expressive"]


# -- regions ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Point(Region):
    """A point (an entity of a knowledge graph model), optionally with a *bump* vector
    (BoxE).

    :param centre: ``(B, d)``.
    :param bump: ``(B, d)`` or ``None``.
    """

    centre: th.Tensor
    bump: th.Tensor = None

    def translate(self, vector):
        return Point(self.centre + vector, self.bump)


@dataclass(frozen=True)
class Related(Region):
    """The symbolic region :math:`\\exists r.C` of a relation that is a set of pairs: the
    relations ``ids`` and the region ``filler`` of the fillers. Only
    :class:`TripleScore` reads it (a point is *in* it when the triple scores well).
    """

    relations: object
    ids: th.Tensor
    filler: Region

    def __len__(self):
        return len(self.ids)


def _centre(region):
    return region.centre


def _bump(region):
    bump = getattr(region, "bump", None)
    return th.zeros_like(region.centre) if bump is None else bump


# -- class model -----------------------------------------------------------------------------
class PyKEENTable(nn.Module):
    """A PyKEEN representation used like an ``nn.Embedding`` (``table(ids)``); the
    parameters are the representation's own, not a copy."""

    def __init__(self, representation):
        super().__init__()
        self.representation = representation

    @property
    def num_embeddings(self):
        return self.representation.max_id

    @property
    def embedding_dim(self):
        return int(self.representation.shape[0]) if self.representation.shape else 1

    def forward(self, ids):
        return self.representation(indices=ids)


def _table(size, dim, scale=1.0):
    table = nn.Embedding(size, dim)
    with th.no_grad():
        nn.init.uniform_(table.weight, a=-scale, b=scale)
    return table


class PointClasses(Classes):
    """A class (or individual) is a point, as in a knowledge graph embedding.

    :param bump: Whether a point has a bump vector as well (BoxE).
    :param centres, bumps: Existing tables to use instead of new ones.
    """

    region = Point
    part = "point"
    exact_conjunction = False

    def __init__(self, size=None, embed_dim=None, bump=False, centres=None, bumps=None):
        super().__init__()
        self.centres = centres if centres is not None else _table(size, embed_dim)
        if bumps is None and bump:
            bumps = _table(len(self), self.embed_dim)
        self.bumps = bumps

    def forward(self, ids):
        return Point(self.centres(ids), None if self.bumps is None else self.bumps(ids))


register_class_model("point")(PointClasses)
register_class_model("point with bump")(lambda size, dim: PointClasses(size, dim, bump=True))


# -- relation models: a relation is a set of pairs ------------------------------------------
class _PairRelations(Relations):
    """Base of relation models that are sets of pairs of points.

    The last row of the tables is the relation ``subclass``, used by :class:`TripleScore`
    for :math:`C \\sqsubseteq D` and :math:`C(a)` (the projection's ``subclassof`` edge).
    """

    def __len__(self):
        return self.vectors.num_embeddings - self.reserved

    #: Rows reserved at the end of the tables (the ``subclass`` relation).
    reserved = 1

    @property
    def subclass(self):
        return len(self)

    def image(self, ids, region):
        raise ConstructorUnavailable(
            f"{type(self).__name__}: a relation is a set of pairs, the image of a region "
            f"is not a region of the class model.")

    def preimage(self, ids, region):
        return Related(self, ids, region)

    def cost(self, ids, head, tail):
        """Cost ``(B, 1)`` of ``r(head, tail)`` for the relations ``ids``: minus the score
        of the triple. Zero or small when the pair is in the region of the relation."""
        raise NotImplementedError


class BoxERelations(_PairRelations):
    """BoxE (Abboud et al. 2020): a relation is a *head box* and a *tail box*;
    :math:`r(h, t)` holds when :math:`h` bumped by :math:`t` lies in the head box and
    :math:`t` bumped by :math:`h` in the tail box. The score is that of PyKEEN's
    :class:`~pykeen.nn.modules.BoxEInteraction` (whose functions compute the boxes and the
    distance), so costs equal those of a PyKEEN BoxE with the same tables.

    Over classes without a bump (balls, boxes) only the centres are used and the bump is
    zero; that is an adaptation, BoxE's entities have a bump.

    :param tables: Six existing tables (head base, head delta, head size, tail base, tail
        delta, tail size), e.g. :class:`PyKEENTable` of a PyKEEN BoxE; a relation index
        ``i`` of the geometry is row ``i`` of every table. Then no ``subclass`` row is
        reserved (``reserved = 0``) and ``subclass_id`` must be given.
    :param tanh_map, p, power_norm: As in PyKEEN's interaction.
    """

    part = "head and tail boxes"

    def __init__(self, size=None, embed_dim=None, tables=None, tanh_map=True, p=2,
                 power_norm=False, subclass_id=None):
        super().__init__()
        from pykeen.nn.modules import BoxEInteraction
        if tables is None:
            n = size + 1
            tables = [_table(n, embed_dim), _table(n, embed_dim), _table(n, 1),
                      _table(n, embed_dim), _table(n, embed_dim), _table(n, 1)]
            self.reserved = 1
        else:
            self.reserved = 0
            self._subclass_id = subclass_id
        (self.head_base, self.head_delta, self.head_size,
         self.tail_base, self.tail_delta, self.tail_size) = tables
        self.vectors = self.head_base
        self.interaction = BoxEInteraction(tanh_map=tanh_map, p=p, power_norm=power_norm)

    @property
    def subclass(self):
        if self.reserved == 0:
            if self._subclass_id is None:
                raise ConstructorUnavailable("No subclass relation was given for shared "
                                             "tables.")
            return self._subclass_id
        return len(self)

    def cost(self, ids, head, tail):
        r = tuple(t(ids) for t in (self.head_base, self.head_delta, self.head_size,
                                   self.tail_base, self.tail_delta, self.tail_size))
        score = self.interaction(h=(_centre(head), _bump(head)), r=r,
                                 t=(_centre(tail), _bump(tail)))
        return -score.reshape(-1, 1)


class ExpressivERelations(_PairRelations):
    """ExpressivE (Pavlovic and Sallinger 2023): a relation is a hyper-parallelogram in the
    space of pairs :math:`\\mathbb{R}^{2d}`, given by slopes, centres and widths for the
    head and the tail position (a raw vector of size ``6 d``). The score is that of
    :class:`mowl.nn.graph.expressive.ExpressivEInteraction`, so costs equal those of an
    ExpressivE with the same table. Only centres of the classes are used.

    :param table: An existing table of raw relation vectors (``6 d`` wide).
    :param p, tanh_map, min_denom: As in the interaction.
    """

    part = "region over pairs"

    def __init__(self, size=None, embed_dim=None, table=None, p=2, tanh_map=True,
                 min_denom=0.5, subclass_id=None):
        super().__init__()
        from mowl.nn.graph.expressive.module import ExpressivEInteraction
        if table is None:
            table = _table(size + 1, 6 * embed_dim, 0.5)
            self.reserved = 1
        else:
            self.reserved = 0
            self._subclass_id = subclass_id
        self.vectors = table
        self.interaction = ExpressivEInteraction(p=p, tanh_map=tanh_map, min_denom=min_denom)

    @property
    def subclass(self):
        if self.reserved == 0:
            if self._subclass_id is None:
                raise ConstructorUnavailable("No subclass relation was given for shared "
                                             "tables.")
            return self._subclass_id
        return len(self)

    def cost(self, ids, head, tail):
        score = self.interaction(_centre(head), self.vectors(ids), _centre(tail))
        return -score.reshape(-1, 1)


register_relation_model("head and tail boxes")(BoxERelations)
register_relation_model("region over pairs")(ExpressivERelations)


# -- measure ---------------------------------------------------------------------------------
class TripleScore(RegionMeasure):
    """The cost of a triple, as a measure: ``inside(a, exists r.C)`` is the cost of
    ``(a, r, C)``, ``inside(a, b)`` that of ``(a, subclass, b)``. A margin does not enter
    (the triple score is not a containment); ``apart`` is the negative triple of a margin
    ranking loss, ``max(0, margin - cost)``.

    Relations without a ``cost`` of their own (a :class:`~mowl.representation.Translation`
    makes TransE) are scored by the distance of the centre of the filler to the image of
    the centre of the head.

    :param relations: The relation model that gives the ``subclass`` relation; set by
        :class:`TripleGeometry`.
    """

    accepts = (Point, Ball, Box)
    part = "triple score"
    exact_intersection = False

    def __init__(self, relations=None):
        self.relations = relations

    def bind(self, relations):
        self.relations = relations
        return self

    def _cost(self, a, b):
        if isinstance(a, Related):
            raise ConstructorUnavailable("A triple score has no region for exists r.C on "
                                         "the left of an inclusion.")
        if isinstance(b, Related):
            relations, ids, filler = b.relations, b.ids, b.filler
        else:
            relations, ids, filler = self.relations, None, b
            if relations is None:
                raise ConstructorUnavailable(
                    "TripleScore needs a relation model for the subclass relation: build "
                    "the geometry with TripleGeometry or the methods graph_kge/boxe/"
                    "expressive.")
            ids = th.full((len(a),), relations.subclass, dtype=th.long,
                          device=a.centre.device)
        if hasattr(relations, "cost"):
            return relations.cost(ids, a, filler)
        moved = relations.image(ids, Point(_centre(a)))
        return th.linalg.norm(moved.centre - _centre(filler), dim=1, keepdim=True)

    def inside(self, a, b, margin=0.0):
        return self._cost(a, b)

    def apart(self, a, b, margin=0.0):
        return th.relu(margin - self._cost(a, b))

    def meets(self, a, b, margin=0.0):
        raise ConstructorUnavailable("TripleScore does not define overlap.")

    def empty(self, a):
        raise ConstructorUnavailable("TripleScore does not define emptiness.")

    nonempty = empty

    def intersection_inside(self, a, b, c, margin=0.0):
        raise ConstructorUnavailable("TripleScore does not define conjunction.")

    intersection_apart = intersection_nonempty = intersection_inside


register_measure("triple score", default_for=(Point,))(TripleScore)


class TripleGeometry(ComposedGeometry):
    """A :class:`~mowl.representation.ComposedGeometry` of points and a triple score. The
    forms it implements are the ones a projection turns into triples: ``gci0``, ``gci2``,
    ``class_assertion`` and ``object_property_assertion``; the others raise
    :class:`~mowl.representation.ConstructorUnavailable`."""

    def __init__(self, classes, relations, measure, objective=None, individuals=None,
                 regularizers=()):
        super().__init__(classes, relations, measure, objective, individuals, regularizers)
        if isinstance(measure, TripleScore):
            measure.bind(relations)
        self.neg_capable_gcis = frozenset({"gci0", "gci2", "object_property_assertion"}) \
            if self.objective.negatives == "all" else frozenset(
                {"gci2", "object_property_assertion"})
        self.scored_forms = ("gci0", "gci2", "class_assertion", "object_property_assertion")

    def member(self, individual_ids, region):
        return self.inside(self.individuals(individual_ids), region)

    @property
    def parts(self):
        out = super().parts
        out["measure"] = "triple score"
        out["conjunction"] = "none"
        return out


_DEFAULT_RELATIONS = {"graph_kge": "translation", "boxe": "head and tail boxes",
                      "expressive": "region over pairs"}


def _build(key, nb_classes, nb_relations, nb_individuals, embed_dim, margin, classes,
           relations, objective, tables, relation_options):
    from mowl.representation import methods as M
    if tables is not None:
        raise TypeError(f"{key}: tables must be a PyKEEN model; use boxe(...) / "
                        f"expressive(...) with tables, or geometry_of(model).")
    class_factory = M.CLASS_MODELS[classes] if isinstance(classes, str) else classes
    if classes == "point" and key == "boxe":
        class_factory = M.CLASS_MODELS["point with bump"]
    relation_factory = M.RELATION_MODELS[relations] if isinstance(relations, str) \
        else relations
    class_model = class_factory(nb_classes, embed_dim)
    relation_model = relation_factory(nb_relations, embed_dim, **relation_options) \
        if relation_options else relation_factory(nb_relations, embed_dim)
    if relation_model is not None and not hasattr(relation_model, "subclass"):
        # a relation model without a reserved subclass row (translation): add one
        relation_model = _WithSubclass(relation_model, embed_dim)
    individual_model = class_factory(nb_individuals, embed_dim) if nb_individuals else None
    if objective is None:
        objective = Objective(margin, negatives="gci2")
    return TripleGeometry(class_model, relation_model, TripleScore(), objective,
                          individuals=individual_model)


class _WithSubclass(Relations):
    """A relation model plus one more relation, ``subclass``, of the same kind."""

    def __init__(self, relations, embed_dim):
        super().__init__()
        self.inner = relations
        extra = _table(1, embed_dim)
        extra.weight.data /= th.linalg.norm(extra.weight.data, dim=1, keepdim=True)
        self.extra = extra
        self.part = relations.part

    @property
    def vectors(self):
        return self.inner.vectors

    def __len__(self):
        return len(self.inner)

    @property
    def subclass(self):
        return len(self.inner)

    def _all(self):
        return th.cat([self.inner.vectors.weight, self.extra.weight])

    def image(self, ids, region):
        return region.translate(self._all()[ids])

    def preimage(self, ids, region):
        return Related(self, ids, region)


@register_method("graph_kge")
def graph_kge(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=1.0,
              classes="point", relations="translation", objective=None, tables=None,
              **relation_options):
    """Any knowledge graph model on a projection, as parts: classes and individuals are
    entities, a relation model says what a relation is, the measure is the triple score.
    The default (point classes, translation) is TransE; ``relations="head and tail
    boxes"`` is BoxE, ``"region over pairs"`` is ExpressivE, and ``classes="ball"`` or
    ``"box"`` use the centres of those regions as the entities.

    A PyKEEN model that is not one of these is not rebuilt: :func:`geometry_of` gives the
    geometry of the trained model, with PyKEEN's own interaction.
    """
    return _build("graph_kge", nb_classes, nb_relations, nb_individuals, embed_dim, margin,
                  classes, relations, objective, tables, relation_options)


@register_method("boxe")
def boxe(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=1.0,
         classes="point", objective=None, tables=None, **relation_options):
    """BoxE from parts: points with a bump, head and tail boxes, the triple score.

    :param classes: ``"point"`` (the default, with a bump) or any class model name
        (``"ball"``, ``"box"``): then only the centres are used.
    :param tables: A trained PyKEEN :class:`~pykeen.models.BoxE` whose entity and relation
        tables are shared; classes and relations are then indexed by the PyKEEN ids
        (``nb_*`` are ignored) and the ``subclass`` relation is ``subclass_id``.
    """
    if tables is not None:
        subclass_id = relation_options.pop("subclass_id", None)
        reps = tables.relation_representations
        classes_model = PointClasses(
            centres=PyKEENTable(tables.entity_representations[0]),
            bumps=PyKEENTable(tables.entity_representations[1]))
        relations_model = BoxERelations(
            tables=[PyKEENTable(r) for r in reps], subclass_id=subclass_id,
            tanh_map=tables.interaction.tanh_map, p=tables.interaction.p,
            power_norm=tables.interaction.power_norm)
        return TripleGeometry(classes_model, relations_model, TripleScore(),
                              objective or Objective(margin, negatives="gci2"),
                              individuals=classes_model)
    return _build("boxe", nb_classes, nb_relations, nb_individuals, embed_dim, margin,
                  classes, "head and tail boxes", objective, None, relation_options)


@register_method("expressive")
def expressive(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=1.0,
               classes="point", objective=None, tables=None, **relation_options):
    """ExpressivE from parts: points, parallelograms over pairs, the triple score. See
    :func:`boxe` for ``classes`` and ``tables`` (a trained PyKEEN ExpressivE)."""
    if tables is not None:
        subclass_id = relation_options.pop("subclass_id", None)
        classes_model = PointClasses(centres=PyKEENTable(tables.entity_representations[0]))
        inter = tables.interaction
        relations_model = ExpressivERelations(
            table=PyKEENTable(tables.relation_representations[0]), p=inter.p,
            tanh_map=inter.tanh_map, min_denom=inter.min_denom, subclass_id=subclass_id)
        return TripleGeometry(classes_model, relations_model, TripleScore(),
                              objective or Objective(margin, negatives="gci2"),
                              individuals=classes_model)
    return _build("expressive", nb_classes, nb_relations, nb_individuals, embed_dim, margin,
                  classes, "region over pairs", objective, None, relation_options)


# -- the geometry of a trained PyKEEN model --------------------------------------------------
class Entities(Region):
    """Entities of the projected graph: ``nodes`` are their ids in the PyKEEN model
    (``-1`` for a class or individual that is not a node of the graph).

    :ivar vectors: The entity representations of the model (the tables, so gradients flow;
        for a model that encodes the graph, such as RESHUFFLE, the input features).
    """

    def __init__(self, nodes, geometry):
        self.nodes = nodes
        self._geometry = geometry

    def __len__(self):
        return len(self.nodes)

    @property
    def vectors(self):
        rep = self._geometry.kge.entity_representations[0]
        return rep(indices=self.nodes.clamp(min=0).to(self._geometry.device))


class ExistsEntities(Region):
    """:math:`\\exists r.C` for named ``C`` in a projected graph: the edge label ``r`` of
    the model and the filler."""

    def __init__(self, relations, filler):
        self.relations = relations
        self.filler = filler

    def __len__(self):
        return len(self.relations)


class PyKEENGeometry(Geometry):
    """The geometry of a trained ``GraphPlusPyKEENModel``
    (:class:`mowl.models.graph_kge.graph_pykeen_model.GraphPlusPyKEENModel`).

    Entities are the nodes of the projection; their regions are points
    (:class:`Entities`). The score is PyKEEN's, in its two steps: ``kge._get_representations``
    gives the representations of head, relation and tail, and ``kge.interaction`` scores
    them (:meth:`triple_cost`); no parameter is copied. Costs are minus the score, as in
    ``axiom_costs``, ``inf`` for a class, individual or role that is not in the graph.

    ``inside(C, D)`` is the triple ``(C, subclassof, D)``, ``inside(C, exists(r, D))`` the
    triple ``(C, r, D)``, ``member(a, C)`` the triple ``(a, type, C)`` (the type edge of the
    projector; ``inf`` if the projection has none) and ``member(a, exists(r, b))`` the
    triple ``(a, r, b)``. The costs are *scores of triples*, not zero for entailed axioms.
    """

    def __init__(self, model):
        self.model = model
        self.kge = model.kge_method
        self.device = next(self.kge.parameters()).device
        factory = model.triples_factory
        nodes, relations = factory.entity_to_id, factory.relation_to_id
        self._inverse = bool(getattr(model, "create_inverse_triples", False))
        self._classes = self._map(model.class_index_dict, nodes)
        self._individuals = self._map(model.individual_index_dict, nodes)
        self._roles = self._map(model.object_property_index_dict, relations)
        factor = 2 if self._inverse else 1   # PyKEEN: relation r is 2r, its inverse 2r + 1
        self._subclass = self._scale(relations.get(model.SUBCLASS_RELATION, -1), factor)
        self._type = self._scale(relations.get("http://type", -1), factor)

    @staticmethod
    def _scale(relation, factor):
        return relation * factor if relation >= 0 else relation

    @staticmethod
    def _map(index_dict, to_id):
        out = th.full((max(index_dict.values(), default=-1) + 1,), -1, dtype=th.long)
        for iri, i in index_dict.items():
            out[i] = to_id.get(iri, -1)
        return out

    # -- the PyKEEN split -------------------------------------------------------------------
    @property
    def interaction(self):
        """PyKEEN's interaction function (scores representations)."""
        return self.kge.interaction

    def representations(self, heads, relations, tails):
        """PyKEEN's ``(head, relation, tail)`` representations of graph ids."""
        return self.kge._get_representations(h=heads, r=relations, t=tails, mode=None)

    def triple_cost(self, heads, relations, tails):
        """Minus the score of the triples (graph ids, ``(B,)`` each) as ``(B, 1)``; ``inf``
        where an id is ``-1``."""
        heads, relations, tails = (x.to(self.device) for x in (heads, relations, tails))
        known = (heads >= 0) & (relations >= 0) & (tails >= 0)
        out = th.full((len(heads), 1), float("inf"), device=self.device)
        if known.any():
            h, r, t = self.representations(heads[known], relations[known], tails[known])
            score = self.interaction.score_hrt(h=h, r=r, t=t)
            out[known] = -score.reshape(-1, 1).to(out.dtype)
        return out

    def _relation_ids(self, relation_ids):
        r = self._roles[relation_ids.cpu()]
        if self._inverse:
            r = th.where(r >= 0, 2 * r, r)
        return r

    # -- Geometry -----------------------------------------------------------------------------
    def classes(self, ids):
        return Entities(self._classes[ids.cpu()], self)

    def individuals(self, ids):
        return Entities(self._individuals[ids.cpu()], self)

    def exists(self, relation_ids, region):
        if not isinstance(region, Entities):
            raise ConstructorUnavailable("The filler of an existential restriction must be "
                                         "a named class or an individual.")
        return ExistsEntities(self._relation_ids(relation_ids), region)

    def _relation_to(self, a, b):
        if isinstance(b, ExistsEntities):
            return a.nodes, b.relations, b.filler.nodes
        sub = th.full_like(a.nodes, self._subclass)
        return a.nodes, sub, b.nodes

    def inside(self, a, b):
        if isinstance(a, ExistsEntities):
            raise ConstructorUnavailable("exists r.C on the left of an inclusion has no "
                                         "triple.")
        return self.triple_cost(*self._relation_to(a, b))

    def member(self, individual_ids, region):
        a = self.individuals(individual_ids)
        if isinstance(region, ExistsEntities):
            return self.inside(a, region)
        return self.triple_cost(a.nodes, th.full_like(a.nodes, self._type), region.nodes)

    def apart(self, a, b):
        raise ConstructorUnavailable("A knowledge graph model has no disjointness.")

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def num_individuals(self):
        return len(self._individuals)

    @property
    def parts(self):
        return {"classes": "point", "relations": getattr(self, "relation_part", "none"),
                "individuals": "point", "measure": "triple score", "conjunction": "none"}

    @property
    def relation_part(self):
        name = type(self.kge).__name__
        return {"BoxE": "head and tail boxes", "ExpressivE": "region over pairs",
                "Reshuffle": "matrix"}.get(name, "model of PyKEEN")

    def as_parts(self):
        """The same model as a :class:`TripleGeometry` of parts that share its tables
        (``BoxE`` and ``ExpressivE`` only), indexed by *graph* ids: classes are the nodes,
        relations the relation ids of the graph."""
        name = type(self.kge).__name__
        if name == "BoxE":
            return boxe(0, 0, tables=self.kge, subclass_id=self._subclass)
        if name == "ExpressivE":
            return expressive(0, 0, tables=self.kge, subclass_id=self._subclass)
        raise NotImplementedError(f"{name} is not a combination of the parts of this "
                                  f"module.")


register_adapter("GraphPlusPyKEENModel", "RegionKGEModel", "BoxE", "ExpressivE",
                 "Reshuffle")(PyKEENGeometry)
