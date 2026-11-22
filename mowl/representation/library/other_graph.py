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

import contextlib
import functools
import inspect
import math
import warnings
from dataclasses import dataclass

import torch as th
import torch.nn as nn

from mowl.representation.adapters import register_adapter
from mowl.representation.geometry import ComposedGeometry, Geometry
from mowl.representation.methods import (register_measure,
                                         register_method, register_relation_model)
from mowl.representation.measures import RegionMeasure
from mowl.representation.objectives import Objective
from mowl.representation.regions import Ball, Box, ConstructorUnavailable, Region
from mowl.representation.class_models import Classes
from mowl.representation.library.shapes_classes import Points
from mowl.representation.library.shapes_regions import Point
from mowl.representation.relation_models import Relations

__all__ = ["Point", "PointClasses", "Related", "TripleScore", "BoxERelations",
           "ExpressivERelations", "triple_geometry", "TripleGeometry", "PyKEENGeometry",
           "Entities", "ExistsEntities", "PyKEENTable", "graph_kge", "gda_projection", "boxe",
           "expressive", "reshuffle", "TripleLoss", "SelfAdversarial", "MarginRanking",
           "PyKEENLoss", "TRIPLE_LOSSES", "triple_loss", "UniformCorruption",
           "ReshuffleRelations", "GraphEncodedPoints"]


# -- regions ---------------------------------------------------------------------------------
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


def PointClasses(size=None, embed_dim=None, bump=False, centres=None, bumps=None):
    """A class (or individual) is a point, as in a knowledge graph embedding: the class model
    :class:`~mowl.representation.library.shapes_classes.Points` of the family ``"point"``
    with the uniform initialisation of the knowledge graph models, and with a bump vector
    for BoxE (``bump=True``)."""
    return Points(size, embed_dim, family="point", centres=centres, init="uniform",
                  bump=bump, bumps=bumps)


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
    :math:`t` bumped by :math:`h` in the tail box.

    This is the other of the two head-and-tail-box relation models of the library, and the two
    differ in what they are, not in how they are written: the boxes of
    :class:`~mowl.representation.library.multibox_parts.HeadTailRelations` (Box2EL,
    MultiBoxEL; key ``"head and tail boxes"``) are given by centre and half-width and the
    relation is a *region* statement (a class is related if its region lies in the head box
    moved by the bump of the filler: containment of regions), while here the boxes are given
    by the base, delta and size of BoxE and the cost is a *distance score* of two points
    (PyKEEN's interaction), which is not a containment and cannot be written with a region
    measure. Used by: ``boxe``. The score is that of PyKEEN's
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
        self._subclass_id = subclass_id
        if tables is None:
            # a graph names its ``subclass`` relation (``subclass_id``, a row among the
            # ``size`` relations); without a graph one more row is reserved for it
            self.reserved = 1 if subclass_id is None else 0
            n = size + self.reserved
            tables = [_table(n, embed_dim), _table(n, embed_dim), _table(n, 1),
                      _table(n, embed_dim), _table(n, embed_dim), _table(n, 1)]
        else:
            self.reserved = 0
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
    :class:`mowl.representation.library.graph_interactions.ExpressivEInteraction`, so costs equal those of an
    ExpressivE with the same table. Only centres of the classes are used.

    :param table: An existing table of raw relation vectors (``6 d`` wide).
    :param p, tanh_map, min_denom: As in the interaction.

    Part (vocabulary value): ``region over pairs``. Used by: ``expressive``.
    """

    part = "region over pairs"

    def __init__(self, size=None, embed_dim=None, table=None, p=2, tanh_map=True,
                 min_denom=0.5, subclass_id=None):
        super().__init__()
        from mowl.representation.library.graph_interactions import ExpressivEInteraction
        self._subclass_id = subclass_id
        if table is None:
            self.reserved = 1 if subclass_id is None else 0
            table = _table(size + self.reserved, 6 * embed_dim, 0.5)
        else:
            self.reserved = 0
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


register_relation_model("head and tail boxes (distance score)")(BoxERelations)
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
        :func:`triple_geometry`.

    Part (vocabulary value): ``triple score``. Used by: ``boxe``, ``expressive``, ``graph_kge``.
    """

    accepts = (Point, Ball, Box, Related)
    part = "triple score"
    reading = "score"
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
                    "the geometry with triple_geometry or the methods graph_kge/boxe/"
                    "expressive.")
            if not hasattr(relations, "subclass"):
                raise ConstructorUnavailable(
                    f"{type(relations).__name__} has no subclass relation, so a triple score "
                    f"cannot read an inclusion of classes.")
            ids = th.full((len(a),), relations.subclass, dtype=th.long,
                          device=a.centre.device)
        return self.triple(relations, ids, a, filler)

    @staticmethod
    def triple(relations, ids, head, tail):
        """The cost ``(B, 1)`` of the triples ``(head, ids, tail)``: the relation model's own
        ``cost``, or the distance of the centre of the tail to the image of the centre of the
        head (a translation: TransE)."""
        if hasattr(relations, "cost"):
            return relations.cost(ids, head, tail)
        moved = relations.image(ids, Point(_centre(head)))
        return th.linalg.norm(moved.centre - _centre(tail), dim=1, keepdim=True)

    def inside(self, a, b, margin=0.0):
        return self._cost(a, b)

    def apart(self, a, b, margin=0.0):
        return th.relu(margin - self._cost(a, b))

    def meets(self, a, b, margin=0.0):
        raise ConstructorUnavailable("TripleScore does not define overlap.")

    def empty(self, a, margin=0.0):
        raise ConstructorUnavailable("TripleScore does not define emptiness.")

    nonempty = empty

    def intersection_inside(self, a, b, c, margin=0.0):
        raise ConstructorUnavailable("TripleScore does not define conjunction.")

    intersection_apart = intersection_nonempty = intersection_inside


register_measure("triple score")(TripleScore)


# -- objective: the loss on scores of triples -------------------------------------------------
def _reduce(values, reduction):
    return values.sum() if reduction == "sum" else values.mean()


class TripleLoss:
    """The objective of a knowledge graph model: a loss on the scores of a batch of positive
    triples ``(B, 1)`` and of their corrupted triples ``(B, K)`` (a score is minus the cost
    of the triple). Subclasses are the losses of the knowledge graph embedding literature;
    they are written out here, not taken from PyKEEN, so that the combination does not depend
    on a library's loss class (:class:`PyKEENLoss` adapts one).

    :param reduction: ``"mean"`` or ``"sum"`` over the batch.

    Part (vocabulary value): ``triple loss``.
    """

    name = ""

    def __init__(self, reduction="mean"):
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'.")
        self.reduction = reduction

    def __call__(self, positive_scores, negative_scores):
        raise NotImplementedError

    def describe(self):
        return {"loss": self.name, "reduction": self.reduction}


class SelfAdversarial(TripleLoss):
    r"""The self-adversarial negative sampling loss (Sun et al. 2019, used by BoxE and
    ExpressivE): :math:`-\log\sigma(\gamma + s^+) - \sum_i w_i \log\sigma(-s_i^- - \gamma)`
    with the weights :math:`w = \mathrm{softmax}(\alpha s^-)` of the corrupted triples
    (no gradient through the weights). With the reduction ``"mean"`` the sum of the two
    terms is halved.

    :param margin: :math:`\gamma`. :param temperature: :math:`\alpha`.
    """

    name = "self-adversarial negative sampling"

    def __init__(self, margin=3.0, temperature=2.0, reduction="sum"):
        super().__init__(reduction)
        self.margin, self.temperature = margin, temperature

    def __call__(self, positive_scores, negative_scores):
        positive = -_reduce(nn.functional.logsigmoid(self.margin + positive_scores),
                            self.reduction)
        weights = negative_scores.detach().mul(self.temperature).softmax(dim=-1)
        negative = -nn.functional.logsigmoid(-negative_scores - self.margin)
        negative = _reduce((weights * negative).sum(dim=-1), self.reduction)
        factor = 0.5 if self.reduction == "mean" else 1.0
        return factor * (positive + negative)

    def describe(self):
        return dict(super().describe(), margin=self.margin, temperature=self.temperature)


class MarginRanking(TripleLoss):
    r"""The pairwise margin ranking loss :math:`h(s^- - s^+ + \gamma)` with :math:`h` the
    hinge (``"relu"``, TransE and RESHUFFLE) or the soft hinge (``"softplus"``).

    :param margin: :math:`\gamma`.
    """

    name = "margin ranking"

    def __init__(self, margin=1.0, reduction="mean", activation="relu"):
        super().__init__(reduction)
        if activation not in ("relu", "softplus"):
            raise ValueError("activation must be 'relu' or 'softplus'.")
        self.margin, self.activation = margin, activation

    def __call__(self, positive_scores, negative_scores):
        gap = negative_scores - positive_scores + self.margin
        gap = th.relu(gap) if self.activation == "relu" else nn.functional.softplus(gap)
        return _reduce(gap, self.reduction)

    def describe(self):
        return dict(super().describe(), margin=self.margin, activation=self.activation)


class PyKEENLoss(TripleLoss):
    """Any loss of PyKEEN (:mod:`pykeen.losses`) as the objective: the adapter of the
    knowledge graph embeddings of that library."""

    name = "pykeen loss"

    def __init__(self, loss, num_entities=None):
        super().__init__(getattr(loss, "reduction", "mean"))
        self.loss = loss
        self.num_entities = num_entities

    def __call__(self, positive_scores, negative_scores):
        return self.loss.process_slcwa_scores(
            positive_scores=positive_scores, negative_scores=negative_scores,
            label_smoothing=0.0, batch_filter=None, num_entities=self.num_entities,
            pos_weights=None, neg_weights=None)

    def describe(self):
        return {"loss": type(self.loss).__name__, "reduction": self.reduction}


#: Name -> loss of the triples.
TRIPLE_LOSSES = {"self-adversarial": SelfAdversarial, "margin ranking": MarginRanking}


def triple_loss(spec, **options):
    """A loss from a name of :data:`TRIPLE_LOSSES`, a class or an instance."""
    if isinstance(spec, TripleLoss):
        if options:
            raise TypeError("Options are given with a name or a class, not an instance.")
        return spec
    if isinstance(spec, str):
        if spec not in TRIPLE_LOSSES:
            raise ValueError(f"Unknown loss {spec!r}; available: {sorted(TRIPLE_LOSSES)}.")
        spec = TRIPLE_LOSSES[spec]
    return spec(**options)


# -- negative sampler ------------------------------------------------------------------------
class UniformCorruption:
    """Corrupt a positive triple by replacing its head or its tail (and the relation, if
    asked) with a uniformly drawn entity, never with the one it had: ``num_negs_per_pos``
    corrupted triples per positive, the sides taking equal shares of them (the basic
    sampler of the knowledge graph embedding literature, drawn in the order PyKEEN's is, so
    that equal seeds give equal triples).

    :param num_negs_per_pos: Corrupted triples per positive (``None``: 1).
    :param sides: Which of ``"h"``, ``"r"``, ``"t"`` are corrupted.

    Part (vocabulary value): ``negative sampler``.
    """

    _COLUMN = {"h": 0, "r": 1, "t": 2}

    def __init__(self, num_negs_per_pos=None, sides=("h", "t")):
        self.num_negs_per_pos = 1 if num_negs_per_pos is None else int(num_negs_per_pos)
        if not set(sides) <= set(self._COLUMN) or not sides:
            raise ValueError("sides is a non-empty subset of 'h', 'r', 't'.")
        self.sides = tuple(sides)

    def __call__(self, positives, num_entities, num_relations):
        """``(B, K, 3)`` corrupted triples of the positives ``(B, 3)``."""
        batch = positives.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0)
        total = batch.shape[0]
        step = int(math.ceil(total / len(self.sides)))
        for side, start in zip(self.sides, range(0, total, step), strict=False):
            column = self._COLUMN[side]
            stop = min(start + step, total)
            high = (num_relations if column == 1 else num_entities) - 1
            replacement = th.randint(high=high, size=(stop - start,), device=batch.device)
            replacement += (replacement >= batch[start:stop, column]).long()
            batch[start:stop, column] = replacement
        return batch.view(*positives.shape[:-1], self.num_negs_per_pos, 3)

    def describe(self):
        return {"sampler": "uniform corruption", "num_negs_per_pos": self.num_negs_per_pos,
                "sides": self.sides}


class _TripleBatches(th.utils.data.IterableDataset):
    """Shuffled batches of the triples, one pass per epoch: the positives and, sampled by
    ``sample``, their corrupted triples."""

    def __init__(self, triples, batch_size, drop_last, sample):
        self.triples, self.batch_size = triples, batch_size
        self.drop_last, self.sample = drop_last, sample

    def __iter__(self):
        indices = th.utils.data.BatchSampler(
            sampler=th.utils.data.RandomSampler(data_source=range(len(self.triples))),
            batch_size=self.batch_size, drop_last=self.drop_last)
        for ids in indices:
            positives = self.triples[ids]
            yield positives, self.sample(positives)

    def __len__(self):
        batches, rest = divmod(len(self.triples), self.batch_size)
        return batches + (1 if rest and not self.drop_last else 0)


# -- the geometry of triples -----------------------------------------------------------------
class TripleGeometry(ComposedGeometry):
    """The geometry of a model trained on the triples of a graph: entities, a relation model
    (what a relation is), the triple score, the objective of a :class:`TripleLoss` and a
    negative sampler. It is a :class:`~mowl.representation.ComposedGeometry` of the parts of
    :func:`triple_geometry`, so it also reads the axioms that are triples (``gci0``, ``gci2``,
    ``class_assertion``, ``object_property_assertion``) and :class:`~mowl.models.ComposedEL`
    trains it on them; a model of the projection (:mod:`mowl.models.graph_kge`) trains it on
    the triples of the graph instead (:meth:`triple_cost`).

    :param loss: A :class:`TripleLoss` (default: margin ranking with the margin of the
        objective).
    :param sampler: The negative sampler (default: :class:`UniformCorruption`).
    :param projection: The name of the projection (:mod:`graph_projection`) the method is
        defined with, for a model that is built without one.
    """

    def __init__(self, classes, relations, measure, objective=None, individuals=None,
                 regularizers=(), supported_forms=None, neg_capable_gcis=None, loss=None,
                 sampler=None, projection="owl2vec_star"):
        super().__init__(classes, relations, measure, objective, individuals, regularizers,
                         supported_forms, neg_capable_gcis)
        self.loss = loss if loss is not None else MarginRanking(self.objective.margin)
        self.sampler = sampler if sampler is not None else UniformCorruption()
        self.projection = projection
        #: Function without arguments that draws all tables again as at the start (set by
        #: the builder of a method; the default draws every table uniformly in [-1, 1]).
        self.initialiser = None

    def reset_parameters(self):
        """Draw the parameters again, as they were drawn when the geometry was built."""
        if self.initialiser is not None:
            self.initialiser()
            return
        with th.no_grad():
            for part in (self.class_model, self.relation_model, self.individual_model):
                if part is None:
                    continue
                for child in part.modules():
                    if isinstance(child, nn.Embedding) and child.weight.requires_grad:
                        nn.init.uniform_(child.weight, a=-1, b=1)

    def triple_cost(self, heads, relations, tails):
        """Minus the score of the triples with these entity and relation ids, ``(B, 1)``."""
        with self.encoding():
            head, tail = self.classes(heads), self.classes(tails)
            return self.measure.triple(self.relation_model, relations, head, tail)

    def trainable_parameters(self):
        """The parameters an optimiser trains (a graph network's fixed features are not)."""
        return [p for p in self.parameters() if p.requires_grad]

    def encoding(self):
        """Compute what the entities share over a whole batch once (the encoding of a graph
        network) and reuse it inside the block."""
        encoding = getattr(self.class_model, "encoding", None)
        return contextlib.nullcontext() if encoding is None else encoding()

    @property
    def parts(self):
        parts = super().parts
        parts["objective"] = self.loss.describe()["loss"]
        parts["negative sampler"] = self.sampler.describe()["sampler"]
        parts["projection"] = self.projection
        return parts


def triple_geometry(classes, relations, measure, objective=None, individuals=None,
                    regularizers=(), loss=None, sampler=None, projection="owl2vec_star"):
    """The :class:`~mowl.representation.ComposedGeometry` of points and a triple score. The
    forms it reads are the ones a projection turns into triples: ``gci0``, ``gci2``,
    ``class_assertion`` and ``object_property_assertion`` (``supported_forms``); the others
    raise :class:`~mowl.representation.ConstructorUnavailable`. The measure is bound to the
    relation model by the geometry (the ``subclass`` relation)."""
    objective = objective if objective is not None else Objective()
    neg = {"gci0", "gci2", "object_property_assertion"} if objective.negatives == "all" \
        else {"gci2", "object_property_assertion"}
    return TripleGeometry(
        classes, relations, measure, objective, individuals, regularizers,
        supported_forms=("gci0", "gci2", "class_assertion", "object_property_assertion"),
        neg_capable_gcis=neg, loss=loss, sampler=sampler, projection=projection)


# -- RESHUFFLE: a relation is a matrix, an entity is encoded by the graph --------------------
class ReshuffleRelations(_PairRelations):
    r"""RESHUFFLE (Pavlovic et al. 2025): a relation is a soft reshuffling of coordinates,
    an :math:`l \times l` matrix :math:`R_r` obtained from a raw :math:`(l+1) \times l`
    parameter by a column-wise softmax whose last row is dropped. An entity is a
    :math:`k \times l` matrix and a triple is scored
    :math:`-\lVert \mathrm{ReLU}(x_h R_r - x_t) \rVert_2`
    (:func:`mowl.representation.library.graph_interactions.reshuffle_interaction`), which is zero exactly when
    :math:`x_h R_r \le x_t` coordinate-wise.

    The entities that meet this relation model are encoded by a graph network that uses its
    matrices (:class:`GraphEncodedPoints`).

    :param size: Number of rows of the matrices. With ``inverse=True`` the relations of a graph
        are the rows ``2r`` (forward) and ``2r + 1`` (inverse) and ``size`` is twice their
        number; with ``inverse=False`` relation ``r`` is row ``2 r`` and the odd rows only
        carry messages.
    :param l: Number of coordinates.
    :param inverse: Whether the triples carry inverse relations (the setting of the authors'
        code).
    :param row_function: ``"softmax"`` or ``"square"`` (the RESHUFFLE2 ablation).
    :param subclass_id: The row of the relation ``subclass``.

    Part (vocabulary value): ``matrix``. Used by: ``reshuffle``.
    """

    part = "matrix"

    def __init__(self, size=None, l=25, inverse=True,  # noqa: E741
                 row_function="softmax", subclass_id=None):
        super().__init__()
        from mowl.representation.library.graph_interactions import relation_matrices
        self._matrices = relation_matrices
        self.l, self.inverse, self.row_function = l, bool(inverse), row_function
        self.reserved = 0
        self._subclass_id = subclass_id
        self.vectors = nn.Embedding(size, (l + 1) * l)
        with th.no_grad():
            nn.init.normal_(self.vectors.weight)
        self.loop_raw = nn.Parameter(th.randn(l + 1, l) * 0.1)
        self._shared = None

    def __len__(self):
        return self.vectors.num_embeddings // (1 if self.inverse else 2)

    @property
    def subclass(self):
        return -1 if self._subclass_id is None else self._subclass_id

    def raw(self):
        """The raw parameters ``(rows, l + 1, l)``."""
        return self.vectors.weight.view(-1, self.l + 1, self.l)

    def matrices(self):
        """``(rows, l, l)`` matrices of all rows and the loop matrix ``(l, l)`` (inside
        :meth:`shared`, those computed when the block was entered)."""
        if self._shared is not None:
            return self._shared
        return (self._matrices(self.raw(), self.row_function),
                self._matrices(self.loop_raw, self.row_function))

    @contextlib.contextmanager
    def shared(self):
        """Compute the matrices once for everything inside the block: the entity encoding
        and the cost of a triple use the same tensors, so the gradient of both flows into
        the parameters through one node."""
        outer = self._shared
        if outer is None:
            self._shared = self.matrices()
        try:
            yield self
        finally:
            self._shared = outer

    def row(self, ids):
        """The rows of the relation ids (relation ``r`` is row ``2 r`` without inverses)."""
        return ids if self.inverse else 2 * ids

    def cost(self, ids, head, tail):
        from mowl.representation.library.graph_interactions import reshuffle_interaction
        matrix = self.matrices()[0][self.row(ids)]
        k = head.centre.shape[1] // self.l
        score = reshuffle_interaction(head.centre.view(-1, k, self.l), matrix,
                                      tail.centre.view(-1, k, self.l))
        return -score.reshape(-1, 1)

    def reset_parameters(self):
        with th.no_grad():
            nn.init.normal_(self.vectors.weight)
            nn.init.normal_(self.loop_raw, std=0.1)


class GraphEncodedPoints(Classes):
    r"""Entities computed by a graph network (the class model of RESHUFFLE): every entity has
    a fixed random binary feature matrix :math:`k \times l` and the model is the features
    after ``num_layers`` rounds of message passing along the edges of the graph, each message
    the features of a neighbour multiplied by the matrix of the relation
    (:func:`mowl.representation.library.graph_interactions.propagate`). So an entity is not a row of a table:
    an entity that is new to the graph gets an encoding as well
    (:meth:`encode_graph`).

    The matrices are those of the :class:`ReshuffleRelations` it is bound to
    (:meth:`bind`); the encoding of a whole batch is computed once inside
    :meth:`encoding`.

    :param size: Number of entities.
    :param l, k: Coordinates and rows of the feature matrices.
    :param triples: ``(E, 3)`` integer tensor of the edges ``[source, relation, target]`` of the
        graph (forward relation ids ``0 .. R-1``).
    :param num_layers: Message passing layers.
    :param aggregation: ``"max_self_and_selfLoop"`` or ``"max_self_noLoop"``.
    :param density: Probability of a 1 in the random features (the authors' code draws
        ``N(0, 1) > 0.5``, 0.3085; the paper's text says 0.5).
    :param feature_seed: Seed of the random features.

    Part (vocabulary value): ``point``; parametrisation ``graph encoder``. Used by:
    ``reshuffle``.
    """

    region = Point
    part = "point"
    exact_conjunction = False

    def __init__(self, size=None, l=25, k=80, triples=None, num_layers=4,  # noqa: E741
                 aggregation="max_self_and_selfLoop", density=0.3085, feature_seed=0):
        super().__init__()
        from mowl.representation.library.graph_interactions import random_features
        if aggregation not in ("max_self_and_selfLoop", "max_self_noLoop"):
            raise ValueError("aggregation must be 'max_self_and_selfLoop' or "
                             "'max_self_noLoop'.")
        self.l, self.k, self.num_layers = l, k, num_layers
        self.aggregation, self.density = aggregation, density
        generator = th.Generator().manual_seed(feature_seed)
        features = random_features(size, k, l, density, generator)
        # fixed: only the relation matrices (and the loop) are learned
        self.centres = nn.Embedding.from_pretrained(features.reshape(size, k * l), freeze=True)
        triples = th.zeros((0, 3), dtype=th.long) if triples is None else triples
        self.register_buffer("edge_index", th.stack([triples[:, 0], triples[:, 2]]))
        self.register_buffer("edge_type", triples[:, 1].clone())
        self._relations = None
        self._frozen = None

    def bind(self, relations):
        """Use the matrices of this relation model (not registered as a sub-module: its
        parameters are those of the geometry's relation model)."""
        if not isinstance(relations, ReshuffleRelations):
            raise TypeError("GraphEncodedPoints is bound to ReshuffleRelations.")
        object.__setattr__(self, "_relations", relations)
        return self

    def encode(self, features=None, edge_index=None, edge_type=None, all_relations=False):
        """Entity matrices ``(n, k, l)`` after message passing, and the relation matrices
        (``all_relations``: all rows, relation ``i`` at ``2i`` and its inverse at ``2i + 1``;
        otherwise the forward ones). Without arguments: the features and the graph of the
        model."""
        from mowl.representation.library.graph_interactions import propagate
        if self._relations is None:
            raise TypeError("GraphEncodedPoints needs the relation model: bind it.")
        features = self.centres.weight.view(-1, self.k, self.l) if features is None \
            else features
        edge_index = self.edge_index if edge_index is None else edge_index
        edge_type = self.edge_type if edge_type is None else edge_type
        rel, loop = self._relations.matrices()
        x = propagate(features, rel, loop, edge_index, edge_type, self.num_layers,
                      self.aggregation)
        return x, (rel if all_relations else rel[0::2])

    def encode_graph(self, graph_triples, num_entities, seed=0, all_relations=False):
        """Entity matrices and relation matrices of another graph over the same relations,
        with fresh random features drawn with ``seed``: the inductive setting of the paper,
        where the entities of the test graph are not seen in training."""
        from mowl.representation.library.graph_interactions import random_features
        generator = th.Generator().manual_seed(seed)
        features = random_features(num_entities, self.k, self.l, self.density, generator)
        device = self.edge_index.device
        features, graph_triples = features.to(device), graph_triples.to(device)
        return self.encode(features, th.stack([graph_triples[:, 0], graph_triples[:, 2]]),
                           graph_triples[:, 1], all_relations=all_relations)

    @contextlib.contextmanager
    def encoding(self):
        outer = self._frozen
        with self._relations.shared():
            if outer is None:
                self._frozen = self.encode(all_relations=self._relations.inverse)[0]
            try:
                yield self
            finally:
                self._frozen = outer

    def forward(self, ids):
        x = self._frozen if self._frozen is not None \
            else self.encode(all_relations=self._relations.inverse)[0]
        return Point(x[ids].reshape(len(ids), -1))

    def tables(self):
        return {}


# -- initialisation of the tables ------------------------------------------------------------
def _init_boxe(classes, relations):
    """The initialisation of BoxE (the authors' and PyKEEN's, drawn in PyKEEN's order):
    unit-norm entity base and bump vectors, unit-norm relation position and shape vectors,
    relation sizes uniform in [0, 1]."""
    from pykeen.nn.init import uniform_norm_
    with th.no_grad():
        for table in (classes.centres, classes.bumps):
            if table is not None:
                table.weight.copy_(uniform_norm_(th.empty_like(table.weight)))
        for table, size in ((relations.head_base, False), (relations.head_delta, False),
                            (relations.head_size, True), (relations.tail_base, False),
                            (relations.tail_delta, False), (relations.tail_size, True)):
            if size:
                nn.init.uniform_(table.weight)
            else:
                table.weight.copy_(uniform_norm_(th.empty_like(table.weight)))


def _init_normal(classes, relations):
    """The initialisation of ExpressivE (the default of PyKEEN's representations)."""
    with th.no_grad():
        nn.init.normal_(classes.centres.weight)
        nn.init.normal_(relations.vectors.weight)


# -- the methods -----------------------------------------------------------------------------
_DEFAULT_RELATIONS = {"graph_kge": "translation",
                      "boxe": "head and tail boxes (distance score)",
                      "expressive": "region over pairs"}

#: The loss each method is trained with (the loss of its publication).
_DEFAULT_LOSS = {
    "graph_kge": lambda margin: MarginRanking(margin),
    "gda_projection": lambda margin: MarginRanking(margin),
    "boxe": lambda margin: SelfAdversarial(3.0, 2.0, "sum"),
    "expressive": lambda margin: SelfAdversarial(3.0, 2.0, "sum"),
    "reshuffle": lambda margin: MarginRanking(2.0, "mean"),
}


def _accepted_options(relations, options):
    """The options the relation model takes. A named method passes the options of its own
    relation model (``min_denom`` of ExpressivE); when another relation model is swapped in
    that does not have them they are dropped with a warning instead of failing."""
    from mowl.representation import methods as M
    factory = M.RELATION_MODELS.get(relations, relations) if isinstance(relations, str) \
        else relations
    if not options or not isinstance(factory, type):
        return options
    parameters = inspect.signature(factory.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return options
    dropped = sorted(set(options) - set(parameters))
    if dropped:
        warnings.warn(f"{factory.__name__} has no option {dropped}; not used.", UserWarning,
                      stacklevel=3)
    return {k: v for k, v in options.items() if k in parameters}


def _build(key, nb_classes, nb_relations, nb_individuals, embed_dim, margin, classes,
           relations, objective, tables, relation_options, subclass_id=-2, loss=None,
           sampler=None, regularizers=(), projection="owl2vec_star", init=True):
    """The geometry of a method from parts. With ``subclass_id`` (an int, ``-1`` if the
    graph has no such edge) the relations are those of a graph, indexed by its ids, and
    ``subclass_id`` is the row of the ``subclass`` relation; without it one more relation
    is reserved for ``subclass`` (the geometry reads axioms)."""
    from mowl.representation import methods as M
    if tables is not None:
        raise TypeError(f"{key}: tables must be a PyKEEN model; use boxe(...) / "
                        f"expressive(...) with tables, or geometry_of(model).")
    graph = subclass_id != -2
    relation_options = _accepted_options(relations, relation_options)
    pair_options = dict(relation_options)
    if graph and subclass_id >= 0:
        pair_options["subclass_id"] = subclass_id
    elif graph:
        pair_options["subclass_id"] = -1
    if classes == "point":
        # the entities of a knowledge graph model: points (with a bump for BoxE)
        def class_factory(size, dim):
            return PointClasses(size, dim, bump=key == "boxe")
    else:
        class_factory = M.CLASS_MODELS[classes] if isinstance(classes, str) else classes
    relation_factory = M.RELATION_MODELS[relations] if isinstance(relations, str) \
        else relations
    class_model = class_factory(nb_classes, embed_dim)
    if relation_factory is None:
        relation_model = None
    elif graph and not (isinstance(relation_factory, type)
                        and issubclass(relation_factory, _PairRelations)):
        relation_model = relation_factory(nb_relations, embed_dim, **relation_options)
    else:
        relation_model = relation_factory(nb_relations, embed_dim, **pair_options) \
            if pair_options else relation_factory(nb_relations, embed_dim)
    if relation_model is not None and not graph and not hasattr(relation_model, "subclass"):
        # a relation model without a reserved subclass row (translation): add one
        relation_model = _WithSubclass(relation_model, embed_dim)
    initialiser = None
    if init and classes == "point" and isinstance(class_model, Points):
        if isinstance(relation_model, BoxERelations) and key == "boxe":
            initialiser = functools.partial(_init_boxe, class_model, relation_model)
        elif isinstance(relation_model, ExpressivERelations) and key == "expressive":
            initialiser = functools.partial(_init_normal, class_model, relation_model)
        if initialiser is not None:
            initialiser()
    individual_model = class_factory(nb_individuals, embed_dim) if nb_individuals else None
    if objective is None:
        objective = Objective(margin, negatives="gci2")
    if loss is None:
        loss = _DEFAULT_LOSS.get(key, _DEFAULT_LOSS["graph_kge"])(margin)
    else:
        loss = triple_loss(loss)
    geometry = triple_geometry(class_model, relation_model, TripleScore(), objective,
                               individuals=individual_model, regularizers=regularizers,
                               loss=loss, sampler=sampler, projection=projection)
    geometry.initialiser = initialiser
    return geometry


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
              subclass_id=-2, loss=None, sampler=None, regularizers=(),
              projection="owl2vec_star", **relation_options):
    """Any knowledge graph model on a projection, as parts: classes and individuals are
    entities, a relation model says what a relation is, the measure is the triple score.
    The default (point classes, translation) is TransE; ``relations="head and tail
    boxes (distance score)"`` is BoxE, ``"region over pairs"`` is ExpressivE, and
    ``classes="ball"`` or ``"box"`` use the centres of those regions as the entities.

    :param subclass_id: ``-2`` (default): the geometry reads axioms and reserves a relation
        for ``subclass``. A graph (:class:`~mowl.models.graph_kge.TripleGraphModel`) passes
        the row of the ``subclass`` edge (``-1`` if it has none) and the relations are its
        own.
    :param loss: A :class:`TripleLoss` or its name (default: margin ranking with ``margin``).
    :param sampler: The negative sampler (default: :class:`UniformCorruption`).
    :param projection: The projection the method is defined with (:mod:`graph_projection`).

    A PyKEEN model that is not one of these is not rebuilt: :func:`geometry_of` gives the
    geometry of the trained model, with PyKEEN's own interaction.
    """
    return _build("graph_kge", nb_classes, nb_relations, nb_individuals, embed_dim, margin,
                  classes, relations, objective, tables, relation_options, subclass_id,
                  loss, sampler, regularizers, projection)


@register_method("gda_projection")
def gda_projection(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=1.0,
                   classes="point", relations="translation", objective=None, tables=None,
                   subclass_id=-2, loss=None, sampler=None, regularizers=(),
                   projection="gda", **relation_options):
    """The GDA projection with a knowledge graph embedding: the combination of
    :func:`graph_kge` with the projection that unfolds an existential restriction through
    intersections and nested existentials until a named class is reached
    (:class:`mowl.projection.GDAProjector`; one relation per composed role path). The
    relation model, the objective and the sampler are those of :func:`graph_kge` (TransE
    with the margin ranking loss by default); only the projection differs."""
    return _build("gda_projection", nb_classes, nb_relations, nb_individuals, embed_dim,
                  margin, classes, relations, objective, tables, relation_options,
                  subclass_id, loss, sampler, regularizers, projection)


@register_method("boxe")
def boxe(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=1.0,
         classes="point", objective=None, tables=None, subclass_id=-2, loss=None,
         sampler=None, regularizers=(), projection="owl2vec_star", relations=None,
         **relation_options):
    """BoxE from parts: points with a bump, head and tail boxes, the triple score, the
    self-adversarial negative sampling loss (margin 3, temperature 2, summed), uniform
    corruption of heads and tails.

    :param classes: ``"point"`` (the default, with a bump) or any class model name
        (``"ball"``, ``"box"``): then only the centres are used.
    :param relations: A relation model name (default: head and tail boxes with the
        distance score); any other relation model of triples is a swap.
    :param tables: A trained PyKEEN :class:`~pykeen.models.BoxE` whose entity and relation
        tables are shared; classes and relations are then indexed by the PyKEEN ids
        (``nb_*`` are ignored) and the ``subclass`` relation is ``subclass_id``.
    :param subclass_id: See :func:`graph_kge`.
    """
    if tables is not None:
        subclass_id = relation_options.pop("subclass_id", None if subclass_id == -2
                                           else subclass_id)
        reps = tables.relation_representations
        classes_model = PointClasses(
            centres=PyKEENTable(tables.entity_representations[0]),
            bumps=PyKEENTable(tables.entity_representations[1]))
        relations_model = BoxERelations(
            tables=[PyKEENTable(r) for r in reps], subclass_id=subclass_id,
            tanh_map=tables.interaction.tanh_map, p=tables.interaction.p,
            power_norm=tables.interaction.power_norm)
        return triple_geometry(classes_model, relations_model, TripleScore(),
                               objective or Objective(margin, negatives="gci2"),
                               individuals=classes_model)
    return _build("boxe", nb_classes, nb_relations, nb_individuals, embed_dim, margin,
                  classes, relations or "head and tail boxes (distance score)", objective,
                  None, relation_options, subclass_id, loss, sampler, regularizers,
                  projection)


@register_method("expressive")
def expressive(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=1.0,
               classes="point", objective=None, tables=None, subclass_id=-2, loss=None,
               sampler=None, regularizers=(), projection="owl2vec_star", relations=None,
               **relation_options):
    """ExpressivE from parts: points, parallelograms over pairs, the triple score, the
    self-adversarial negative sampling loss (margin 3, temperature 2, summed). See
    :func:`boxe` for ``classes``, ``relations``, ``subclass_id`` and ``tables`` (a trained
    PyKEEN ExpressivE)."""
    if tables is not None:
        subclass_id = relation_options.pop("subclass_id", None if subclass_id == -2
                                           else subclass_id)
        classes_model = PointClasses(centres=PyKEENTable(tables.entity_representations[0]))
        inter = tables.interaction
        relations_model = ExpressivERelations(
            table=PyKEENTable(tables.relation_representations[0]), p=inter.p,
            tanh_map=inter.tanh_map, min_denom=inter.min_denom, subclass_id=subclass_id)
        return triple_geometry(classes_model, relations_model, TripleScore(),
                               objective or Objective(margin, negatives="gci2"),
                               individuals=classes_model)
    return _build("expressive", nb_classes, nb_relations, nb_individuals, embed_dim, margin,
                  classes, relations or "region over pairs", objective, None,
                  relation_options, subclass_id, loss, sampler, regularizers, projection)


@register_method("reshuffle")
def reshuffle(nb_classes, nb_relations, nb_individuals=0, embed_dim=None, margin=2.0,
              objective=None, tables=None, subclass_id=-2, loss=None, sampler=None,
              regularizers=(), projection="owl2vec_star", triples=None, l=25,  # noqa: E741
              k=80, num_layers=4, aggregation="max_self_and_selfLoop", density=0.3085,
              feature_seed=0, row_function="softmax", inverse_triples=True):
    """RESHUFFLE from parts: entities encoded by a graph network from fixed random binary
    features (:class:`GraphEncodedPoints`), relations that are soft coordinate reshufflings
    (:class:`ReshuffleRelations`), the triple score, the margin ranking loss with margin 2
    and the uniform corruption of heads and tails.

    It is defined on a graph: ``triples`` (``(E, 3)`` ids of its edges) is required, and
    ``nb_relations`` is the number of relations of the graph (the rows of the matrices are
    twice that, a relation and its inverse).

    :param l, k, num_layers, aggregation, density, feature_seed: See
        :class:`GraphEncodedPoints`.
    :param row_function: See :class:`ReshuffleRelations`.
    :param inverse_triples: Whether the triples carry the inverse relations (ids ``2r`` and
        ``2r + 1``); ``False`` trains on the forward triples only.
    """
    if triples is None:
        raise ConstructorUnavailable(
            "reshuffle is defined on a graph: its entities are encoded by message passing "
            "over the edges. Pass triples=..., or use mowl.models.Reshuffle.")
    if tables is not None:
        raise TypeError("reshuffle: tables are not supported.")
    relation_model = ReshuffleRelations(
        2 * nb_relations, l, inverse_triples, row_function,
        subclass_id=None if subclass_id < 0 else (2 * subclass_id if inverse_triples
                                                  else subclass_id))
    class_model = GraphEncodedPoints(nb_classes, l, k, th.as_tensor(triples), num_layers,
                                     aggregation, density, feature_seed)
    class_model.bind(relation_model)
    if objective is None:
        objective = Objective(margin, negatives="gci2")
    loss = triple_loss(loss) if loss is not None else _DEFAULT_LOSS["reshuffle"](margin)
    geometry = triple_geometry(class_model, relation_model, TripleScore(), objective,
                               regularizers=regularizers, loss=loss, sampler=sampler,
                               projection=projection)
    geometry.initialiser = relation_model.reset_parameters
    return geometry


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
        super().__init__()
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
        self.sampler = UniformCorruption(model.num_negs_per_pos)
        self.projection = type(model.projector).__name__

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

    # -- the training protocol of a triple model (see TripleGeometry) -----------------------
    @property
    def loss(self):
        """The objective: the loss of the PyKEEN model (:class:`PyKEENLoss`)."""
        return PyKEENLoss(self.kge.loss, self.kge.num_entities)

    def parameters(self):
        return self.kge.parameters()

    def trainable_parameters(self):
        return self.kge.get_grad_params()

    def regularization_loss(self):
        """The regularisation term of the PyKEEN model."""
        return self.kge.collect_regularization_term()

    def reset_parameters(self):
        """Draw the parameters of the PyKEEN model again."""
        self.kge.reset_parameters_()

    def project(self):
        """Apply the constraints of the PyKEEN model after an optimisation step."""
        self.kge.post_parameter_update()

    @property
    def training(self):
        """Whether the PyKEEN model is in training mode."""
        return self.kge.training

    def train(self, mode=True):
        self.kge.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        return self.kge.state_dict()

    def encoding(self):
        return contextlib.nullcontext()

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
                "individuals": "point", "measure": "triple score", "conjunction": "none",
                "objective": self.loss.describe()["loss"],
                "negative sampler": self.sampler.describe()["sampler"],
                "projection": self.projection}

    @property
    def relation_part(self):
        name = type(self.kge).__name__
        return {"BoxE": "head and tail boxes", "ExpressivE": "region over pairs",
                "Reshuffle": "matrix"}.get(name, "model of PyKEEN")

    def as_parts(self):
        """The same model as a :func:`triple_geometry` of parts that share its tables
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
