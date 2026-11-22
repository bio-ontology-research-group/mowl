"""FaithEL (Lacerda, Ozaki and Guimarães, RuleML+RR 2024) as a combination of parts.

The TBox of FaithEL is *constructed*, not learned: the convex geometric model of the
Definition 26 canonical model is strongly faithful, so a TBox axiom holds in it exactly when
the ontology entails it. Only the ABox is learned. The method is the combination

========================  ======================================================================
TBox                      :class:`FaithfulTBox`: a constructed model (fixed by reasoning) that
                          answers the TBox forms exactly (cost 0.0 or 1.0)
class model               :class:`AnchoredClasses`: a class has a *moving* point, kept within a
                          ``radius`` of the centroid of its fixed convex region
relation model            :class:`AnchoredPairs`: a relation has a moving head and tail point,
                          anchored at the centroid of its region over pairs
individual model          :class:`~mowl.representation.library.shapes_classes.Points`: a point
measure                   :class:`AnchorDistance`: the distance of a point to the moving point
                          plus the distance to the centroid
objective                 :class:`AnchoredObjective`: mean squared errors pulling the individual
                          and the moving points to the centroid of what they belong to
training regime           :class:`AlternatingMinibatches` (the authors' loop; or
                          :class:`FullBatch`)
========================  ======================================================================

Everything is a :class:`~mowl.representation.ComposedGeometry` with the exception of the
constructed TBox, which is held next to it (``geometry.tbox``) and read by the TBox forms
``gci0`` to ``gci3``. The tables of the learned parts keep the names of the module of the
earlier implementation (``individuals``, ``concepts``, ``role_head``, ``role_tail``,
``concept_centroid``, ``role_centroid``).
"""

from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn

from mowl.representation.class_models import Classes
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.other_constructed import (ConstructedGeometry,
                                                           NormalFormBackend)
from mowl.representation.library.shapes_classes import Points
from mowl.representation.library.shapes_regions import Point
from mowl.representation.measures import RegionMeasure
from mowl.representation.methods import register_method
from mowl.representation.objectives import Objective
from mowl.representation.regions import ConstructorUnavailable, Region
from mowl.representation.relation_models import Relations

__all__ = ["Anchored", "AnchoredPair", "AnchoredExists", "AnchoredClasses", "AnchoredPairs",
           "AnchorDistance", "AnchoredObjective", "FaithfulTBox", "FaithELGeometry",
           "AlternatingMinibatches", "FullBatch", "faithel", "faith_centroids"]


# -- regions -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Anchored(Region):
    """The region of a class: its moving point (learned) and the centroid of its fixed
    convex region (``(B, d)`` each)."""

    moving: th.Tensor
    centroid: th.Tensor

    def __len__(self):
        return self.moving.shape[0]


@dataclass(frozen=True)
class AnchoredPair(Region):
    """The region of a relation: the moving head and tail points and the centroid of its
    region over pairs (``(B, d)``, ``(B, d)`` and ``(B, 2d)``)."""

    head: th.Tensor
    tail: th.Tensor
    centroid: th.Tensor

    def __len__(self):
        return self.head.shape[0]


@dataclass(frozen=True)
class AnchoredExists(Region):
    """:math:`\\exists r.\\{b\\}`: the pair region of the relations and the individual
    ``filler`` (a point)."""

    pair: AnchoredPair
    filler: Point

    def __len__(self):
        return len(self.pair)


# -- the constructed TBox ------------------------------------------------------------------------
class _FaithBackend(NormalFormBackend):
    """The TBox model of FaithEL: an entity outside its signature costs 1.0 (the top class
    is known)."""

    def _known(self, expr):
        from mowl.geometric.normal_forms import TOP
        sig = set(self.model.signature) | {TOP}
        return all(not (e[0] == "name" and e[1] not in sig) for e in _atoms(expr))

    def _roles_known(self, *exprs):
        return all(e[1] in self.model.role_signature for e in exprs if e[0] == "exists")

    def subsumed(self, a, b):
        if not (self._known(a) and self._known(b) and self._roles_known(a, b)):
            return False
        return super().subsumed(a, b)


def _atoms(expr):
    if expr[0] in ("and", "or"):
        return _atoms(expr[1]) + _atoms(expr[2])
    if expr[0] == "exists":
        return _atoms(expr[2])
    if expr[0] == "not":
        return _atoms(expr[1])
    return [expr]


class FaithfulTBox(ConstructedGeometry):
    """The constructed part: the strongly TBox faithful convex model of the canonical model
    (:class:`~mowl.geometric.faithfulness.StronglyFaithfulModel`), answering ``gci0`` to
    ``gci3`` by what the ontology entails: cost 0.0 where the axiom holds, 1.0 where it does
    not, and for any name outside the signature of the TBox.

    :param model: A ``StronglyFaithfulModel`` (or anything with ``satisfies``, ``signature``
        and ``role_signature``).
    :param names, roles: The class names and role names by id.
    """

    def __init__(self, model, names, roles, individuals=()):
        backend = _FaithBackend(model, names=list(names), roles=list(roles),
                                individuals=list(individuals), key="faithel")
        super().__init__(backend)
        self.model = model

    def cost(self, form, rows):
        """``(B, 1)`` costs of the TBox axioms ``rows`` of ``form`` (``gci0`` ... ``gci3``)."""
        if form == "gci0":
            return self.inside(self.classes(rows[:, 0]), self.classes(rows[:, 1]))
        if form == "gci1":
            both, _ = self.conjunction(self.classes(rows[:, 0]), self.classes(rows[:, 1]))
            return self.inside(both, self.classes(rows[:, 2]))
        if form == "gci2":
            return self.inside(self.classes(rows[:, 0]),
                               self.exists(rows[:, 1], self.classes(rows[:, 2])))
        if form == "gci3":
            return self.inside(self.exists(rows[:, 0], self.classes(rows[:, 1])),
                               self.classes(rows[:, 2]))
        raise ConstructorUnavailable(f"The constructed TBox has no cost for {form!r}.")


def faith_centroids(tbox, class_index_dict, role_index_dict):
    """The centroids of the convex region of every class (``(n, d)``) and of the region over
    pairs of every role (``(m, 2d)``) in the constructed model ``tbox``."""
    eta, d = tbox.eta, tbox.dimension
    cents = np.zeros((len(class_index_dict), d), dtype=np.float32)
    for name, idx in class_index_dict.items():
        pts = eta.class_region(name).vertices
        if len(pts):
            cents[idx] = pts.mean(axis=0)
    rcents = np.zeros((len(role_index_dict), 2 * d), dtype=np.float32)
    for name, idx in role_index_dict.items():
        pts = eta.role_region(name).points
        if len(pts):
            rcents[idx] = pts.mean(axis=0)
    return cents, rcents


# -- class model and relation model --------------------------------------------------------------
def _ball_projection(weight, centre, radius):
    diff = weight - centre
    dist = diff.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return centre + diff * th.clamp(radius / dist, max=1.0)


class AnchoredClasses(Classes):
    """A class is a *moving point* of a table, initialised at the centroid of the fixed convex
    region of the class and kept within ``radius`` of it (:meth:`project`). The region
    (:class:`Anchored`) carries the moving point and the centroid.

    :param centroids: ``(n, d)`` centroids of the fixed regions.
    :param radius: Largest distance of a moving point from its centroid.

    Part (vocabulary value): ``convex region``. Used by: ``faithel``.
    """

    region = Anchored
    part = "convex region"
    exact_conjunction = True

    def __init__(self, centroids, radius=0.6):
        super().__init__()
        centroids = th.as_tensor(centroids, dtype=th.float32)
        self.radius = radius
        self.register_buffer("centroid", centroids.clone())
        self.moving = nn.Embedding.from_pretrained(centroids.clone(), freeze=False)

    def __len__(self):
        return self.moving.num_embeddings

    @property
    def embed_dim(self):
        return self.moving.embedding_dim

    @property
    def centres(self):
        return self.moving

    def forward(self, ids):
        return Anchored(self.moving(ids), self.centroid[ids])

    def project(self):
        """Keep each moving point within ``radius`` of the centroid of its region."""
        with th.no_grad():
            self.moving.weight.copy_(_ball_projection(self.moving.weight, self.centroid,
                                                      self.radius))


class AnchoredPairs(Relations):
    """A relation is a moving head point and a moving tail point (two tables), anchored at the
    centroid of the relation's region over pairs (a vector of twice the dimension); the pair
    of moving points is kept within ``radius`` of it.

    Part (vocabulary value): ``region over pairs``. Used by: ``faithel``.
    """

    part = "region over pairs"

    def __init__(self, centroids, radius=0.6):
        super().__init__()
        centroids = th.as_tensor(centroids, dtype=th.float32)
        d = centroids.shape[1] // 2
        self.radius = radius
        self.register_buffer("centroid", centroids.clone())
        self.head = nn.Embedding.from_pretrained(centroids[:, :d].clone(), freeze=False)
        self.tail = nn.Embedding.from_pretrained(centroids[:, d:].clone(), freeze=False)

    def __len__(self):
        return self.head.num_embeddings

    def tables(self):
        return {"head": self.head.weight, "tail": self.tail.weight}

    def region(self, ids):
        """The :class:`AnchoredPair` of the relations ``ids``."""
        return AnchoredPair(self.head(ids), self.tail(ids), self.centroid[ids])

    def preimage(self, ids, region):
        if not isinstance(region, Point):
            raise ConstructorUnavailable(
                "FaithEL reads exists r.{b} for an individual b; a class has no region of "
                "exists r.C here (the TBox is the constructed model).")
        return AnchoredExists(self.region(ids), region)

    def project(self):
        with th.no_grad():
            both = th.cat([self.head.weight, self.tail.weight], dim=1)
            both = _ball_projection(both, self.centroid, self.radius)
            d = self.head.embedding_dim
            self.head.weight.copy_(both[:, :d])
            self.tail.weight.copy_(both[:, d:])


# -- measure and objective -----------------------------------------------------------------------
class AnchorDistance(RegionMeasure):
    """A point compared with the region of a class (or of :math:`\\exists r.\\{b\\}`): the
    distance to the moving point, plus (``centroid_score``) the distance to the centroid of
    the fixed region; for a relation, the same on the concatenated vectors of the two
    individuals. A ranking score, not a truth condition (``reading = "score"``).

    Part (vocabulary value): ``point-to-region distance``. Used by: ``faithel``.
    """

    accepts = (Point, Anchored, AnchoredPair, AnchoredExists)
    part = "point-to-region distance"
    reading = "score"

    def __init__(self, centroid_score=True):
        self.centroid_score = bool(centroid_score)

    def distance(self, x, region):
        """``(B,)`` costs of the points ``x`` (or the concatenated pair vectors) against a
        region."""
        if isinstance(region, AnchoredExists):
            pair = th.cat([x, region.filler.centre], dim=1)
            return self.pair_cost(pair, region.pair)
        cost = (x - region.moving).norm(dim=1)
        if self.centroid_score:
            cost = cost + (x - region.centroid).norm(dim=1)
        return cost

    def pair_cost(self, pair, region):
        par = th.cat([region.head, region.tail], dim=1)
        cost = (pair - par).norm(dim=1)
        if self.centroid_score:
            cost = cost + (pair - region.centroid).norm(dim=1)
        return cost

    def member(self, a, b, margin=0.0):
        return self.distance(a.centre, b).reshape(-1, 1)

    def inside(self, a, b, margin=0.0):
        raise ConstructorUnavailable("FaithEL's TBox is the constructed model: inclusions of "
                                     "classes are read there, not as a distance.")

    meets = apart = empty = nonempty = inside


def _mse(a, b):
    """Row-wise mean squared error ``(B, 1)``: its mean is ``torch.nn.functional.mse_loss``."""
    return ((a - b) ** 2).mean(dim=1, keepdim=True)


class AnchoredObjective(Objective):
    """The loss of the authors' code, per assertion (``(B, 1)``; its mean over a batch is
    the mean squared error of the code).

    :math:`C(a)`: :math:`\\mathrm{mse}(x_a, c_C) + \\gamma\\,\\mathrm{mse}(p_C, x_a) +
    \\phi\\,\\mathrm{mse}(p_C, c_C) - \\psi\\,\\mathrm{mse}(x_a, x_{neg})`, with :math:`x_a` the individual, :math:`c_C` the
    centroid, :math:`p_C` the moving point and :math:`x_{neg}` a uniformly drawn individual
    (``neg_sampling``); :math:`r(a, b)` the same on the pair of individuals, the moving head
    and tail and the centroid of the relation. The TBox forms are read from the constructed
    TBox (0.0 or 1.0).

    :param phi: Weight of the moving-point-to-centroid term.
    :param gamma: Weight of the moving-point-to-individual terms.
    :param psi: Weight of the negative term.
    :param neg_sampling: Use the negative term.
    """

    def __init__(self, phi=1.0, gamma=1.0, psi=0.7, neg_sampling=True, **options):
        options.setdefault("negatives", "none")
        super().__init__(**options)
        self.phi, self.gamma, self.psi, self.neg_sampling = phi, gamma, psi, neg_sampling

    def _negative(self, g, n, device):
        ids = th.randint(0, len(g.individual_model), (n,), device=device)
        return g.individuals(ids).centre

    def class_assertion(self, g, rows, neg=False):
        x = g.individuals(rows[:, 0]).centre
        region = g.classes(rows[:, 1])
        p, c = region.moving, region.centroid
        loss = _mse(x, c) + self.gamma * _mse(p, x) + self.phi * _mse(p, c)
        if self.neg_sampling:
            loss = loss - self.psi * _mse(x.detach(), self._negative(g, len(rows), rows.device))
        return loss

    def object_property_assertion(self, g, rows, neg=False):
        xs, xo = g.individuals(rows[:, 0]).centre, g.individuals(rows[:, 2]).centre
        region = g.relation_model.region(rows[:, 1])
        ph, pt, c = region.head, region.tail, region.centroid
        pair = th.cat([xs, xo], dim=1)
        loss = _mse(pair, c)
        loss = loss + self.gamma * _mse(th.cat([ph, xo], dim=1), pair)
        loss = loss + self.gamma * _mse(th.cat([xs, pt], dim=1), pair)
        loss = loss + self.phi * _mse(th.cat([ph, pt], dim=1), c)
        if self.neg_sampling:
            neg_x = self._negative(g, len(rows), rows.device)
            loss = loss - self.psi * _mse(pair.detach(), th.cat([xs.detach(), neg_x], dim=1))
        return loss

    @staticmethod
    def _tbox(g, form, rows):
        return g.tbox.cost(form, rows.cpu()).to(rows.device)

    def gci0(self, g, rows, neg=False):
        return self._tbox(g, "gci0", rows)

    def gci1(self, g, rows, neg=False):
        return self._tbox(g, "gci1", rows)

    def gci2(self, g, rows, neg=False):
        return self._tbox(g, "gci2", rows)

    def gci3(self, g, rows, neg=False):
        return self._tbox(g, "gci3", rows)


# -- the geometry --------------------------------------------------------------------------------
class FaithELGeometry(ComposedGeometry):
    """FaithEL as a combination of parts (module docstring). ``tbox`` is the constructed
    TBox (:class:`FaithfulTBox`, or ``None`` for a geometry built from sizes only: the TBox
    forms then raise :class:`~mowl.representation.ConstructorUnavailable`)."""

    def __init__(self, classes, relations, individuals, measure, objective, tbox=None,
                 regularizers=()):
        super().__init__(classes, relations, measure, objective, individuals=individuals,
                         regularizers=regularizers,
                         supported_forms=("class_assertion", "object_property_assertion",
                                          "gci0", "gci1", "gci2", "gci3"),
                         neg_capable_gcis=())
        self.tbox = tbox

    # names of the tables of the module of the earlier implementation (its ``individuals``
    # table is ``individual_model.centres``: ``individuals`` is the method of the interface)
    @property
    def concepts(self):
        return self.class_model.moving

    @property
    def role_head(self):
        return self.relation_model.head

    @property
    def role_tail(self):
        return self.relation_model.tail

    @property
    def concept_centroid(self):
        return self.class_model.centroid

    @property
    def role_centroid(self):
        return self.relation_model.centroid

    def exists(self, relation_ids, region):
        return self.relation_model.preimage(relation_ids, region)

    def cost(self, form, rows, neg=False):
        if form in ("gci0", "gci1", "gci2", "gci3") and self.tbox is None:
            raise ConstructorUnavailable("This FaithEL geometry has no constructed TBox "
                                         "(it was built from sizes only).")
        return super().cost(form, rows, neg)

    def class_assertion_costs(self, rows):
        """Cost of :math:`C(a)` for rows ``(a, C)``: the distance to the moving point (plus
        the distance to the centroid), lower is more plausible; ``(B,)``."""
        return self.member(rows[:, 0], self.classes(rows[:, 1])).reshape(-1)

    def role_assertion_costs(self, rows):
        """Cost of :math:`r(a, b)` for rows ``(a, r, b)`` (the same on the concatenated
        vectors); ``(B,)``."""
        region = self.exists(rows[:, 1], self.individuals(rows[:, 2]))
        return self.member(rows[:, 0], region).reshape(-1)

    def project(self):
        for part in (self.class_model, self.relation_model):
            part.project()

    @property
    def parts(self):
        return {"classes": self.class_model.part, "relations": self.relation_model.part,
                "individuals": "point", "measure": self.measure.part,
                "conjunction": self.class_model.conjunction_kind, "existential": "exact"}


# -- training regimes -------------------------------------------------------------------------------
class AlternatingMinibatches:
    """The loop of the authors' code: per epoch, the concept assertions and then the role
    assertions, each in a random order and in minibatches; one Adam step and one projection
    per minibatch. ``model.loss_history`` holds the mean loss of every epoch.

    :param batch_size: Assertions per step (default: the model's).
    :param forms: The forms and the columns of their rows.
    """

    key = "alternating-minibatches"
    forms = (("class_assertion", 2), ("object_property_assertion", 3))

    def __init__(self, batch_size=None):
        self.batch_size = batch_size

    def run(self, model, epochs, epoch_callback=None):
        batch = self.batch_size or model.batch_size
        data = {}
        for name, width in self.forms:
            ds = model.training_datasets.get(name)
            data[name] = ds[:].to(model.device) if ds is not None and len(ds) else \
                th.empty((0, width), dtype=th.long, device=model.device)
        opt = th.optim.Adam(model.module.parameters(), lr=model.learning_rate)
        model.loss_history = []
        for epoch in range(epochs):
            total, steps = 0.0, 0
            for name, _ in self.forms:
                rows = data[name]
                perm = th.randperm(len(rows), device=model.device)
                for s in range(0, len(rows), batch):
                    opt.zero_grad()
                    loss = model.module.cost(name, rows[perm[s:s + batch]]).mean()
                    loss.backward()
                    opt.step()
                    model.module.project()
                    total += float(loss.detach())
                    steps += 1
            model.loss_history.append(total / max(steps, 1))
            if epoch_callback is not None:
                epoch_callback(epoch, model)


class FullBatch:
    """One step per epoch on all assertions of both forms (the mean of the two losses): the
    generic loop of the other models, as a swap for :class:`AlternatingMinibatches`."""

    key = "full-batch"
    forms = AlternatingMinibatches.forms

    def run(self, model, epochs, epoch_callback=None):
        data = {}
        for name, width in self.forms:
            ds = model.training_datasets.get(name)
            data[name] = ds[:].to(model.device) if ds is not None and len(ds) else None
        opt = th.optim.Adam(model.module.parameters(), lr=model.learning_rate)
        model.loss_history = []
        for epoch in range(epochs):
            opt.zero_grad()
            loss = sum(model.module.cost(name, rows).mean()
                       for name, rows in data.items() if rows is not None)
            if not isinstance(loss, th.Tensor):
                model.loss_history.append(0.0)
                continue
            loss.backward()
            opt.step()
            model.module.project()
            model.loss_history.append(float(loss.detach()))
            if epoch_callback is not None:
                epoch_callback(epoch, model)


# -- the method ------------------------------------------------------------------------------------
@register_method("faithel")
def faithel(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, radius=0.6, phi=1.0,
            gamma=1.0, psi=0.7, neg_sampling=True, centroid_score=True, tbox=None,
            centroids=None, role_centroids=None, classes=None, relations=None,
            individuals=None, measure=None, objective=None, regularizers=(), tables=None):
    """FaithEL from its parts: a constructed strongly faithful TBox (``tbox``), moving points
    anchored at the centroids of the fixed regions (:class:`AnchoredClasses`,
    :class:`AnchoredPairs`), point individuals, the anchor distance and the mean squared
    errors of the authors' code (:class:`AnchoredObjective`).

    Without ``tbox`` and centroids (a geometry built from sizes) the centroids are zero and
    the TBox forms are unavailable; :class:`mowl.models.FaithEL` passes them.

    :param embed_dim: The dimension of the space (the model takes it from the constructed
        model: one less than the number of elements of the canonical model).
    :param centroids, role_centroids: ``(nb_classes, d)`` and ``(nb_relations, 2d)``.
    :param classes, relations, individuals, measure, objective: Replace a part (an instance).
    :param tables: Not used (FaithEL has no module of its own whose tables could be shared).
    """
    if tables is not None:
        raise TypeError("FaithEL has no module whose tables could be shared.")
    d = embed_dim
    if centroids is None:
        centroids = np.zeros((nb_classes, d), dtype=np.float32)
    if role_centroids is None:
        role_centroids = np.zeros((nb_relations, 2 * d), dtype=np.float32)
    if classes is None:
        classes = AnchoredClasses(centroids, radius)
    if relations is None:
        relations = AnchoredPairs(role_centroids, radius)
    if individuals is None:
        # the initialisation (and the random numbers it draws) of the authors' code
        table = nn.Embedding(nb_individuals, d)
        with th.no_grad():
            table.weight.normal_(mean=0.5, std=0.7)
        individuals = Points(family="point", centres=table)
    return FaithELGeometry(
        classes, relations, individuals,
        measure if measure is not None else AnchorDistance(centroid_score),
        objective if objective is not None else AnchoredObjective(phi, gamma, psi, neg_sampling),
        tbox=tbox, regularizers=regularizers)
