"""Embed2Reason (E2R) from parts: classes are axis-parallel subspaces.

E2R reads *equations* of the paper (a class equals the meet, join or complement of others; a
relation is a subspace of pairs; structure terms keep the indicators in :math:`\\{0, 1\\}`
and they are clipped after every step), and it has no existential restriction. The parts:

* class model :class:`SubspaceClasses` (the 0/1 indicator over the :math:`2d` axes of
  :math:`\\mathbb{C}^d` simulated as :math:`\\mathbb{R}^{2d}`, region :class:`Subspace`),
  relation model :class:`SubspaceRelations` (the subspace of *pairs*), individual model
  :class:`PointIndividuals` (the vector of an individual, region :class:`Point`);
* the operators of the subspace algebra
  (:mod:`~mowl.representation.library.logic_subspace_algebra`): conjunction is the
  intersection of subspaces (product of indicators), disjunction the span (maximum), negation
  the orthogonal complement inside the concept universe (the real half);
* the measure :class:`ResidualMeasure`: ``inside`` of two subspaces is eq. (5), ``apart`` the
  disjointness loss (eq. (6) with :math:`\\bot = \\{0\\}`), ``member`` the residual of the
  projection of the vector onto the subspace (eq. (4) and Sec. 5);
* the objective :class:`SubspaceObjective`: which equation each kind of axiom is;
* the regularisers :class:`Orthogonality` and :class:`Separation` (eqs. (11), (12)) and the
  structure terms (:class:`SubspaceStructure`, eqs. (1)-(3));

put together by :func:`e2r` (registered as the method ``e2r``) into a
:class:`SubspaceGeometry`. :class:`E2RGeometry` describes the parameters of the earlier
hand-written module (:func:`~mowl.representation.geometry_of`).

There is no ``exists``: the paper has no existential restriction (a relation is a subspace of
pairs, :meth:`E2RGeometry.relation_member` scores a pair of individuals against it).
"""

from dataclasses import dataclass

import math

import torch as th
import torch.nn as nn

from mowl.nn.alc.module import ALCModule
from mowl.representation.class_models import Classes
from mowl.representation.geometry import ComposedGeometry, Geometry
from mowl.representation.library import logic_subspace_algebra as _algebra
from mowl.representation.measures import RegionMeasure
from mowl.representation.methods import (register_class_model, register_method,
                                          register_relation_model)
from mowl.representation.objectives import Objective
from mowl.representation.regions import Region
from mowl.representation.relation_models import Relations
from mowl.training.regularization import Regularizer

__all__ = ["Subspace", "Point", "E2RGeometry", "SubspaceClasses", "SubspaceRelations",
           "PointIndividuals", "ResidualMeasure", "SubspaceObjective", "SubspaceStructure",
           "Orthogonality", "Separation", "SubspaceGeometry", "e2r", "e2r_parts",
           "AXIOM_KINDS"]


@dataclass(frozen=True)
class Subspace(Region):
    """An axis-parallel subspace: the 0/1 (or relaxed) indicator of its axes, ``(B, 2d)``."""

    indicator: th.Tensor

    def __len__(self):
        return self.indicator.shape[0]

    def intersect(self, other):
        meet = self.indicator * other.indicator
        # the intersection is {0} (empty in the sense of the paper) unless one axis is left
        gap = th.relu(1 - meet.sum(-1, keepdim=True))
        return Subspace(meet), gap

    def complement(self):
        """The orthogonal complement inside the concept universe (the real half, which holds
        every individual vector): the imaginary half is ``0``."""
        d = self.indicator.shape[-1] // 2
        out = th.zeros_like(self.indicator)
        out[..., :d] = 1 - self.indicator[..., :d]
        return Subspace(out)

    def union(self, other):
        """The span :math:`S_1 + S_2` (maximum of the indicators)."""
        return Subspace(th.maximum(self.indicator, other.indicator))


@dataclass(frozen=True)
class Point(Region):
    """The vector of an individual, ``(B, 2d)``."""

    vector: th.Tensor

    def __len__(self):
        return self.vector.shape[0]


def _losses():
    return _algebra


class E2RGeometry(Geometry):
    """The geometry of an earlier hand-written E2R module
    (:class:`mowl.baselines.nn.alc.e2r.module.E2RModule`); shares its parameters.

    :param module: The module.
    :param unit_length: Add the unit-length term :math:`(1 - x^\\top x)^2` of eq. (4) to the
        membership cost, which makes ``member`` the module's ``class_assertion_loss``. The
        default leaves it out: ``member`` is then the squared residual that
        :meth:`~mowl.models.E2R.rank_members` sorts by, and it does not depend on how long an
        individual's vector is.
    """

    def __init__(self, module, unit_length=False):
        self.module = module
        self.unit_length = unit_length

    def classes(self, ids):
        return Subspace(self.module.concept(ids))

    def individuals(self, ids):
        return Point(self.module.entity(ids))

    def relations(self, ids):
        """The subspaces of pairs of the relations ``ids``."""
        return Subspace(self.module.relation(ids))

    def conjunction(self, a, b):
        return a.intersect(b)

    def disjunction(self, a, b):
        return Subspace(_losses().join(a.indicator, b.indicator))

    def negation(self, a):
        d = self.module.dim
        out = th.zeros_like(a.indicator)
        out[..., :d] = 1 - a.indicator[..., :d]
        return Subspace(out)

    def inside(self, a, b):
        L = _losses()
        if isinstance(a, Point):
            cost = L.residual_sq(b.indicator, a.vector)
            if self.unit_length:
                cost = cost + L.unit_length_loss(a.vector)
            return cost.unsqueeze(-1)
        return L.inclusion_loss(a.indicator, b.indicator).unsqueeze(-1)

    def apart(self, a, b):
        return _losses().disjointness_loss(a.indicator, b.indicator).unsqueeze(-1)

    def member(self, individual_ids, region):
        return self.inside(self.individuals(individual_ids), region)

    def relation_member(self, subject_ids, relation_ids, object_ids):
        """Cost of the pairs being in the relations (eq. (4) for role assertions)."""
        L, m = _losses(), self.module
        return L.role_membership_loss(
            m.relation(relation_ids), m.entity(subject_ids), m.entity(object_ids), m.dim,
            unit_length=self.unit_length).unsqueeze(-1)

    @property
    def num_classes(self):
        return self.module.c_embedding.num_embeddings

    @property
    def num_individuals(self):
        return self.module.e_embedding.num_embeddings

    # -- reasoning and scoring on a trained embedding -------------------------------------
    def _device(self):
        return next(self.parameters()).device

    def indicator(self, tree, binary=False):
        """Indicator vector of a class expression given as a tree of class indices: ``("class",
        index)``, ``("and", [trees])`` (product, :math:`\sqcap`), ``("or", [trees])`` (maximum,
        :math:`\sqcup`) and ``("not", tree)`` (complement in the real half) -- Sec. 4,
        "Membership" task.

        :param binary: Round the indicators of the named classes at 0.5 first.
        """
        kind, arg = tree
        if kind == "class":
            y = self.concept(th.tensor([arg], device=self._device()))[0].detach()
            return (y > 0.5).to(y) if binary else y
        if kind == "and":
            return _algebra.meet(*[self.indicator(t, binary) for t in arg])
        if kind == "or":
            return _algebra.join(*[self.indicator(t, binary) for t in arg])
        if kind == "not":
            y = self.indicator(arg, binary)
            out = th.zeros_like(y)
            out[:self.dim] = 1 - y[:self.dim]
            return out
        raise NotImplementedError(f"A subspace geometry has no indicator for {kind!r}.")

    def membership_scores(self, y, individual_ids):
        """Residual length :math:`\|\overline{y} \odot x_a\|` of the individuals
        ``individual_ids`` (a list of indices) for the indicator ``y`` (lower is better;
        Sec. 5), a numpy array."""
        idx = th.as_tensor(individual_ids, dtype=th.long, device=self._device())
        with th.no_grad():
            x = self.entity(idx)
            return self.membership_residual(y.unsqueeze(0), x).cpu().numpy()

    def role_membership_scores(self, relation_id, subject_ids, object_ids):
        """Residual lengths of the pair vectors of the individuals for the relation
        ``relation_id`` (lower is better), a numpy array."""
        dev = self._device()
        with th.no_grad():
            z = self.relation(th.tensor([relation_id], device=dev))
            s = th.as_tensor(subject_ids, dtype=th.long, device=dev)
            o = th.as_tensor(object_ids, dtype=th.long, device=dev)
            xpq = _algebra.pair_vector(self.entity(s), self.entity(o), self.dim)
            return self.membership_residual(z, xpq).cpu().numpy()

    def inclusion_cost(self, y_sub, y_sup):
        """Inclusion loss of eq. (5) of two indicators (``0`` iff the subspace of the first
        lies in that of the second for 0/1 indicators)."""
        return float(_algebra.inclusion_loss(y_sub, y_sup))

    @th.no_grad()
    def axiom_costs(self, kind, rows):
        """Cost of every candidate axiom in ``rows`` (model indices), lower is more
        plausible: the inclusion loss of eq. (5) for ``gci0`` rows ``[C, D]`` and the
        residual length of Sec. 5 for ``class_assertion`` rows ``[a, C]``."""
        rows = rows.to(self._device())
        if kind == "gci0":
            return _algebra.inclusion_loss(self.concept(rows[:, 0]), self.concept(rows[:, 1]))
        if kind == "class_assertion":
            return self.membership_residual(self.concept(rows[:, 1]), self.entity(rows[:, 0]))
        from mowl.evaluation.model_scoring import UnsupportedAxiomForm
        raise UnsupportedAxiomForm(f"A subspace geometry has no score for {kind!r} axioms.")

    def discretise(self, threshold=0.5):
        """0/1 indicators of all concepts and relations and the entity vectors, as numpy
        arrays ``(concepts, relations, entities)`` in index order."""
        with th.no_grad():
            c = self.concept(th.arange(self.c_embedding.weight.shape[0],
                                       device=self._device()))
            r = self.r_embedding.weight
            return ((c > threshold).cpu().numpy().astype(int),
                    (r > threshold).cpu().numpy().astype(int),
                    self.e_embedding.weight.detach().cpu().numpy())

    @property
    def parts(self):
        return {"classes": "subspace", "relations": "subspace", "individuals": "point",
                "measure": "point-to-region distance", "conjunction": "exact",
                "negation": "exact", "disjunction": "exact"}


# =========================================================================================
# the parts of E2R
# =========================================================================================

#: The kinds of axiom the objective reads (rows of class and relation indices), each the
#: equation of the paper that its loss is.
AXIOM_KINDS = (
    "class_assertion", "role_assertion",
    "concept_inclusion", "concept_conjunction", "concept_disjunction",
    "concept_negation", "concept_disjointness",
    "role_inclusion", "role_conjunction", "role_disjunction", "role_negation",
    "range",
)


class SubspaceClasses(Classes):
    """A class is an axis-parallel subspace of the real half of :math:`\\mathbb{R}^{2d}`: a
    relaxed 0/1 indicator over the :math:`2d` axes (the imaginary half is pushed to ``0`` by
    eq. (2)). ``owl:Thing`` (if its index is given) is the concept universe
    :math:`(1_d, 0_d)` and ``owl:Nothing`` is :math:`\\{0\\}`. The indicators are kept in
    :math:`[0, 1]` by :meth:`project` (the paper does not say how).

    :param size: Number of classes; :param dim: :math:`d`.
    :param centres: An existing table ``(size, 2d)`` to share.

    Part (vocabulary value): ``subspace``. Used by: ``e2r``.
    """

    region = Subspace
    part = "subspace"
    exact_conjunction = True
    #: The class model of the individuals that goes with this one (a vector per individual).
    individual_factory = None        # set below, once PointIndividuals exists

    def __init__(self, size=None, dim=None, centres=None, top_index=None, bottom_index=None):
        super().__init__()
        if centres is None:
            centres = nn.Embedding(max(size, 1), 2 * dim)
            with th.no_grad():
                centres.weight.uniform_(0, 1)
                centres.weight[:, dim:] = 0
        self.dim = centres.embedding_dim // 2
        self.centres = centres
        self.top_index, self.bottom_index = top_index, bottom_index

    def forward(self, ids):
        y = self.centres(ids)
        d = self.dim
        if self.top_index is not None:
            top = th.cat([th.ones(d), th.zeros(d)]).to(y)
            y = th.where((ids == self.top_index).unsqueeze(-1), top, y)
        if self.bottom_index is not None:
            y = th.where((ids == self.bottom_index).unsqueeze(-1), th.zeros_like(y), y)
        return Subspace(y)

    @th.no_grad()
    def project(self):
        """Clip the indicators to :math:`[0, 1]` in place."""
        self.centres.weight.clamp_(0, 1)


class BinarySubspaceClasses(SubspaceClasses):
    """The subspaces of :class:`SubspaceClasses` initialised as random axis sets (every axis
    is in the subspace with probability one half, an indicator of exactly 0 or 1) instead of
    uniform relaxed indicators in :math:`[0, 1]`. The region, the equations and the measure
    are the same; only the starting point of the optimisation differs, a start that is
    already a subspace of the lattice that the structure term :math:`\\Omega` (eq. (3))
    pushes the relaxed ones towards.

    Part (vocabulary value): ``subspace``. Used by: ``e2r`` (as a swap).
    """

    def __init__(self, size=None, dim=None, centres=None, top_index=None, bottom_index=None):
        super().__init__(size, dim, centres, top_index, bottom_index)
        if centres is None:
            with th.no_grad():
                self.centres.weight[:, :self.dim] = (
                    th.rand(self.centres.weight.shape[0], self.dim) < 0.5).float()


class SubspaceRelations(Relations):
    """A relation is an axis-parallel subspace of the space of *pairs* of individuals (both
    halves of :math:`\\mathbb{R}^{2d}`): the pair :math:`(a, b)` is the vector
    :math:`(x_{aR}, x_{bR})` and :math:`r(a, b)` holds iff it lies in the subspace. There is
    no image or preimage (the paper has no existential restriction); roles are read by
    :meth:`~SubspaceGeometry.role_assertion_loss`.

    :param size: Number of relations; :param dim: :math:`d`.

    Part (vocabulary value): ``subspace``. Used by: ``e2r``.
    """

    part = "subspace"

    def __init__(self, size=None, dim=None, vectors=None):
        super().__init__()
        if vectors is None:
            vectors = nn.Embedding(max(size, 1), 2 * dim)
            with th.no_grad():
                vectors.weight.uniform_(0, 1)
        self.vectors = vectors
        self.dim = vectors.embedding_dim // 2

    def indicators(self, ids):
        """The indicators of the relations ``ids``, ``(B, 2d)``."""
        return self.vectors(ids)

    @th.no_grad()
    def project(self):
        """Clip the indicators to :math:`[0, 1]` in place."""
        self.vectors.weight.clamp_(0, 1)


class PointIndividuals(Classes):
    """An individual is a vector :math:`x = (x_R, 0)`; the imaginary half is pushed to ``0``
    by eq. (1) and the length to ``1`` by the unit-length terms of eq. (4). Initialised as
    random unit vectors in the real half.

    :param size: Number of individuals; :param dim: :math:`d`.

    Part (vocabulary value): ``point``. Used by: ``e2r``.
    """

    region = Point
    part = "point"

    def __init__(self, size=None, dim=None, centres=None):
        super().__init__()
        if centres is None:
            centres = nn.Embedding(max(size, 1), 2 * dim)
            with th.no_grad():
                x = th.randn(centres.weight.shape[0], dim)
                x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
                centres.weight.zero_()
                centres.weight[:, :dim] = x
        self.dim = centres.embedding_dim // 2
        self.centres = centres

    def forward(self, ids):
        return Point(self.centres(ids))


SubspaceClasses.individual_factory = PointIndividuals
register_class_model("subspace")(SubspaceClasses)
register_class_model("binary subspace")(BinarySubspaceClasses)
register_relation_model("subspace")(SubspaceRelations)


class ResidualMeasure(RegionMeasure):
    """The measure of E2R. ``inside`` of two subspaces is eq. (5), :math:`\\|y_i \\odot
    \\overline{y}_j\\|^2`, zero iff the first lies in the second for 0/1 indicators;
    ``inside`` of a point and a subspace is the squared residual of the projection of the point
    onto the subspace, :math:`\\|\\overline{y} \\odot x\\|^2`, zero iff it lies in it (eq. (4),
    Sec. 5); ``apart`` is the disjointness loss :math:`\\|y_a \\odot y_b\\|^2` (eq. (6) with
    :math:`\\bot = \\{0\\}`); ``empty`` is :math:`\\|y\\|^2`.

    :param unit_length: Add the unit-length term :math:`(1 - x^\\top x)^2` of eq. (4) to the cost
        of a point in a subspace. Default ``False``: the cost does not depend on how long the
        vector is (the residual that ``rank_members`` of E2R sorts by).

    Part (vocabulary value): ``point-to-region distance``. Used by: ``e2r``.
    """

    accepts = (Subspace, Point)
    part = "point-to-region distance"

    def __init__(self, unit_length=False):
        self.unit_length = unit_length

    def inside(self, a, b, margin=0.0):
        L = _algebra
        if isinstance(a, Point):
            cost = L.residual_sq(b.indicator, a.vector)
            if self.unit_length:
                cost = cost + L.unit_length_loss(a.vector)
            return cost.unsqueeze(-1)
        return L.inclusion_loss(a.indicator, b.indicator).unsqueeze(-1)

    def member(self, point, region, margin=0.0):
        return self.inside(point, region, margin)

    def apart(self, a, b, margin=0.0):
        return _algebra.disjointness_loss(a.indicator, b.indicator).unsqueeze(-1)

    def empty(self, a, margin=0.0):
        return (a.indicator ** 2).sum(-1, keepdim=True)

    def residual(self, y, x):
        """The length :math:`\\|\\overline{y} \\odot x\\|` (the non-fitment score of
        Sec. 5; lower is better) of the vectors ``x`` for the indicators ``y``."""
        return _algebra.residual_sq(y, x).clamp_min(0).sqrt()


class SubspaceObjective(Objective):
    """The reading of the axioms of E2R: each kind of axiom (:data:`AXIOM_KINDS`) is one
    equation of Sec. 4 (:meth:`kind_cost`): eq. (4) for assertions (with the unit-length
    terms), eq. (5) for inclusions of concepts and of relations, eqs. (6) and (7) for a class
    defined as a meet or a join of others, eq. (8) for a negation, eq. (6) with
    :math:`\\bot = \\{0\\}` for disjointness, eq. (9) for ranges. The normal forms of
    :class:`~mowl.representation.ComposedGeometry` are read as in
    :class:`~mowl.representation.Objective` with the measure :class:`ResidualMeasure`.

    :param role_unit_length: Add the unit-length terms of both entities to the loss of a role
        assertion (eq. (4) does not print them; without them an entity that occurs only in role
        assertions can shrink to ``0``, which lies in every subspace).
    """

    def __init__(self, margin=0.0, role_unit_length=True, **kwargs):
        if margin:
            raise ValueError(
                "The equations of E2R have no margin (their losses are squared residuals that "
                "are zero exactly when the axiom holds), so margin must be 0; got "
                f"{margin}.")
        super().__init__(margin, **kwargs)
        self.role_unit_length = bool(role_unit_length)

    def kind_cost(self, g, kind, rows, neg=False):
        """The loss of the rows of an axiom kind, one value per row ``(N,)``; ``neg`` for
        the assertions gives the negated membership loss of a corrupted assertion."""
        L = _algebra
        if kind not in AXIOM_KINDS:
            raise ValueError(f"Unknown E2R axiom kind {kind!r}.")
        x = rows
        if kind == "class_assertion":
            y, e = g.concept(x[:, 1]), g.entity(x[:, 0])
            return L.negative_membership_loss(y, e) if neg else L.membership_loss(y, e)
        if kind == "role_assertion":
            z = g.relation(x[:, 1])
            xp, xq = g.entity(x[:, 0]), g.entity(x[:, 2])
            if neg:
                return L.negative_role_membership_loss(z, xp, xq, g.dim)
            return L.role_membership_loss(z, xp, xq, g.dim, unit_length=self.role_unit_length)
        if kind == "concept_inclusion":
            return L.inclusion_loss(g.concept(x[:, 0]), g.concept(x[:, 1]))
        if kind == "concept_conjunction":
            return L.conjunction_loss(g.concept(x[:, 0]),
                                      *[g.concept(x[:, i]) for i in range(1, x.shape[1])])
        if kind == "concept_disjunction":
            return L.disjunction_loss(g.concept(x[:, 0]),
                                      *[g.concept(x[:, i]) for i in range(1, x.shape[1])])
        if kind == "concept_negation":
            return L.negation_loss(g.concept(x[:, 0]), g.concept(x[:, 1]), dim=g.dim)
        if kind == "concept_disjointness":
            return L.disjointness_loss(g.concept(x[:, 0]), g.concept(x[:, 1]))
        if kind == "role_inclusion":
            return L.inclusion_loss(g.relation(x[:, 0]), g.relation(x[:, 1]))
        if kind == "role_conjunction":
            return L.conjunction_loss(g.relation(x[:, 0]),
                                      *[g.relation(x[:, i]) for i in range(1, x.shape[1])])
        if kind == "role_disjunction":
            return L.disjunction_loss(g.relation(x[:, 0]),
                                      *[g.relation(x[:, i]) for i in range(1, x.shape[1])])
        if kind == "role_negation":
            return L.negation_loss(g.relation(x[:, 0]), g.relation(x[:, 1]))
        return L.range_loss(g.concept(x[:, 1]), g.relation(x[:, 0]), g.dim)       # range


class SubspaceStructure(Regularizer):
    """The structure terms of E2R, eqs. (1)-(3): the imaginary half of every individual and of
    every concept is ``0``, a relation has at least one real and one imaginary axis, and the
    indicators are 0/1. The mean over the individuals, over the classes (except ``owl:Thing``
    and ``owl:Nothing``) and over the relations, summed.

    :param weight: Factor of the penalty.
    """

    def __init__(self, weight=1.0):
        super().__init__("class_embed", None, weight)

    def penalty(self, model):
        g = getattr(model, "module", model)
        L, d = _algebra, g.dim
        classes = g.class_model
        special = {classes.top_index, classes.bottom_index}
        ids = th.tensor([i for i in range(len(classes)) if i not in special],
                        dtype=th.long, device=g.device)
        x, r = g.individual_model.centres.weight, g.relation_model.vectors.weight
        loss = L.entity_structure_loss(x, d).mean()
        if len(ids) > 0:
            c = classes.centres(ids)
            loss = loss + L.concept_structure_loss(c, d).mean() + L.binary_loss(c).mean()
        loss = loss + L.relation_structure_loss(r, d).mean() + L.binary_loss(r).mean()
        return self.weight * loss


class _PairPenalty(Regularizer):
    """Base of the penalties over pairs of classes and pairs of relations that the model reads
    from the told hierarchy: ``pairs`` and ``relation_pairs`` are ``LongTensor (P, 2)``
    (``None``: no such pairs; the model of E2R sets them)."""

    def __init__(self, weight=1.0, pairs=None, relation_pairs=None):
        super().__init__("class_embed", None, weight)
        self.pairs, self.relation_pairs = pairs, relation_pairs

    def _term(self, g, pairs, get):
        raise NotImplementedError

    def terms(self, model):
        """The weighted mean over the class pairs, then over the relation pairs (those that
        exist), one scalar tensor each."""
        g = getattr(model, "module", model)
        out = []
        if self.weight > 0:
            for pairs, get in ((self.pairs, g.concept), (self.relation_pairs, g.relation)):
                if pairs is not None and len(pairs) > 0:
                    out.append(self.weight * self._term(pairs, get).mean())
        return out

    def penalty(self, model):
        terms = self.terms(model)
        if not terms:
            return None
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total


class Orthogonality(_PairPenalty):
    """:math:`\\Omega_{\\text{orth}}` (eq. (11)): the indicators of siblings of the told
    hierarchy (concepts with a common parent, relations likewise) are orthogonal,
    :math:`(y_i^\\top y_j)^2`."""

    def _term(self, pairs, get):
        return _algebra.orthogonality_loss(get(pairs[:, 0]), get(pairs[:, 1]))


class Separation(_PairPenalty):
    """:math:`\\Omega_{\\text{sep}}` (eq. (12)): for :math:`C_i \\sqsubseteq C_j` the
    indicator of :math:`C_j` has at least ``gap`` axes more than that of :math:`C_i`,
    :math:`(g - (y_j - y_i)^\\top 1)^2`.

    :param gap: The constant :math:`g`; ``None``: :math:`\\sqrt{d}` as in the paper (any
        constant :math:`\\geq 1` is allowed).
    """

    def __init__(self, weight=1.0, gap=None, pairs=None, relation_pairs=None):
        super().__init__(weight, pairs, relation_pairs)
        self.gap = gap

    def _term(self, pairs, get):
        return _algebra.separation_loss(get(pairs[:, 0]), get(pairs[:, 1]), self.gap)

    def terms(self, model):
        if self.gap is None:
            self.gap = math.sqrt(getattr(model, "module", model).dim)
        return super().terms(model)


class SubspaceGeometry(ComposedGeometry, ALCModule):
    """E2R from parts: classes, relations (of pairs) and individuals (vectors) that are
    subspaces and points, the measure :class:`ResidualMeasure`, the objective
    :class:`SubspaceObjective`.

    ``forward(kind, rows, e_emb=None, neg=False)`` is the loss of the rows of an axiom kind
    (:data:`AXIOM_KINDS`, :meth:`SubspaceObjective.kind_cost`), one value per row; the
    normal forms are read as everywhere else (``forward(rows, "gci0")``).
    """

    #: Role inclusion is an equation of E2R (``role_inclusion``), not an EL++ role axiom of the
    #: normal forms (``role_inclusion`` and ``role_chain`` of ``ComposedGeometry``).
    role_axiom_capable = False

    def __init__(self, classes, relations, measure, objective=None, individuals=None,
                 regularizers=(), **kwargs):
        ComposedGeometry.__init__(self, classes, relations, measure, objective,
                                  individuals=individuals, regularizers=regularizers,
                                  **kwargs)
        if not hasattr(self.objective, "kind_cost"):
            raise TypeError("The objective of a subspace geometry reads the axiom kinds of "
                            "E2R: use SubspaceObjective.")
        self.dim = classes.dim

    # -- the equations ---------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            kind, rows = args[0], args[1]
            neg = kwargs.pop("neg", False)
            if len(args) > 3:
                neg = args[3]
            return self.objective.kind_cost(self, kind, rows, neg=neg)
        return ComposedGeometry.forward(self, *args, **kwargs)

    def forward_fs(self, cexpr, x, e_emb, cur_index=0):
        raise NotImplementedError(
            "E2R does not use fuzzy-set semantics; see SubspaceGeometry.forward.")

    # -- accessors ---------------------------------------------------------------------------
    def entity(self, idx):
        """The vectors of the individuals ``idx``."""
        return self.individual_model.centres(idx)

    def concept(self, idx):
        """The indicator vectors of the classes ``idx`` (``owl:Thing`` and ``owl:Nothing``
        fixed)."""
        return self.class_model(idx).indicator

    def relation(self, idx):
        """The indicator vectors of the relations ``idx``."""
        return self.relation_model.vectors(idx)

    @property
    def c_embedding(self):
        return self.class_model.centres

    @property
    def r_embedding(self):
        return self.relation_model.vectors

    @property
    def e_embedding(self):
        return self.individual_model.centres

    @property
    def top_index(self):
        return self.class_model.top_index

    @property
    def bottom_index(self):
        return self.class_model.bottom_index

    def project_(self):
        """Clip all indicator vectors to :math:`[0, 1]` in place."""
        self.project()

    def reset_parameters(self):
        """Entities: random unit vectors in the real half. Concepts: real half
        :math:`U(0, 1)`, imaginary half ``0``. Relations: :math:`U(0, 1)` everywhere."""
        d = self.dim
        with th.no_grad():
            x = th.randn(self.e_embedding.weight.shape[0], d)
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.e_embedding.weight.zero_()
            self.e_embedding.weight[:, :d] = x
            self.c_embedding.weight.uniform_(0, 1)
            self.c_embedding.weight[:, d:] = 0
            self.r_embedding.weight.uniform_(0, 1)

    # -- regularisers --------------------------------------------------------------------------
    def structure_loss(self, class_idx=None):
        """The mean of eqs. (1), (2) and (3) over all individuals, classes ``class_idx``
        (default: all except ``owl:Thing`` and ``owl:Nothing``) and relations, summed."""
        L, d = _algebra, self.dim
        x, r = self.e_embedding.weight, self.r_embedding.weight
        if class_idx is None:
            class_idx = th.arange(self.c_embedding.weight.shape[0], device=x.device)
        c = self.c_embedding(class_idx)
        loss = L.entity_structure_loss(x, d).mean()
        if len(class_idx) > 0:
            loss = loss + L.concept_structure_loss(c, d).mean() + L.binary_loss(c).mean()
        return loss + L.relation_structure_loss(r, d).mean() + L.binary_loss(r).mean()

    def orthogonality_loss(self, pairs, relations=False):
        """Eq. (11) summands for sibling pairs (``[N, 2]``)."""
        get = self.relation if relations else self.concept
        return _algebra.orthogonality_loss(get(pairs[:, 0]), get(pairs[:, 1]))

    def separation_loss(self, pairs, gap, relations=False):
        """Eq. (12) summands for (sub, super) pairs (``[N, 2]``)."""
        get = self.relation if relations else self.concept
        return _algebra.separation_loss(get(pairs[:, 0]), get(pairs[:, 1]), gap)

    def membership_residual(self, y, x):
        """The residual length :math:`\\|\\overline{y} \\odot x\\|` (non-fitment score,
        Sec. 5; lower is better)."""
        return self.measure.residual(y, x)

    # -- reasoning and scoring on a trained embedding -------------------------------------
    def _device(self):
        return next(self.parameters()).device

    def indicator(self, tree, binary=False):
        """Indicator vector of a class expression given as a tree of class indices: ``("class",
        index)``, ``("and", [trees])`` (product, :math:`\sqcap`), ``("or", [trees])`` (maximum,
        :math:`\sqcup`) and ``("not", tree)`` (complement in the real half) -- Sec. 4,
        "Membership" task.

        :param binary: Round the indicators of the named classes at 0.5 first.
        """
        kind, arg = tree
        if kind == "class":
            y = self.concept(th.tensor([arg], device=self._device()))[0].detach()
            return (y > 0.5).to(y) if binary else y
        if kind == "and":
            return _algebra.meet(*[self.indicator(t, binary) for t in arg])
        if kind == "or":
            return _algebra.join(*[self.indicator(t, binary) for t in arg])
        if kind == "not":
            y = self.indicator(arg, binary)
            out = th.zeros_like(y)
            out[:self.dim] = 1 - y[:self.dim]
            return out
        raise NotImplementedError(f"A subspace geometry has no indicator for {kind!r}.")

    def membership_scores(self, y, individual_ids):
        """Residual length :math:`\|\overline{y} \odot x_a\|` of the individuals
        ``individual_ids`` (a list of indices) for the indicator ``y`` (lower is better;
        Sec. 5), a numpy array."""
        idx = th.as_tensor(individual_ids, dtype=th.long, device=self._device())
        with th.no_grad():
            x = self.entity(idx)
            return self.membership_residual(y.unsqueeze(0), x).cpu().numpy()

    def role_membership_scores(self, relation_id, subject_ids, object_ids):
        """Residual lengths of the pair vectors of the individuals for the relation
        ``relation_id`` (lower is better), a numpy array."""
        dev = self._device()
        with th.no_grad():
            z = self.relation(th.tensor([relation_id], device=dev))
            s = th.as_tensor(subject_ids, dtype=th.long, device=dev)
            o = th.as_tensor(object_ids, dtype=th.long, device=dev)
            xpq = _algebra.pair_vector(self.entity(s), self.entity(o), self.dim)
            return self.membership_residual(z, xpq).cpu().numpy()

    def inclusion_cost(self, y_sub, y_sup):
        """Inclusion loss of eq. (5) of two indicators (``0`` iff the subspace of the first
        lies in that of the second for 0/1 indicators)."""
        return float(_algebra.inclusion_loss(y_sub, y_sup))

    @th.no_grad()
    def axiom_costs(self, kind, rows):
        """Cost of every candidate axiom in ``rows`` (model indices), lower is more
        plausible: the inclusion loss of eq. (5) for ``gci0`` rows ``[C, D]`` and the
        residual length of Sec. 5 for ``class_assertion`` rows ``[a, C]``."""
        rows = rows.to(self._device())
        if kind == "gci0":
            return _algebra.inclusion_loss(self.concept(rows[:, 0]), self.concept(rows[:, 1]))
        if kind == "class_assertion":
            return self.membership_residual(self.concept(rows[:, 1]), self.entity(rows[:, 0]))
        from mowl.evaluation.model_scoring import UnsupportedAxiomForm
        raise UnsupportedAxiomForm(f"A subspace geometry has no score for {kind!r} axioms.")

    def discretise(self, threshold=0.5):
        """0/1 indicators of all concepts and relations and the entity vectors, as numpy
        arrays ``(concepts, relations, entities)`` in index order."""
        with th.no_grad():
            c = self.concept(th.arange(self.c_embedding.weight.shape[0],
                                       device=self._device()))
            r = self.r_embedding.weight
            return ((c > threshold).cpu().numpy().astype(int),
                    (r > threshold).cpu().numpy().astype(int),
                    self.e_embedding.weight.detach().cpu().numpy())

    @property
    def parts(self):
        return {"classes": "subspace", "relations": "subspace", "individuals": "point",
                "measure": "point-to-region distance", "conjunction": "exact",
                "negation": "exact", "disjunction": "exact"}


def _per_kind_losses():
    for kind in AXIOM_KINDS:
        def loss(self, x, neg=False, _kind=kind):
            return self.objective.kind_cost(self, _kind, x, neg=neg)
        loss.__name__ = f"{kind}_loss"
        loss.__doc__ = f"Loss of the rows ``x`` of kind ``{kind}`` (see :data:`AXIOM_KINDS`)."
        setattr(SubspaceGeometry, f"{kind}_loss", loss)


_per_kind_losses()


def e2r_parts(nb_classes, nb_relations, nb_individuals=0, embed_dim=100, top_index=None,
              bottom_index=None, role_unit_length=True, unit_length=False,
              regularization=True, tables=None):
    """The parts of E2R, as the keyword arguments of :class:`SubspaceGeometry` (see
    :func:`e2r`)."""
    if tables is not None:
        t = tables
        classes = SubspaceClasses(centres=t.c_embedding, top_index=t.top_index,
                                  bottom_index=t.bottom_index)
        relations = SubspaceRelations(vectors=t.r_embedding)
        individuals = PointIndividuals(centres=t.e_embedding)
        role_unit_length = t.role_unit_length
    else:
        d = embed_dim
        # entities first, as the earlier module drew its parameters
        individuals = PointIndividuals(max(nb_individuals, 1), d)
        classes = SubspaceClasses(nb_classes, d, top_index=top_index,
                                  bottom_index=bottom_index)
        relations = SubspaceRelations(max(nb_relations, 1), d)
    regularizers = (SubspaceStructure(), Orthogonality(), Separation()) if regularization \
        else ()
    return dict(classes=classes, relations=relations, measure=ResidualMeasure(unit_length),
                objective=SubspaceObjective(0.0, role_unit_length), individuals=individuals,
                regularizers=regularizers)


@register_method("e2r", trained_on="expressions")
def e2r(nb_classes, nb_relations, nb_individuals=0, embed_dim=100, margin=0.0,
        top_index=None, bottom_index=None, role_unit_length=True, unit_length=False,
        regularization=True, tables=None):
    """Embed2Reason (Garg et al. 2019): axis-parallel subspaces for classes and relations (of
    pairs), vectors for individuals, the quantum-logic operators of the subspace algebra, the
    residual of the projection as the measure, one equation of Sec. 4 per kind of axiom as the
    objective, and the structure terms and regularisers of eqs. (1)-(3), (11), (12).

    :param embed_dim: :math:`d`; the vectors have ``2 d`` real coordinates.
    :param margin: Must be 0: the losses of E2R have no margin, and a margin that no part
        reads is refused rather than ignored.
    :param role_unit_length: Unit-length terms in the loss of role assertions, see
        :class:`SubspaceObjective`.
    :param unit_length: Unit-length term in the cost of a point in a subspace
        (:class:`ResidualMeasure`); the equations of the objective have it in any case.
    :param regularization: Add :class:`SubspaceStructure`, :class:`Orthogonality` and
        :class:`Separation`; the last two need the pairs of the told hierarchy, which the
        model sets (:class:`~mowl.models.E2R`).
    :param tables: A :class:`~mowl.baselines.nn.alc.e2r.module.E2RModule` (the direct
        implementation) to share parameters with.
    """
    if margin:
        raise ValueError(f"E2R has no margin (its losses are zero exactly when the axiom "
                         f"holds), so margin must be 0; got {margin}.")
    return SubspaceGeometry(**e2r_parts(nb_classes, nb_relations, nb_individuals, embed_dim,
                                        top_index, bottom_index, role_unit_length, unit_length,
                                        regularization, tables))
