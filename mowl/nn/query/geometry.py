"""The geometries of the query embedding methods: how each represents an entity and a
query, and the four operators (projection, intersection, negation) on it.

Every geometry is a :class:`QueryGeometry`: ``anchor(entity_ids)`` gives the state of the
query that is one entity, ``project``, ``intersect`` and ``negate`` are the operators and
``logit(state, entity_ids)`` is the score (higher: more likely an answer). A *state* is a
tuple of tensors with a leading batch dimension. The generic evaluation of a query tree
(:meth:`QueryGeometry.embed`) does not depend on the geometry. Unions are not operators:
the models take the best branch of the disjunctive normal form
(:func:`mowl.nn.query.expression.to_dnf`).

Reference implementations followed (state, operators, distance):

* :class:`Query2BoxGeometry`: Ren, Hu, Leskovec (ICLR 2020), and the code of
  github.com/snap-stanford/KGReasoning (``geo="box"``).
* :class:`BetaEGeometry`: Ren, Leskovec (NeurIPS 2020), same code (``geo="beta"``).
* :class:`ConEGeometry`: Zhang et al. (NeurIPS 2021), github.com/MIRALab-USTC/QE-ConE.
* :class:`AConEGeometry`: He et al. (2024), github.com/RoyaHe/AConE.
"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from mowl.nn.query.expression import from_signature  # noqa: F401  (re-export convenience)

pi = math.pi

__all__ = ["QueryGeometry", "Query2BoxGeometry", "BetaEGeometry", "ConEGeometry",
           "AConEGeometry", "UnsupportedOperator"]


class UnsupportedOperator(ValueError):
    """The geometry has no such operator (Query2Box has no negation)."""


def _clamp_positive(x):
    return th.clamp(x + 1.0, 0.05, 1e9)


class QueryGeometry(nn.Module):
    """Base class.

    :param num_entities: Number of entities (individuals and named classes).
    :param num_relations: Number of relations (roles, their inverses, ``isa`` and its
        inverse).
    :param dim: Embedding dimension.
    :param gamma: Margin of the score, ``logit = gamma - distance``.
    """

    #: Whether :meth:`negate` is defined.
    supports_negation = False

    def __init__(self, num_entities, num_relations, dim, gamma):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.register_buffer("gamma", th.tensor(float(gamma)))
        #: Range of the uniform initialisation (both reference codes).
        self.embedding_range = (float(gamma) + 2.0) / dim

    # -- to be provided -----------------------------------------------------------------
    def anchor(self, entity_ids):
        raise NotImplementedError

    def project(self, state, relation_ids):
        raise NotImplementedError

    def intersect(self, states):
        """``states`` is a list of states (at least two)."""
        raise NotImplementedError

    def negate(self, state):
        raise UnsupportedOperator(f"{type(self).__name__} has no negation operator.")

    def distance(self, state, entity_ids):
        """Distance of entities ``(B, K)`` to the queries ``(B,)`` in ``state``: ``(B, K)``."""
        raise NotImplementedError

    def entity_vectors(self):
        """One flat vector per entity (for inspection), ``(num_entities, *)``."""
        raise NotImplementedError

    def relation_vectors(self):
        raise NotImplementedError

    # -- shared -------------------------------------------------------------------------
    def logit(self, state, entity_ids):
        return self.gamma - self.distance(state, entity_ids)

    def embed(self, signature, id_matrix, _cursor=None):
        """State of a batch of union-free queries of one shape.

        :param signature: From :func:`mowl.nn.query.expression.signature`.
        :param id_matrix: ``LongTensor (B, n)`` with the ids of
            :func:`mowl.nn.query.expression.ids`, one row per query.
        """
        cursor = [0] if _cursor is None else _cursor
        kind = signature[0]
        children = [self.embed(s, id_matrix, cursor) for s in signature[1:]]
        if kind in ("a", "p"):
            col = id_matrix[:, cursor[0]]
            cursor[0] += 1
        if kind == "a":
            return self.anchor(col)
        if kind == "p":
            return self.project(children[0], col)
        if kind == "i":
            return self.intersect(children)
        if kind == "n":
            return self.negate(children[0])
        raise ValueError("Unions are not operators: use the disjunctive normal form.")


# ---------------------------------------------------------------------------------------
# Query2Box
# ---------------------------------------------------------------------------------------

class Query2BoxGeometry(QueryGeometry):
    """Axis-aligned boxes ``(center, offset)``; entities are points (offset 0).

    Projection adds the relation's center and (activated) offset: the box is translated
    and grows. Intersection: the center is an attention-weighted average of the centers
    (weights from a one-layer network), the offset is the minimum of the offsets times a
    sigmoid gate (DeepSets). Score ``gamma - d_out - cen * d_in`` with the out-of-box and
    in-box L1 distances. No negation.

    :param center_weight: ``cen``, the weight of the in-box distance (0.02 in the paper).
    :param activation: ``"relu"``, ``"softplus"`` or ``"none"``, applied to the relation
        offsets (the reference code offers all three).
    :param clamp_extent: ``False`` (default, as in the reference code): the offset of a
        projected box is the sum of the offsets, which with ``activation="none"`` can become
        negative, a box with negative extent (the distances then still compute, but the
        region is not a box). ``True``: offsets are clamped at 0 after each projection, so
        every state is a box (a negative sum becomes a point). With ``"relu"`` and
        ``"softplus"`` the offsets are never negative and the option changes nothing.
    """

    def __init__(self, num_entities, num_relations, dim, gamma=24.0, center_weight=0.02,
                 activation="relu", clamp_extent=False):
        super().__init__(num_entities, num_relations, dim, gamma)
        self.clamp_extent = clamp_extent
        if activation not in ("relu", "softplus", "none"):
            raise ValueError("activation must be 'relu', 'softplus' or 'none'.")
        self.cen = center_weight
        self.activation = {"relu": F.relu, "softplus": F.softplus,
                           "none": lambda x: x}[activation]
        r = self.embedding_range
        self.entity_embedding = nn.Parameter(th.empty(num_entities, dim).uniform_(-r, r))
        self.relation_embedding = nn.Parameter(th.empty(num_relations, dim).uniform_(-r, r))
        self.offset_embedding = nn.Parameter(th.empty(num_relations, dim).uniform_(0, r))
        self.center_layer1 = nn.Linear(dim, dim)
        self.center_layer2 = nn.Linear(dim, dim)
        self.offset_layer1 = nn.Linear(dim, dim)
        self.offset_layer2 = nn.Linear(dim, dim)
        for layer in (self.center_layer1, self.center_layer2, self.offset_layer1,
                      self.offset_layer2):
            nn.init.xavier_uniform_(layer.weight)

    def anchor(self, entity_ids):
        c = self.entity_embedding[entity_ids]
        return c, th.zeros_like(c)

    def project(self, state, relation_ids):
        c, o = state
        o = o + self.activation(self.offset_embedding[relation_ids])
        if self.clamp_extent:
            o = F.relu(o)
        return c + self.relation_embedding[relation_ids], o

    def intersect(self, states):
        centers = th.stack([s[0] for s in states])
        offsets = th.stack([s[1] for s in states])
        attention = F.softmax(self.center_layer2(F.relu(self.center_layer1(centers))), dim=0)
        center = (attention * centers).sum(0)
        gate = th.sigmoid(self.offset_layer2(F.relu(self.offset_layer1(offsets)).mean(0)))
        return center, offsets.min(0)[0] * gate

    def distance(self, state, entity_ids):
        c, o = state
        e = self.entity_embedding[entity_ids]
        delta = (e - c.unsqueeze(1)).abs()
        o = o.unsqueeze(1)
        return F.relu(delta - o).sum(-1) + self.cen * th.min(delta, o).sum(-1)

    def entity_vectors(self):
        return self.entity_embedding.detach()

    def relation_vectors(self):
        return th.cat([self.relation_embedding, self.offset_embedding], -1).detach()


# ---------------------------------------------------------------------------------------
# BetaE
# ---------------------------------------------------------------------------------------

class BetaEGeometry(QueryGeometry):
    """Products of Beta distributions ``(alpha, beta)``, one factor per dimension.

    Entities are Beta distributions (parameters clamped to ``[0.05, 1e9]``). Projection is
    an MLP on the distribution parameters and the relation embedding. Intersection is the
    weighted average of the parameters with attention weights from an MLP. Negation
    inverts the parameters (``alpha, beta -> 1/alpha, 1/beta``): a density that is high
    where the original is low (Proposition 1: involution and idempotent intersection).
    The score is ``gamma`` minus the KL divergence of the entity's distribution from the
    query's, summed over dimensions.

    :param hidden: Width of the projection MLP (1600 in the reference code).
    :param layers: Number of hidden layers of the projection MLP (2 in the code).
    """

    supports_negation = True

    def __init__(self, num_entities, num_relations, dim, gamma=24.0, hidden=128, layers=1):
        super().__init__(num_entities, num_relations, dim, gamma)
        r = self.embedding_range
        self.entity_embedding = nn.Parameter(th.empty(num_entities, 2 * dim).uniform_(-r, r))
        self.relation_embedding = nn.Parameter(th.empty(num_relations, dim).uniform_(-r, r))
        self.proj_in = nn.Linear(3 * dim, hidden)
        self.proj_hidden = nn.ModuleList([nn.Linear(hidden, hidden)
                                          for _ in range(layers - 1)])
        self.proj_out = nn.Linear(hidden, 2 * dim)
        self.int_layer1 = nn.Linear(2 * dim, 2 * dim)
        self.int_layer2 = nn.Linear(2 * dim, dim)
        for layer in (self.proj_in, self.proj_out, self.int_layer1, self.int_layer2,
                      *self.proj_hidden):
            nn.init.xavier_uniform_(layer.weight)

    def anchor(self, entity_ids):
        e = _clamp_positive(self.entity_embedding[entity_ids])
        return e[..., :self.dim], e[..., self.dim:]

    def project(self, state, relation_ids):
        x = th.cat([state[0], state[1], self.relation_embedding[relation_ids]], -1)
        x = F.relu(self.proj_in(x))
        for layer in self.proj_hidden:
            x = F.relu(layer(x))
        x = _clamp_positive(self.proj_out(x))
        return x[..., :self.dim], x[..., self.dim:]

    def intersect(self, states):
        alpha = th.stack([s[0] for s in states])
        beta = th.stack([s[1] for s in states])
        attention = F.softmax(
            self.int_layer2(F.relu(self.int_layer1(th.cat([alpha, beta], -1)))), dim=0)
        return (attention * alpha).sum(0), (attention * beta).sum(0)

    def negate(self, state):
        return 1.0 / state[0], 1.0 / state[1]

    def distance(self, state, entity_ids):
        ea, eb = self.anchor(entity_ids)
        qa, qb = state[0].unsqueeze(1), state[1].unsqueeze(1)
        return _beta_kl(ea, eb, qa, qb).sum(-1)

    def entity_vectors(self):
        return _clamp_positive(self.entity_embedding.detach())

    def relation_vectors(self):
        return self.relation_embedding.detach()


def _beta_kl(a1, b1, a2, b2):
    """``KL(Beta(a1, b1) || Beta(a2, b2))``, elementwise."""
    lbeta = lambda a, b: th.lgamma(a) + th.lgamma(b) - th.lgamma(a + b)  # noqa: E731
    return (lbeta(a2, b2) - lbeta(a1, b1) + (a1 - a2) * th.digamma(a1)
            + (b1 - b2) * th.digamma(b1) + (a2 - a1 + b2 - b1) * th.digamma(a1 + b1))


# ---------------------------------------------------------------------------------------
# ConE and AConE
# ---------------------------------------------------------------------------------------

def _to_axis(x):
    return th.tanh(x) * pi


def _to_aperture(x):
    return th.tanh(2 * x) * pi / 2 + pi / 2


class ConEGeometry(QueryGeometry):
    """Cartesian products of two-dimensional sector-cones ``(axis, aperture)``.

    One sector per dimension: the axis angle in ``[-pi, pi]`` and the aperture in
    ``[0, pi]`` (the cone covers angles within ``aperture`` of the axis). Entities are
    sectors with aperture 0. Projection is an MLP on the sum of the (axis, aperture)
    embeddings and the relation's. Intersection: the axis is the angle of the
    attention-weighted average of the unit vectors of the axes (SemanticAverage), the
    aperture the minimum of the apertures times a gate (CardMin). Negation is the
    closure-complement: axis moved by ``pi``, aperture ``pi - aperture``. Distance:
    ``d_o + cen * d_i`` with the chord distance of the entity to the nearest boundary
    (``d_o``, 0 inside) and to the axis or the boundary (``d_i``); the score is
    ``gamma - modulus * distance`` with a learnable modulus.

    :param center_weight: ``cen``, the weight of the inside distance (0.02 in the code).
    :param hidden: Width of the projection MLP (1600 in the code, 2 layers).
    :param layers: Number of layers of the projection MLP.
    :param dropout: Dropout on the apertures in the intersection (0.2 in the code).
    :param signed_negation: ``False`` (default): the negation uses the absolute value of the
        aperture, clipped to ``pi`` (the cone of aperture ``a`` is the cone of ``|a|``), so
        the aperture of a complement is in ``[0, pi]`` also when a state has a negative
        aperture (possible in :class:`AConEGeometry`). ``True``: ``pi - aperture``, the
        behaviour of earlier mOWL versions (identical for apertures in ``[0, pi]``, as in
        ConE's own code); for a negative aperture it exceeds ``pi`` and the complement has
        the wrong boundaries.
    """

    supports_negation = True

    def __init__(self, num_entities, num_relations, dim, gamma=20.0, center_weight=0.02,
                 hidden=128, layers=1, dropout=0.0, signed_negation=False):
        super().__init__(num_entities, num_relations, dim, gamma)
        self.signed_negation = signed_negation
        self.cen = center_weight
        r = self.embedding_range
        self.entity_embedding = nn.Parameter(th.empty(num_entities, dim).uniform_(-r, r))
        self.axis_embedding = nn.Parameter(th.empty(num_relations, dim).uniform_(-r, r))
        self.aperture_embedding = nn.Parameter(th.empty(num_relations, dim).uniform_(-r, r))
        self.modulus = nn.Parameter(th.tensor(0.5 * r))
        self.axis_layer1 = nn.Linear(2 * dim, dim)
        self.aperture_layer1 = nn.Linear(2 * dim, dim)
        self.axis_layer2 = nn.Linear(dim, dim)
        self.aperture_layer2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.axis_layer1, self.aperture_layer1, self.axis_layer2,
                      self.aperture_layer2):
            nn.init.xavier_uniform_(layer.weight)
        self._make_projection(dim, hidden, layers)

    def _make_projection(self, dim, hidden, layers):
        self.proj_in = nn.Linear(2 * dim, hidden)
        self.proj_hidden = nn.ModuleList([nn.Linear(hidden, hidden)
                                          for _ in range(layers - 1)])
        self.proj_out = nn.Linear(hidden, 2 * dim)
        for layer in (self.proj_in, self.proj_out, *self.proj_hidden):
            nn.init.xavier_uniform_(layer.weight)

    def _angle(self, x):
        return x / self.embedding_range

    def _relation(self, relation_ids):
        return (_to_axis(self._angle(self.axis_embedding[relation_ids])),
                _to_axis(self._angle(self.aperture_embedding[relation_ids])))

    def anchor(self, entity_ids):
        axis = _to_axis(self._angle(self.entity_embedding[entity_ids]))
        return axis, th.zeros_like(axis)

    def project(self, state, relation_ids):
        axis, aperture = state
        r_axis, r_aperture = self._relation(relation_ids)
        x = th.cat([axis + r_axis, aperture + r_aperture], -1)
        x = F.relu(self.proj_in(x))
        for layer in self.proj_hidden:
            x = F.relu(layer(x))
        x = self.proj_out(x)
        return _to_axis(x[..., :self.dim]), _to_aperture(x[..., self.dim:])

    def intersect(self, states):
        axes = th.stack([s[0] for s in states])
        apertures = th.stack([s[1] for s in states])
        feats = th.cat([axes - apertures, axes + apertures], -1)
        attention = F.softmax(self.axis_layer2(F.relu(self.axis_layer1(feats))), dim=0)
        x = (attention * th.cos(axes)).sum(0)
        y = (attention * th.sin(axes)).sum(0)
        axis = th.atan2(y, x)
        gate = th.sigmoid(self.aperture_layer2(F.relu(self.aperture_layer1(feats)).mean(0)))
        aperture = self.dropout(apertures).min(0)[0] * gate
        return axis, aperture

    def negate(self, state):
        axis, aperture = state
        axis = th.where(axis >= 0, axis - pi, axis + pi)
        if self.signed_negation:
            return axis, pi - aperture
        return axis, pi - aperture.abs().clamp(max=pi)

    def distance(self, state, entity_ids):
        axis, aperture = state
        axis, aperture = axis.unsqueeze(1), aperture.unsqueeze(1)
        e = _to_axis(self._angle(self.entity_embedding[entity_ids]))
        lower, upper = axis - aperture, axis + aperture
        to_axis = th.abs(th.sin((e - axis) / 2))
        base = th.abs(th.sin(aperture / 2))
        outside = th.min(th.abs(th.sin((e - lower) / 2)), th.abs(th.sin((e - upper) / 2)))
        outside = th.where(to_axis < base, th.zeros_like(outside), outside)
        inside = th.min(to_axis, base)
        return self.modulus * (outside.sum(-1) + self.cen * inside.sum(-1))

    def entity_vectors(self):
        return _to_axis(self._angle(self.entity_embedding.detach()))

    def relation_vectors(self):
        return th.cat(self._relation(th.arange(self.num_relations,
                                               device=self.axis_embedding.device)),
                      -1).detach()


class AConEGeometry(ConEGeometry):
    """ConE with relations as rotations (AConE, aperture-additive).

    A relation is a pair of angles per dimension; projection adds them to the axis and to
    the aperture of the query cone (multiplication of unit complex numbers, wrapped to
    ``(-pi, pi]``), so ``r`` rotates the cone and widens or narrows it. There is no
    projection network, which is what makes the relation patterns of the paper
    (composition, inversion, hierarchy: Propositions 2-5, Corollaries 6-7) readable from
    the angles. The other operators and the score are those of ConE (the reference code
    shares them).

    :param aperture: How the sum of apertures is kept an angle. ``"wrap"`` (default): wrapped
        to ``(-pi, pi]`` like the axis, so an aperture can be negative (a cone of aperture
        ``a`` is the cone of ``|a|``) and a sum above ``pi`` comes back small; this is
        mOWL's reading of the complex-number multiplication and could not be checked
        against the authors' code (the public repository has the ConE projection network
        only). ``"clip"``: clipped to ``[0, pi]``, the range of the aperture of every other
        state: a negative sum is a point, a sum above ``pi`` the whole circle.
    """

    def __init__(self, *args, aperture="wrap", **kwargs):
        if aperture not in ("wrap", "clip"):
            raise ValueError("aperture must be 'wrap' or 'clip'.")
        super().__init__(*args, **kwargs)
        self.aperture_mode = aperture

    def _make_projection(self, dim, hidden, layers):
        pass

    def project(self, state, relation_ids):
        axis, aperture = state
        r_axis, r_aperture = self._relation(relation_ids)
        if self.aperture_mode == "clip":
            return _wrap(axis + r_axis), (aperture + r_aperture).clamp(0.0, pi)
        return _wrap(axis + r_axis), _wrap(aperture + r_aperture)


def _wrap(angle):
    """Angle in ``(-pi, pi]`` of the unit complex number ``exp(i * angle)``."""
    return th.atan2(th.sin(angle), th.cos(angle))
