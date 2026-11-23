"""Fuzzy and logit semantics for full class expressions: the parts of DF-ALC, FALCON and
Al-cone.

The three methods share one idea: a class is a *graded* subset of a finite set of elements
(a fuzzy set, or its logit), and the constructors are the operators of a fuzzy logic. What
differs is independent of that:

=================  ==================  =======================  ==========================
part               DF-ALC              FALCON                   Al-cone
=================  ==================  =======================  ==========================
elements           domain elements     individuals (+ anon.)    half-axes of :math:`R^n`
class model        truth table         membership network       half-axis logits
relation model     truth matrix        network over pairs       incidence logits
operator family    Goedel              product (or others)      Goedel on logits
measure            mean violation      conflict (``-log``)      either-or hinge
=================  ==================  =======================  ==========================

Each cell is a small class below. They are combined by
:class:`~mowl.representation.library.logic_alc.ExpressionGeometry` (a
:class:`~mowl.representation.ComposedGeometry`; the region family gives it negation and
disjunction, and it reads axioms with class expressions of any depth through
:class:`FuzzyObjective`); the three methods are the functions :func:`dfalc`, :func:`falcon` and
:func:`alcone` (registered in :data:`mowl.representation.METHODS`; the classes
:class:`~mowl.models.DFALCModel`, :class:`~mowl.models.FALCONModel` and
:class:`~mowl.models.ALConeModel` are these combinations), and
:func:`~mowl.representation.geometry_of` builds them over the tables of the direct
implementations (:mod:`mowl.baselines`). The operator family is a part of its own
(:data:`FAMILIES`): every method can be built with every family, and the class models,
relation models and measures can be exchanged (see the tests).

Regions are :class:`FuzzySet` objects: degrees (or logits) per element, together with the
family that produced them, so that conjunction and the other constructors need no further
argument. A degree is a number in ``[0, 1]`` for the families of fuzzy logic, a real number
for the logit family.
"""

from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn

from mowl.owlapi import ClassExpressionType
from mowl.representation.class_models import Classes
from mowl.representation.library.logic_alc import (
    AnnealedRelaxation, ExpressionGeometry, ExpressionReading, Relaxation)
from mowl.representation.library.logic_expressions import pattern_region
from mowl.representation.measures import RegionMeasure
from mowl.representation.regions import MarginUnavailable
from mowl.representation.methods import (register_class_model, register_measure,
                                         register_method, register_relation_model)
from mowl.representation.objectives import Objective
from mowl.representation.regions import ConstructorUnavailable, Region
from mowl.representation.relation_models import Relations
from mowl.representation.vocabulary import ClassModel, Construction, RelationModel
from mowl.representation.adapters import register_adapter
from mowl.training.regularization import Regularizer

__all__ = ["FuzzySet", "LogitSet", "Family", "Product", "Goedel", "Lukasiewicz", "LogitGoedel",
           "FAMILIES", "get_family", "register_family", "TruthTableClasses",
           "HalfAxisClasses", "NetworkClasses", "DomainElements", "MembershipNetwork",
           "Domain", "TruthMatrix", "IncidenceMatrices", "PairNetworkRelations",
           "MeanViolation", "ConflictMeasure", "ConeMeasure", "FuzzyObjective",
           "NonDegeneracy", "empty_cone_penalty", "rule_based_loss", "rule_nf4", "rule_nf5", "rule_nf6", "rule_nf7",
           "dfalc", "falcon", "alcone"]

#: Logit of the constants owl:Thing and owl:Nothing of the Al-cone module.
DEFAULT_CONSTANT_LOGIT = 1.0


# =========================================================================================
# the rule-based losses of DF-ALC (Wu et al. 2022, Eqs. 9-13)
# =========================================================================================

def _g(v, w):
    """G(v, w) = ReLU(v - w), excluded from the gradient (paper, after Eq. 9)."""
    return th.relu(v - w).detach()


def rule_based_loss(c, d):
    """Eq. 9 for an inclusion :math:`C \\sqsubseteq D` without quantifier:
    :math:`\\sum_s (1 - D(s)) G(C(s), D(s))`, averaged over the domain (one value per row);
    ``c``, ``d``: ``(B, N)``."""
    return ((1 - d) * th.relu(c - d).detach()).mean(dim=-1)


def rule_nf4(b, a, r, alpha):
    """Eq. 10 for :math:`B \\sqsubseteq \\exists r.A` (``alpha`` is :math:`\\alpha'`), per row.
    Product t-norm. ``b``, ``a``: ``(B, n)``; ``r``: ``(B, n, n)`` with ``r[:, s, a] =
    r(s, a)``. First term: where :math:`A(s)` is below the threshold but some :math:`a` has
    :math:`\\sum_a B(a) r(a, s)` above it, raise :math:`A(s)`. Second term (summed over
    :math:`a`, read from the garbled index structure of the printed formula): raise
    :math:`r(s, a)` when :math:`B(s) A(a)` exceeds it."""
    pred = th.einsum("ba,bas->bs", b, r)
    t1 = (1 - a) * _g(alpha, a) * _g(pred, alpha)
    t2 = ((1 - r) * _g(b.unsqueeze(-1) * a.unsqueeze(1), r)).mean(dim=-1)
    return (t1 + t2).mean(dim=-1)


def rule_nf5(b, a, r, alpha):
    """Eq. 11 for :math:`B \\sqsubseteq \\forall r.A`: the first term of NF4 with threshold
    :math:`\\alpha`."""
    pred = th.einsum("ba,bas->bs", b, r)
    return ((1 - a) * _g(alpha, a) * _g(pred, alpha)).mean(dim=-1)


def rule_nf6(a, b, r, alpha):
    """Eq. 12 for :math:`\\exists r.A \\sqsubseteq B`: raise :math:`B(s)` where it is below the
    threshold and :math:`\\sum_a A(a) r(s, a)` is above it."""
    succ = th.einsum("ba,bsa->bs", a, r)
    return ((1 - b) * _g(alpha, b) * _g(succ, alpha)).mean(dim=-1)


def rule_nf7(a, b, r, alpha):
    """Eq. 13 for :math:`\\forall r.A \\sqsubseteq B`: the NF6 term plus the NF4/NF5 term
    raising :math:`A(s)` from :math:`\\sum_a B(a) r(a, s)`."""
    succ = th.einsum("ba,bsa->bs", a, r)
    pred = th.einsum("ba,bas->bs", b, r)
    t = (1 - b) * _g(alpha, b) * _g(succ, alpha) + (1 - a) * _g(alpha, a) * _g(pred, alpha)
    return t.mean(dim=-1)


_QUANTIFIERS = (ClassExpressionType.OBJECT_SOME_VALUES_FROM,
                ClassExpressionType.OBJECT_ALL_VALUES_FROM)


def has_quantifier(cexpr):
    """Whether the class expression contains ``∃`` or ``∀``."""
    t = cexpr.getClassExpressionType()
    if t in _QUANTIFIERS:
        return True
    if t in (ClassExpressionType.OBJECT_INTERSECTION_OF, ClassExpressionType.OBJECT_UNION_OF):
        return any(has_quantifier(o) for o in cexpr.getOperandsAsList())
    if t == ClassExpressionType.OBJECT_COMPLEMENT_OF:
        return has_quantifier(cexpr.getOperand())
    return False


def _is_literal(e):
    t = e.getClassExpressionType()
    if t == ClassExpressionType.OBJECT_COMPLEMENT_OF:
        return e.getOperand().getClassExpressionType() == ClassExpressionType.OWL_CLASS
    return t == ClassExpressionType.OWL_CLASS


class _CL:
    """The operators of Al-cone on logits: smooth maximum and minimum with a temperature,
    and the either-or hinge of its losses."""

    @staticmethod
    def smax(x, y, tau):
        if tau <= 0:
            return th.maximum(x, y)
        return tau * th.logaddexp(x / tau, y / tau)

    @staticmethod
    def godel_and(x, y, tau=0.0):
        return -_CL.smax(-x, -y, tau)

    @staticmethod
    def godel_or(x, y, tau=0.0):
        return _CL.smax(x, y, tau)

    @staticmethod
    def reduce_max(x, dim, tau):
        if tau <= 0:
            return x.max(dim=dim)[0]
        return tau * th.logsumexp(x / tau, dim=dim)

    @staticmethod
    def either(x, y, margin):
        return th.relu(margin - x) * th.relu(margin - y)


CL = _CL


# =========================================================================================
# regions
# =========================================================================================

@dataclass(frozen=True)
class FuzzySet(Region):
    """Degrees of membership of the elements of a finite domain.

    :param values: ``(B, N)``.
    :param family: The :class:`Family` whose operators apply to these values.
    :param index: For the crisp region of one domain element (an individual): its position
        in the domain, ``(B,)``; ``None`` otherwise.
    """

    values: th.Tensor
    family: "Family"
    index: th.Tensor = None

    def __len__(self):
        return self.values.shape[0]

    def intersect(self, other):
        # A fuzzy set is empty only if every degree is 0; the intersection is a region of
        # the same kind, so the gap is zero.
        out = FuzzySet(self.family.and_(self.values, other.values), self.family)
        return out, th.zeros(out.values.shape[0], 1, device=out.values.device)

    def complement(self):
        return FuzzySet(self.family.not_(self.values), self.family)

    def union(self, other):
        return FuzzySet(self.family.or_(self.values, other.values), self.family)


@dataclass(frozen=True)
class LogitSet(FuzzySet):
    """A fuzzy set given by logits (the half-axis cones of Al-cone): an element belongs to
    the set iff its logit is positive. Its own region class, so that its default measure
    (:class:`ConeMeasure`) is not the measure of degrees in ``[0, 1]``."""


# =========================================================================================
# the operator family
# =========================================================================================

class Family:
    """The operators of a fuzzy logic: conjunction, disjunction, negation, implication, and
    the supremum and infimum over the domain (existential and universal restriction).

    :cvar part: The vocabulary entry of the construction.
    """

    part = Construction.FUZZY.value
    name = ""
    #: Whether the values are logits (real numbers, an element belongs iff its logit is
    #: positive) rather than degrees in ``[0, 1]``.
    logits = False

    def and_(self, x, y):
        raise NotImplementedError

    def or_(self, x, y):
        raise NotImplementedError

    def not_(self, x):
        raise NotImplementedError

    def implies(self, x, y):
        """The S-implication :math:`\\neg x \\sqcup y` (``notCorD`` of FALCON, and the
        implication of DF-ALC and Al-cone)."""
        return self.or_(self.not_(x), y)

    def sup(self, x, dim):
        return x.amax(dim=dim)

    def inf(self, x, dim):
        return x.amin(dim=dim)

    def exists(self, relation, fuzzy_set):
        """:math:`\\sup_b r(a, b) \\sqcap C(b)` for ``relation`` ``(B, N, N)`` (row: source)
        and ``fuzzy_set`` ``(B, N)``."""
        return self.sup(self.and_(relation, fuzzy_set.unsqueeze(-2)), -1)

    def forall(self, relation, fuzzy_set):
        """:math:`\\inf_b r(a, b) \\to C(b)`."""
        return self.inf(self.implies(relation, fuzzy_set.unsqueeze(-2)), -1)


class Product(Family):
    """Product t-norm."""
    name = "product"

    def and_(self, x, y):
        return x * y

    def or_(self, x, y):
        return x + y - x * y

    def not_(self, x):
        return 1 - x


class Goedel(Family):
    """Goedel (minimum / maximum) operators."""
    name = "godel"

    def and_(self, x, y):
        return th.minimum(x, y)

    def or_(self, x, y):
        return th.maximum(x, y)

    def not_(self, x):
        return 1 - x


class Lukasiewicz(Family):
    """Lukasiewicz t-norm."""
    name = "lukasiewicz"

    def and_(self, x, y):
        return th.relu(x + y - 1)

    def or_(self, x, y):
        return th.clamp(x + y, max=1.0)

    def not_(self, x):
        return 1 - x


class LogitGoedel(Family):
    """Goedel operators on *logits* (Al-cone): minimum, maximum, negation of the sign. With
    ``tau > 0`` the smooth bounds (log-sum-exp) are used.

    :param tau: Temperature, or ``source`` (an object with a ``temperature`` attribute that
        is read at every call, as the Al-cone module's is changed during training).
    """

    name = "godel (logits)"
    logits = True

    def __init__(self, tau=0.0, source=None):
        self._tau = tau
        self.source = source

    @property
    def tau(self):
        return self.source.temperature if self.source is not None else self._tau

    def and_(self, x, y):
        return CL.godel_and(x, y, self.tau)

    def or_(self, x, y):
        return CL.godel_or(x, y, self.tau)

    def not_(self, x):
        return -x

    def sup(self, x, dim):
        return CL.reduce_max(x, dim, self.tau)

    def inf(self, x, dim):
        return -CL.reduce_max(-x, dim, self.tau)


#: Operator families by name.
FAMILIES = {"product": Product, "godel": Goedel, "lukasiewicz": Lukasiewicz,
            "godel (logits)": LogitGoedel}
_ALIASES = {"minmax": "godel", "Łukasiewicz": "lukasiewicz", "goedel": "godel"}


def register_family(name):
    """Decorator: make an operator family available under ``name``."""
    def decorate(factory):
        FAMILIES[name] = factory
        return factory
    return decorate


def get_family(family):
    """A family object from a name (``"product"``, ``"godel"``, ``"lukasiewicz"``, the
    FALCON names ``"minmax"`` and ``"Łukasiewicz"``), a class, or an object."""
    if isinstance(family, str):
        key = _ALIASES.get(family, family)
        if key not in FAMILIES:
            raise ValueError(f"Unknown fuzzy operator family {family!r}; "
                             f"available: {sorted(FAMILIES)}.")
        return FAMILIES[key]()
    if isinstance(family, type):
        return family()
    return family


# =========================================================================================
# class models
# =========================================================================================

def _constants(values, ids, top_index, bottom_index, top, bottom):
    if top_index is not None:
        values = th.where((ids == top_index).unsqueeze(-1), th.full_like(values, top), values)
    if bottom_index is not None:
        values = th.where((ids == bottom_index).unsqueeze(-1),
                          th.full_like(values, bottom), values)
    return values


def _logit_of_degrees(values, eps):
    """``logit(clip(values, eps, 1 - eps))`` as a float tensor: the parameters that give the
    truth values ``values``."""
    degrees = th.as_tensor(np.asarray(values, dtype="float32")).clamp(eps, 1 - eps)
    return th.log(degrees / (1 - degrees))


class _FuzzyClasses:
    """Class models whose regions are fuzzy sets: the operators of the family do conjunction,
    disjunction and negation."""

    conjunction_part = "fuzzy operator"

    @property
    def extra_parts(self):
        return {"negation": "fuzzy operator", "disjunction": "fuzzy operator",
                "family": type(self.family).__name__}

    def degree_at(self, ids, elements):
        """The degree of the domain elements ``elements`` in the classes ``ids`` (any shape,
        both the same): the entries of :meth:`forward` that are asked for. A class model
        that can compute them without the whole domain does so."""
        shape = ids.shape
        values = self.forward(ids.reshape(-1)).values
        return values.gather(1, elements.reshape(-1, 1)).reshape(shape)


class TruthTableClasses(_FuzzyClasses, Classes):
    """A class is a vector of truth values over a finite domain, ``sigmoid(logit)`` per
    element (DF-ALC). ``owl:Thing`` and ``owl:Nothing`` are the constants ``1`` and ``0``.

    :param size: Number of classes.
    :param domain: Size of the domain.
    :param logits: An existing table ``(size, domain)`` to share.
    :param family: Name or object of the operator family (default Goedel).
    """

    region = FuzzySet
    part = ClassModel.FUZZY_SET.value
    exact_conjunction = True

    @property
    def domain_size(self):
        """The number of elements of the domain."""
        return self.centres.embedding_dim

    def __init__(self, size=None, domain=None, logits=None, family="godel", top_index=None,
                 bottom_index=None, init_std=1.0):
        super().__init__()
        if logits is None:
            logits = nn.Embedding(size, domain)
            nn.init.normal_(logits.weight, std=init_std)
        self.centres = logits
        self.family = get_family(family)
        self.top_index, self.bottom_index = top_index, bottom_index

    def forward(self, ids):
        values = th.sigmoid(self.centres(ids))
        values = _constants(values, ids, self.top_index, self.bottom_index, 1.0, 0.0)
        return FuzzySet(values, self.family)

    def set_degrees(self, index, values, eps=1e-3, name=None):
        """Start class ``index`` at the truth values ``values`` (array over the domain,
        in ``[0, 1]``; for example a perceptual grounding): they are clipped to
        ``[eps, 1 - eps]`` and stored as logits, without a gradient."""
        logits = _logit_of_degrees(values, eps)
        if logits.shape != (self.domain_size,):
            raise ValueError(f"Grounding of {name or index} must have shape "
                             f"({self.domain_size},).")
        with th.no_grad():
            self.centres.weight[index] = logits.to(self.centres.weight.device)


class HalfAxisClasses(_FuzzyClasses, Classes):
    """A class is a set of half-axes of :math:`\\mathbb{R}^n` (an axis-aligned cone), given
    by one logit per half-axis; ``> 0`` means the half-axis belongs to the cone (Al-cone).
    ``owl:Thing`` and ``owl:Nothing`` are the constants :math:`\\pm c` (``constant_logit``,
    default 1; the earlier mOWL module used :math:`10^4`). Also used for the
    individuals of Al-cone (the support of the point).

    :param size: Number of classes; :param dim: :math:`n` (there are ``2 n`` logits).
    :param constant_logit: Magnitude of the logits of ``owl:Thing`` (``+``) and
        ``owl:Nothing`` (``-``); it has to be at least the margin, and a large value scales
        every loss that involves them. Default ``1.0``; the earlier mOWL module used ``1e4``.

    Part (vocabulary value): ``axis-aligned cone``. Used by: ``alcone``.
    """

    region = LogitSet
    part = ClassModel.AXIS_CONE.value
    exact_conjunction = True

    def __init__(self, size=None, dim=None, logits=None, family=None, top_index=None,
                 bottom_index=None, init_std=0.5, constant_logit=DEFAULT_CONSTANT_LOGIT):
        super().__init__()
        self.constant_logit = float(constant_logit)
        if logits is None:
            logits = nn.Embedding(size, 2 * dim)
            nn.init.normal_(logits.weight, std=init_std)
        self.centres = logits
        self.family = LogitGoedel() if family is None else get_family(family)
        self.top_index, self.bottom_index = top_index, bottom_index

    def forward(self, ids):
        values = _constants(self.centres(ids), ids, self.top_index, self.bottom_index,
                            self.constant_logit, -self.constant_logit)
        return LogitSet(values, self.family)

    def crisp(self):
        """The discretised cones: a boolean array ``(size, 2 n)``, half-axis ``h`` of row
        ``i`` belongs to the cone iff its logit is ``> 0``."""
        with th.no_grad():
            return self.centres.weight.detach().cpu().numpy() > 0

    def points(self):
        """The point of :math:`\\mathbb{R}^n` whose support every row is (the individuals of
        Al-cone): coordinate ``i`` is ``+1`` if the logit of the positive half-axis is
        ``> 0`` and not below that of the negative one, ``-1`` if only the negative one is
        ``> 0`` or it is the larger, ``0`` if neither is (an array ``(size, n)``)."""
        with th.no_grad():
            pairs = self.centres.weight.detach().cpu().numpy().reshape(
                self.centres.weight.shape[0], -1, 2)
        positive, negative = pairs[..., 0], pairs[..., 1]
        plus = (positive > 0) & ((negative <= 0) | (positive >= negative))
        minus = ~plus & (negative > 0)
        return plus.astype(float) - minus.astype(float)


class MembershipNetwork(nn.Module):
    """The two-layer network of FALCON: ``(concept vector, entity vector) -> degree``.
    Pass the layers of an existing module to share them."""

    def __init__(self, embed_dim=None, fc_0=None, fc_1=None):
        super().__init__()
        if fc_0 is None:
            fc_0, fc_1 = nn.Linear(2 * embed_dim, embed_dim), nn.Linear(embed_dim, 1)
            nn.init.xavier_uniform_(fc_0.weight)
            nn.init.xavier_uniform_(fc_1.weight)
        self.fc_0, self.fc_1 = fc_0, fc_1

    def forward(self, concept, entity):
        hidden = nn.functional.leaky_relu(
            self.fc_0(th.cat([concept, entity], dim=-1)), negative_slope=0.1)
        return th.sigmoid(self.fc_1(hidden)).squeeze(-1)


class Domain(nn.Module):
    """The finite domain of a method with a learned vector per element: the table of the
    named individuals, plus ``extra`` anonymous elements (learned rows) when given, or the
    anonymous elements that a training loop samples for one evaluation (``context``)."""

    #: Elements set from the outside for one evaluation (the sampled anonymous entities of
    #: FALCON: the named elements followed by the sample), or ``None``. When set, they *are*
    #: the domain.
    context = None

    def __init__(self, table, extra=0):
        super().__init__()
        self.table = table
        self.extra = None
        if extra:
            self.extra = nn.Parameter(nn.init.xavier_uniform_(
                th.empty(extra, table.embedding_dim)))

    def forward(self):
        if self.context is not None:
            return self.context
        weight = self.table.weight
        return weight if self.extra is None else th.cat([weight, self.extra], dim=0)

    def vectors(self, ids):
        """The vectors of the named elements ``ids`` (whatever the context)."""
        return self.table(ids)

    def __len__(self):
        return self.table.num_embeddings + (0 if self.extra is None else len(self.extra))

    def size(self):
        """The number of elements of the domain now (with the context, if one is set)."""
        return len(self) if self.context is None else self.context.shape[0]


class NetworkClasses(_FuzzyClasses, Classes):
    """A class is a vector; its fuzzy set is the degree the network gives to the pair of
    the class vector and each element of the domain (FALCON).

    :param size: Number of classes; :param embed_dim: Dimension of the vectors.
    :param centres: An existing table to share.
    :param network: A :class:`MembershipNetwork`.
    :param domain: A :class:`Domain`.
    """

    region = FuzzySet
    part = ClassModel.FUZZY_SET.value
    exact_conjunction = True

    def __init__(self, size=None, embed_dim=None, centres=None, network=None, domain=None,
                 family="product", top_index=None, bottom_index=None):
        super().__init__()
        if centres is None:
            centres = nn.Embedding(size, embed_dim)
            nn.init.xavier_uniform_(centres.weight)
        self.centres = centres
        self.network = network
        self.domain = domain
        self.family = get_family(family)
        self.top_index, self.bottom_index = top_index, bottom_index

    def forward(self, ids):
        c = self.centres(ids)
        e = self.domain()
        values = self.network(c.unsqueeze(1).expand(-1, e.shape[0], -1),
                              e.unsqueeze(0).expand(c.shape[0], -1, -1))
        values = _constants(values, ids, self.top_index, self.bottom_index, 1.0, 0.0)
        return FuzzySet(values, self.family)

    def degree_at(self, ids, elements):
        """The network on the pairs (class vector, vector of the named element), without the
        rest of the domain; ``owl:Thing`` and ``owl:Nothing`` are the constants where they
        are."""
        degree = self.network(self.centres(ids), self.domain.vectors(elements))
        return _constants(degree.unsqueeze(-1), ids, self.top_index, self.bottom_index,
                          1.0, 0.0).squeeze(-1)


class DomainElements(_FuzzyClasses, Classes):
    """Individuals that are the elements of the domain: the crisp set of element ``i``
    (a one-hot vector). The table is fixed.

    :param size: Number of individuals; :param domain: Size of the domain (at least
        ``size``).
    :param elements: The :class:`Domain` of a geometry whose domain changes (FALCON: the
        named elements followed by sampled anonymous ones); individual ``i`` is then the
        crisp set of element ``i`` of the domain as it is at the call, so that it can be
        compared with the class expressions evaluated on it.
    """

    region = FuzzySet
    part = ClassModel.FUZZY_SET.value
    exact_conjunction = True

    def __init__(self, size, domain, family="godel", elements=None):
        super().__init__()
        if domain < size:
            raise ValueError(f"The domain has {domain} elements for {size} individuals.")
        self.centres = nn.Embedding.from_pretrained(th.eye(size, domain), freeze=True)
        self.family = get_family(family)
        self.elements = elements

    def forward(self, ids):
        values = self.centres(ids)
        if self.elements is not None:
            extra = self.elements.size() - values.shape[-1]
            if extra > 0:
                values = nn.functional.pad(values, (0, extra))
        return FuzzySet(values, self.family, index=ids)


# =========================================================================================
# relation models
# =========================================================================================

class _FuzzyRelations(Relations):
    """Relations as fuzzy relations on the domain: a table ``(N, N)`` per relation, the
    element ``[a, b]`` being the degree of ``r(a, b)``. :meth:`image` is
    :math:`\\sup_a x(a) \\sqcap r(a, b)`, :meth:`preimage` the existential restriction."""

    part = RelationModel.FUZZY_RELATION.value
    accepts = (FuzzySet,)

    def __init__(self, size=None, domain=None, vectors=None, init_std=1.0):
        super().__init__()
        if vectors is None:
            vectors = nn.Embedding(size, domain * domain)
            nn.init.normal_(vectors.weight, std=init_std)
        self.vectors = vectors
        self.domain = domain if domain is not None else int(round(
            vectors.embedding_dim ** 0.5))

    def matrix(self, ids):
        """The tables of the relations ``ids``: ``(B, N, N)``, source first."""
        raise NotImplementedError

    def _domain_of(self, region):
        if not isinstance(region, FuzzySet) or region.values.shape[-1] != self.domain:
            size = region.values.shape[-1] if isinstance(region, FuzzySet) else None
            raise ConstructorUnavailable(
                f"{type(self).__name__} relates the {self.domain} elements of its domain; "
                f"the region is a {type(region).__name__} over {size} elements.")
        return region.family

    def image(self, ids, region):
        family = self._domain_of(region)
        return FuzzySet(family.sup(family.and_(self.matrix(ids), region.values.unsqueeze(-1)),
                                   -2), family)

    def preimage(self, ids, region):
        family = self._domain_of(region)
        return FuzzySet(family.exists(self.matrix(ids), region.values), family)


class TruthMatrix(_FuzzyRelations):
    """The truth values of the pairs, ``sigmoid(logit)`` (DF-ALC).

    Part (vocabulary value): ``fuzzy relation``. Used by: ``dfalc``.
    """

    def matrix(self, ids):
        n = self.domain
        return th.sigmoid(self.vectors(ids)).view(*ids.shape, n, n)

    def set_degrees(self, index, values, eps=1e-3, name=None):
        """Start relation ``index`` at the truth values ``values`` (a ``(domain, domain)``
        array, source first, in ``[0, 1]``): clipped to ``[eps, 1 - eps]`` and stored as
        logits, without a gradient."""
        n = self.domain
        logits = _logit_of_degrees(values, eps)
        if logits.shape != (n, n):
            raise ValueError(f"Grounding of {name or index} must be {n} x {n}.")
        with th.no_grad():
            self.vectors.weight[index] = logits.reshape(-1).to(self.vectors.weight.device)


class IncidenceMatrices(_FuzzyRelations):
    """Logits of the incidence matrix of a relation between half-axes (Al-cone): entry
    ``[j, i]`` of the stored matrix is the logit of "half-axis ``i`` has successor ``j``",
    so the table of this part (source first) is its transpose.

    :param size: Number of relations; :param dim: :math:`n` (the matrix is
        ``2 n x 2 n``).

    Part (vocabulary value): ``fuzzy relation``. Used by: ``alcone``.
    """

    def __init__(self, size=None, dim=None, vectors=None, init_std=0.5):
        super().__init__(size, None if dim is None else 2 * dim, vectors, init_std)

    def matrix(self, ids):
        return self.stored(ids).transpose(-1, -2)

    def stored(self, ids):
        """The incidence logits of the relations ``ids`` in the stored layout, entry
        ``[j, i]`` for "half-axis ``i`` has successor ``j``": ``ids.shape + (2n, 2n)``."""
        n = self.domain
        return self.vectors(ids).view(*ids.shape, n, n)

    def crisp(self):
        """The discretised incidence matrices, stored layout: a boolean array
        ``(size, 2 n, 2 n)``, an entry is in the relation iff its logit is ``> 0``."""
        n = self.domain
        with th.no_grad():
            return self.vectors.weight.detach().cpu().numpy().reshape(-1, n, n) > 0

    # The operators run over the stored layout (``[successor, source]``) instead of the
    # transposed table: the same numbers, summed in the order of the table the optimiser
    # holds (the smooth relaxation sums in floating point, so the order is visible).
    def image(self, ids, region):
        family = self._domain_of(region)
        stored = self.stored(ids)
        return FuzzySet(family.sup(family.and_(
            stored, region.values.unsqueeze(-2).expand_as(stored)), -1), family)

    def preimage(self, ids, region):
        family = self._domain_of(region)
        stored = self.stored(ids)
        return FuzzySet(family.sup(family.and_(
            stored, region.values.unsqueeze(-1).expand_as(stored)), -2), family)


class PairNetworkRelations(Relations):
    """A relation is a vector; the degree of ``r(a, b)`` is the network applied to
    ``(vector(a) + r, vector(b))`` (FALCON).

    :param role_pairs: Which pairs :math:`\\exists r.C` and :math:`\\forall r.C` range over.
        ``"all"`` (default, as published and as :class:`~mowl.nn.FALCONModule` does by
        default): the degree of element ``a`` is :math:`\\sup_b \\mathrm{net}(e_a + r, e_b)
        \\sqcap C(b)`, over all pairs of the domain (memory quadratic in the domain).
        ``"self"``: :math:`\\sup_b \\mathrm{net}(e_b + r, e_b) \\sqcap C(b)`, the pair of an
        element with itself, one value for every element of the domain. That was a porting
        error of the earlier mOWL module (``role_pairs="self"``); the published method does
        not do it.
    """

    part = RelationModel.NETWORK.value
    accepts = (FuzzySet,)

    def __init__(self, size=None, embed_dim=None, vectors=None, network=None, domain=None,
                 role_pairs="all"):
        super().__init__()
        if role_pairs not in ("all", "self"):
            raise ValueError("role_pairs must be 'all' or 'self'.")
        if vectors is None:
            vectors = nn.Embedding(size, embed_dim)
            nn.init.xavier_uniform_(vectors.weight)
        self.vectors = vectors
        self.network = network
        self.domain = domain
        self.role_pairs = role_pairs

    def degree_at(self, ids, heads, tails):
        """The degree of the pairs of named elements ``(heads, tails)`` in the relations
        ``ids`` (all of one shape): the network on ``(vector(head) + r, vector(tail))``,
        without the rest of the domain."""
        return self.network(self.domain.vectors(heads) + self.vectors(ids),
                            self.domain.vectors(tails))

    def head_degrees(self, head_vectors):
        """The degree of the pairs of every head vector ``(M, D)`` with every element of the
        domain, ``(M, N)``: the network on ``(head, element)``."""
        e = self.domain()
        return self.network(head_vectors.unsqueeze(1).expand(-1, e.shape[0], -1),
                            e.unsqueeze(0).expand(head_vectors.shape[0], -1, -1))

    def _pairs(self, ids):
        """``(B, N, N)``: degree of ``r(x, y)`` for all pairs of the domain."""
        r = self.vectors(ids)
        e = self.domain()
        n = e.shape[0]
        head = (e.unsqueeze(0) + r.unsqueeze(1)).unsqueeze(2).expand(-1, -1, n, -1)
        tail = e.unsqueeze(0).unsqueeze(0).expand(r.shape[0], n, -1, -1)
        return self.network(head, tail)

    def image(self, ids, region):
        family = region.family
        r, e = self.vectors(ids), self.domain()
        n = e.shape[0]
        if region.index is not None:           # an element: one row of the pair table
            head = e[region.index] + r
            return FuzzySet(self.network(head.unsqueeze(1).expand(-1, n, -1),
                                         e.unsqueeze(0).expand(r.shape[0], -1, -1)), family)
        pairs = self._pairs(ids)
        return FuzzySet(family.sup(family.and_(pairs, region.values.unsqueeze(-1)), -2),
                        family)

    def preimage(self, ids, region):
        family = region.family
        if self.role_pairs == "all":
            return FuzzySet(family.exists(self._pairs(ids), region.values), family)
        r, e = self.vectors(ids), self.domain()
        n = e.shape[0]
        r_fs = self.network((e.unsqueeze(0) + r.unsqueeze(1)),
                            e.unsqueeze(0).expand(r.shape[0], -1, -1))
        value = family.sup(family.and_(r_fs, region.values), -1).unsqueeze(-1)
        return FuzzySet(value.expand(-1, n), family)


# =========================================================================================
# measures
# =========================================================================================

class _FuzzyMeasure(RegionMeasure):
    accepts = (FuzzySet,)
    #: Whether the measure compares logits (:class:`LogitGoedel`) rather than degrees.
    logits = False

    def refuses(self, region):
        if region.family.logits != self.logits:
            return (f"compares fuzzy sets of {'logits' if self.logits else 'degrees'}, "
                    f"not of {'logits' if region.family.logits else 'degrees'} (family "
                    f"{type(region.family).__name__}).")
        return None

    def member(self, point, region, margin=0.0):
        """Cost of the individual ``point`` being in ``region``."""
        return self.inside(point, region, margin)

    def validity(self, point, margin=0.0):
        """Cost of the individual being well formed (zero by default)."""
        return th.zeros(len(point), 1, device=point.values.device)

    def meets(self, a, b, margin=0.0):
        self.check(a, b)
        s = a.family.sup(a.family.and_(a.values, b.values), -1).unsqueeze(-1)
        return th.relu(margin - s)

    def nonempty(self, a, epsilon=0.0):
        self.check(a)
        return th.relu(epsilon - a.family.sup(a.values, -1).unsqueeze(-1))


class MeanViolation(_FuzzyMeasure):
    """DF-ALC (Eq. 8): :math:`A \\subseteq B` is scored by the mean over the domain of
    :math:`\\mathrm{ReLU}(A(x) - B(x))`; zero iff :math:`A \\le B` pointwise. An element
    ``a`` is a member of :math:`C` with the cost :math:`1 - C(a)` (a fuzzy assertion of
    degree 1). A positive margin makes the inclusion strict; for degrees in ``[0, 1]`` it
    makes disjointness unsatisfiable, so use margin 0 or a negative margin (slack).

    Part (vocabulary value): ``fuzzy implication``. Used by: ``dfalc``.
    """

    part = "fuzzy implication"

    def inside(self, a, b, margin=0.0):
        self.check(a, b)
        return th.relu(a.values - b.values + margin).mean(dim=-1, keepdim=True)

    def apart(self, a, b, margin=0.0):
        self.check(a, b)
        return th.relu(a.family.and_(a.values, b.values) + margin).mean(dim=-1, keepdim=True)

    def empty(self, a, margin=0.0):
        self.check(a)
        return a.values.mean(dim=-1, keepdim=True)

    def member(self, point, region, margin=0.0):
        if point.index is None:
            return self.inside(point, region, margin)
        degree = region.values.gather(1, point.index.view(-1, 1))
        return th.relu(1 - degree + margin)

    @staticmethod
    def inside_degrees(c, d, margin=0.0):
        """The cost of :meth:`inside` on the degree tensors ``c`` and ``d`` themselves (no
        regions): the mean over the last axis of :math:`\\mathrm{ReLU}(c - d + margin)`, one
        value per row (DF-ALC's hierarchical loss, Eq. 8)."""
        return th.relu(c - d + margin).mean(dim=-1)

    def member_at(self, degree):
        """The cost :math:`1 - C(a)` of an element whose degree in :math:`C` is ``degree``
        (any shape; :meth:`member` of the element)."""
        return th.relu(1 - degree)

    def apart_at(self, degree):
        """The cost :math:`C(a)` of an element that is not to be in :math:`C`."""
        return th.relu(degree)


class ConflictMeasure(_FuzzyMeasure):
    """FALCON: a statement is scored by the conflict it leaves, :math:`-\\log(1 - m)` where
    ``m`` is the largest degree (``max_measure="max"``) or the p-mean
    (``"pmean2"``...) over the domain of the *counterexample* region: :math:`A \\sqcap \\neg B`
    for :math:`A \\subseteq B`, :math:`A \\sqcap B` for disjointness, :math:`A` for
    emptiness. An element is a member of :math:`C` with the cost :math:`-\\log C(a)`.
    There is no margin: a non-zero margin raises ``ValueError``.

    :param stabiliser: ``"clamp"`` (default): :math:`-\\log \\max(1 - m, 10^{-10})`, never
        negative. ``"add"``: :math:`-\\log(1 - m + 10^{-10})` as in the FALCON code (down to
        :math:`-10^{-10}` for a perfect fit).

    Part (vocabulary value): ``fuzzy implication``. Used by: ``falcon``.
    """

    part = "fuzzy implication"

    def __init__(self, max_measure="max", stabiliser="clamp"):
        if stabiliser not in ("clamp", "add"):
            raise ValueError("stabiliser must be 'clamp' or 'add'.")
        is_pmean = max_measure[:5] == "pmean" and max_measure[5:].isdigit()
        if max_measure != "max" and not is_pmean:
            raise ValueError("max_measure must be 'max' or 'pmean<p>'.")
        self.max_measure = max_measure
        self.stabiliser = stabiliser

    def _conflict(self, fs):
        if self.max_measure == "max":
            top = fs.max(dim=-1)[0]
        else:
            p = int(self.max_measure[-1])
            top = ((fs ** p).mean(dim=-1)) ** (1 / p)
        rest = 1 - top
        rest = rest + 1e-10 if self.stabiliser == "add" else rest.clamp_min(1e-10)
        return -th.log(rest).unsqueeze(-1)

    def _stabilised(self, rest):
        rest = rest + 1e-10 if self.stabiliser == "add" else rest.clamp_min(1e-10)
        return -th.log(rest)

    def member_at(self, degree):
        """The cost :math:`-\\log C(a)` of an element whose degree in :math:`C` is ``degree``
        (any shape; the same as :meth:`member` of the crisp set of the element)."""
        return self._stabilised(degree)

    def apart_at(self, degree):
        """The cost :math:`-\\log(1 - C(a))` of an element that is not to be in :math:`C`."""
        return self._stabilised(1 - degree)

    @staticmethod
    def _no_margin(margin):
        if margin:
            raise MarginUnavailable("ConflictMeasure has no margin (the conflict is graded).")

    def inside(self, a, b, margin=0.0):
        self.check(a, b)
        self._no_margin(margin)
        return self._conflict(a.family.and_(a.values, a.family.not_(b.values)))

    def apart(self, a, b, margin=0.0):
        self.check(a, b)
        self._no_margin(margin)
        return self._conflict(a.family.and_(a.values, b.values))

    def empty(self, a, margin=0.0):
        self.check(a)
        return self._conflict(a.values)


class ConeMeasure(_FuzzyMeasure):
    """Al-cone: statements about half-axis logits with a strict margin
    :math:`\\gamma > 0`. :math:`A \\subseteq B` is the mean over the half-axes of
    :math:`\\mathrm{relu}(\\gamma + a)\\,\\mathrm{relu}(\\gamma - b)`: zero iff at every
    half-axis :math:`a \\le -\\gamma` or :math:`b \\ge \\gamma`, which implies the inclusion
    of the discretised cones. Disjointness uses :math:`-b`. An individual is the support of
    a point; :meth:`validity` penalises supports that are empty or contain both half-axes of
    a coordinate (Def. 2 of the paper).

    :param empty_margin: Margin used by :meth:`empty` (which gets none).

    Part (vocabulary value): ``fuzzy implication``. Used by: ``alcone``.
    """

    part = "fuzzy implication"
    logits = True

    def __init__(self, empty_margin=0.1, tau=0.0, source=None):
        self.empty_margin = empty_margin
        self._tau = tau
        self.source = source

    @property
    def tau(self):
        return self.source.temperature if self.source is not None else self._tau

    def inside(self, a, b, margin=0.0):
        self.check(a, b)
        return CL.either(-a.values, b.values, margin).mean(dim=-1, keepdim=True)

    def apart(self, a, b, margin=0.0):
        self.check(a, b)
        return CL.either(-a.values, -b.values, margin).mean(dim=-1, keepdim=True)

    def empty(self, a, margin=0.0):
        self.check(a)
        return th.relu(self.empty_margin + a.values).mean(dim=-1, keepdim=True)

    def validity(self, point, margin=0.0):
        u = point.values
        nonzero = th.relu(margin - CL.reduce_max(u, -1, self.tau))
        pairs = u.view(u.shape[0], -1, 2)
        exclusive = CL.either(-pairs[..., 0], -pairs[..., 1], margin).mean(dim=-1)
        return (nonzero + exclusive).unsqueeze(-1)


register_measure("mean violation", default_for=(FuzzySet,))(MeanViolation)
register_measure("conflict")(ConflictMeasure)
register_measure("half-axis hinge", default_for=(LogitSet,))(ConeMeasure)


# =========================================================================================
# the objective and the geometry
# =========================================================================================

class FuzzyObjective(ExpressionReading, Objective):
    """The reading of the axiom forms, and of ALC axioms, for fuzzy geometries.

    Besides the options of :class:`~mowl.representation.Objective`:

    :param rule: ``True``: the rule-based losses of DF-ALC (Eq. 9 for :math:`C \\sqsubseteq D`
        with the gradient only through :math:`D`, Eq. 10 for :math:`C \\sqsubseteq \\exists r.D`,
        Eq. 12 for :math:`\\exists r.C \\sqsubseteq D`; for ALC axioms also Eqs. 11 and 13 for
        :math:`\\forall`).
    :param threshold: The threshold :math:`\\alpha` of Eqs. 10 to 13.
    :param validity: Add the well-formedness cost of the individuals to assertions (Al-cone).
    :param role_assertion: ``"preimage"`` (default): :math:`a \\in \\exists r.\\{b\\}`.
        ``"image"``: :math:`b` lies in the image of :math:`a` under :math:`r`. Equal for
        crisp relations; the published losses of all three methods read it as ``"image"``.
    :param abox: How the assertions of ALC axioms are read. ``"semantic"`` (default): an
        individual is a member of the class expression, the measure says how much.
        ``"sampled"``: the reading of FALCON: every assertion is contrasted with corrupted
        ones (``num_negs`` individuals drawn at random for a class assertion, heads and
        tails not asserted for a role assertion), the loss is the mean cost of the positives
        plus that of the negatives, halved, and an assertion with a compound class expression
        pairs the individual with every element of the domain through the relation network.
        Needs :meth:`attach_assertions` for the role assertions.
    :param num_negs: Corrupted assertions per assertion (``abox="sampled"``).
    :param abox_head: ``"object"``: the role assertion rows of the ALC dataset are
        ``[object, role, subject]`` and the relation network reads ``(vector(object) + r,
        vector(subject))``, as FALCON's code does it (this is the opposite direction of
        :math:`\\exists r.C`, which reads ``(vector(x) + r, vector(y))`` for a pair
        :math:`r(x, y)`). ``"subject"`` (default): ``(vector(subject) + r, vector(object))``.
    :param role_loss: The loss of a sampled role assertion: ``"contrast"`` (default): the
        positive and the corrupted ones separately, as for class assertions; ``"ranking"``:
        :math:`-\\log\\sigma(\\mathrm{pos} - \\mathrm{neg})` (Bayesian personalised ranking).
    :param weighting: How the costs of the axiom patterns are combined into the loss of a
        training step, an
        :class:`~mowl.representation.library.logic_alc.AxiomKindWeighting`; ``None``
        (default): the one the model gives (its ``alpha`` and ``beta``).
    """

    def __init__(self, margin=0.0, rule=False, threshold=0.8, validity=False,
                 role_assertion="preimage", abox="semantic", num_negs=4, abox_head="subject",
                 role_loss="contrast", weighting=None, **kwargs):
        super().__init__(margin, **kwargs)
        self.weighting = weighting
        if role_assertion not in ("preimage", "image"):
            raise ValueError("role_assertion must be 'preimage' or 'image'.")
        if abox not in ("semantic", "sampled"):
            raise ValueError("abox must be 'semantic' or 'sampled'.")
        if abox_head not in ("subject", "object"):
            raise ValueError("abox_head must be 'subject' or 'object'.")
        if role_loss not in ("contrast", "ranking"):
            raise ValueError("role_loss must be 'contrast' or 'ranking'.")
        self.role_loss = role_loss
        self.rule, self.threshold = bool(rule), float(threshold)
        self.validity, self.role_assertion = bool(validity), role_assertion
        self.abox, self.num_negs, self.abox_head = abox, int(num_negs), abox_head
        #: The asserted ``(tail, role) -> heads`` and ``(head, role) -> tails``, and the
        #: number of named individuals negatives are drawn from (``abox="sampled"``).
        self.heads, self.tails, self.n_individuals = {}, {}, 0

    @classmethod
    def dfalc(cls, loss_type="hierarchical", threshold=0.8):
        return cls(0.0, rule=loss_type == "rule", threshold=threshold, role_assertion="image",
                   negatives="all")

    @classmethod
    def falcon(cls, num_negs=4, abox_head="object", role_loss="contrast"):
        return cls(0.0, role_assertion="image", negatives="all", abox="sampled",
                   num_negs=num_negs, abox_head=abox_head, role_loss=role_loss)

    @classmethod
    def alcone(cls, margin=0.1):
        return cls(margin, validity=True, role_assertion="image", negatives="all")

    def requirements(self, g):
        """Why the geometry ``g`` cannot be read by this objective (a sentence), or ``None``.

        The semantic reading needs nothing beyond the constructors and the measure (a part
        that lacks one says so when it is asked), but some options need more:

        * ``rule=True`` (the rule-based losses of DF-ALC) reads truth degrees in ``[0, 1]``
          and the truth matrices of the relations (``matrix``);
        * ``validity=True`` adds the well-formedness cost of the individuals, which the
          measure has to define (:meth:`ConeMeasure.validity`);
        * the sampled reading of assertions (``abox="sampled"``, FALCON) contrasts degrees
          of named elements with those of corrupted ones, so it needs a class model that
          gives the degree of a class at an element (``degree_at``), a relation model over
          pairs of element vectors (``degree_at``, ``head_degrees``, ``vectors``), a domain
          of element vectors, and a measure with costs of single degrees (``member_at``,
          ``apart_at``)."""
        reasons = []
        if self.rule:
            if getattr(getattr(g.class_model, "family", None), "logits", True):
                reasons.append(f"rule=True (the rule-based losses of DF-ALC) reads truth "
                               f"degrees in [0, 1]; the class model "
                               f"{type(g.class_model).__name__} gives logits or no fuzzy set")
            if not hasattr(g.relation_model, "matrix"):
                reasons.append(f"rule=True needs the truth matrices of the relations "
                               f"(matrix); the relation model "
                               f"{type(g.relation_model).__name__} has none")
        if self.validity and getattr(type(g.measure), "validity", None) in (
                None, _FuzzyMeasure.validity):
            reasons.append(f"validity=True adds the well-formedness cost of the individuals "
                           f"of the measure; {type(g.measure).__name__} has none (it is zero), "
                           f"so the option would change nothing")
        if self.abox == "sampled":
            sampled = self._sampled_requirements(g)
            if sampled:
                reasons.append(sampled)
        return "; ".join(reasons) + "." if reasons else None

    def _sampled_requirements(self, g):
        missing = []
        if not hasattr(g.class_model, "degree_at"):
            missing.append(f"a class model with the degree of a class at an element "
                           f"(degree_at), not {type(g.class_model).__name__}")
        relations = g.relation_model
        if relations is None or not all(hasattr(relations, name) for name in
                                        ("degree_at", "head_degrees", "vectors")):
            missing.append(f"a relation model that scores pairs of element vectors "
                           f"(degree_at, head_degrees, vectors; PairNetworkRelations), not "
                           f"{type(relations).__name__}")
        if not hasattr(getattr(g, "domain", None), "vectors"):
            missing.append("a domain of element vectors (Domain)")
        if not (hasattr(g.measure, "member_at") and hasattr(g.measure, "apart_at")):
            missing.append(f"a measure with the cost of a single degree (member_at, "
                           f"apart_at), not {type(g.measure).__name__}")
        if not missing:
            return None
        return (f"{type(self).__name__}(abox='sampled'), the reading of assertions against "
                "corrupted ones (FALCON), needs " + ", ".join(missing) + " (the semantic "
                "reading, objective_options={'abox': 'semantic'}, reads assertions as "
                "memberships and needs none of these)")

    def attach_assertions(self, heads, tails, n_individuals):
        """The role assertions of the training ontology, so that the corrupted assertions of
        ``abox="sampled"`` are not asserted ones: ``heads[(tail, role)]`` lists the heads of
        the pair, ``tails[(head, role)]`` the tails, with *head* the first column of the
        dataset rows (the object); ``n_individuals`` is the number of named individuals."""
        self.heads, self.tails, self.n_individuals = heads, tails, int(n_individuals)

    # -- rule-based losses --------------------------------------------------------------
    def gci0(self, g, rows, neg=False):
        if self.rule and not neg:
            c, d = g.classes(rows[:, 0]), g.classes(rows[:, 1])
            return rule_based_loss(c.values, d.values).unsqueeze(-1)
        return super().gci0(g, rows, neg)

    def gci1(self, g, rows, neg=False):
        if self.rule and not neg:
            c, d, e = (g.classes(rows[:, i]) for i in range(3))
            both, _ = c.intersect(d)
            return rule_based_loss(both.values, e.values).unsqueeze(-1)
        return super().gci1(g, rows, neg)

    def gci2(self, g, rows, neg=False):
        if self.rule and not neg:
            b, a = g.classes(rows[:, 0]), g.classes(rows[:, 2])
            r = g.relation_model.matrix(rows[:, 1])
            return rule_nf4(b.values, a.values, r, self.threshold).unsqueeze(-1)
        return super().gci2(g, rows, neg)

    def gci3(self, g, rows, neg=False):
        if self.rule and not neg:
            a, b = g.classes(rows[:, 1]), g.classes(rows[:, 2])
            r = g.relation_model.matrix(rows[:, 0])
            return rule_nf6(a.values, b.values, r, self.threshold).unsqueeze(-1)
        return super().gci3(g, rows, neg)

    def gci0_bot(self, g, rows, neg=False):
        if self.rule and not neg:
            c = g.classes(rows[:, 0])
            return rule_based_loss(c.values, th.zeros_like(c.values)).unsqueeze(-1)
        return super().gci0_bot(g, rows, neg)

    # -- assertions ---------------------------------------------------------------------
    def class_assertion(self, g, rows, neg=False):
        a, c = g.individuals(rows[:, 0]), g.classes(rows[:, 1])
        if self._neg(neg, "class_assertion"):
            return g.measure.apart(a, c, self.margin)
        cost = g.measure.member(a, c, self.m)
        if self.validity:
            cost = cost + g.measure.validity(a, self.m)
        return cost

    def object_property_assertion(self, g, rows, neg=False):
        a, b = g.individuals(rows[:, 0]), g.individuals(rows[:, 2])
        if self.role_assertion == "image":
            subject, target = b, g.relation_model.image(rows[:, 1], a)
        else:
            subject, target = a, g.exists(rows[:, 1], b)
        if self._neg(neg, "object_property_assertion"):
            return g.measure.apart(subject, target, self.margin)
        cost = g.measure.member(subject, target, self.m) + self.gci2_offset
        if self.validity:
            cost = cost + g.measure.validity(a, self.m) + g.measure.validity(b, self.m)
        return cost

    # -- ALC axioms ---------------------------------------------------------------------
    def _rule_nf47(self, g, sub, sup, x):
        """The rule-based loss of normal forms 4-7 (Eqs. 10-13) if the inclusion has one of
        the shapes ``L ⊑ ∃r.L``, ``L ⊑ ∀r.L``, ``∃r.L ⊑ L``, ``∀r.L ⊑ L`` (L a concept name
        or its negation), else ``None``."""
        some = ClassExpressionType.OBJECT_SOME_VALUES_FROM
        st, ut = sub.getClassExpressionType(), sup.getClassExpressionType()
        if _is_literal(sub) and ut in _QUANTIFIERS and _is_literal(sup.getFiller()):
            b, nxt = pattern_region(g, sub, x, 0)
            r = g.relation_model.matrix(x[:, nxt])
            a, _ = pattern_region(g, sup.getFiller(), x, nxt + 1)
            rule = rule_nf4 if ut == some else rule_nf5
            return rule(b.values, a.values, r, self.threshold)
        if _is_literal(sup) and st in _QUANTIFIERS and _is_literal(sub.getFiller()):
            r = g.relation_model.matrix(x[:, 0])
            a, nxt = pattern_region(g, sub.getFiller(), x, 1)
            b, _ = pattern_region(g, sup, x, nxt)
            rule = rule_nf6 if st == some else rule_nf7
            return rule(a.values, b.values, r, self.threshold)
        return None

    def inclusion(self, g, sub, sup, x):
        if self.rule:
            out = self._rule_nf47(g, sub, sup, x)
            if out is not None:
                return out
            if not (has_quantifier(sub) or has_quantifier(sup)):
                a, b, _ = self.regions(g, sub, sup, x)
                return rule_based_loss(a.values, b.values)
        return super().inclusion(g, sub, sup, x)

    def equivalence(self, g, first, second, x):
        if not self.rule:
            return super().equivalence(g, first, second, x)
        # C ⊑ D with every rule that applies; D ⊑ C only with the rule for inclusions
        # without a quantifier (the shapes of normal forms 4-7 are not mirrored). The two
        # directions read the expressions again, as DF-ALC's code does.
        forward = self.inclusion(g, first, second, x)
        a, b, _ = self.regions(g, first, second, x)
        if self.rule and not (has_quantifier(first) or has_quantifier(second)):
            backward = rule_based_loss(b.values, a.values)
        else:
            backward = g.inside(b, a).reshape(len(x), -1).sum(-1)
        return forward + backward

    def class_assertion_pattern(self, g, expression, x):
        if self.abox == "sampled":
            return self._sampled_class_assertion(g, expression, x)
        region, _ = pattern_region(g, expression, x, 1)
        a = g.individuals(x[:, 0])
        cost = g.measure.member(a, region, self.m)
        if self.validity:
            cost = cost + g.measure.validity(a, self.m)
        return cost.reshape(len(x), -1).sum(-1)

    def role_assertion_pattern(self, g, x):
        if self.abox == "sampled":
            return self._sampled_role_assertion(g, x)
        # the rows of the ALC dataset are [object, role, subject]
        return self.object_property_assertion(g, x[:, [2, 1, 0]]).reshape(len(x), -1).sum(-1)

    # -- the sampled reading of assertions (FALCON) --------------------------------------
    def _sample_negatives(self, e, r, used):
        """For every row, ``num_negs`` named individuals that are not in ``used[(e, r)]``."""
        ret = th.zeros((e.shape[0], self.num_negs), dtype=th.int64)
        for i in range(e.shape[0]):
            taken = th.tensor(used[(e[i].item(), r[i].item())])
            pool = th.ones(self.n_individuals)
            pool[taken] = 0
            pool = pool.nonzero()
            ret[i, :] = pool[th.randint(len(pool), (self.num_negs,))].flatten()
        return ret

    def _sampled_class_assertion(self, g, expression, x):
        n = self.num_negs
        x = x.unsqueeze(dim=1)
        size = [1] * len(x.size())
        size[1] = n
        negative = x.repeat(size)
        negative[:, :, 0] = th.randint(self.n_individuals, (x.shape[0], n))
        x = th.cat([x, negative], dim=1)
        measure = g.measure
        if expression.getClassExpressionType() == ClassExpressionType.OWL_CLASS:
            # a plain assertion: the degree of the individual in the class, towards 1 for
            # the asserted individual and towards 0 for the corrupted ones
            degree = g.class_model.degree_at(x[:, :, 1], x[:, :, 0])
            return self._contrast(measure, degree)
        # an assertion with a compound class expression ``e : ∃R.C`` (or ``e : C``): the
        # individual, shifted by the role if there is one, is paired with every element of
        # the domain through the relation network, and the best pair with an element of C
        some = expression.getClassExpressionType() == ClassExpressionType.OBJECT_SOME_VALUES_FROM
        if some:
            relation, filler, columns = x[:, :, 1], expression.getFiller(), x[:, 0, 2:]
        else:
            relation, filler, columns = None, expression, x[:, 0, 1:]
        region, _ = pattern_region(g, filler, columns, 0)
        head = g.domain.vectors(x[:, :, 0])
        if relation is not None:
            head = head + g.relation_model.vectors(relation)
        pairs = g.relation_model.head_degrees(head.reshape(-1, head.shape[-1]))
        pairs = pairs.reshape(head.shape[0], head.shape[1], -1)
        family = region.family
        degree = family.and_(pairs, region.values.unsqueeze(dim=1)).max(dim=-1)[0]
        return self._contrast(measure, degree)

    def role_scores(self, g, x):
        """The degrees of the role assertions ``x`` and of their corrupted assertions
        (``abox="sampled"``), flattened (for evaluating a trained model)."""
        return self._sampled_role_degrees(g, x).flatten()

    def _sampled_role_assertion(self, g, x):
        degree = self._sampled_role_degrees(g, x)
        if self.role_loss == "ranking":
            diff = degree[:, 0].unsqueeze(dim=-1) - degree[:, 1:]
            return -th.nn.functional.logsigmoid(diff).mean()
        return self._contrast(g.measure, degree)

    @staticmethod
    def _contrast(measure, degree):
        """The mean cost of the first column (the assertion, to be true) plus that of the
        others (the corrupted assertions, to be false), halved."""
        positive = measure.member_at(degree[:, 0]).mean()
        negative = measure.apart_at(degree[:, 1:]).mean()
        return (positive + negative) / 2

    def _sampled_role_degrees(self, g, x):
        heads, tails = self.heads, self.tails
        if self.abox_head == "subject":
            x, heads, tails = x[:, [2, 1, 0]], tails, heads
        n = self.num_negs
        x = x.unsqueeze(dim=1)
        size = [1] * len(x.size())
        size[1] = n
        negative_head = x.repeat(size)
        negative_head[:, :, 0] = self._sample_negatives(x[:, :, 2], x[:, :, 1], heads)
        negative_tail = x.repeat(size)
        negative_tail[:, :, 2] = self._sample_negatives(x[:, :, 0], x[:, :, 1], tails)
        x = th.cat([x, negative_head, negative_tail], dim=1)
        return g.relation_model.degree_at(x[:, :, 1], x[:, :, 0], x[:, :, 2])


def empty_cone_penalty(logits, margin, temperature=0.0):
    """Mean over the rows of ``logits`` (``(n, half-axes)``) of
    :math:`\\mathrm{relu}(\\gamma - \\max_h c_h)`: zero for a class that has a half-axis with
    logit at least ``margin`` (so is not the empty cone). ``temperature`` is that of the smooth
    maximum of the geometry (0: exact)."""
    return th.relu(margin - CL.reduce_max(logits, -1, temperature)).mean()


class NonDegeneracy(Regularizer):
    """Pushes every named class away from the empty cone :math:`\\{0\\}`: the mean over the
    classes (not ``owl:Thing`` and ``owl:Nothing``) of :math:`\\mathrm{relu}(\\gamma - \\max_h
    c_h)` on the half-axis logits, so a class has a half-axis with logit at least the margin.
    It is not part of the semantics: it assumes that the named classes are satisfiable. Needs
    a half-axis class model (:class:`HalfAxisClasses`).

    :param weight: Factor of the penalty.
    :param margin: :math:`\\gamma`; ``None``: the margin of the geometry's objective.
    """

    def __init__(self, weight=0.1, margin=None):
        super().__init__("class_embed", None, weight)
        self.margin = margin

    def penalty(self, model):
        g = getattr(model, "module", model)
        classes = g.class_model
        special = {classes.top_index, classes.bottom_index}
        ids = [i for i in range(len(classes)) if i not in special]
        if self.weight <= 0 or not ids:
            return None
        ids = th.tensor(ids, dtype=th.long, device=g.device)
        margin = g.objective.margin if self.margin is None else self.margin
        tau = g.temperature
        return self.weight * empty_cone_penalty(classes.centres(ids), margin, tau)


# =========================================================================================
# registered parts (names to combine with ``compose`` / ``ComposedEL``)
# =========================================================================================

def _truth_table(size, embed_dim, family="godel"):
    return TruthTableClasses(size, embed_dim, family=family)


def _fuzzy_set(family, operators, used_by):
    """A registered fuzzy-set class model with the operators of ``family``."""
    def factory(size, embed_dim):
        return _truth_table(size, embed_dim, family)
    factory.part = ClassModel.FUZZY_SET.value
    factory.__doc__ = (
        f"A class is a fuzzy set over a finite domain (a truth degree per element) with the "
        f"{operators} operators (:class:`TruthTableClasses`, ``family={family!r}``); "
        f"part: ``fuzzy set``. Used by: {used_by}.")
    return factory


register_class_model("fuzzy set")(_fuzzy_set("godel", "Goedel", "``dfalc`` (Goedel is its "
                                             "default family)"))
register_class_model("fuzzy set (product)")(_fuzzy_set("product", "product", "``falcon``"))
register_class_model("fuzzy set (lukasiewicz)")(_fuzzy_set("lukasiewicz", "Lukasiewicz",
                                                           "no method (a combination)"))
register_class_model("half-axis cone")(HalfAxisClasses)
register_relation_model("fuzzy relation")(TruthMatrix)
register_relation_model("incidence matrix")(IncidenceMatrices)


# =========================================================================================
# the methods
# =========================================================================================

def _domain_size(nb_individuals, embed_dim, domain):
    n = domain if domain is not None else max(embed_dim, nb_individuals)
    if n < max(nb_individuals, 1):
        raise ValueError("The domain must hold every individual.")
    return n


def truth_tables(nclasses, nrelations, ndomain, init_std=1.0):
    """The parameter tables of a DF-ALC interpretation over ``ndomain`` elements, drawn in
    the order of the earlier module (both tables, then the logits of both, normal with
    ``init_std``): ``(classes (nclasses, n), relations (max(nrelations, 1), n * n))``."""
    classes = nn.Embedding(nclasses, ndomain)
    relations = nn.Embedding(max(nrelations, 1), ndomain * ndomain)
    for table in (classes, relations):
        nn.init.normal_(table.weight, std=init_std)
    return classes, relations


def half_axis_tables(nclasses, nrelations, nindividuals, dim, init_std=0.5):
    """The parameter tables of an Al-cone interpretation in :math:`\\mathbb{R}^{dim}`
    (``2 dim`` half-axes), drawn in the order of the earlier module (all three tables, then
    the logits of all three, normal with ``init_std``): ``(classes (nclasses, 2 dim),
    relations (max(nrelations, 1), (2 dim)^2), individuals (max(nindividuals, 1), 2 dim))``."""
    nhalf = 2 * dim
    classes = nn.Embedding(nclasses, nhalf)
    relations = nn.Embedding(max(nrelations, 1), nhalf * nhalf)
    individuals = nn.Embedding(max(nindividuals, 1), nhalf)
    for table in (classes, relations, individuals):
        nn.init.normal_(table.weight, std=init_std)
    return classes, relations, individuals


def dfalc_parts(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, domain=None,
                loss_type="hierarchical", threshold=0.8, family="godel", top_index=None,
                bottom_index=None, init_std=1.0, tables=None):
    """The parts of DF-ALC, as the keyword arguments of
    :class:`~mowl.representation.library.logic_alc.ExpressionGeometry` (see :func:`dfalc`)."""
    t = tables
    if t is not None:
        n = t.ndomain
        classes = TruthTableClasses(logits=t.c_embedding, family=family,
                                    top_index=t.top_index, bottom_index=t.bottom_index)
        relations = TruthMatrix(vectors=t.r_embedding, domain=n)
        loss_type, threshold = t.loss_type, t.threshold
    else:
        n = _domain_size(nb_individuals, embed_dim, domain)
        classes = TruthTableClasses(nb_classes, n, family=family, top_index=top_index,
                                    bottom_index=bottom_index, init_std=init_std)
        relations = TruthMatrix(max(nb_relations, 1), n, init_std=init_std)
    return dict(classes=classes, relations=relations, measure=MeanViolation(),
                objective=FuzzyObjective.dfalc(loss_type, threshold),
                individuals=DomainElements(n, n, family=family))


@register_method("dfalc")
def dfalc(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.0,
          domain=None, loss_type="hierarchical", threshold=0.8, family="godel",
          top_index=None, bottom_index=None, init_std=1.0, tables=None):
    """DF-ALC (Wu et al. 2022): truth tables over a finite domain, truth matrices for the
    roles, Goedel operators, the mean violation as the measure. Individual ``i`` is domain
    element ``i``.

    The geometry reads ALC axioms (``forward(axiom, rows)``) as well as normal forms.

    :param embed_dim: Size of the domain if ``domain`` is not given (at least the number of
        individuals).
    :param family: The operator family; ``"godel"`` is the published method, any member of
        :data:`FAMILIES` can be used instead.
    :param loss_type: ``"hierarchical"`` (Eq. 8) or ``"rule"`` (Eqs. 9 to 13).
    :param tables: A :class:`~mowl.baselines.nn.alc.dfalc.module.DFALCModule` (the direct
        implementation) to share parameters with.
    """
    return ExpressionGeometry(**dfalc_parts(
        nb_classes, nb_relations, nb_individuals, embed_dim, domain, loss_type, threshold,
        family, top_index, bottom_index, init_std, tables))


def falcon_parts(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, t_norm="product",
                 max_measure="max", stabiliser="add", role_pairs="all", anon_domain=0,
                 top_index=None, bottom_index=None, thing="learned", nothing="empty",
                 num_negs=4, abox_head="object", role_loss="contrast", tables=None):
    """The parts of FALCON, as the keyword arguments of
    :class:`~mowl.representation.library.logic_alc.ExpressionGeometry` (see :func:`falcon`)."""
    if thing not in ("learned", "all") or nothing not in ("empty", "learned"):
        raise ValueError("thing must be 'learned' or 'all', nothing 'empty' or 'learned'.")
    top_index = top_index if thing == "all" else None
    bottom_index = bottom_index if nothing == "empty" else None
    objective = FuzzyObjective.falcon(num_negs, abox_head, role_loss)
    if tables is not None:
        t = tables
        top_index, bottom_index = t.top_index, t.bottom_index
        role_pairs = t.role_pairs
        network = MembershipNetwork(fc_0=t.fc_0, fc_1=t.fc_1)
        domain = Domain(t.e_embedding)
        classes = NetworkClasses(centres=t.c_embedding, network=network, domain=domain,
                                 family=t_norm if t_norm is not None else t.t_norm,
                                 top_index=top_index, bottom_index=bottom_index)
        relations = PairNetworkRelations(vectors=t.r_embedding, network=network,
                                         domain=domain, role_pairs=role_pairs)
        measure = ConflictMeasure(t.max_measure, stabiliser)
        objective.num_negs = t.num_negs
        objective.attach_assertions(t.heads_dict, t.tails_dict, t.nentities)
        table = t.e_embedding
    else:
        network = MembershipNetwork(embed_dim)
        table = nn.Embedding(nb_individuals, embed_dim)
        nn.init.xavier_uniform_(table.weight)
        domain = Domain(table, anon_domain)
        if len(domain) < 1:
            raise ValueError("FALCON needs a domain: individuals or anon_domain > 0.")
        classes = NetworkClasses(nb_classes, embed_dim, network=network, domain=domain,
                                 family=t_norm, top_index=top_index,
                                 bottom_index=bottom_index)
        relations = PairNetworkRelations(max(nb_relations, 1), embed_dim, network=network,
                                         domain=domain, role_pairs=role_pairs)
        measure = ConflictMeasure(max_measure, stabiliser)
        objective.n_individuals = table.num_embeddings
    individuals = DomainElements(table.num_embeddings, len(domain), family=t_norm,
                                 elements=domain)
    return dict(classes=classes, relations=relations, measure=measure, objective=objective,
                individuals=individuals, domain=domain)


@register_method("falcon")
def falcon(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.0,
           t_norm="product", max_measure="max", stabiliser="add", role_pairs="all",
           anon_domain=0, top_index=None, bottom_index=None, thing="learned",
           nothing="empty", num_negs=4, abox_head="object", role_loss="contrast",
           tables=None):
    """FALCON (Tang et al. 2022): class vectors, a membership network over (vector, element
    vector) pairs, a network over pairs for roles, a t-norm, the conflict of the
    counterexample as the measure, and assertions contrasted with corrupted ones. The domain
    is the table of the individuals (plus ``anon_domain`` learned anonymous elements; a
    training loop may also sample anonymous elements for one evaluation, see
    :class:`~mowl.representation.library.logic_fuzzy.Domain`).

    The geometry reads ALC axioms (``forward(axiom, rows, e_emb)``) as well as normal forms.

    :param t_norm: The operator family (``"product"``, ``"minmax"``, ``"Łukasiewicz"`` or any
        name of :data:`FAMILIES`).
    :param stabiliser: ``"add"``, the published reading that is not the default of its part
        (``-log(1 - m + 1e-10)``).
    :param role_pairs: ``"all"`` (default, as published): :math:`\\exists r.C` ranges over
        all pairs of elements; ``"self"``: over the pair of an element with itself (the
        earlier mOWL module, a porting error), see :class:`PairNetworkRelations`.
    :param top_index, bottom_index: Class ids of ``owl:Thing`` and ``owl:Nothing``.
    :param thing: ``"learned"`` (default, as published): ``owl:Thing`` is an ordinary learned
        class; ``"all"``: the constant set of degree 1.
    :param nothing: ``"empty"`` (default, as published): ``owl:Nothing`` is the constant empty
        set; ``"learned"``: an ordinary learned class (the earlier mOWL module).
    :param num_negs: Corrupted assertions per assertion, see :class:`FuzzyObjective`.
    :param abox_head: ``"object"`` (as the FALCON code reads the role assertions of the ALC
        dataset, the one the named method selects) or ``"subject"`` (the part's default: the
        direction of :math:`\\exists r.C`), see :class:`FuzzyObjective`.
    :param role_loss: ``"contrast"`` or ``"ranking"``, see :class:`FuzzyObjective`.
    :param tables: A :class:`~mowl.baselines.nn.alc.falcon.module.FALCONModule` (the direct
        implementation) to share parameters with.
    """
    parts = falcon_parts(nb_classes, nb_relations, nb_individuals, embed_dim, t_norm,
                         max_measure, stabiliser, role_pairs, anon_domain, top_index,
                         bottom_index, thing, nothing, num_negs, abox_head, role_loss,
                         tables)
    geometry = ExpressionGeometry(**parts)
    geometry.network = parts["classes"].network
    return geometry


def alcone_parts(nb_classes, nb_relations, nb_individuals=0, embed_dim=8, margin=0.1,
                 temperature=0.0, family=None, top_index=None, bottom_index=None,
                 constant_logit=DEFAULT_CONSTANT_LOGIT, init_std=0.5, anneal=0.3, noise=0.3,
                 nondegeneracy=0.1, tables=None):
    """The parts of Al-cone, as the keyword arguments of
    :class:`~mowl.representation.library.logic_alc.ExpressionGeometry` (see :func:`alcone`)."""
    t = tables
    if t is not None:
        relaxation = getattr(t, "relaxation", None) or _ModuleTemperature(t)
        fam = LogitGoedel(source=relaxation) if family is None else get_family(family)
        classes = HalfAxisClasses(logits=t.c_embedding, family=fam, top_index=t.top_index,
                                  bottom_index=t.bottom_index,
                                  constant_logit=t.constant_logit)
        relations = IncidenceMatrices(vectors=t.r_embedding, dim=t.dim)
        individuals = HalfAxisClasses(logits=t.e_embedding, family=fam)
        measure = ConeMeasure(source=relaxation)
        margin = t.margin
    else:
        relaxation = Relaxation(temperature)
        fam = LogitGoedel(source=relaxation) if family is None else get_family(family)
        classes = HalfAxisClasses(nb_classes, embed_dim, family=fam, top_index=top_index,
                                  bottom_index=bottom_index, constant_logit=constant_logit,
                                  init_std=init_std)
        relations = IncidenceMatrices(max(nb_relations, 1), embed_dim, init_std=init_std)
        individuals = HalfAxisClasses(max(nb_individuals, 1), embed_dim, family=fam,
                                      init_std=init_std)
        measure = ConeMeasure(source=relaxation)
    return dict(classes=classes, relations=relations, measure=measure,
                objective=FuzzyObjective.alcone(margin), individuals=individuals,
                regularizers=(NonDegeneracy(nondegeneracy),) if nondegeneracy else (),
                schedule=None if anneal is None else AnnealedRelaxation(anneal, noise),
                relaxation=relaxation)


class _ModuleTemperature:
    """The temperature of an existing module, read and written live (a geometry that
    describes the module shares its relaxation)."""

    def __init__(self, module):
        self.module = module

    @property
    def temperature(self):
        return self.module.temperature

    @temperature.setter
    def temperature(self, value):
        self.module.temperature = value


@register_method("alcone")
def alcone(nb_classes, nb_relations, nb_individuals=0, embed_dim=8, margin=0.1,
           temperature=0.0, family=None, top_index=None, bottom_index=None,
           constant_logit=DEFAULT_CONSTANT_LOGIT, init_std=0.5, anneal=0.3, noise=0.3,
           nondegeneracy=0.1, tables=None):
    """Learnable Al-cone (adaptation of Oezcep, Leemhuis and Wolter 2023): half-axis logits
    for classes and individuals, incidence logits for roles, Goedel operators on logits, the
    half-axis hinge as the measure, a well-formedness cost on individuals, a penalty against
    empty classes, and a training loop that anneals a smoothing of the operators.

    The geometry reads ALC axioms (``forward(axiom, rows)``) as well as normal forms.

    :param embed_dim: :math:`n`; there are ``2 n`` half-axes.
    :param margin: The strict margin :math:`\\gamma > 0`.
    :param temperature: The current smoothing of the operators (``0``: exact), a number that
        the training schedule changes.
    :param constant_logit: Magnitude of the logits of ``owl:Thing`` and ``owl:Nothing``, see
        :class:`HalfAxisClasses`.
    :param family: Another operator family on logits instead of Goedel (any
        :data:`FAMILIES` entry whose values are logits).
    :param anneal: Initial temperature of the training schedule
        (:class:`~mowl.representation.library.logic_alc.AnnealedRelaxation`), falling to ``0``
        over the epochs; ``None``: no schedule.
    :param noise: Scale of the parameter noise of the schedule (times the temperature).
    :param nondegeneracy: Weight of :class:`NonDegeneracy`; ``0``: no such penalty.
    :param tables: A :class:`~mowl.baselines.nn.alc.alcone.module.ALConeModule` (the direct
        implementation) to share parameters with.
    """
    return ExpressionGeometry(**alcone_parts(
        nb_classes, nb_relations, nb_individuals, embed_dim, margin, temperature, family,
        top_index, bottom_index, constant_logit, init_std, anneal, noise, nondegeneracy,
        tables))


@register_adapter("DFALCModule")
def _dfalc_adapter(module):
    return dfalc(None, None, tables=module)


@register_adapter("FALCONModule")
def _falcon_adapter(module):
    return falcon(None, None, t_norm=module.t_norm, max_measure=module.max_measure,
                  tables=module)


@register_adapter("ALConeModule")
def _alcone_adapter(module):
    return alcone(None, None, anneal=None, nondegeneracy=0.0, tables=module)
