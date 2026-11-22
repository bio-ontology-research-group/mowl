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
:class:`~mowl.representation.ComposedGeometry` (the region family gives it negation and
disjunction); the three methods are the functions :func:`dfalc`, :func:`falcon` and
:func:`alcone` (registered in :data:`mowl.representation.METHODS`), and
:func:`~mowl.representation.geometry_of` builds them over the tables of the existing
modules. The operator family is a part of its own
(:data:`FAMILIES`): every method can be built with every family, and the class models,
relation models and measures can be exchanged (see the tests).

Regions are :class:`FuzzySet` objects: degrees (or logits) per element, together with the
family that produced them, so that conjunction and the other constructors need no further
argument. A degree is a number in ``[0, 1]`` for the families of fuzzy logic, a real number
for the logit family.
"""

from dataclasses import dataclass

import torch as th
import torch.nn as nn

from mowl.representation.class_models import Classes
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.measures import RegionMeasure
from mowl.representation.methods import (register_class_model, register_measure,
                                         register_method, register_relation_model)
from mowl.representation.objectives import Objective
from mowl.representation.regions import Region
from mowl.representation.relation_models import Relations
from mowl.representation.vocabulary import ClassModel, Construction, RelationModel
from mowl.representation.adapters import register_adapter

__all__ = ["FuzzySet", "Family", "Product", "Goedel", "Lukasiewicz", "LogitGoedel",
           "FAMILIES", "get_family", "register_family", "TruthTableClasses",
           "HalfAxisClasses", "NetworkClasses", "DomainElements", "MembershipNetwork",
           "Domain", "TruthMatrix", "IncidenceMatrices", "PairNetworkRelations",
           "MeanViolation", "ConflictMeasure", "ConeMeasure", "FuzzyObjective",
           "dfalc", "falcon", "alcone"]

#: Logit of the constants owl:Thing and owl:Nothing of the Al-cone module.
DEFAULT_CONSTANT_LOGIT = 1.0


def _df():
    # The DF-ALC module imports the OWL API, which needs the JVM; import it when used.
    from mowl.nn.alc.dfalc import module
    return module


class _CL:
    """The operators of :mod:`mowl.nn.alc.alcone.losses` (same formulas; that module
    cannot be imported without the JVM)."""

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
    ``tau > 0`` the smooth bounds of :mod:`mowl.nn.alc.alcone.losses` are used.

    :param tau: Temperature, or ``source`` (an object with a ``temperature`` attribute that
        is read at every call, as the Al-cone module's is changed during training).
    """

    name = "godel (logits)"

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


class _FuzzyClasses:
    """Class models whose regions are fuzzy sets: the operators of the family do conjunction,
    disjunction and negation."""

    conjunction_part = "fuzzy operator"

    @property
    def extra_parts(self):
        return {"negation": "fuzzy operator", "disjunction": "fuzzy operator",
                "family": type(self.family).__name__}


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


class HalfAxisClasses(_FuzzyClasses, Classes):
    """A class is a set of half-axes of :math:`\\mathbb{R}^n` (an axis-aligned cone), given
    by one logit per half-axis; ``> 0`` means the half-axis belongs to the cone (Al-cone).
    ``owl:Thing`` and ``owl:Nothing`` are the constants :math:`\\pm 10^4`. Also used for the
    individuals of Al-cone (the support of the point).

    :param size: Number of classes; :param dim: :math:`n` (there are ``2 n`` logits).
    :param constant_logit: Magnitude of the logits of ``owl:Thing`` (``+``) and
        ``owl:Nothing`` (``-``); it has to be at least the margin, and a large value scales
        every loss that involves them. Default ``1.0``; the earlier mOWL module used ``1e4``.
    """

    region = FuzzySet
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
        return FuzzySet(values, self.family)


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
    named individuals, plus ``extra`` anonymous elements (learned rows) when given."""

    def __init__(self, table, extra=0):
        super().__init__()
        self.table = table
        self.extra = None
        if extra:
            self.extra = nn.Parameter(nn.init.xavier_uniform_(
                th.empty(extra, table.embedding_dim)))

    def forward(self):
        weight = self.table.weight
        return weight if self.extra is None else th.cat([weight, self.extra], dim=0)

    def __len__(self):
        return self.table.num_embeddings + (0 if self.extra is None else len(self.extra))


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


class DomainElements(_FuzzyClasses, Classes):
    """Individuals that are the elements of the domain: the crisp set of element ``i``
    (a one-hot vector). The table is fixed.

    :param size: Number of individuals; :param domain: Size of the domain (at least
        ``size``).
    """

    region = FuzzySet
    part = ClassModel.FUZZY_SET.value
    exact_conjunction = True

    def __init__(self, size, domain, family="godel"):
        super().__init__()
        if domain < size:
            raise ValueError(f"The domain has {domain} elements for {size} individuals.")
        self.centres = nn.Embedding.from_pretrained(th.eye(size, domain), freeze=True)
        self.family = get_family(family)

    def forward(self, ids):
        return FuzzySet(self.centres(ids), self.family, index=ids)


# =========================================================================================
# relation models
# =========================================================================================

class _FuzzyRelations(Relations):
    """Relations as fuzzy relations on the domain: a table ``(N, N)`` per relation, the
    element ``[a, b]`` being the degree of ``r(a, b)``. :meth:`image` is
    :math:`\\sup_a x(a) \\sqcap r(a, b)`, :meth:`preimage` the existential restriction."""

    part = RelationModel.FUZZY_RELATION.value

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

    def image(self, ids, region):
        family = region.family
        return FuzzySet(family.sup(family.and_(self.matrix(ids), region.values.unsqueeze(-1)),
                                   -2), family)

    def preimage(self, ids, region):
        return FuzzySet(region.family.exists(self.matrix(ids), region.values), region.family)


class TruthMatrix(_FuzzyRelations):
    """The truth values of the pairs, ``sigmoid(logit)`` (DF-ALC)."""

    def matrix(self, ids):
        n = self.domain
        return th.sigmoid(self.vectors(ids)).view(*ids.shape, n, n)


class IncidenceMatrices(_FuzzyRelations):
    """Logits of the incidence matrix of a relation between half-axes (Al-cone): entry
    ``[j, i]`` of the stored matrix is the logit of "half-axis ``i`` has successor ``j``",
    so the table of this part (source first) is its transpose.

    :param size: Number of relations; :param dim: :math:`n` (the matrix is
        ``2 n x 2 n``).
    """

    def __init__(self, size=None, dim=None, vectors=None, init_std=0.5):
        super().__init__(size, None if dim is None else 2 * dim, vectors, init_std)

    def matrix(self, ids):
        n = self.domain
        return self.vectors(ids).view(*ids.shape, n, n).transpose(-1, -2)


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

    @staticmethod
    def _no_margin(margin):
        if margin:
            raise ValueError("ConflictMeasure has no margin (the conflict is graded).")

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
    """

    part = "fuzzy implication"

    def __init__(self, empty_margin=0.1, tau=0.0, source=None):
        self.empty_margin = empty_margin
        self._tau = tau
        self.source = source

    @property
    def tau(self):
        return self.source.temperature if self.source is not None else self._tau

    def inside(self, a, b, margin=0.0):
        self.check(a, b)
        return (th.relu(margin + a.values) * th.relu(margin - b.values)).mean(
            dim=-1, keepdim=True)

    def apart(self, a, b, margin=0.0):
        self.check(a, b)
        return (th.relu(margin + a.values) * th.relu(margin + b.values)).mean(
            dim=-1, keepdim=True)

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
register_measure("half-axis hinge")(ConeMeasure)


# =========================================================================================
# the objective and the geometry
# =========================================================================================

class FuzzyObjective(Objective):
    """The reading of the axiom forms for fuzzy geometries.

    Besides the options of :class:`~mowl.representation.Objective`:

    :param rule: ``True``: the rule-based losses of DF-ALC (Eq. 9 for :math:`C \\sqsubseteq D`
        with the gradient only through :math:`D`, Eq. 10 for :math:`C \\sqsubseteq \\exists r.D`,
        Eq. 12 for :math:`\\exists r.C \\sqsubseteq D`).
    :param threshold: The threshold :math:`\\alpha` of Eqs. 10 and 12.
    :param validity: Add the well-formedness cost of the individuals to assertions (Al-cone).
    :param role_assertion: ``"preimage"`` (default): :math:`a \\in \\exists r.\\{b\\}`.
        ``"image"``: :math:`b` lies in the image of :math:`a` under :math:`r`. Equal for
        crisp relations; the published losses of all three methods read it as ``"image"``.
    """

    def __init__(self, margin=0.0, rule=False, threshold=0.8, validity=False,
                 role_assertion="preimage", **kwargs):
        super().__init__(margin, **kwargs)
        if role_assertion not in ("preimage", "image"):
            raise ValueError("role_assertion must be 'preimage' or 'image'.")
        self.rule, self.threshold = bool(rule), float(threshold)
        self.validity, self.role_assertion = bool(validity), role_assertion

    @classmethod
    def dfalc(cls, loss_type="hierarchical", threshold=0.8):
        return cls(0.0, rule=loss_type == "rule", threshold=threshold, role_assertion="image",
                   negatives="all")

    @classmethod
    def falcon(cls):
        return cls(0.0, role_assertion="image", negatives="all")

    @classmethod
    def alcone(cls, margin=0.1):
        return cls(margin, validity=True, role_assertion="image", negatives="all")

    # -- rule-based losses --------------------------------------------------------------
    def gci0(self, g, rows, neg=False):
        if self.rule and not neg:
            c, d = g.classes(rows[:, 0]), g.classes(rows[:, 1])
            return _df().rule_based_loss(c.values, d.values).unsqueeze(-1)
        return super().gci0(g, rows, neg)

    def gci1(self, g, rows, neg=False):
        if self.rule and not neg:
            c, d, e = (g.classes(rows[:, i]) for i in range(3))
            both, _ = c.intersect(d)
            return _df().rule_based_loss(both.values, e.values).unsqueeze(-1)
        return super().gci1(g, rows, neg)

    def gci2(self, g, rows, neg=False):
        if self.rule and not neg:
            b, a = g.classes(rows[:, 0]), g.classes(rows[:, 2])
            r = g.relation_model.matrix(rows[:, 1])
            return _df().rule_nf4(b.values, a.values, r, self.threshold).unsqueeze(-1)
        return super().gci2(g, rows, neg)

    def gci3(self, g, rows, neg=False):
        if self.rule and not neg:
            a, b = g.classes(rows[:, 1]), g.classes(rows[:, 2])
            r = g.relation_model.matrix(rows[:, 0])
            return _df().rule_nf6(a.values, b.values, r, self.threshold).unsqueeze(-1)
        return super().gci3(g, rows, neg)

    def gci0_bot(self, g, rows, neg=False):
        if self.rule and not neg:
            c = g.classes(rows[:, 0])
            return _df().rule_based_loss(c.values, th.zeros_like(c.values)).unsqueeze(-1)
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


# =========================================================================================
# registered parts (names to combine with ``compose`` / ``ComposedEL``)
# =========================================================================================

def _truth_table(size, embed_dim, family="godel"):
    return TruthTableClasses(size, embed_dim, family=family)


register_class_model("fuzzy set")(_truth_table)
register_class_model("fuzzy set (product)")(lambda s, d: _truth_table(s, d, "product"))
register_class_model("fuzzy set (lukasiewicz)")(
    lambda s, d: _truth_table(s, d, "lukasiewicz"))
register_class_model("half-axis cone")(
    lambda s, d: HalfAxisClasses(s, d))
register_relation_model("fuzzy relation")(lambda s, d: TruthMatrix(s, d))
register_relation_model("incidence matrix")(lambda s, d: IncidenceMatrices(s, d))


# =========================================================================================
# the methods
# =========================================================================================

def _domain_size(nb_individuals, embed_dim, domain):
    n = domain if domain is not None else max(embed_dim, nb_individuals)
    if n < max(nb_individuals, 1):
        raise ValueError("The domain must hold every individual.")
    return n


@register_method("dfalc")
def dfalc(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.0,
          domain=None, loss_type="hierarchical", threshold=0.8, family="godel",
          top_index=None, bottom_index=None, init_std=1.0, tables=None):
    """DF-ALC (Wu et al. 2022): truth tables over a finite domain, truth matrices for the
    roles, Goedel operators, the mean violation as the measure. Individual ``i`` is domain
    element ``i``.

    :param embed_dim: Size of the domain if ``domain`` is not given (at least the number of
        individuals).
    :param family: The operator family; ``"godel"`` is the published method, any member of
        :data:`FAMILIES` can be used instead.
    :param loss_type: ``"hierarchical"`` (Eq. 8) or ``"rule"`` (Eqs. 9, 10, 12).
    :param tables: A :class:`~mowl.nn.alc.dfalc.DFALCModule` to share parameters with.
    """
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
    individuals = DomainElements(n, n, family=family)
    return ComposedGeometry(classes, relations, MeanViolation(),
                            FuzzyObjective.dfalc(loss_type, threshold), individuals=individuals)


@register_method("falcon")
def falcon(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.0,
           t_norm="product", max_measure="max", stabiliser="add", role_pairs="all",
           anon_domain=0, top_index=None, bottom_index=None, thing="learned",
           nothing="empty", tables=None):
    """FALCON (Tang et al. 2022): class vectors, a membership network over (vector, element
    vector) pairs, a network over pairs for roles, a t-norm, the conflict of the
    counterexample as the measure. The domain is the table of the individuals (plus
    ``anon_domain`` learned anonymous elements).

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
    :param tables: A :class:`~mowl.nn.alc.falcon.FALCONModule` to share parameters with.
    """
    if thing not in ("learned", "all") or nothing not in ("empty", "learned"):
        raise ValueError("thing must be 'learned' or 'all', nothing 'empty' or 'learned'.")
    top_index = top_index if thing == "all" else None
    bottom_index = bottom_index if nothing == "empty" else None
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
        n = len(domain)
        measure = ConflictMeasure(t.max_measure, stabiliser)
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
        n = len(domain)
        measure = ConflictMeasure(max_measure, stabiliser)
    individuals = DomainElements(table.num_embeddings if tables is None
                                 else tables.e_embedding.num_embeddings, n, family=t_norm)
    geometry = ComposedGeometry(classes, relations, measure, FuzzyObjective.falcon(),
                                individuals=individuals)
    geometry.domain = domain
    geometry.network = network
    return geometry


@register_method("alcone")
def alcone(nb_classes, nb_relations, nb_individuals=0, embed_dim=8, margin=0.1,
           temperature=0.0, family=None, top_index=None, bottom_index=None,
           constant_logit=DEFAULT_CONSTANT_LOGIT, tables=None):
    """Learnable Al-cone (adaptation of Oezcep, Leemhuis and Wolter 2023): half-axis logits
    for classes and individuals, incidence logits for roles, Goedel operators on logits, the
    half-axis hinge as the measure, and a well-formedness cost on individuals.

    :param embed_dim: :math:`n`; there are ``2 n`` half-axes.
    :param margin: The strict margin :math:`\\gamma > 0`.
    :param temperature: Smoothing of the operators (``0``: exact); a module's temperature is
        read live.
    :param constant_logit: Magnitude of the logits of ``owl:Thing`` and ``owl:Nothing``, see
        :class:`HalfAxisClasses`.
    :param family: Another operator family on logits instead of Goedel (any
        :data:`FAMILIES` entry whose values are logits).
    :param tables: A :class:`~mowl.nn.alc.alcone.ALConeModule` to share parameters with.
    """
    t = tables
    if t is not None:
        fam = LogitGoedel(source=t) if family is None else get_family(family)
        classes = HalfAxisClasses(logits=t.c_embedding, family=fam, top_index=t.top_index,
                                  bottom_index=t.bottom_index,
                                  constant_logit=t.constant_logit)
        relations = IncidenceMatrices(vectors=t.r_embedding, dim=t.dim)
        individuals = HalfAxisClasses(logits=t.e_embedding, family=fam)
        measure = ConeMeasure(source=t)
        margin = t.margin
    else:
        fam = LogitGoedel(temperature) if family is None else get_family(family)
        classes = HalfAxisClasses(nb_classes, embed_dim, family=fam, top_index=top_index,
                                  bottom_index=bottom_index, constant_logit=constant_logit)
        relations = IncidenceMatrices(max(nb_relations, 1), embed_dim)
        individuals = HalfAxisClasses(max(nb_individuals, 1), embed_dim, family=fam)
        measure = ConeMeasure(tau=temperature)
    return ComposedGeometry(classes, relations, measure, FuzzyObjective.alcone(margin),
                            individuals=individuals)


@register_adapter("DFALCModule")
def _dfalc_adapter(module):
    return dfalc(None, None, tables=module)


@register_adapter("FALCONModule")
def _falcon_adapter(module):
    return falcon(None, None, t_norm=module.t_norm, max_measure=module.max_measure,
                  tables=module)


@register_adapter("ALConeModule")
def _alcone_adapter(module):
    return alcone(None, None, tables=module)
