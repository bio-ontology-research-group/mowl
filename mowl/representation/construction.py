"""Constructed methods as parts: a source, a reading, a measure.

Some methods are not trained: a model of the ontology is *constructed* (by completion, by a
solver, in closed form) and then read as regions. They are combinations of three parts:

* a **source** (:class:`Source`) builds the model from the data: the canonical model of an
  EL++ ontology by completion, the least model of indexed tables, the atoms of a Boolean
  TBox, an ultimately periodic temporal model, a grounding of an attributed ontology, a box
  model found by a mixed-integer programme. It states the logic it covers and the reasoner
  it needs. Most sources produce a :class:`FiniteModel` (a finite interpretation: a domain
  ``0 .. n - 1``, the extension of every concept and role, the element of every individual);
  a temporal source a :class:`TemporalModel`; a solver its own result (a set of boxes);
* a **reading** (:class:`Reading`) turns what the source built into regions of one family
  (element sets, axis-aligned cones, unions of intervals, faces of a simplex, boxes): the
  class model, relation model and individual model of the method;
* the **measure** (:class:`ExactMeasure`) decides inclusion, disjointness, emptiness and
  membership of these regions exactly: a cost is 0.0 where the relation holds and 1.0
  where it does not.

The three are put together into a :class:`~mowl.representation.ComposedGeometry`
(:class:`ConstructionGeometry`) with the default, semantic objective, so that a constructed
model answers the same questions as a trained one (``module(rows, "gci0")``, ``classes``,
``exists``, ``conjunction``, ``inside``, ...) and the evaluation regimes rank with it as with
any other model. :func:`construct` is the one entry point; a named method is a
:class:`Recipe` (a source and a reading with options) registered under its catalogue key
with :func:`~mowl.representation.register_method` (``trained_on="construction"``).

**Readings of a finite model.** A reading of a finite model maps every set ``S`` of domain
elements (more precisely: of *cells*, the elements or the classes of elements the reading
places in space) to a region :math:`\\iota(S)`, with
:math:`\\iota(S) \\subseteq \\iota(T) \\iff S \\subseteq T` and
:math:`\\iota(S \\cap T) = \\iota(S) \\cap \\iota(T)`. The region of a concept name is
:math:`\\iota(A^{\\mathcal{I}})`, so every normal-form axiom holds in the regions iff it
holds in the finite model. Such regions are :class:`CellRegion` objects: they keep the set
of cells (``mask``) they are made of, next to the family's own geometry (``contains``), and
the operations of the family act on the cells. Where the family is not closed under an
operation (a union of two faces of a simplex is not convex) the region raises
:class:`~mowl.representation.ConstructorUnavailable`. A reading that does not apply to what
a source built (a set of boxes has no domain elements to place on half-axes; a finite model
cannot in general be read as single boxes, because boxes have the Helly property) raises
the same error when the combination is put together.

**Time.** A :class:`CellRegion` has a time axis: ``mask`` is ``(B, T, n)``, ``T = 1`` for a
model without time. ``next`` and ``until`` act on it (a model without time is a constant
trace, so ``next`` is the identity there); a relation between regions holds when it holds
at every time point (the semantics of a GCI), unless the measure is asked about one time
point (:attr:`ExactMeasure.time`).
"""

from dataclasses import dataclass, field
from typing import Optional

import torch as th

from mowl.representation.class_models import Classes
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.measures import RegionMeasure
from mowl.representation.objectives import Objective
from mowl.representation.regions import Box, ConstructorUnavailable, Region
from mowl.representation.relation_models import Relations

__all__ = ["FiniteModel", "TemporalModel", "Guarantee", "Source", "Reading", "ReadModel",
           "Points", "CellRegion", "CellRelations", "RegionTable", "ExactMeasure",
           "ConstructionGeometry", "Recipe", "construct", "SOURCES", "READINGS",
           "RECIPES", "register_source", "register_reading", "register_recipe", "TOP",
           "BOTTOM"]

#: The names of the top and bottom class (those of :mod:`mowl.geometric.normal_forms`).
TOP = "http://www.w3.org/2002/07/owl#Thing"
BOTTOM = "http://www.w3.org/2002/07/owl#Nothing"


# -- what a source builds ---------------------------------------------------------------------
def _frozen(values):
    return frozenset(int(v) for v in values)


@dataclass
class FiniteModel:
    """A finite interpretation.

    :param size: Number of domain elements; the domain is ``0 .. size - 1``.
    :param concepts: ``{name: set of elements}``.
    :param roles: ``{name: set of pairs (x, y)}``.
    :param individuals: ``{name: element}``, or ``{name: set of elements}`` for an
        individual that is only known to be one of several elements (the strongly faithful
        individuals of the Boolean al-cone construction, JAIR Prop. 8: membership in a
        class means that *all* of them belong to it).
    :param origin: What the model was read from (a canonical model, a least model, ...),
        for parts that need more than the interpretation.
    """

    size: int
    concepts: dict = field(default_factory=dict)
    roles: dict = field(default_factory=dict)
    individuals: dict = field(default_factory=dict)
    origin: object = None

    def __post_init__(self):
        if self.size < 1:
            raise ValueError("A finite model needs at least one element.")
        self.concepts = {k: _frozen(v) for k, v in self.concepts.items()}
        self.roles = {k: frozenset((int(x), int(y)) for x, y in v)
                      for k, v in self.roles.items()}
        self.individuals = {k: (int(v) if not isinstance(v, (set, frozenset, list, tuple))
                                else _frozen(v)) for k, v in self.individuals.items()}
        for name, ext in self.concepts.items():
            if any(not 0 <= x < self.size for x in ext):
                raise ValueError(f"Concept {name!r} has elements outside the domain.")

    # -- conversions -----------------------------------------------------------------------
    @classmethod
    def from_canonical_model(cls, cm, nominals=True):
        """The canonical model (:class:`~mowl.geometric.completion.CanonicalModel`): one
        element per satisfiable name (positions of ``cm.names``), every concept name of
        its signature (and, with ``nominals``, the nominal names ``N_a`` of the
        individuals), every role, and the individuals as the elements of their nominals."""
        names = set(cm.concept_names) | (set(cm.names) if nominals else set())
        concepts = {n: cm.extension(n) for n in names if n not in (TOP, BOTTOM)}
        roles = {r: cm.role(r) for r in cm.role_names}
        individuals = {a: cm.element_of(a) for a in cm.individuals}
        return cls(len(cm.names), concepts, roles, individuals, origin=cm)

    @classmethod
    def of(cls, model):
        """A :class:`FiniteModel` from any finite interpretation of the library: this
        class, a canonical model, the finite interpretations of
        :mod:`mowl.geometric.convex_regions` and :mod:`mowl.geometric.cones` (with their
        individuals) and a :class:`~mowl.geometric.least_model.LeastModel` (domain: its
        live blocks)."""
        if isinstance(model, FiniteModel):
            return model
        name = type(model).__name__
        if name == "CanonicalModel":
            return cls.from_canonical_model(model)
        if name == "LeastModel":
            return cls._from_least_model(model)
        size = getattr(model, "size", getattr(model, "domain_size", None))
        if size is None or not hasattr(model, "concepts"):
            raise TypeError(f"{name} is not a finite interpretation.")
        return cls(size, dict(model.concepts), dict(getattr(model, "roles", {}) or {}),
                   dict(getattr(model, "individuals", {}) or {}), origin=model)

    @classmethod
    def _from_least_model(cls, lm):
        import numpy as np
        live = np.flatnonzero(~np.asarray(lm.dead, dtype=bool))
        index = {int(b): i for i, b in enumerate(live)}
        labels = lm.labels.tocsc()
        concepts = {}
        for c in range(lm.nb_classes):
            rows = labels[:, c].nonzero()[0]
            concepts[c] = {index[int(b)] for b in rows if int(b) in index}
        roles = {r: set() for r in range(lm.nb_rels)}
        for x, y, r in lm.edges.tolist():
            if x in index and y in index:
                roles[int(r)].add((index[x], index[y]))
        individuals = {k: index[int(lm.element_map[lm.nb_classes + k])]
                       for k in range(lm.nb_individuals)
                       if int(lm.element_map[lm.nb_classes + k]) in index}
        return cls(max(len(live), 1), concepts, roles, individuals, origin=lm)

    # -- reading -----------------------------------------------------------------------------
    def extension(self, name):
        """The elements of ``name``: all for ``TOP``, none for ``BOTTOM`` and unknown
        names."""
        if name == TOP:
            return frozenset(range(self.size))
        if name == BOTTOM:
            return frozenset()
        return self.concepts.get(name, frozenset())

    def role(self, name):
        return self.roles.get(name, frozenset())

    @property
    def partial_individuals(self):
        """Whether some individual is a set of several elements."""
        return any(isinstance(v, frozenset) for v in self.individuals.values())

    def element_set(self, individual):
        value = self.individuals[individual]
        return value if isinstance(value, frozenset) else frozenset((value,))


@dataclass
class TemporalModel:
    """An ultimately periodic temporal model over one domain (constant domain, rigid
    individuals): ``steps[t]`` is the :class:`FiniteModel` at time point ``t``
    (``0 .. L - 1``) and time point ``L - 1`` is followed by ``loop_start``."""

    steps: tuple
    loop_start: int = 0
    individuals: dict = field(default_factory=dict)
    origin: object = None

    def __post_init__(self):
        self.steps = tuple(self.steps)
        if not self.steps or len({s.size for s in self.steps}) != 1:
            raise ValueError("Every time point needs a finite model over the same domain.")
        if not 0 <= self.loop_start < len(self.steps):
            raise ValueError("loop_start must be a time point.")

    @property
    def size(self):
        return self.steps[0].size

    @property
    def length(self):
        return len(self.steps)

    def successors(self):
        """The time point after every time point, ``(L,)``."""
        n = self.length
        return th.tensor([t + 1 if t + 1 < n else self.loop_start for t in range(n)])


@dataclass(frozen=True)
class Guarantee:
    """A guarantee of a source or a reading, as data.

    :param kind: A value of :class:`mowl.catalog.GuaranteeKind` (``"soundness"``,
        ``"strong_faithfulness"``, ``"completeness"``, ``"closure"``, ...).
    :param statement: What holds.
    :param logic: For which logic or axiom forms.
    :param condition: When (``"by construction"``, ``"verified on construction"``, ...).
    """

    kind: str
    statement: str
    logic: str
    condition: str = "by construction"


# -- the parts ----------------------------------------------------------------------------------
class Source:
    """Where the model comes from.

    :cvar key: Name in :data:`SOURCES`.
    :cvar kind: ``"completion"``, ``"least model"``, ``"closed form"``, ``"search"``,
        ``"grounding"`` or ``"solver"``.
    :cvar logic: The logic of the data it accepts.
    :cvar reasoner: The reasoning the construction needs (and which mOWL code does it).
    :cvar produces: The type of what :meth:`build` returns.
    :cvar guarantees: :class:`Guarantee` objects that hold for what it builds.
    """

    key = ""
    kind = ""
    logic = ""
    reasoner = ""
    produces = FiniteModel
    guarantees = ()

    def build(self, data):
        """The model built from ``data``."""
        raise NotImplementedError

    def name_of(self, name):
        """The name of the built model for a name given to the geometry (identity; the
        attributed grounding maps ``(concept, specifier)`` to its grounded name)."""
        return name

    def default_names(self, built):
        """``(class names, role names, individual names)`` of the built model, in the
        order of their ids."""
        from mowl.geometric.normal_forms import is_nominal_name
        concepts = getattr(built, "concepts", {})
        names = sorted((n for n in concepts if not (isinstance(n, str) and
                                                    is_nominal_name(n))), key=str)
        return names, sorted(getattr(built, "roles", {}), key=str), \
            sorted(getattr(built, "individuals", {}), key=str)

    def __repr__(self):
        options = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in options.items())})"


@dataclass
class ReadModel:
    """What a reading returns: the regions of the named classes (one row per class id),
    the relation model (``None`` if the reading has none), the individuals (a
    :class:`Points` region, or ``None``), the universe and the empty region (one row
    each), and the names in the order of the ids."""

    classes: Region
    relations: Optional[Relations]
    individuals: Optional[Region]
    universe: Region
    empty: Region
    names: list
    roles: list
    individual_names: list


class Reading:
    """How what a source built becomes regions of one family.

    :cvar key: Name in :data:`READINGS`.
    :cvar region: The :class:`~mowl.representation.Region` class of its regions.
    :cvar accepts: The types of built models it can read.
    :cvar class_part, relation_part, individual_part: Vocabulary values
        (:class:`~mowl.representation.ClassModel`, ``RelationModel``, ``IndividualModel``).
    :cvar constructors: ``{"conjunction": ..., "existential": ..., "negation": ...,
        "disjunction": ...}`` (:class:`~mowl.representation.Construction` values).
    :cvar guarantees: :class:`Guarantee` objects of the reading (what it preserves).
    """

    key = ""
    region = Region
    accepts = (FiniteModel, TemporalModel)
    class_part = ""
    relation_part = "none"
    individual_part = "none"
    constructors = {}
    guarantees = ()

    def check(self, built):
        """Raise :class:`~mowl.representation.ConstructorUnavailable` if this reading does
        not apply to ``built``."""
        if not isinstance(built, self.accepts):
            raise ConstructorUnavailable(
                f"The {self.key} reading reads {[c.__name__ for c in self.accepts]}, not "
                f"{type(built).__name__}.")

    def read(self, built, names, roles, individuals, source):
        """A :class:`ReadModel` of ``built`` for the given class names, role names and
        individual names (``source.name_of`` maps a given name to the built model's)."""
        raise NotImplementedError

    def __repr__(self):
        options = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in options.items())})"


# -- regions read off cells -------------------------------------------------------------------
def _flag(value):
    return value.to(th.get_default_dtype()).reshape(-1, 1)


@dataclass(init=False, eq=False)
class Points(Region):
    """Points of the space of a reading (the individuals): ``centre`` ``(B, d)``, and the
    cells ``mask`` ``(B, T, n)`` the points belong to (for relation models over cells)."""

    centre: th.Tensor
    mask: Optional[th.Tensor] = None

    def __init__(self, centre, mask=None):
        self.centre, self.mask = centre, mask

    def __len__(self):
        return self.centre.shape[0]

    def take(self, ids):
        return Points(self.centre[ids], None if self.mask is None else self.mask[ids])

    def tensors(self):
        return {"points": self.centre}


@dataclass(init=False, eq=False)
class CellRegion(Region):
    """A region made of cells of a reading (see the module docstring).

    :param mask: ``(B, T, n)`` boolean (or ``(B, n)`` for one time point): the cells of every
        row at every time point.
    :param geometry: The geometry of the cells, shared by all regions of one reading
        (``successor``: ``(T,)`` long, the time point after every time point; the
        family's own entries); further keyword arguments are added to it.

    Subclasses give the geometry of the cells (:meth:`contains`, :meth:`point`,
    :meth:`tensors`) and say under which operations the family is closed.
    """

    mask: th.Tensor
    geometry: dict

    #: Whether the union of two regions of the family is one (a set union, or the convex
    #: union where that is the family's disjunction).
    union_closed = False
    #: Whether the complement (relative to the universe of the reading) is a region.
    complement_closed = False
    #: Whether the preimage of a region under a relation of cells is the region of the
    #: preimage cells (exact :math:`\\exists r.C`).
    exists_exact = True

    def __init__(self, mask, geometry=None, **extra):
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        self.mask = mask.to(th.bool)
        if geometry is None or extra:
            geometry = dict(geometry or {}, **extra)
        geometry.setdefault("successor", th.zeros(self.mask.shape[1], dtype=th.long))
        self.geometry = geometry

    def __len__(self):
        return self.mask.shape[0]

    @property
    def successor(self):
        return self.geometry["successor"]

    @property
    def n_cells(self):
        return self.mask.shape[-1]

    def with_mask(self, mask):
        """A region of the same family and geometry with other cells."""
        return type(self)(mask, self.geometry)

    def take(self, ids):
        return self.with_mask(self.mask[ids])

    # -- operations on cells -----------------------------------------------------------------
    def _same(self, other):
        if not isinstance(other, CellRegion) or other.n_cells != self.n_cells:
            raise ConstructorUnavailable(
                f"{type(self).__name__} and {type(other).__name__} are regions of different "
                f"readings.")
        if len(self) != len(other) and min(len(self), len(other)) != 1:
            raise ValueError("Both sides need the same batch size.")

    def intersect(self, other):
        """Exact: the cells of both; ``gap`` is 1.0 where the result is empty."""
        self._same(other)
        region = self.with_mask(self.mask & other.mask)
        return region, _flag(region.is_empty())

    def union(self, other):
        if not self.union_closed:
            raise ConstructorUnavailable(
                f"The union of two {type(self).__name__} regions is not a region of the "
                f"family.")
        self._same(other)
        return self.with_mask(self.mask | other.mask)

    def complement(self):
        if not self.complement_closed:
            raise ConstructorUnavailable(
                f"{type(self).__name__} regions are not closed under complement.")
        return self.with_mask(~self.mask)

    def next(self):
        """:math:`\\circ C`: the cells of the next time point."""
        return self.with_mask(self.mask[:, self.successor])

    def until(self, other):
        """:math:`C\\,\\mathcal{U}\\,D` at every time point ``i``: the union over
        ``m >= i`` of :math:`D(m) \\cap \\bigcap_{i \\le k < m} C(k)`, unrolled over the
        ``T`` positions reachable from ``i``. Needs a family closed under union."""
        if not self.union_closed:
            raise ConstructorUnavailable(
                f"until needs unions, and {type(self).__name__} regions are not closed under "
                f"union.")
        self._same(other)
        steps = self.mask.shape[1]
        pos = th.arange(steps)
        out = th.zeros_like(self.mask | other.mask)
        prefix = th.ones_like(out)
        for _ in range(steps):
            out = out | (prefix & other.mask[:, pos])
            prefix = prefix & self.mask[:, pos]
            pos = self.successor[pos]
        return self.with_mask(out)

    # -- relations (exact) ---------------------------------------------------------------------
    def at(self, time=None):
        """The mask ``(B, T', n)`` at one time point (``T' = 1``) or at all
        (``time=None``)."""
        if time is None:
            return self.mask
        return self.mask[:, [int(time) % self.mask.shape[1]]]

    def is_empty(self, time=None):
        """``(B,)``: no cell (at any time point, or at ``time``)."""
        return ~self.at(time).any(dim=(1, 2))

    def subset_of(self, other, time=None):
        """``(B,)``: the cells of every row lie among those of ``other`` (at every time
        point, or at ``time``). By the property of the reading this is inclusion of the
        regions."""
        self._same(other)
        return ~(self.at(time) & ~other.at(time)).any(dim=(1, 2))

    def meets(self, other, time=None):
        """``(B,)``: a common cell (at some time point, or at ``time``)."""
        self._same(other)
        return (self.at(time) & other.at(time)).any(dim=(1, 2))

    def contains(self, points, time=0):
        """``(B, P)``: whether each point ``(P, d)`` lies in the region of each row at time
        ``time`` (the family's geometry, not its cells)."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no point-membership test.")

    def contains_rows(self, points, time=0):
        """``(B,)``: whether point ``b`` lies in region ``b``."""
        out = self.contains(points, time)
        if out.shape[0] == 1:
            return out[0]
        if out.shape[1] == 1:
            return out[:, 0]
        return out.diagonal()

    def point(self, cells):
        """The point of the space that stands for the set of cells ``cells`` (an
        individual)."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no points.")

    def tensors(self):
        """The region as tensors (for comparisons and export)."""
        return {"mask": self.mask}


class CellRelations(Relations):
    """A relation is a set of pairs of cells: ``pairs`` ``(R, n, n)`` boolean,
    ``pairs[r, x, y]`` iff :math:`(x, y) \\in r`. Read off the roles of a finite model.

    The preimage of a region (:math:`\\exists r.C`) is the region of the cells with an
    ``r``-successor among the cells of the region; it is exact where the family says so
    (``exists_exact``). Role inclusion and role chains are decided on the pairs.

    :param part: The vocabulary value of what a relation is in the reading (an incidence
        matrix of half-axes, a region over pairs, a relation between elements).
    """

    def __init__(self, pairs, part="fixed by reasoning"):
        super().__init__()
        self.register_buffer("pairs", pairs.to(th.bool))
        self.part = part

    def __len__(self):
        return self.pairs.shape[0]

    def _cells(self, region):
        if isinstance(region, Points):
            if region.mask is None:
                raise ConstructorUnavailable("These points do not belong to cells.")
            return region.mask
        if not isinstance(region, CellRegion):
            raise ConstructorUnavailable(
                f"{type(self).__name__} maps regions made of cells, not "
                f"{type(region).__name__}.")
        if not region.exists_exact:
            raise ConstructorUnavailable(
                f"exists r.C of {type(region).__name__} regions is not a region of the "
                f"family (only of the regions spanned by cells).")
        return region.mask

    def preimage(self, ids, region):
        mask = self._cells(region).to(th.float64)            # (B, T, n)
        pairs = self.pairs[ids].to(th.float64)                # (B, n, n)
        out = th.einsum("bxy,bty->btx", pairs, mask) > 0
        template = region if isinstance(region, CellRegion) else self._owner
        return template.with_mask(out)

    def image(self, ids, region):
        mask = self._cells(region).to(th.float64)
        pairs = self.pairs[ids].to(th.float64)
        out = th.einsum("bxy,btx->bty", pairs, mask) > 0
        template = region if isinstance(region, CellRegion) else self._owner
        return template.with_mask(out)

    def bind_template(self, region):
        """The (empty) region whose family the image of points is given in."""
        self._owner = region
        return self

    def role_inclusion(self, sub, sup, measure=None, margin=0.0):
        a, b = self.pairs[sub], self.pairs[sup]
        return _flag((a & ~b).flatten(1).any(dim=1))

    def role_chain(self, first, second, sup, measure=None, margin=0.0):
        composed = (self.pairs[first].to(th.float64) @ self.pairs[second].to(th.float64)) > 0
        return _flag((composed & ~self.pairs[sup]).flatten(1).any(dim=1))


def _take(region, ids):
    if hasattr(region, "take"):
        return region.take(ids)
    from dataclasses import fields, is_dataclass, replace
    if is_dataclass(region):
        values = {f.name: getattr(region, f.name)[ids] for f in fields(region)
                  if isinstance(getattr(region, f.name), th.Tensor)}
        return replace(region, **values)
    raise TypeError(f"Cannot take rows of {type(region).__name__}.")


class RegionTable(Classes):
    """A class model without parameters: the regions read off a constructed model, one row
    per class (or individual) id. ``forward(ids)`` returns their rows.

    :param regions: A region with one row per id.
    :param part: The vocabulary value of the class model.
    :param exact: Whether the family is closed under intersection.
    """

    def __init__(self, regions, part="fixed by reasoning", exact=True, extra=None):
        super().__init__()
        self.regions = regions
        self.part = part
        self.exact_conjunction = exact
        self.region = type(regions)
        self._extra = dict(extra or {})

    def forward(self, ids, detach_size=False):
        return _take(self.regions, ids.to("cpu"))

    def __len__(self):
        return len(self.regions)

    @property
    def extra_parts(self):
        return dict(self._extra)

    def tables(self):
        return {}


# -- the measure ----------------------------------------------------------------------------------
def _box_empty(a, tol):
    return (a.lower > a.upper + tol).any(dim=-1)


class ExactMeasure(RegionMeasure):
    """Inclusion, overlap, disjointness, emptiness and membership decided exactly: costs
    0.0 where the relation holds and 1.0 where it does not (no margin, no gradient).

    Regions made of cells (:class:`CellRegion`) are compared on their cells (see the module
    docstring); boxes (:class:`~mowl.representation.Box`) by their corners with the
    tolerance ``tol`` of :class:`~mowl.geometric.checker.ModelChecker`; an individual
    (:class:`Points`) is a member of a region when the region contains its point.

    :param tol: Tolerance of the box comparisons.
    :param time: ``None``: a relation holds when it holds at every time point (the
        semantics of a GCI); an int: at that time point only. Membership is read at time
        ``0`` (the convention of the temporal constructions) unless ``time`` is given.
    """

    accepts = (CellRegion, Box, Points)
    part = "exact"
    reading = "truth"

    def __init__(self, tol=1e-9, time=None):
        self.tol = float(tol)
        self.time = time

    @staticmethod
    def _margin(margin):
        if margin:
            raise ValueError("An exact measure has no margin (the relations hold or do not).")

    # -- the three questions, as booleans ------------------------------------------------------
    def _empty(self, a):
        if isinstance(a, CellRegion):
            return a.is_empty(self.time)
        if isinstance(a, Box):
            return _box_empty(a, self.tol)
        raise ConstructorUnavailable(f"{type(a).__name__} cannot say whether it is empty.")

    def _subset(self, a, b):
        if isinstance(a, Points):
            return self._contains(b, a)
        if isinstance(a, CellRegion):
            return a.subset_of(b, self.time)
        if isinstance(a, Box) and isinstance(b, Box):
            inside = ((b.lower - self.tol <= a.lower) & (a.upper <= b.upper + self.tol)).all(-1)
            return _box_empty(a, self.tol) | (inside & ~_box_empty(b, self.tol))
        raise ConstructorUnavailable(f"No exact inclusion of {type(a).__name__} in "
                                     f"{type(b).__name__}.")

    def _meet(self, a, b):
        if isinstance(a, Points):
            return self._contains(b, a)
        if isinstance(b, Points):
            return self._contains(a, b)
        if isinstance(a, CellRegion):
            return a.meets(b, self.time)
        if isinstance(a, Box) and isinstance(b, Box):
            low, high = th.maximum(a.lower, b.lower), th.minimum(a.upper, b.upper)
            return ~(_box_empty(a, self.tol) | _box_empty(b, self.tol)
                     | (low > high + self.tol).any(-1))
        raise ConstructorUnavailable(f"No exact overlap of {type(a).__name__} and "
                                     f"{type(b).__name__}.")

    def _contains(self, region, points):
        """``(B,)``: the point of individual ``b`` lies in region ``b`` (the family's own
        point-membership test, at time 0 or at :attr:`time`)."""
        known = ~th.isnan(points.centre).any(-1)          # an individual of no cell
        if isinstance(region, CellRegion):
            time = 0 if self.time is None else self.time
            return region.contains_rows(th.nan_to_num(points.centre), time) & known
        if isinstance(region, Box):
            return _contains_rows(region, th.nan_to_num(points.centre)) & known
        raise ConstructorUnavailable(f"{type(region).__name__} has no membership test.")

    # -- as costs ------------------------------------------------------------------------------
    def inside(self, a, b, margin=0.0):
        self._margin(margin)
        return _flag(~self._subset(a, b))

    def meets(self, a, b, margin=0.0):
        self._margin(margin)
        return _flag(~self._meet(a, b))

    def apart(self, a, b, margin=0.0):
        self._margin(margin)
        return _flag(self._meet(a, b))

    def empty(self, a, margin=0.0):
        self._margin(margin)
        return _flag(~self._empty(a))

    def nonempty(self, a, epsilon=0.0):
        return _flag(self._empty(a))

    def member(self, a, b, margin=0.0):
        self._margin(margin)
        if isinstance(a, Points):
            return _flag(~self._contains(b, a))
        return self.inside(a, b)

    def outside(self, a, b, margin=0.0):
        self._margin(margin)
        return _flag(self._subset(a, b))

    def intersect(self, a, b):
        if isinstance(a, Box) and isinstance(b, Box):
            low, high = th.maximum(a.lower, b.lower), th.minimum(a.upper, b.upper)
            empty = (low > high + self.tol).any(-1, keepdim=True) | \
                _box_empty(a, self.tol).unsqueeze(-1) | _box_empty(b, self.tol).unsqueeze(-1)
            low = th.where(empty, th.full_like(low, float("inf")), low)
            high = th.where(empty, th.full_like(high, float("-inf")), th.maximum(low, high))
            return _corner_box(low, high), _flag(empty.reshape(-1))
        return a.intersect(b)

    # an empty conjunction is included in every class: no gap is added
    def intersection_inside(self, a, b, c, margin=0.0):
        region, _ = self.intersect(a, b)
        return self.inside(region, c, margin)

    def intersection_apart(self, a, b, c, margin=0.0):
        region, _ = self.intersect(a, b)
        return self.apart(region, c, margin)

    def intersection_nonempty(self, a, b, margin=0.0, epsilon=0.0):
        region, _ = self.intersect(a, b)
        return self.nonempty(region)


def _corner_box(low, high):
    from mowl.representation.regions import CornerBox
    finite = th.isfinite(low) & th.isfinite(high)
    centre = th.where(finite, (low + high) / 2, th.zeros_like(low))
    offset = th.where(finite, (high - low) / 2, th.full_like(low, float("-inf")))
    return CornerBox(centre, offset, low, high)


def _contains_rows(region, x):
    """``(B,)``: point ``b`` in box ``b``."""
    inside = (x >= region.lower) & (x <= region.upper)
    return inside.all(-1) & ~_box_empty(region, 0.0)


# -- the geometry ---------------------------------------------------------------------------
class ConstructionGeometry(ComposedGeometry):
    """The geometry of a constructed method: a :class:`~mowl.representation.ComposedGeometry`
    whose class model and individual model are the tables of regions read off the
    constructed model (:class:`RegionTable`), whose relation model is the reading's
    (:class:`CellRelations`, or none), whose measure is :class:`ExactMeasure` and whose
    objective is the default, semantic one with margin 0 (the cost of an axiom is 0.0 iff
    it holds in the constructed model).

    Built by :func:`construct`. Besides the interface of every geometry it has ``top``,
    ``bottom``, ``next`` and ``until`` (the latter two on models with time), the parts
    ``source`` and ``reading``, the built model (``built``), the guarantees of the
    combination (``guarantees``) and the names of the ids (``names``, ``roles``,
    ``individual_names``).
    """

    def __init__(self, read, source, reading, built, measure=None, objective=None):
        classes = RegionTable(read.classes, reading.class_part,
                              reading.constructors.get("conjunction", "exact") == "exact",
                              extra={k: v for k, v in reading.constructors.items()
                                     if k != "conjunction"})
        individuals = None
        if read.individuals is not None and len(read.individuals):
            individuals = RegionTable(read.individuals, reading.individual_part)
            individuals.region = Points
        measure = measure if measure is not None else ExactMeasure()
        super().__init__(classes, read.relations, measure,
                         objective if objective is not None else Objective(),
                         individuals=individuals)
        self.read = read
        self.source = source
        self.reading = reading
        self.built = built
        self.names = list(read.names)
        self.roles = list(read.roles)
        self.individual_names = list(read.individual_names)

    # names of earlier constructed geometries
    @property
    def _names(self):
        return self.names

    @property
    def _roles(self):
        return self.roles

    @property
    def _individuals(self):
        return self.individual_names

    @property
    def guarantees(self):
        """The guarantees of the source and of the reading (:class:`Guarantee`)."""
        return tuple(self.source.guarantees) + tuple(self.reading.guarantees)

    @property
    def time(self):
        """The time point relations are read at (``None``: all time points)."""
        return self.measure.time

    @time.setter
    def time(self, value):
        self.measure.time = value

    def class_id(self, name):
        return self.names.index(name)

    def top(self, n=1):
        return _take(self.read.universe, th.zeros(n, dtype=th.long))

    def bottom(self, n=1):
        return _take(self.read.empty, th.zeros(n, dtype=th.long))

    def next(self, a):
        if not hasattr(a, "next"):
            raise ConstructorUnavailable(f"{type(a).__name__} regions have no time.")
        return a.next()

    def until(self, a, b):
        if not hasattr(a, "until"):
            raise ConstructorUnavailable(f"{type(a).__name__} regions have no time.")
        return a.until(b)

    def nonempty(self, a):
        return self.measure.nonempty(a)

    def regions(self):
        """The regions of all classes as tensors (``{name of tensor: tensor}``)."""
        return self.read.classes.tensors()

    @property
    def parts(self):
        parts = super().parts
        parts.update({"source": self.source.key, "reading": self.reading.key,
                      "individuals": self.reading.individual_part
                      if self.individual_model is not None else "none",
                      "relations": self.reading.relation_part
                      if self.relation_model is not None else "none"})
        if self.relation_model is None:
            parts["existential"] = "none"
        return parts


# -- registries and the entry point ---------------------------------------------------------------
#: Source name -> class.
SOURCES = {}
#: Reading name -> class.
READINGS = {}
#: Catalogue key -> :class:`Recipe`.
RECIPES = {}


def register_source(cls):
    """Class decorator: make a :class:`Source` available under its ``key``."""
    SOURCES[cls.key] = cls
    return cls


def register_reading(cls):
    """Class decorator: make a :class:`Reading` available under its ``key``."""
    READINGS[cls.key] = cls
    return cls


def _part(value, registry, kind, options):
    if isinstance(value, str):
        if value not in registry:
            raise ValueError(f"Unknown {kind} {value!r}; available: {sorted(registry)}.")
        return registry[value](**options)
    if options:
        raise TypeError(f"Options {sorted(options)} need a {kind} given by name.")
    return value


def construct(data, source, reading, names=None, roles=None, individuals=None,
              source_options=None, reading_options=None, measure=None, built=None):
    """Build a model of ``data`` with ``source`` and read it with ``reading``.

    :param data: What the source builds from (an ontology, a dataset, a canonical model, a
        finite model, ...; see the sources).
    :param source: A :class:`Source` or its name in :data:`SOURCES`.
    :param reading: A :class:`Reading` or its name in :data:`READINGS`.
    :param names, roles, individuals: The names of the class, role and individual ids
        (default: those of the built model).
    :param built: A model the source has already built (``data`` is then ignored).
    :raises ConstructorUnavailable: If the reading does not apply to what the source
        builds.
    :rtype: ConstructionGeometry
    """
    source = _part(source, SOURCES, "source", dict(source_options or {}))
    reading = _part(reading, READINGS, "reading", dict(reading_options or {}))
    if isinstance(source.produces, tuple):
        produced = source.produces
    else:
        produced = (source.produces,)
    if not any(issubclass(p, reading.accepts) for p in produced):
        raise ConstructorUnavailable(
            f"The {reading.key!r} reading reads {[c.__name__ for c in reading.accepts]}; "
            f"the {source.key!r} source builds {[p.__name__ for p in produced]}."
            + getattr(reading, "why_not", lambda p: "")(produced))
    if built is None:
        built = source.build(data)
    reading.check(built)
    default = source.default_names(built)
    read = reading.read(built, default[0] if names is None else list(names),
                        default[1] if roles is None else list(roles),
                        default[2] if individuals is None else list(individuals), source)
    return ConstructionGeometry(read, source, reading, built, measure=measure)


@dataclass
class Recipe:
    """A named constructed method: a source and a reading with their options.

    :param key: Catalogue key.
    :param source, reading: Names in :data:`SOURCES` and :data:`READINGS`.
    :param source_options, reading_options: Their options.
    """

    key: str
    source: str
    reading: str
    source_options: dict = field(default_factory=dict)
    reading_options: dict = field(default_factory=dict)

    def build(self, data, source=None, reading=None, source_options=None,
              reading_options=None, **names):
        """The geometry of the method on ``data``; ``source`` / ``reading`` swap a part
        (with its own options), the ``*_options`` replace the recipe's options of the part
        they name."""
        s_opts = dict(self.source_options if source is None else {})
        s_opts.update(source_options or {})
        r_opts = dict(self.reading_options if reading is None else {})
        r_opts.update(reading_options or {})
        return construct(data, source or self.source, reading or self.reading,
                         source_options=s_opts, reading_options=r_opts, **names)

    @property
    def parts(self):
        return {"source": self.source, "reading": self.reading,
                "source_options": dict(self.source_options),
                "reading_options": dict(self.reading_options)}


def register_recipe(recipe):
    """Register a :class:`Recipe` under its key, as a method trained on
    ``"construction"`` (:func:`~mowl.representation.register_method`)."""
    from mowl.representation.methods import register_method
    RECIPES[recipe.key] = recipe
    register_method(recipe.key, trained_on="construction")(recipe.build)
    return recipe
