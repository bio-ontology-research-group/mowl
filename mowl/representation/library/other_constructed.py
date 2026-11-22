"""The exact constructions of :mod:`mowl.geometric` in the common interface.

``canonical_model``, ``least_model``, ``box_milp``, ``interval_union``,
``convex_polyhedral``, ``convex_regions``, ``alcone-exact``, ``temporal_cones`` and
``attributed_dl`` build a model by a construction (a theorem), not by training. They are
not combinations of parts: nothing is learned, so there is nothing to swap. What they
share with the trained methods is the questions one asks of a model: what is the region of
a class, of an individual, of :math:`\\exists r.C` or :math:`C \\sqcap D`; does one lie
inside another. :class:`ConstructedGeometry` answers them for a constructed model.

**Costs are 0.0 where the relation holds and 1.0 where it does not**, computed by the
exact checker of the construction (:class:`~mowl.geometric.checker.ModelChecker`, the
finite model, the cone algebra); there is no gradient and no degree. ``classes``,
``individuals``, ``exists``, ``conjunction``, ``negation`` and ``disjunction`` return
:class:`Symbols`, symbolic descriptions of the regions, because the constructions decide
relations between concept descriptions rather than hand out tensors; what a construction
cannot decide raises :class:`~mowl.representation.ConstructorUnavailable` (box models have
no roles; the normal-form checkers decide only the forms of an EL normal form: names,
conjunctions of two names, ``C ⊑ ∃r.D`` and ``∃r.C ⊑ D``; the al-cone constructions decide
full ALC).

``apart`` and ``nonempty`` are about the *constructed model* (two classes that no element
of the model belongs to are apart there), not about entailment: only inclusion is
faithful to entailment, and only for the strongly faithful constructions (see their
cards).

Class and role ids are positions in ``names`` and ``roles`` (given or, by default, sorted
names of the model); individual ids are positions in ``individuals``.
"""

from dataclasses import dataclass

import numpy as np
import torch as th

from mowl.representation.adapters import register_adapter
from mowl.representation.geometry import Geometry
from mowl.representation.regions import ConstructorUnavailable, Region

__all__ = ["Symbols", "ConstructedGeometry", "NormalFormBackend", "FiniteBackend",
           "LeastModelBackend", "ALConeBackend", "TemporalBackend", "AttributedBackend",
           "constructed_geometry"]


@dataclass(frozen=True)
class Symbols(Region):
    """A batch of symbolic regions: ``exprs`` is a tuple of expressions, each a tuple
    ``("name", n)``, ``("nominal", individual)``, ``("top",)``, ``("bot",)``,
    ``("and", a, b)``, ``("or", a, b)``, ``("not", a)``, ``("exists", role, a)``,
    ``("next", a)`` or ``("until", a, b)``."""

    exprs: tuple

    def __len__(self):
        return len(self.exprs)


def _binary(op, a, b):
    if len(a) != len(b):
        raise ValueError("Both sides need the same batch size.")
    return Symbols(tuple((op, x, y) for x, y in zip(a.exprs, b.exprs)))


# -- backends: what a construction can decide ---------------------------------------------
class Backend:
    """What a construction decides about symbolic regions. Subclasses implement the
    methods; the defaults raise :class:`ConstructorUnavailable`."""

    key = ""

    def names(self):
        raise NotImplementedError

    def roles(self):
        return []

    def individuals(self):
        return []

    def subsumed(self, a, b):
        """``True`` iff the region of ``a`` lies inside that of ``b``."""
        raise ConstructorUnavailable(f"{type(self).__name__} cannot decide inclusion.")

    def disjoint(self, a, b):
        raise ConstructorUnavailable(f"{type(self).__name__} cannot decide disjointness.")

    def satisfiable(self, a):
        raise ConstructorUnavailable(f"{type(self).__name__} cannot decide emptiness.")

    def instance(self, individual, a):
        raise ConstructorUnavailable(f"{type(self).__name__} has no individuals.")


def _atom(expr):
    """``("name", n)`` / top / bottom / nominal as a pair ``(kind, value)``, else ``None``."""
    return expr if expr[0] in ("name", "top", "bot", "nominal") else None


class NormalFormBackend(Backend):
    """A model with ``satisfies(NF)`` (the exact checkers of the box, interval-union,
    convex-polyhedral and convex-region constructions, and finite interpretations). It
    decides the normal forms: ``A ⊑ B``, ``A ⊓ B ⊑ C`` (also ``⊑ ⊥``), ``A ⊑ ∃r.B`` and
    ``∃r.A ⊑ B`` over names, ⊤, ⊥ and nominals of individuals.

    :param model: The model.
    :param names, roles, individuals: The concept names (default: the model's names that
        are not nominals), role names and individuals, by id.
    """

    def __init__(self, model, names=None, roles=None, individuals=None, key=""):
        self.model = model
        self.key = key
        self._names = list(names) if names is not None else None
        self._roles = list(roles) if roles is not None else None
        self._individuals = list(individuals) if individuals is not None else None

    # naming ------------------------------------------------------------------------------
    def _model_names(self):
        names = getattr(self.model, "names", None)
        if names is None:
            names = getattr(self.model, "concepts", {})
        return list(names)

    def names(self):
        if self._names is None:
            from mowl.geometric.normal_forms import TOP, BOTTOM, is_nominal_name
            self._names = sorted(n for n in self._model_names()
                                 if n not in (TOP, BOTTOM) and not is_nominal_name(n))
        return self._names

    def roles(self):
        if self._roles is None:
            roles = getattr(self.model, "role_regions", None)
            if roles is None:
                roles = getattr(self.model, "roles", {})
            self._roles = sorted(roles)
        return self._roles

    def individuals(self):
        if self._individuals is None:
            from mowl.geometric.normal_forms import individual_of_name, is_nominal_name
            self._individuals = sorted(individual_of_name(n) for n in self._model_names()
                                       if is_nominal_name(n))
        return self._individuals

    def name_of(self, name):
        """The name of the model for a name given to the geometry (identity here)."""
        return name

    def _nf(self, expr):
        from mowl.geometric.normal_forms import BOTTOM, TOP, nominal_name
        kind = expr[0]
        if kind == "name":
            return self.name_of(expr[1])
        if kind == "top":
            return TOP
        if kind == "bot":
            return BOTTOM
        if kind == "nominal":
            return nominal_name(expr[1])
        return None

    def _satisfies(self, axiom):
        from mowl.geometric.errors import UnsupportedAxiom, UnsupportedRegion
        try:
            return bool(self.model.satisfies(axiom))
        except (UnsupportedAxiom, UnsupportedRegion) as error:
            raise ConstructorUnavailable(str(error)) from error

    # relations ---------------------------------------------------------------------------
    def subsumed(self, a, b):
        from mowl.geometric.normal_forms import BOTTOM, NF1, NF2, NF3, NF4
        sa, sb = self._nf(a), self._nf(b)
        if sa is not None and sb is not None:
            return self._satisfies(NF1(sa, sb))
        if a[0] == "and" and sb is not None and self._nf(a[1]) and self._nf(a[2]):
            return self._satisfies(NF2(self._nf(a[1]), self._nf(a[2]), sb))
        if sa is not None and b[0] == "exists" and self._nf(b[2]) is not None:
            return self._satisfies(NF3(sa, b[1], self._nf(b[2])))
        if a[0] == "exists" and self._nf(a[2]) is not None and sb is not None:
            return self._satisfies(NF4(a[1], self._nf(a[2]), sb))
        del BOTTOM
        raise ConstructorUnavailable(
            f"{type(self).__name__} decides A <= B, A and B <= C, A <= exists r.B and "
            f"exists r.A <= B over names; not {a[0]!r} <= {b[0]!r}.")

    def disjoint(self, a, b):
        from mowl.geometric.normal_forms import BOTTOM, NF2
        sa, sb = self._nf(a), self._nf(b)
        if sa is None or sb is None:
            raise ConstructorUnavailable("Disjointness is decided for names only.")
        return self._satisfies(NF2(sa, sb, BOTTOM))

    def satisfiable(self, a):
        from mowl.geometric.normal_forms import BOTTOM, NF1
        if a[0] == "and":
            return not self.disjoint(a[1], a[2])
        sa = self._nf(a)
        if sa is None:
            raise ConstructorUnavailable("Emptiness is decided for names and conjunctions "
                                         "of two names only.")
        return not self._satisfies(NF1(sa, BOTTOM))

    def instance(self, individual, a):
        return self.subsumed(("nominal", individual), a)


class FiniteBackend(NormalFormBackend):
    """The canonical model (or any finite interpretation): the normal forms are checked in
    the finite model, instances by the entailment of the canonical model."""

    def __init__(self, model, canonical=None, **kwargs):
        from mowl.geometric.convex_regions import FiniteInterpretation
        if canonical is None and not isinstance(model, FiniteInterpretation):
            canonical = model
            model = FiniteInterpretation.from_canonical_model(canonical)
        super().__init__(model, **kwargs)
        self.canonical = canonical

    def names(self):
        if self._names is None:
            self._names = sorted(self.model.concepts)
            if self.canonical is not None:
                self._names = sorted(self.canonical.concept_names)
        return self._names

    def roles(self):
        if self._roles is None:
            self._roles = sorted(self.model.roles)
        return self._roles

    def individuals(self):
        if self._individuals is None:
            self._individuals = sorted(self.canonical.individuals) \
                if self.canonical is not None else []
        return self._individuals

    def instance(self, individual, a):
        if self.canonical is None:
            raise ConstructorUnavailable("A finite interpretation without a canonical "
                                         "model has no individuals.")
        name = self._nf(a)
        if name is None:
            raise ConstructorUnavailable("Instances are decided for names only.")
        return bool(self.canonical.instance_of(individual, name))


class AttributedBackend(NormalFormBackend):
    """Attributed DL (:class:`~mowl.geometric.attributed.AttributedConvexModel`): a name
    is a pair ``(concept, specifier)``; the regions are those of the grounded convex model.

    :param names: List of ``(concept, specifier)`` (default: the plain concept names,
        under the open empty specifier).
    """

    def __init__(self, attributed, names=None, individuals=None, key="attributed_dl"):
        from mowl.geometric.attributed import ANY
        if names is None:
            names = [(n, ANY) for n, is_role in sorted(attributed.grounding.specifiers)
                     if not is_role]
        super().__init__(attributed.model, names=names,
                         roles=sorted(attributed.canonical_model.role_names),
                         individuals=individuals if individuals is not None
                         else sorted(attributed.canonical_model.individuals), key=key)
        self.attributed = attributed

    def name_of(self, name):
        from mowl.geometric.attributed import _name
        concept, spec = name
        return _name(concept, spec)

    def instance(self, individual, a):
        if a[0] != "name":
            raise ConstructorUnavailable("Instances are decided for names only.")
        concept, spec = a[1]
        return bool(self.attributed.holds_concept(individual, concept, spec))


class LeastModelBackend(Backend):
    """The finite least model (:class:`~mowl.geometric.least_model.LeastModel`): classes
    are its class ids, roles its role ids, individuals its individual ids. It decides the
    same normal forms as :class:`NormalFormBackend`, on the pruned model (an unsatisfiable
    class has the empty extension)."""

    key = "least_model"

    def __init__(self, model):
        self.model = model
        labels = model.labels.toarray().astype(bool)           # (blocks, classes)
        self._ext = labels.T                                    # (classes, blocks)
        self._edges = model.edges

    def names(self):
        return list(range(self.model.nb_classes))

    def roles(self):
        return list(range(self.model.nb_rels))

    def individuals(self):
        return list(range(self.model.nb_individuals))

    def _extension(self, expr):
        kind = expr[0]
        if kind == "name":
            return self._ext[int(expr[1])]
        if kind == "top":
            top = self.model.top_id
            return self._ext[top] if top is not None else ~self.model.dead
        if kind == "bot":
            return np.zeros(self.model.n_blocks, dtype=bool)
        if kind == "nominal":
            block = self.model.element_map[self.model.nb_classes + int(expr[1])]
            out = np.zeros(self.model.n_blocks, dtype=bool)
            out[block] = True
            return out
        if kind == "and":
            return self._extension(expr[1]) & self._extension(expr[2])
        if kind == "or":
            return self._extension(expr[1]) | self._extension(expr[2])
        if kind == "exists":
            r, filler = int(expr[1]), self._extension(expr[2])
            edges = self._edges[self._edges[:, 2] == r]
            out = np.zeros(self.model.n_blocks, dtype=bool)
            out[edges[filler[edges[:, 1]], 0]] = True
            return out
        raise ConstructorUnavailable(f"The least model has no {kind!r} constructor.")

    def subsumed(self, a, b):
        return bool(np.all(~self._extension(a) | self._extension(b)))

    def disjoint(self, a, b):
        return not bool(np.any(self._extension(a) & self._extension(b)))

    def satisfiable(self, a):
        return bool(np.any(self._extension(a)))

    def instance(self, individual, a):
        return bool(self._extension(("nominal", individual)).__and__(
            self._extension(a)).any())


def _to_cones(expr):
    """The expression as a :mod:`mowl.geometric.cones` concept."""
    from mowl.geometric import cones as C
    kind = expr[0]
    if kind == "name":
        return C.Atomic(expr[1])
    if kind == "top":
        return C.TOP_CONCEPT
    if kind == "bot":
        return C.BOTTOM_CONCEPT
    if kind == "and":
        return C.And((_to_cones(expr[1]), _to_cones(expr[2])))
    if kind == "or":
        return C.Or((_to_cones(expr[1]), _to_cones(expr[2])))
    if kind == "not":
        return C.Not(_to_cones(expr[1]))
    if kind == "exists":
        return C.Exists(expr[1], _to_cones(expr[2]))
    if kind in ("next", "until"):
        from mowl.geometric import temporal_cones as T
        if kind == "next":
            return T.Next(_to_cones(expr[1]))
        return T.Until(_to_cones(expr[1]), _to_cones(expr[2]))
    raise ConstructorUnavailable(f"No al-cone for {kind!r}.")


class ALConeBackend(Backend):
    """Al-cone interpretations (:class:`~mowl.geometric.cones.ALConeInterpretation`): the
    region of any ALC concept is an al-cone, and inclusion, disjointness, emptiness and
    membership of individuals (rays) are decided exactly."""

    key = "alcone-exact"

    def __init__(self, interpretation):
        self.interpretation = interpretation
        self.checker = interpretation.checker()

    def names(self):
        return sorted(self.interpretation.concepts)

    def roles(self):
        return sorted(self.interpretation.roles)

    def individuals(self):
        return sorted(self.interpretation.individuals)

    def _cone(self, expr):
        try:
            return self.checker.cone(_to_cones(expr))
        except KeyError as error:
            raise ValueError(str(error)) from error

    def subsumed(self, a, b):
        return bool(self._cone(a).subset_of(self._cone(b)))

    def disjoint(self, a, b):
        return bool(self._cone(("and", a, b)).is_bottom())

    def satisfiable(self, a):
        return not self._cone(a).is_bottom()

    def instance(self, individual, a):
        return bool(self.checker.instance_of(individual, _to_cones(a)))


class TemporalBackend(ALConeBackend):
    """Temporal al-cones
    (:class:`~mowl.geometric.temporal_cones.TemporalALConeInterpretation`): one al-cone
    interpretation per time point; ``next`` and ``until`` are available. Costs are read at
    time ``time`` (inclusion: at that time point only; the construction's own satisfaction
    is at all of them)."""

    key = "temporal_cones"

    def __init__(self, interpretation, time=0):
        self.interpretation = interpretation
        self.time = time

    def names(self):
        return sorted(self.interpretation.atoms)

    def roles(self):
        return []

    def individuals(self):
        return sorted(self.interpretation.individuals)

    def _cone(self, expr):
        return self.interpretation.cone(_to_cones(expr), self.time)

    def instance(self, individual, a):
        return bool(self.interpretation.instance_of(individual, _to_cones(a), self.time))


# -- the geometry -------------------------------------------------------------------------
class ConstructedGeometry(Geometry):
    """The geometry of a constructed model: exact answers, as costs 0.0 (holds) and 1.0
    (does not hold).

    :param backend: A :class:`Backend`.
    :param names, roles, individuals: Override the ids of classes, roles and individuals.
    """

    def __init__(self, backend, names=None, roles=None, individuals=None):
        self.backend = backend
        self._names = list(names) if names is not None else list(backend.names())
        self._roles = list(roles) if roles is not None else list(backend.roles())
        self._individuals = list(individuals) if individuals is not None \
            else list(backend.individuals())
        self.time = getattr(backend, "time", None)

    # -- regions ------------------------------------------------------------------------------
    def classes(self, ids):
        return Symbols(tuple(("name", self._names[int(i)]) for i in ids.tolist()))

    def individuals(self, ids):
        return Symbols(tuple(("nominal", self._individuals[int(i)]) for i in ids.tolist()))

    def exists(self, relation_ids, region):
        if not self._roles:
            raise ConstructorUnavailable(f"{type(self.backend).__name__} has no roles.")
        return Symbols(tuple(("exists", self._roles[int(r)], e)
                             for r, e in zip(relation_ids.tolist(), region.exprs)))

    def conjunction(self, a, b):
        region = _binary("and", a, b)
        gap = th.tensor([[0.0 if self.backend.satisfiable(e) else 1.0]
                         for e in region.exprs])
        return region, gap

    def negation(self, a):
        return Symbols(tuple(("not", e) for e in a.exprs))

    def disjunction(self, a, b):
        return _binary("or", a, b)

    def next(self, a):
        """The next time point (temporal cones)."""
        return Symbols(tuple(("next", e) for e in a.exprs))

    def until(self, a, b):
        """``a until b`` (temporal cones)."""
        return _binary("until", a, b)

    def top(self, n=1):
        return Symbols((("top",),) * n)

    def bottom(self, n=1):
        return Symbols((("bot",),) * n)

    # -- costs --------------------------------------------------------------------------------
    @staticmethod
    def _cost(flags):
        return th.tensor([[0.0 if f else 1.0] for f in flags])

    def inside(self, a, b):
        """0.0 where the region of ``a`` lies inside that of ``b``, else 1.0."""
        return self._cost(self.backend.subsumed(x, y) for x, y in zip(a.exprs, b.exprs))

    def apart(self, a, b):
        """0.0 where the two regions are disjoint, else 1.0."""
        return self._cost(self.backend.disjoint(x, y) for x, y in zip(a.exprs, b.exprs))

    def nonempty(self, a):
        """0.0 where the region is not empty, else 1.0."""
        return self._cost(self.backend.satisfiable(x) for x in a.exprs)

    def member(self, individual_ids, region):
        """0.0 where the individual is in the region (the construction's own check)."""
        return self._cost(
            self.backend.instance(self._individuals[int(i)], e)
            for i, e in zip(individual_ids.tolist(), region.exprs))

    @property
    def num_classes(self):
        return len(self._names)

    @property
    def num_individuals(self):
        return len(self._individuals)

    @property
    def parts(self):
        from mowl.representation.catalog import REPRESENTATIONS
        rep = REPRESENTATIONS.get(self.backend.key)
        return {"classes": "fixed by reasoning" if rep is None else rep.classes.value,
                "relations": "fixed by reasoning", "individuals": "region",
                "measure": "exact", "conjunction": "exact"}


def constructed_geometry(model, **options):
    """The :class:`ConstructedGeometry` of a constructed model, by its type."""
    name = type(model).__name__
    if name == "AttributedConvexModel":
        backend = AttributedBackend(model, **{k: v for k, v in options.items()
                                              if k in ("names", "individuals")})
        return ConstructedGeometry(backend)
    if name == "CanonicalModel":
        return ConstructedGeometry(FiniteBackend(model, key="canonical_model"))
    if name == "FiniteInterpretation":
        return ConstructedGeometry(FiniteBackend(model, canonical=options.get("canonical")))
    if name == "LeastModel":
        return ConstructedGeometry(LeastModelBackend(model))
    if name == "ALConeInterpretation":
        return ConstructedGeometry(ALConeBackend(model))
    if name == "TemporalALConeInterpretation":
        return ConstructedGeometry(TemporalBackend(model, options.get("time", 0)))
    keys = {"BoxModel": "box_milp", "IntervalUnionModel": "interval_union",
            "ConvexPolyhedralModel": "convex_polyhedral",
            "ConvexRegionModel": "convex_regions"}
    if name in keys:
        return ConstructedGeometry(NormalFormBackend(model, key=keys[name]))
    raise NotImplementedError(f"No geometry for {name}.")


# The geometry of these models is now the combination of a source and a reading
# (mowl.representation.library.construct_methods registers it); this direct geometry of
# backends stays for the geometries built on it (FaithEL) and as a reference.
del register_adapter
