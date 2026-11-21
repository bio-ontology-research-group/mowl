"""Convex geometric model of an arbitrary finite model (Gutiérrez-Basulto & Schockaert, KR 2018).

Proposition 3 of the paper: for a quasi-chained ontology and a finite model ``M`` of a
knowledge base there is a convex geometric model ``η`` with ``φ(η) = M``. The proof
constructs ``η`` from ``M`` itself, not from a canonical model: with ``m`` the number of
objects of ``M``, object ``x_i`` becomes the unit vector ``e_i`` of ``R^m``, a concept
becomes the convex hull of its members and a ``k``-ary relation the convex hull of the
concatenated vectors of its tuples.

:class:`ConvexRegionModel` implements this for the unary and binary predicates of OWL
(concepts and roles). Coordinates are those of
:class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`: the standard simplex of
``R^(m-1)``, an affine image of the paper's ``m`` unit vectors on the hyperplane
``sum = 1`` (the paper notes that dimension ``m - 1`` suffices for datalog). The regions
are the :class:`~mowl.geometric.regions.Face` and
:class:`~mowl.geometric.regions.PairHull` objects the
:class:`~mowl.geometric.checker.ModelChecker` decides exactly.

Guarantees:
- ``φ(η) = M`` (:meth:`ConvexRegionModel.phi`), checked geometrically by convex-hull
  membership and not by reading the construction back.
- Consequently ``η`` satisfies a normal-form axiom iff ``M`` does (for the forms
  ``NF1`` to ``NF7``), so ``η`` is a model of every axiom set that ``M`` is a model of.
- The construction does not need the ontology: it needs a finite model. The canonical
  model gives the strongly faithful case (see :mod:`mowl.geometric.faithfulness`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, FrozenSet, Iterable, Optional, Tuple

import numpy as np

from mowl.geometric.checker import DictInterpretation, ModelChecker
from mowl.geometric.errors import InternalConsistencyError
from mowl.geometric.normal_forms import (
    BOTTOM,
    NF1,
    NF2,
    NF3,
    NF4,
    NF6,
    NF7,
    TOP,
    Axiom,
)
from mowl.geometric.regions import Face, PairHull, simplex_points

if TYPE_CHECKING:  # pragma: no cover
    from mowl.geometric.completion import CanonicalModel


@dataclass
class FiniteInterpretation:
    """A finite classical interpretation: a domain ``0..size-1``, concept and role extensions."""

    size: int
    concepts: Dict[str, FrozenSet[int]] = field(default_factory=dict)
    roles: Dict[str, FrozenSet[Tuple[int, int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("a finite model needs at least one element")
        self.concepts = {k: frozenset(v) for k, v in self.concepts.items()}
        self.roles = {k: frozenset(map(tuple, v)) for k, v in self.roles.items()}
        for name, ext in self.concepts.items():
            if any(not 0 <= x < self.size for x in ext):
                raise ValueError(f"concept {name!r} has elements outside the domain")
        for name, rel in self.roles.items():
            if any(not (0 <= x < self.size and 0 <= y < self.size) for x, y in rel):
                raise ValueError(f"role {name!r} has pairs outside the domain")

    @classmethod
    def from_canonical_model(cls, cm: "CanonicalModel") -> "FiniteInterpretation":
        """The canonical model as a finite interpretation (domain = ``cm.names`` positions)."""
        return cls(
            len(cm.names),
            {n: cm.extension(n) for n in cm.concept_names if n not in (TOP, BOTTOM)},
            {r: cm.role(r) for r in cm.role_names},
        )

    def extension(self, name: str) -> FrozenSet[int]:
        if name == TOP:
            return frozenset(range(self.size))
        if name == BOTTOM:
            return frozenset()
        return self.concepts.get(name, frozenset())

    def role(self, name: str) -> FrozenSet[Tuple[int, int]]:
        return self.roles.get(name, frozenset())

    def satisfies(self, axiom: Axiom) -> bool:
        """Classical satisfaction of a normal-form axiom (``NF1``-``NF4``, ``NF6``, ``NF7``)."""
        if isinstance(axiom, NF1):
            return self.extension(axiom.sub) <= self.extension(axiom.sup)
        if isinstance(axiom, NF2):
            return (self.extension(axiom.sub1) & self.extension(axiom.sub2)) <= self.extension(
                axiom.sup)
        if isinstance(axiom, NF3):
            rel, fill = self.role(axiom.role), self.extension(axiom.filler)
            return all(any((x, y) in rel for y in fill) for x in self.extension(axiom.sub))
        if isinstance(axiom, NF4):
            fill, sup = self.extension(axiom.filler), self.extension(axiom.sup)
            return all(x in sup for x, y in self.role(axiom.role) if y in fill)
        if isinstance(axiom, NF6):
            return self.role(axiom.sub) <= self.role(axiom.sup)
        if isinstance(axiom, NF7):
            sup = self.role(axiom.sup)
            r2 = self.role(axiom.r2)
            return all((x, z) in sup for x, y in self.role(axiom.r1) for y2, z in r2 if y == y2)
        raise TypeError(f"unsupported axiom {axiom!r}")

    def is_model(self, axioms: Iterable[Axiom]) -> bool:
        return all(self.satisfies(a) for a in axioms)

    def same_as(self, other: "FiniteInterpretation") -> bool:
        """Equality of interpretations on the predicates of either (missing means empty)."""
        if self.size != other.size:
            return False
        names = set(self.concepts) | set(other.concepts)
        roles = set(self.roles) | set(other.roles)
        same_concepts = all(self.extension(n) == other.extension(n) for n in names)
        same_roles = all(self.role(r) == other.role(r) for r in roles)
        return same_concepts and same_roles


class ConvexRegionModel(DictInterpretation):
    """Convex geometric model ``η`` of a finite interpretation ``M`` (Proposition 3).

    :param model: the finite interpretation ``M``.
    :param axioms: optional axioms; ``M`` must be a model of them, and the construction is
        then verified to satisfy them too (:class:`InternalConsistencyError` otherwise).
    """

    def __init__(self, model: FiniteInterpretation, axioms: Optional[Iterable[Axiom]] = None):
        m = model.size
        self.finite_model: FiniteInterpretation = model
        self.dimension: int = m - 1
        self._points = simplex_points(m)
        universe = Face.from_simplex(self._points, range(m))
        classes = {
            name: Face.from_simplex(self._points, sorted(ext))
            for name, ext in model.concepts.items()
        }
        roles = {}
        for name, rel in model.roles.items():
            rows = [np.concatenate([self._points[i], self._points[j]]) for i, j in sorted(rel)]
            roles[name] = PairHull(np.vstack(rows) if rows
                                   else np.zeros((0, 2 * self.dimension)))
        super().__init__("polyhedral", universe, classes, roles, names=list(classes))
        self._checker: Optional[ModelChecker] = None
        if axioms is not None:
            axioms = list(axioms)
            if not model.is_model(axioms):
                raise ValueError("the finite interpretation is not a model of the axioms")
            bad = self.check().violations(axioms)
            if bad:
                raise InternalConsistencyError(
                    f"ConvexRegionModel violates {len(bad)} axiom(s): {bad!r}")

    @classmethod
    def from_canonical_model(cls, cm: "CanonicalModel", verify: bool = True):
        """Apply the construction to the canonical model of an ontology."""
        return cls(FiniteInterpretation.from_canonical_model(cm),
                   cm.axioms if verify else None)

    def check(self) -> ModelChecker:
        if self._checker is None:
            self._checker = ModelChecker(self)
        return self._checker

    def satisfies(self, axiom: Axiom) -> bool:
        return self.check().satisfies(axiom)

    def element_point(self, i: int) -> np.ndarray:
        """The vector of object ``i``."""
        return self._points[i].copy()

    def phi(self) -> FiniteInterpretation:
        """The classical interpretation ``φ(η)`` read off the regions by membership tests."""
        m = self.finite_model.size
        concepts: Dict[str, FrozenSet[int]] = {}
        for name in self.finite_model.concepts:
            reg = self.class_region(name)
            concepts[name] = frozenset(
                i for i in range(m) if reg.contains(self._points[i]))
        roles: Dict[str, FrozenSet[Tuple[int, int]]] = {}
        chk = self.check()
        for name in self.finite_model.roles:
            hull = self.role_region(name)
            roles[name] = frozenset(
                (i, j) for i in range(m) for j in range(m)
                if chk._point_in_pair_hull(
                    np.concatenate([self._points[i], self._points[j]]), hull))
        return FiniteInterpretation(m, concepts, roles)
