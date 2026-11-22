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

from typing import TYPE_CHECKING, Dict, FrozenSet, Iterable, Optional, Tuple

import numpy as np

from mowl.geometric.checker import DictInterpretation, ModelChecker
from mowl.geometric.convex_regions import FiniteInterpretation
from mowl.geometric.errors import InternalConsistencyError
from mowl.geometric.normal_forms import Axiom
from mowl.geometric.regions import Face, PairHull, simplex_points

if TYPE_CHECKING:  # pragma: no cover
    from mowl.geometric.completion import CanonicalModel


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
