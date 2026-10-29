"""Exact geometric model checker for :mod:`mowl.geometric` region families.

Semantics and guarantees (Spec §5.2):
- :class:`GeometricInterpretation`: abstract base class for region assignments
  :math:`\\eta` over a universe :math:`U = \\eta(\\top)` in one of the three
  families ``{"interval", "box", "polyhedral"}``. Maps ``TOP`` to
  ``eta.universe`` and ``BOTTOM`` (or any concept/role name absent from the
  interpretation) to the family's canonical empty region.
- :class:`DictInterpretation`: lightweight concrete subclass backed by
  dictionaries of class and role regions, suitable for unit tests and custom
  region assignments.
- :class:`ModelChecker`: decides axiom satisfaction (:meth:`ModelChecker.satisfies`),
  collects violated axioms (:meth:`ModelChecker.violations`), tests whether an
  interpretation is a model (:meth:`ModelChecker.is_model`), and checks
  non-degeneracy (:meth:`ModelChecker.is_nondegenerate`) and strong
  faithfulness on concept names
  (:meth:`ModelChecker.is_strongly_faithful_on_names`).

Decision procedures by family:
- ``"interval"`` (:class:`~mowl.geometric.regions.IntervalUnion` for classes,
  :class:`~mowl.geometric.regions.RectUnion` for roles):
  all normal forms (:class:`NF1`, :class:`NF2`, :class:`NF3`, :class:`NF4`,
  :class:`NF6`, :class:`NF7` and :math:`\\bot` forms) are decided
  combinatorially via exact 1-D interval and 2-D rectangle set algebra
  (:meth:`IntervalUnion.is_subset`, :meth:`IntervalUnion.intersection`,
  :meth:`RectUnion.exists_left`, :meth:`RectUnion.compose`,
  :meth:`RectUnion.is_subset`).
- ``"box"`` (:class:`~mowl.geometric.regions.Box` for classes):
  propositional axioms (:class:`NF1`, :class:`NF2` and their :math:`\\bot`
  forms) are decided combinatorially via corner comparisons
  (:meth:`Box.is_subset`, :meth:`Box.intersection`). Role axioms
  (:class:`NF3`, :class:`NF4`, :class:`NF6`, :class:`NF7`) are outside the v1
  propositional box fragment and raise
  :class:`~mowl.geometric.errors.UnsupportedAxiom`.
- ``"polyhedral"`` (:class:`~mowl.geometric.regions.Face` or H-rep polyhedra
  for classes, :class:`~mowl.geometric.regions.PairHull` for roles):
  - :class:`NF1` and :class:`NF2` on :class:`Face` instances of a single
    simplex are decided combinatorially and exactly via ``vertex_ids``
    (:meth:`Face.is_subset`, :meth:`Face.intersection`). When regions are
    general H-rep polyhedra (e.g. thickened faces), inclusion and intersection
    emptiness are decided by linear programming (`scipy.optimize.linprog` with
    ``method="highs"``) against the target's :meth:`Face.h_rep`.
  - :class:`NF3` (GCI2, :math:`A \\sqsubseteq \\exists r.B`): relies on
    :math:`\\eta(A)` being a :class:`Face` (the convex hull of its stored
    ``vertices``) and raises :class:`~mowl.geometric.errors.UnsupportedRegion`
    otherwise; for each vertex :math:`v` of :math:`\\eta(A)`, checks LP
    feasibility of convex weights :math:`\\mu \\ge 0, \\sum \\mu = 1` over
    ``PairHull(r).points`` such that the left half equals :math:`v` and the
    right half :math:`u` satisfies the H-representation of :math:`\\eta(B)`.
  - :class:`NF4` (GCI3, :math:`\\exists r.A \\sqsubseteq B`): over
    :math:`\\mu \\ge 0, \\sum \\mu = 1` with right half :math:`u \\in \\eta(A)`
    (encoded via ``eta.class_region(A).h_rep()``), checks feasibility first (if
    :math:`\\exists r.A` is empty the axiom holds trivially; if non-empty and
    :math:`\\eta(B)` is empty it fails), and otherwise maximises each facet/equality
    direction of ``eta.class_region(B).h_rep()`` on the left half :math:`v`,
    requiring every optimum to exceed the RHS by at most ``lp_tol``.
  - :class:`NF6` (:math:`r \\sqsubseteq s`): checks each generating point of
    ``PairHull(r)`` for membership in ``conv(PairHull(s).points)`` by LP
    feasibility (exact by convexity).
  - :class:`NF7` (:math:`r \\circ s \\sqsubseteq t`): for each generating pair
    :math:`(v, u)` of :math:`r` and :math:`(u', w)` of :math:`s` whose stored
    intermediate points match (:math:`\\|u - u'\\|_\\infty \\le \\text{tol}`),
    checks :math:`(v, w) \\in \\text{conv}(\\text{PairHull}(t).\\text{points})`
    by the same LP. This is exact when role hulls are spanned by canonical
    element-pair points, as in the Gutiérrez-Basulto & Schockaert (2018)
    construction (the barycentric coordinates of an intermediate point are then
    unique); raises :class:`~mowl.geometric.errors.UnsupportedRegion` if a role
    region lacks stored ``points``.
  - Role-axiom checking (:class:`NF3`, :class:`NF4`, :class:`NF6`,
    :class:`NF7`) is exact only for simplex-vertex regions: every
    :class:`~mowl.geometric.regions.Face` vertex row and every
    :class:`~mowl.geometric.regions.PairHull` left/right half must be a row of
    the canonical simplex (a one-hot unit vector or the zero vector, see
    :func:`~mowl.geometric.regions.is_simplex_vertex_rows`); otherwise these
    checks raise :class:`~mowl.geometric.errors.UnsupportedRegion`.

References:
- Gutiérrez-Basulto & Schockaert, KR 2018 (arXiv:1805.10461), Props. 3, 4.
- Lacerda, Ozaki & Guimarães, TGDK 2(3):2, 2024, Defs. 2, 8, 12, Thms. 13, 16, 18, 25.
- Leemhuis & Kutz, NeSy 2025 (PMLR 284:303–321), Defs. 1, 3, 6, Props. 4, 5, 8.
- Bourgaux, Guimarães, Koudijs, Lacerda & Ozaki, KR 2024, Defs. 5–6, Remark 2.
- Kazakov, Krötzsch & Simančík, JAR 2014 (The Incredible ELK).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from scipy.optimize import linprog

from mowl.geometric.errors import UnsupportedAxiom, UnsupportedRegion
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
from mowl.geometric.regions import (
    EMPTY_1D,
    Box,
    Face,
    IntervalUnion,
    PairHull,
    RectUnion,
    is_simplex_vertex_rows,
)

if TYPE_CHECKING:  # pragma: no cover
    from mowl.geometric.completion import CanonicalModel

VALID_FAMILIES = frozenset({"interval", "box", "polyhedral"})
_LP_OPT_TOL: float = 1e-7


class GeometricInterpretation(ABC):
    """Abstract base class for geometric interpretations over a universe :math:`U`.

    Subclasses specify ``family`` in ``{"interval", "box", "polyhedral"}``,
    ``universe`` (:math:`\\eta(\\top)`), ``names`` (the concept names in the
    interpretation's signature), :meth:`class_region`, and :meth:`role_region`.
    """

    @property
    @abstractmethod
    def family(self) -> str:
        """Return the region family: ``"interval"``, ``"box"``, or ``"polyhedral"``."""

    @property
    @abstractmethod
    def universe(self) -> Any:
        """Return the universe region :math:`U = \\eta(\\top)`."""

    @property
    @abstractmethod
    def names(self) -> Iterable[str]:
        """Return the concept names known to this interpretation."""

    @abstractmethod
    def class_region(self, name: str) -> Any:
        """Return :math:`\\eta(\\text{name})`, or the empty region for ``BOTTOM`` / unknown names."""

    @abstractmethod
    def role_region(self, role: str) -> Any:
        """Return :math:`\\eta(\\text{role})`, or the empty role region for unknown roles."""


class DictInterpretation(GeometricInterpretation):
    """Concrete :class:`GeometricInterpretation` backed by dictionaries of regions.

    Parameters
    ----------
    family : str
        One of ``"interval"``, ``"box"``, or ``"polyhedral"``.
    universe : Any
        Universe region :math:`U = \\eta(\\top)`.
    classes : Dict[str, Any]
        Mapping from concept name to region. ``TOP`` defaults to ``universe`` if
        not explicitly provided; ``BOTTOM`` and unmapped names return the
        family's empty region.
    roles : Dict[str, Any]
        Mapping from role name to role region. Unmapped roles return the
        family's empty role region.
    names : Optional[Iterable[str]]
        Concept names in the signature (defaults to ``classes.keys()``).
    """

    def __init__(
        self,
        family: str,
        universe: Any,
        classes: Optional[Dict[str, Any]] = None,
        roles: Optional[Dict[str, Any]] = None,
        names: Optional[Iterable[str]] = None,
    ) -> None:
        if family not in VALID_FAMILIES:
            raise ValueError(
                f"Unknown geometric family {family!r}; expected one of {sorted(VALID_FAMILIES)}"
            )
        self._family: str = family
        self._universe: Any = universe
        self._classes: Dict[str, Any] = dict(classes or {})
        self._roles: Dict[str, Any] = dict(roles or {})
        if names is not None:
            self._names: List[str] = list(names)
        else:
            self._names = [k for k in self._classes.keys() if k != BOTTOM]

    @property
    def family(self) -> str:
        return self._family

    @property
    def universe(self) -> Any:
        return self._universe

    @property
    def names(self) -> Iterable[str]:
        return list(self._names)

    def _empty_class_region(self) -> Any:
        if self._family == "interval":
            return EMPTY_1D
        if self._family == "box":
            d = int(self._universe.lo.shape[0])
            return Box.empty(d)
        # polyhedral
        k = int(self._universe.dim)
        return Face(k, np.zeros((0, k), dtype=float), frozenset())

    def _empty_role_region(self) -> Any:
        if self._family == "interval":
            return RectUnion([])
        if self._family == "polyhedral":
            k = int(self._universe.dim)
            return PairHull(np.zeros((0, 2 * k), dtype=float))
        raise UnsupportedAxiom("box family: role axioms not supported in v1")

    def class_region(self, name: str) -> Any:
        if name == BOTTOM:
            return self._empty_class_region()
        if name == TOP and TOP not in self._classes:
            return self._universe
        if name in self._classes:
            return self._classes[name]
        return self._empty_class_region()

    def role_region(self, role: str) -> Any:
        if role in self._roles:
            return self._roles[role]
        return self._empty_role_region()


class ModelChecker:
    """Exact model checker for :class:`GeometricInterpretation` instances.

    Decides axiom satisfaction (up to numerical tolerances ``tol`` and
    ``lp_tol``) across the three region families:

    - **Interval family** (``family == "interval"``):
      All normal forms (:class:`NF1`, :class:`NF2`, :class:`NF3`, :class:`NF4`,
      :class:`NF6`, :class:`NF7`, and :math:`\\bot` forms) are decided
      combinatorially on :class:`~mowl.geometric.regions.IntervalUnion` and
      :class:`~mowl.geometric.regions.RectUnion`.
    - **Box family** (``family == "box"``):
      Propositional axioms (:class:`NF1`, :class:`NF2`, and :math:`\\bot` forms)
      are decided combinatorially on :class:`~mowl.geometric.regions.Box`.
      Role axioms (:class:`NF3`, :class:`NF4`, :class:`NF6`, :class:`NF7`) raise
      :class:`~mowl.geometric.errors.UnsupportedAxiom` with message
      ``"box family: role axioms not supported in v1"``.
    - **Polyhedral family** (``family == "polyhedral"``):
      - :class:`NF1` and :class:`NF2`: decided combinatorially via ``vertex_ids``
        when the regions are exact :class:`~mowl.geometric.regions.Face`
        objects (`type(reg) is Face`), and by linear programming
        (`scipy.optimize.linprog`, ``method="highs"``) on :meth:`Face.h_rep`
        when regions are general H-rep polyhedra (such as ``ThickFace``).
      - :class:`NF3` (GCI2, :math:`A \\sqsubseteq \\exists r.B`): requires the
        left-hand class region :math:`\\eta(A)` to be a :class:`Face` (the
        convex hull of its stored ``vertices``) and raises
        :class:`~mowl.geometric.errors.UnsupportedRegion` otherwise. Decided by
        LP feasibility for each vertex :math:`v` of :math:`\\eta(A)`.
      - :class:`NF4` (GCI3, :math:`\\exists r.A \\sqsubseteq B`): decided by LP
        over convex combinations :math:`(v, u) = \\mu^\\top P` of
        ``PairHull(r).points`` constrained by :math:`u \\in \\eta(A)`,
        maximising each H-representation constraint of :math:`\\eta(B)` on
        :math:`v`.
      - :class:`NF6` (:math:`r \\sqsubseteq s`) and :class:`NF7`
        (:math:`r \\circ s \\sqsubseteq t`): decided by LP convex-hull
        membership of generating pairs. For :class:`NF7`, pairs :math:`(v, u)`
        of :math:`r` and :math:`(u', w)` of :math:`s` with matching intermediate
        point (:math:`\\|u - u'\\|_\\infty \\le \\text{tol}`) are composed into
        :math:`(v, w)` and tested for membership in :math:`\\eta(t)`. This is
        exact for the Gutiérrez-Basulto & Schockaert (2018) construction where
        role regions are convex hulls of canonical element-pair points; raises
        :class:`~mowl.geometric.errors.UnsupportedRegion` if a role region lacks
        stored ``points``.
      - All four role-axiom checks raise
        :class:`~mowl.geometric.errors.UnsupportedRegion` unless every
        :class:`Face` vertex row and every :class:`PairHull` left/right half is
        a canonical simplex vertex (one-hot or zero); role-axiom checking is
        exact only for such simplex-vertex regions.
    """

    def __init__(self, eta: GeometricInterpretation, tol: float = 1e-9) -> None:
        if eta.family not in VALID_FAMILIES:
            raise ValueError(
                f"Unsupported interpretation family {eta.family!r}; expected one of {sorted(VALID_FAMILIES)}"
            )
        self.eta: GeometricInterpretation = eta
        self.tol: float = float(tol)
        self.lp_tol: float = max(self.tol, _LP_OPT_TOL)

    def _class_region(self, name: str) -> Any:
        if name == TOP:
            reg = self.eta.class_region(TOP)
            if reg is not None and not getattr(reg, "is_empty", False):
                return reg
            return self.eta.universe
        return self.eta.class_region(name)

    def satisfies(self, axiom: Axiom) -> bool:
        """Return ``True`` iff ``self.eta`` satisfies the normalised ``axiom``."""
        fam = self.eta.family
        if fam == "interval":
            return self._satisfies_interval(axiom)
        if fam == "box":
            return self._satisfies_box(axiom)
        if fam == "polyhedral":
            return self._satisfies_polyhedral(axiom)
        raise ValueError(f"Unsupported family: {fam!r}")  # pragma: no cover

    def violations(self, axioms: Iterable[Axiom]) -> List[Axiom]:
        """Return the list of axioms in ``axioms`` not satisfied by ``self.eta``."""
        return [ax for ax in axioms if not self.satisfies(ax)]

    def is_model(self, axioms: Iterable[Axiom]) -> bool:
        """Return ``True`` iff every axiom in ``axioms`` is satisfied by ``self.eta``."""
        return all(self.satisfies(ax) for ax in axioms)

    def is_nondegenerate(self, cm: CanonicalModel) -> bool:
        """Return ``True`` iff ``class_region(name)`` is non-empty for every satisfiable name in ``cm.names``."""
        for name in cm.names:
            reg = self._class_region(name)
            if self._region_is_empty(reg):
                return False
        return True

    def is_strongly_faithful_on_names(self, cm: CanonicalModel) -> bool:
        """Return ``True`` iff for all ``a, b`` in ``cm.names``, ``region(a) <= region(b) <=> cm.entails(a, b)``."""
        names = list(cm.names)
        regions = {name: self._class_region(name) for name in names}
        for a in names:
            reg_a = regions[a]
            for b in names:
                reg_b = regions[b]
                geom_sub = self._class_subset(reg_a, reg_b)
                ont_sub = cm.entails(a, b)
                if geom_sub != ont_sub:
                    return False
        return True

    # ------------------------------------------------------------------
    # Family dispatchers
    # ------------------------------------------------------------------

    def _satisfies_interval(self, axiom: Axiom) -> bool:
        if isinstance(axiom, NF1):
            reg_sub: IntervalUnion = self._class_region(axiom.sub)
            reg_sup: IntervalUnion = self._class_region(axiom.sup)
            return reg_sub.is_subset(reg_sup)
        if isinstance(axiom, NF2):
            reg_1: IntervalUnion = self._class_region(axiom.sub1)
            reg_2: IntervalUnion = self._class_region(axiom.sub2)
            reg_sup = self._class_region(axiom.sup)
            return reg_1.intersection(reg_2).is_subset(reg_sup)
        if isinstance(axiom, NF3):
            reg_sub = self._class_region(axiom.sub)
            reg_fill: IntervalUnion = self._class_region(axiom.filler)
            role_reg: RectUnion = self.eta.role_region(axiom.role)
            return reg_sub.is_subset(role_reg.exists_left(reg_fill))
        if isinstance(axiom, NF4):
            reg_fill = self._class_region(axiom.filler)
            reg_sup = self._class_region(axiom.sup)
            role_reg = self.eta.role_region(axiom.role)
            return role_reg.exists_left(reg_fill).is_subset(reg_sup)
        if isinstance(axiom, NF6):
            r_sub: RectUnion = self.eta.role_region(axiom.sub)
            r_sup: RectUnion = self.eta.role_region(axiom.sup)
            return r_sub.is_subset(r_sup)
        if isinstance(axiom, NF7):
            r1: RectUnion = self.eta.role_region(axiom.r1)
            r2: RectUnion = self.eta.role_region(axiom.r2)
            r_sup = self.eta.role_region(axiom.sup)
            return r1.compose(r2).is_subset(r_sup)
        raise UnsupportedAxiom(f"Unsupported axiom type: {type(axiom).__name__}")

    def _satisfies_box(self, axiom: Axiom) -> bool:
        if isinstance(axiom, NF1):
            reg_sub: Box = self._class_region(axiom.sub)
            reg_sup: Box = self._class_region(axiom.sup)
            return reg_sub.is_subset(reg_sup, tol=self.tol)
        if isinstance(axiom, NF2):
            reg_1: Box = self._class_region(axiom.sub1)
            reg_2: Box = self._class_region(axiom.sub2)
            reg_sup = self._class_region(axiom.sup)
            inter = self._box_intersection_with_tol(reg_1, reg_2)
            return inter.is_subset(reg_sup, tol=self.tol)
        if isinstance(axiom, (NF3, NF4, NF6, NF7)):
            raise UnsupportedAxiom("box family: role axioms not supported in v1")
        raise UnsupportedAxiom(f"Unsupported axiom type: {type(axiom).__name__}")

    def _box_intersection_with_tol(self, b1: Box, b2: Box) -> Box:
        d = int(b1.lo.shape[0])
        if b1.is_empty or b2.is_empty:
            return Box.empty(d)
        lo = np.maximum(b1.lo, b2.lo)
        hi = np.minimum(b1.hi, b2.hi)
        if np.any(lo > hi + self.tol):
            return Box.empty(d)
        return Box(lo, np.maximum(lo, hi))

    def _satisfies_polyhedral(self, axiom: Axiom) -> bool:
        if isinstance(axiom, NF1):
            reg_sub = self._class_region(axiom.sub)
            reg_sup = self._class_region(axiom.sup)
            return self._poly_subset(reg_sub, reg_sup)
        if isinstance(axiom, NF2):
            reg_1 = self._class_region(axiom.sub1)
            reg_2 = self._class_region(axiom.sub2)
            reg_sup = self._class_region(axiom.sup)
            return self._poly_intersection_subset(reg_1, reg_2, reg_sup)
        if isinstance(axiom, NF3):
            reg_sub = self._class_region(axiom.sub)
            reg_fill = self._class_region(axiom.filler)
            role_reg = self.eta.role_region(axiom.role)
            return self._poly_nf3(reg_sub, role_reg, reg_fill)
        if isinstance(axiom, NF4):
            reg_fill = self._class_region(axiom.filler)
            reg_sup = self._class_region(axiom.sup)
            role_reg = self.eta.role_region(axiom.role)
            return self._poly_nf4(role_reg, reg_fill, reg_sup)
        if isinstance(axiom, NF6):
            r_sub = self.eta.role_region(axiom.sub)
            r_sup = self.eta.role_region(axiom.sup)
            return self._poly_nf6(r_sub, r_sup)
        if isinstance(axiom, NF7):
            r1 = self.eta.role_region(axiom.r1)
            r2 = self.eta.role_region(axiom.r2)
            r_sup = self.eta.role_region(axiom.sup)
            return self._poly_nf7(r1, r2, r_sup)
        raise UnsupportedAxiom(f"Unsupported axiom type: {type(axiom).__name__}")

    # ------------------------------------------------------------------
    # Region helpers across families
    # ------------------------------------------------------------------

    def _region_is_empty(self, reg: Any) -> bool:
        if self.eta.family in ("interval", "box"):
            return bool(reg.is_empty)
        # polyhedral
        if type(reg) is Face:
            return bool(reg.is_empty)
        if getattr(reg, "is_empty", False):
            return True
        if hasattr(reg, "h_rep"):
            A_ub, b_ub, A_eq, b_eq = reg.h_rep()
            return not self._hrep_is_feasible(A_ub, b_ub, A_eq, b_eq)
        raise UnsupportedRegion(
            f"Polyhedral class region {reg!r} must be a Face or provide h_rep()"
        )

    def _class_subset(self, reg_a: Any, reg_b: Any) -> bool:
        fam = self.eta.family
        if fam == "interval":
            return bool(reg_a.is_subset(reg_b))
        if fam == "box":
            return bool(reg_a.is_subset(reg_b, tol=self.tol))
        return self._poly_subset(reg_a, reg_b)

    # ------------------------------------------------------------------
    # Polyhedral exact checks (combinatorial on Face + HiGHS LPs)
    # ------------------------------------------------------------------

    def _poly_subset(self, reg_sub: Any, reg_sup: Any) -> bool:
        if type(reg_sub) is Face and type(reg_sup) is Face:
            return reg_sub.is_subset(reg_sup)
        if getattr(reg_sub, "is_empty", False):
            return True
        if not hasattr(reg_sub, "h_rep") or not hasattr(reg_sup, "h_rep"):
            raise UnsupportedRegion(
                "Polyhedral inclusion requires Face or regions providing h_rep()"
            )
        A_ub_1, b_ub_1, A_eq_1, b_eq_1 = reg_sub.h_rep()
        if not self._hrep_is_feasible(A_ub_1, b_ub_1, A_eq_1, b_eq_1):
            return True
        if getattr(reg_sup, "is_empty", False):
            return False
        A_ub_2, b_ub_2, A_eq_2, b_eq_2 = reg_sup.h_rep()
        if not self._hrep_is_feasible(A_ub_2, b_ub_2, A_eq_2, b_eq_2):
            return False
        return self._hrep_subset(
            A_ub_1, b_ub_1, A_eq_1, b_eq_1, A_ub_2, b_ub_2, A_eq_2, b_eq_2
        )

    def _poly_intersection_subset(
        self, reg_1: Any, reg_2: Any, reg_sup: Any
    ) -> bool:
        if (
            type(reg_1) is Face
            and type(reg_2) is Face
            and type(reg_sup) is Face
        ):
            return reg_1.intersection(reg_2).is_subset(reg_sup)

        if getattr(reg_1, "is_empty", False) or getattr(reg_2, "is_empty", False):
            return True
        if not (hasattr(reg_1, "h_rep") and hasattr(reg_2, "h_rep") and hasattr(reg_sup, "h_rep")):
            raise UnsupportedRegion(
                "Polyhedral conjunction check requires Face or regions providing h_rep()"
            )
        A_ub_1, b_ub_1, A_eq_1, b_eq_1 = reg_1.h_rep()
        A_ub_2, b_ub_2, A_eq_2, b_eq_2 = reg_2.h_rep()
        A_ub_12, b_ub_12 = self._concat_system(A_ub_1, b_ub_1, A_ub_2, b_ub_2)
        A_eq_12, b_eq_12 = self._concat_system(A_eq_1, b_eq_1, A_eq_2, b_eq_2)

        if not self._hrep_is_feasible(A_ub_12, b_ub_12, A_eq_12, b_eq_12):
            return True
        if getattr(reg_sup, "is_empty", False):
            return False
        A_ub_s, b_ub_s, A_eq_s, b_eq_s = reg_sup.h_rep()
        if not self._hrep_is_feasible(A_ub_s, b_ub_s, A_eq_s, b_eq_s):
            return False
        return self._hrep_subset(
            A_ub_12, b_ub_12, A_eq_12, b_eq_12, A_ub_s, b_ub_s, A_eq_s, b_eq_s
        )

    @staticmethod
    def _concat_system(
        A1: np.ndarray, b1: np.ndarray, A2: np.ndarray, b2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        k = A1.shape[1] if A1.ndim == 2 else A2.shape[1]
        rows = []
        rhs = []
        if A1.shape[0] > 0:
            rows.append(A1)
            rhs.append(b1)
        if A2.shape[0] > 0:
            rows.append(A2)
            rhs.append(b2)
        if not rows:
            return np.zeros((0, k), dtype=float), np.zeros((0,), dtype=float)
        return np.vstack(rows), np.concatenate(rhs)

    def _hrep_is_feasible(
        self,
        A_ub: np.ndarray,
        b_ub: np.ndarray,
        A_eq: np.ndarray,
        b_eq: np.ndarray,
    ) -> bool:
        k = int(A_ub.shape[1] if A_ub.ndim == 2 else A_eq.shape[1])
        if k == 0:
            if A_ub.shape[0] > 0 and bool(np.any(-b_ub > self.lp_tol)):
                return False
            if A_eq.shape[0] > 0 and bool(np.any(np.abs(b_eq) > self.lp_tol)):
                return False
            return True
        c = np.zeros(k, dtype=float)
        res = linprog(
            c,
            A_ub=A_ub if A_ub.shape[0] > 0 else None,
            b_ub=b_ub if A_ub.shape[0] > 0 else None,
            A_eq=A_eq if A_eq.shape[0] > 0 else None,
            b_eq=b_eq if A_eq.shape[0] > 0 else None,
            bounds=(None, None),
            method="highs",
        )
        return bool(res.success)

    def _hrep_subset(
        self,
        A_ub_1: np.ndarray,
        b_ub_1: np.ndarray,
        A_eq_1: np.ndarray,
        b_eq_1: np.ndarray,
        A_ub_2: np.ndarray,
        b_ub_2: np.ndarray,
        A_eq_2: np.ndarray,
        b_eq_2: np.ndarray,
    ) -> bool:
        k = int(A_ub_1.shape[1])
        if k == 0:
            return True

        directions: List[Tuple[np.ndarray, float]] = []
        for idx in range(A_ub_2.shape[0]):
            directions.append((A_ub_2[idx], float(b_ub_2[idx])))
        for idx in range(A_eq_2.shape[0]):
            directions.append((A_eq_2[idx], float(b_eq_2[idx])))
            directions.append((-A_eq_2[idx], float(-b_eq_2[idx])))

        for row, rhs in directions:
            res = linprog(
                -row,
                A_ub=A_ub_1 if A_ub_1.shape[0] > 0 else None,
                b_ub=b_ub_1 if A_ub_1.shape[0] > 0 else None,
                A_eq=A_eq_1 if A_eq_1.shape[0] > 0 else None,
                b_eq=b_eq_1 if A_eq_1.shape[0] > 0 else None,
                bounds=(None, None),
                method="highs",
            )
            if not res.success:
                return False
            max_val = float(-res.fun)
            if max_val - rhs > self.lp_tol:
                return False
        return True

    @staticmethod
    def _validate_face_for_nf3(reg: Any) -> Face:
        if type(reg) is not Face:
            raise UnsupportedRegion(
                "polyhedral family: NF3 (A <= exists r.B) requires class_region(A) "
                "to be a Face with stored vertices"
            )
        return reg

    @staticmethod
    def _validate_pair_hull(role_reg: Any) -> PairHull:
        if not isinstance(role_reg, PairHull) or not hasattr(role_reg, "points"):
            raise UnsupportedRegion(
                f"polyhedral family: role region must be a PairHull with stored points, got {role_reg!r}"
            )
        pts = getattr(role_reg, "points", None)
        if not isinstance(pts, np.ndarray) or pts.ndim != 2 or pts.shape[1] % 2 != 0:
            raise UnsupportedRegion(
                "polyhedral family: PairHull.points must be a 2-D numpy array of shape (m, 2k)"
            )
        return role_reg

    @staticmethod
    def _require_simplex_vertex_regions(
        axiom_kind: str, faces: Iterable[Any] = (), hulls: Iterable[PairHull] = ()
    ) -> None:
        """Raise :class:`UnsupportedRegion` unless all Face rows and PairHull halves are canonical simplex vertices.

        Role-axiom checking (NF3, NF4, NF6, NF7) composes and tests generating
        points of :class:`~mowl.geometric.regions.PairHull` regions; this is
        exact only when every generating half is a vertex of the canonical
        simplex (a one-hot unit vector or the zero vector), because then the
        barycentric coordinates of intermediate points are unique.
        """
        bad = False
        for reg in faces:
            if isinstance(reg, Face) and not is_simplex_vertex_rows(
                reg.vertices, reg.dim
            ):
                bad = True
        for hull in hulls:
            k = int(hull.points.shape[1]) // 2
            if not (
                is_simplex_vertex_rows(hull.left(), k)
                and is_simplex_vertex_rows(hull.right(), k)
            ):
                bad = True
        if bad:
            raise UnsupportedRegion(
                f"polyhedral family: {axiom_kind} role-axiom checking is exact only "
                f"for simplex-vertex regions (every Face vertex row and every "
                f"PairHull left/right half must be a one-hot unit vector or the "
                f"zero vector of the canonical simplex)"
            )

    def _poly_nf3(self, reg_sub: Any, role_reg: Any, reg_fill: Any) -> bool:
        """Decide ``A <= exists r.B`` by checking each vertex of ``Face(A)`` via LP."""
        face_a = self._validate_face_for_nf3(reg_sub)
        ph_r = self._validate_pair_hull(role_reg)
        self._require_simplex_vertex_regions(
            "NF3", faces=(face_a, reg_fill), hulls=(ph_r,)
        )
        if not hasattr(reg_fill, "h_rep"):
            raise UnsupportedRegion(
                f"polyhedral family: filler region {reg_fill!r} must provide h_rep()"
            )

        if face_a.is_empty:
            return True
        if getattr(reg_fill, "is_empty", False) or ph_r.is_empty:
            return False

        m = int(ph_r.points.shape[0])
        L = ph_r.left()   # (m, k)
        R = ph_r.right()  # (m, k)
        A_ub_b, b_ub_b, A_eq_b, b_eq_b = reg_fill.h_rep()

        # Constraints on mu in R^m independent of v:
        # 1. sum(mu) = 1
        # 2. (A_eq_b @ R.T) @ mu = b_eq_b
        # 3. (A_ub_b @ R.T) @ mu <= b_ub_b
        A_ub_mu = A_ub_b @ R.T if A_ub_b.shape[0] > 0 else None
        b_ub_mu = b_ub_b if A_ub_b.shape[0] > 0 else None

        base_eq_rows = [np.ones((1, m), dtype=float)]
        base_eq_rhs = [np.array([1.0], dtype=float)]
        if A_eq_b.shape[0] > 0:
            base_eq_rows.append(A_eq_b @ R.T)
            base_eq_rhs.append(b_eq_b)

        c = np.zeros(m, dtype=float)
        for v in face_a.vertices:
            eq_rows = list(base_eq_rows)
            eq_rhs = list(base_eq_rhs)
            if L.shape[1] > 0:
                eq_rows.append(L.T)
                eq_rhs.append(v)
            A_eq = np.vstack(eq_rows)
            b_eq = np.concatenate(eq_rhs)

            res = linprog(
                c,
                A_ub=A_ub_mu,
                b_ub=b_ub_mu,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=(0.0, None),
                method="highs",
            )
            if not res.success:
                return False
        return True

    def _poly_nf4(self, role_reg: Any, reg_fill: Any, reg_sup: Any) -> bool:
        """Decide ``exists r.A <= B`` via LP over ``PairHull(r)`` and ``h_rep(B)``."""
        ph_r = self._validate_pair_hull(role_reg)
        if not hasattr(reg_fill, "h_rep") or not hasattr(reg_sup, "h_rep"):
            raise UnsupportedRegion(
                "polyhedral family: NF4 requires filler and superclass regions to provide h_rep()"
            )
        self._require_simplex_vertex_regions(
            "NF4", faces=(reg_fill, reg_sup), hulls=(ph_r,)
        )

        if ph_r.is_empty or getattr(reg_fill, "is_empty", False):
            return True

        m = int(ph_r.points.shape[0])
        L = ph_r.left()   # (m, k)
        R = ph_r.right()  # (m, k)

        A_ub_a, b_ub_a, A_eq_a, b_eq_a = reg_fill.h_rep()
        A_ub_mu = A_ub_a @ R.T if A_ub_a.shape[0] > 0 else None
        b_ub_mu = b_ub_a if A_ub_a.shape[0] > 0 else None

        eq_rows = [np.ones((1, m), dtype=float)]
        eq_rhs = [np.array([1.0], dtype=float)]
        if A_eq_a.shape[0] > 0:
            eq_rows.append(A_eq_a @ R.T)
            eq_rhs.append(b_eq_a)
        A_eq_mu = np.vstack(eq_rows)
        b_eq_mu = np.concatenate(eq_rhs)

        # First check whether exists r.A is non-empty
        feas = linprog(
            np.zeros(m, dtype=float),
            A_ub=A_ub_mu,
            b_ub=b_ub_mu,
            A_eq=A_eq_mu,
            b_eq=b_eq_mu,
            bounds=(0.0, None),
            method="highs",
        )
        if not feas.success:
            # exists r.A is empty => subset of any B (including empty)
            return True

        if getattr(reg_sup, "is_empty", False):
            return False

        A_ub_b, b_ub_b, A_eq_b, b_eq_b = reg_sup.h_rep()
        if not self._hrep_is_feasible(A_ub_b, b_ub_b, A_eq_b, b_eq_b):
            return False

        # Maximise each constraint row h . v - rhs over mu:
        # Since v = L.T @ mu, h . v = (h @ L.T) @ mu.
        directions: List[Tuple[np.ndarray, float]] = []
        for idx in range(A_ub_b.shape[0]):
            directions.append((A_ub_b[idx], float(b_ub_b[idx])))
        for idx in range(A_eq_b.shape[0]):
            directions.append((A_eq_b[idx], float(b_eq_b[idx])))
            directions.append((-A_eq_b[idx], float(-b_eq_b[idx])))

        for h_row, rhs in directions:
            c_mu = -(h_row @ L.T)
            res = linprog(
                c_mu,
                A_ub=A_ub_mu,
                b_ub=b_ub_mu,
                A_eq=A_eq_mu,
                b_eq=b_eq_mu,
                bounds=(0.0, None),
                method="highs",
            )
            if not res.success:
                return False
            max_val = float(-res.fun)
            if max_val - rhs > self.lp_tol:
                return False
        return True

    def _point_in_pair_hull(self, pt: np.ndarray, hull: PairHull) -> bool:
        """Return ``True`` iff ``pt`` lies in ``conv(hull.points)`` via LP feasibility."""
        if hull.is_empty:
            return False
        P = hull.points  # (m, 2k)
        m, two_k = P.shape
        if two_k == 0:
            return m > 0
        A_eq = np.vstack([np.ones((1, m), dtype=float), P.T])
        b_eq = np.concatenate([np.array([1.0], dtype=float), pt.reshape(two_k)])
        res = linprog(
            np.zeros(m, dtype=float),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0.0, None),
            method="highs",
        )
        return bool(res.success)

    def _poly_nf6(self, r_sub: Any, r_sup: Any) -> bool:
        """Decide ``r <= s`` by testing every generating point of ``PairHull(r)`` in ``PairHull(s)``."""
        ph_sub = self._validate_pair_hull(r_sub)
        ph_sup = self._validate_pair_hull(r_sup)
        self._require_simplex_vertex_regions("NF6", hulls=(ph_sub, ph_sup))
        if ph_sub.is_empty:
            return True
        if ph_sup.is_empty:
            return False
        for pt in ph_sub.points:
            if not self._point_in_pair_hull(pt, ph_sup):
                return False
        return True

    def _poly_nf7(self, r1: Any, r2: Any, r_sup: Any) -> bool:
        """Decide ``r1 o r2 <= sup`` on generating element-pair points."""
        ph_1 = self._validate_pair_hull(r1)
        ph_2 = self._validate_pair_hull(r2)
        ph_sup = self._validate_pair_hull(r_sup)
        self._require_simplex_vertex_regions("NF7", hulls=(ph_1, ph_2, ph_sup))
        if ph_1.is_empty or ph_2.is_empty:
            return True

        L1 = ph_1.left()
        R1 = ph_1.right()
        L2 = ph_2.left()
        R2 = ph_2.right()

        for i in range(ph_1.points.shape[0]):
            v = L1[i]
            u = R1[i]
            for j in range(ph_2.points.shape[0]):
                u_prime = L2[j]
                if u.shape[0] == 0 or bool(np.all(np.abs(u - u_prime) <= self.tol)):
                    w = R2[j]
                    vw = np.concatenate([v, w])
                    if not self._point_in_pair_hull(vw, ph_sup):
                        return False
        return True
