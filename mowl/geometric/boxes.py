"""MILP-based axis-aligned box-model construction with Helly pre-check.

Implements the MILP search for axis-parallel box models of propositional EL
ontologies and the :class:`BoxModel` interpretation of Spec §5.3 (G5).

Contract and mathematical guarantees (Spec §5.3 [v2, contract corrected]):
- By Helly's theorem for axis-parallel boxes in :math:`\\mathbb{R}^d` (Helly
  number 2; Eckhoff 1988; Leemhuis & Kutz, NeSy 2025, Defs. 3, 6; Prop. 4),
  box semantics always satisfy Helly's property. Consequently, an ontology that
  is not Helly-satisfiable has no box model in any dimension (Leemhuis & Kutz,
  NeSy 2025, Prop. 5).
- :class:`BoxModelBuilder` supports the propositional EL fragment only
  (:class:`~mowl.geometric.normal_forms.NF1`,
  :class:`~mowl.geometric.normal_forms.NF2`, and their :math:`\\bot` forms). If
  ``cm.axioms`` contains any role axiom
  (:class:`~mowl.geometric.normal_forms.NF3`,
  :class:`~mowl.geometric.normal_forms.NF4`,
  :class:`~mowl.geometric.normal_forms.NF6`, or
  :class:`~mowl.geometric.normal_forms.NF7`), :meth:`BoxModelBuilder.build`
  raises :class:`~mowl.geometric.errors.UnsupportedAxiom`.
- Before running the MILP, :meth:`BoxModelBuilder.build` runs
  :func:`~mowl.geometric.helly.helly_check`. In both faithfulness modes it
  raises :class:`~mowl.geometric.errors.NotBoxRepresentable` carrying
  ``report`` if and only if ``report.certificates_nondegenerate`` is non-empty.
  Such a certificate is a sound proof that no non-degenerate box model exists
  in any dimension; since every model returned in either mode is
  non-degenerate, it also rules out a strongly faithful one.
- ``report.certificates_strong`` is deliberately *not* used to reject
  ``faithful="strong"``: a strong certificate only excludes
  *conjunction-faithful* box models (every satisfiable conjunction of concept
  names non-empty), which is stronger than strong faithfulness on names. For
  example, :math:`\\{X_1 \\sqcap X_2 \\sqsubseteq Y, Y \\sqcap X_3 \\sqsubseteq \\bot\\}`
  has the strong certificate :math:`(X_1, X_2, X_3)`, yet the 1-D boxes
  :math:`X_1 = [0, 1], X_2 = [2, 3], X_3 = [4, 5], Y = [6, 7]` form a
  non-degenerate model that is strongly faithful on names.
- A passing Helly check does *not* guarantee that an axis-parallel box model
  exists (no such sufficiency theorem is known to us; ``box_representable_* =
  True`` means only that no Helly obstruction of that kind was found). When no
  non-degenerate certificate is found, :class:`BoxModelBuilder` searches for a box
  model by mixed-integer linear programming (`scipy.optimize.milp`, HiGHS) over
  the dimension schedule :math:`d = 1, 2, 4, \\dots, d_{\\max}` (capped at
  ``d_max``).
- If the solver time limit is hit at some :math:`d` with a feasible incumbent,
  the incumbent is turned into a :class:`BoxModel` and post-checked; it is
  returned if it passes, otherwise the search continues with the next
  :math:`d`. Solver status 4 (other problem) is logged at ``logging.INFO``
  and treated as no model found at that :math:`d`.
- If the MILP is infeasible for every :math:`d` in the schedule or the solver
  time limit is exceeded without a passing incumbent,
  :meth:`BoxModelBuilder.build` raises
  :class:`~mowl.geometric.errors.BoxModelNotFound` (an *inconclusive* outcome
  whose message states ``d_max``, ``time_limit``, and that no theorem
  guarantees existence of a box model when the Helly check passes).
- Every returned :class:`BoxModel` is verified before return by
  :class:`~mowl.geometric.checker.ModelChecker`:
  :meth:`~mowl.geometric.checker.ModelChecker.is_model` on ``cm.axioms`` and
  :meth:`~mowl.geometric.checker.ModelChecker.is_nondegenerate` on ``cm`` (plus
  :meth:`~mowl.geometric.checker.ModelChecker.is_strongly_faithful_on_names` on
  ``cm`` when ``faithful="strong"``). A failed post-check is a bug and raises
  :class:`~mowl.geometric.errors.InternalConsistencyError`.
- Scalability note: feasibility is checked by MILP with
  ``scipy.optimize.milp`` (HiGHS); exact up to solver tolerances, exponential
  in the worst case, and intended for small ontologies (:math:`\\le \\sim 200`
  concept names).

References:
- M. Leemhuis & O. Kutz, *Understanding the Expressive Capabilities of
  Knowledge Base Embeddings under Box Semantics*, NeSy 2025 (PMLR
  284:303–321), Definitions 1, 3, 6; Propositions 4, 5, 8.
- Eckhoff, *Intersection properties of boxes. Part I: An upper-bound theorem*,
  Israel Journal of Mathematics 62(3):283–301, 1988.
- V. Lacerda, A. Ozaki & R. Guimarães, *Strong Faithfulness for ELH Ontology
  Embeddings*, TGDK 2(3):2, 2024, Defs. 2, 8, 12.
- V. Gutiérrez-Basulto & S. Schockaert, *From Knowledge Graph Embedding to
  Ontology Embedding? An Analysis of the Compatibility between Vector Space
  Representations and Rules*, KR 2018 (arXiv:1805.10461), Props. 3, 4.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from scipy.sparse import csc_matrix

try:
    from scipy.optimize import Bounds, LinearConstraint, milp as _scipy_milp
except ImportError:  # pragma: no cover
    Bounds = None  # type: ignore[assignment,misc]
    LinearConstraint = None  # type: ignore[assignment,misc]
    _scipy_milp = None  # type: ignore[assignment]

from mowl.geometric.checker import GeometricInterpretation, ModelChecker
from mowl.geometric.errors import (
    BoxModelNotFound,
    InternalConsistencyError,
    MissingDependency,
    NotBoxRepresentable,
    UnsupportedAxiom,
)
from mowl.geometric.helly import helly_check
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
from mowl.geometric.regions import Box

if TYPE_CHECKING:  # pragma: no cover
    from mowl.geometric.completion import CanonicalModel

logger = logging.getLogger(__name__)


def _dimension_schedule(d_max: int) -> List[int]:
    """Return the schedule ``1, 2, 4, ..., d_max`` (including ``d_max`` if not a power of 2)."""
    schedule: List[int] = []
    d = 1
    while d <= d_max:
        schedule.append(d)
        d *= 2
    if not schedule or schedule[-1] != d_max:
        schedule.append(d_max)
    return schedule


class BoxModel(GeometricInterpretation):
    """Axis-aligned box interpretation :math:`\\eta` over universe :math:`[0, M]^d`.

    Produced by :class:`BoxModelBuilder` for ontologies in the propositional EL
    fragment. Every :class:`BoxModel` returned by :class:`BoxModelBuilder` has
    passed :meth:`~mowl.geometric.checker.ModelChecker.is_model` and
    :meth:`~mowl.geometric.checker.ModelChecker.is_nondegenerate` (and
    :meth:`~mowl.geometric.checker.ModelChecker.is_strongly_faithful_on_names`
    when built with ``faithful="strong"``).

    Parameters
    ----------
    cm : CanonicalModel
        Canonical model of the normalised ontology.
    dimension : int
        Ambient Euclidean dimension :math:`d \\ge 1`.
    boxes : Dict[str, Box]
        Mapping from satisfiable concept names (excluding ``BOTTOM``) to
        non-empty :class:`~mowl.geometric.regions.Box` regions in
        :math:`[0, M]^d`. ``TOP`` is always mapped to ``universe = [0, M]^d``.
    M : float, default 100.0
        Upper coordinate bound of the universe box :math:`[0, M]^d`.
    """

    family: str = "box"

    def __init__(
        self,
        cm: CanonicalModel,
        dimension: int,
        boxes: Dict[str, Box],
        M: float = 100.0,
    ) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError(f"BoxModel.dimension must be an int >= 1, got {dimension!r}")
        self.canonical_model: CanonicalModel = cm
        self.dimension: int = int(dimension)
        self.M: float = float(M)
        self._names: List[str] = list(cm.names)
        self._universe: Box = Box(
            np.zeros(self.dimension, dtype=float),
            np.full(self.dimension, self.M, dtype=float),
        )
        self._empty_box: Box = Box.empty(self.dimension)

        self._boxes: Dict[str, Box] = {}
        for name, box in boxes.items():
            if name == BOTTOM or box.is_empty:
                continue
            self._boxes[name] = Box(box.lo.copy(), box.hi.copy())
        self._boxes[TOP] = self._universe

        self._checker: Optional[ModelChecker] = None

    @property
    def universe(self) -> Box:
        """Return the universe bounding box :math:`U = \\eta(\\top) = [0, M]^d`."""
        return self._universe

    @property
    def names(self) -> Iterable[str]:
        """Return the satisfiable concept names in ``canonical_model.names``."""
        return list(self._names)

    @property
    def class_regions(self) -> Dict[str, Box]:
        """Return a copy of the mapping from known concept names to :class:`~mowl.geometric.regions.Box` regions."""
        out: Dict[str, Box] = {
            cname: self.class_region(cname)
            for cname in self.canonical_model.concept_names
        }
        out[TOP] = self._universe
        out[BOTTOM] = self._empty_box
        return out

    @property
    def role_regions(self) -> Dict[str, Any]:
        """Return an empty dict (role regions are not supported in the v1 propositional box model)."""
        return {}

    def class_region(self, name: str) -> Box:
        """Return :math:`\\eta(\\text{name})`: ``universe`` for ``TOP``, empty :class:`~mowl.geometric.regions.Box` for ``BOTTOM`` or unknown/unsatisfiable names."""
        if name == BOTTOM:
            return self._empty_box
        if name == TOP:
            return self._universe
        return self._boxes.get(name, self._empty_box)

    def role_region(self, role: str) -> Any:
        """Raise :class:`~mowl.geometric.errors.UnsupportedAxiom` (role axioms are not supported in v1)."""
        raise UnsupportedAxiom("box family: role axioms not supported in v1")

    def boxes(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return ``{name: (lo, hi)}`` copies for every concept name assigned a non-empty box."""
        return {
            name: (box.lo.copy(), box.hi.copy())
            for name, box in self._boxes.items()
        }

    def check(self) -> ModelChecker:
        """Return the cached :class:`~mowl.geometric.checker.ModelChecker` bound to this interpretation."""
        if self._checker is None:
            self._checker = ModelChecker(self)
        return self._checker

    def satisfies(self, axiom: Axiom) -> bool:
        """Return ``True`` iff this model satisfies ``axiom`` (via :meth:`check`)."""
        return self.check().satisfies(axiom)

    def is_model(self, axioms: Optional[Iterable[Axiom]] = None) -> bool:
        """Return ``True`` iff this interpretation satisfies ``axioms`` (defaults to ``self.canonical_model.axioms``)."""
        target = self.canonical_model.axioms if axioms is None else axioms
        return self.check().is_model(target)

    def is_nondegenerate(self, cm: Optional[CanonicalModel] = None) -> bool:
        """Return ``True`` iff every satisfiable name in ``cm`` (defaults to ``self.canonical_model``) has a non-empty box."""
        target = self.canonical_model if cm is None else cm
        return self.check().is_nondegenerate(target)

    def is_strongly_faithful_on_names(
        self, cm: Optional[CanonicalModel] = None
    ) -> bool:
        """Return ``True`` iff box inclusion on names matches entailment in ``cm`` (defaults to ``self.canonical_model``)."""
        target = self.canonical_model if cm is None else cm
        return self.check().is_strongly_faithful_on_names(target)


class BoxModelBuilder:
    """MILP builder searching for an axis-parallel :class:`BoxModel` after a Helly pre-check.

    Contract (Spec §5.3 [v2, contract corrected]; Leemhuis & Kutz, NeSy 2025,
    Prop. 4 & Prop. 5):
    - Raises :class:`~mowl.geometric.errors.UnsupportedAxiom` if ``cm.axioms``
      contains any role axiom (:class:`~mowl.geometric.normal_forms.NF3`,
      :class:`~mowl.geometric.normal_forms.NF4`,
      :class:`~mowl.geometric.normal_forms.NF6`, or
      :class:`~mowl.geometric.normal_forms.NF7`).
    - Runs :func:`~mowl.geometric.helly.helly_check` first; in both
      faithfulness modes raises
      :class:`~mowl.geometric.errors.NotBoxRepresentable` carrying the
      :class:`~mowl.geometric.helly.HellyReport` iff
      ``report.certificates_nondegenerate`` is non-empty. By Leemhuis & Kutz
      (NeSy 2025, Prop. 4), box semantics satisfy Helly's property (Helly
      number 2), so such a certificate is a proof that no non-degenerate box
      model (hence none of either requested kind) exists in any dimension.
      Strong certificates alone do not reject ``faithful="strong"``: they
      exclude only *conjunction-faithful* box models (every satisfiable
      conjunction of names non-empty), not models strongly faithful on names.
    - When no non-degenerate certificate is found, encodes box
      feasibility as a mixed-integer linear program (`scipy.optimize.milp`;
      raises :class:`~mowl.geometric.errors.MissingDependency` naming
      ``scipy >= 1.9`` if unavailable) over the dimension schedule
      :math:`d = 1, 2, 4, \\dots, d_{\\max}`.
    - If the solver hits the time limit with a feasible incumbent, the
      incumbent is post-checked and returned if it passes; otherwise the
      schedule continues with the next :math:`d`.
    - Raises :class:`~mowl.geometric.errors.BoxModelNotFound` (inconclusive) if
      the MILP is infeasible for all :math:`d` in the schedule or the solver
      time limit is exceeded without a passing incumbent; the exception message states ``d_max``,
      ``time_limit``, and that no theorem guarantees existence of a box model
      when the Helly check passes.
    - Every returned :class:`BoxModel` has passed
      :meth:`~mowl.geometric.checker.ModelChecker.is_model` on ``cm.axioms`` and
      :meth:`~mowl.geometric.checker.ModelChecker.is_nondegenerate` on ``cm``
      (plus
      :meth:`~mowl.geometric.checker.ModelChecker.is_strongly_faithful_on_names`
      on ``cm`` when ``faithful="strong"``); a failed post-check raises
      :class:`~mowl.geometric.errors.InternalConsistencyError`.

    Parameters
    ----------
    cm : CanonicalModel
        Canonical model of a normalised propositional EL ontology.
    d_max : int, default 8
        Maximum dimension in the schedule :math:`1, 2, 4, \\dots, d_{\\max}`.
    time_limit : float, default 60.0
        Total wall-clock time budget in seconds across all dimensions in the
        schedule.
    faithful : str, default "weak"
        ``"weak"`` (axioms + non-degeneracy on satisfiable names) or
        ``"strong"`` (additionally requires witness points separating every
        non-entailed ordered pair of satisfiable names).
    M : float, default 100.0
        Big-M and coordinate upper bound for the bounding box :math:`[0, M]^d`.
        Separation constraints on ``e`` and strong-faithfulness witness
        constraints on ``o`` use big-M coefficient :math:`(M + \\delta)` so
        they are strictly vacuous when their binary is 0:
        :math:`U_A + \\delta \\le L_B + (M + \\delta)(1 - e_{i,0})`,
        :math:`U_B + \\delta \\le L_A + (M + \\delta)(1 - e_{i,1})`,
        :math:`w_{A,B,i} \\le L_B - \\delta + (M + \\delta)(1 - o_{i,0})`, and
        :math:`w_{A,B,i} \\ge U_B + \\delta - (M + \\delta)(1 - o_{i,1})`.
    delta : float, default 1.0
        Strict separation margin :math:`\\delta > 0` for disjointness and
        strong-faithfulness outside-box constraints.
    helly_kwargs : Optional[Dict[str, Any]], default None
        Optional keyword arguments forwarded to
        :func:`~mowl.geometric.helly.helly_check` (e.g. ``k_max``,
        ``max_certificates``).
    """

    def __init__(
        self,
        cm: CanonicalModel,
        d_max: int = 8,
        time_limit: float = 60.0,
        faithful: str = "weak",
        M: float = 100.0,
        delta: float = 1.0,
        helly_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if faithful not in ("weak", "strong"):
            raise ValueError(
                f"faithful must be 'weak' or 'strong', got {faithful!r}"
            )
        if isinstance(d_max, bool) or not isinstance(d_max, int) or d_max < 1:
            raise ValueError(f"d_max must be an int >= 1, got {d_max!r}")
        if float(time_limit) <= 0.0:
            raise ValueError(f"time_limit must be > 0, got {time_limit!r}")
        if float(M) <= 0.0 or float(delta) <= 0.0 or float(delta) >= float(M):
            raise ValueError(
                f"Require 0 < delta < M, got delta={delta!r}, M={M!r}"
            )

        self.cm: CanonicalModel = cm
        self.d_max: int = int(d_max)
        self.time_limit: float = float(time_limit)
        self.faithful: str = faithful
        self.M: float = float(M)
        self.delta: float = float(delta)
        self.helly_kwargs: Dict[str, Any] = dict(helly_kwargs or {})

    def build(self) -> BoxModel:
        """Run the Helly check and MILP dimension schedule to construct a verified :class:`BoxModel`."""
        for ax in self.cm.axioms:
            if isinstance(ax, (NF3, NF4, NF6, NF7)):
                raise UnsupportedAxiom(
                    f"BoxModelBuilder supports only propositional NF1/NF2 axioms; "
                    f"got role axiom {ax!r}"
                )

        if _scipy_milp is None or LinearConstraint is None or Bounds is None:
            raise MissingDependency(
                "BoxModelBuilder requires scipy.optimize.milp (scipy >= 1.9)."
            )

        report = helly_check(self.cm, **self.helly_kwargs)
        # Both modes require non-degeneracy, so only non-degenerate certificates
        # are proofs here; strong certificates exclude only conjunction-faithful
        # box models, not models that are strongly faithful on names.
        if report.certificates_nondegenerate:
            raise NotBoxRepresentable(
                "Ontology has no non-degenerate box model in any dimension "
                "(non-degenerate Helly certificate found).",
                report,
            )

        schedule = _dimension_schedule(self.d_max)
        start_time = time.monotonic()

        for d in schedule:
            elapsed = time.monotonic() - start_time
            remaining = self.time_limit - elapsed
            if remaining <= 0.0:
                raise BoxModelNotFound(
                    f"BoxModelBuilder inconclusive: time_limit={self.time_limit}s "
                    f"exceeded before d={d} (d_max={self.d_max}); no theorem "
                    f"guarantees existence of a box model when the Helly check passes."
                )

            status, boxes_d = self._solve_for_dimension(d, remaining)
            if status == "optimal":
                assert boxes_d is not None
                model = BoxModel(
                    cm=self.cm,
                    dimension=d,
                    boxes=boxes_d,
                    M=self.M,
                )
                self._verify_model(model)
                return model
            if status == "incumbent":
                assert boxes_d is not None
                model = BoxModel(
                    cm=self.cm,
                    dimension=d,
                    boxes=boxes_d,
                    M=self.M,
                )
                if self._passes_post_check(model):
                    return model
                logger.info(
                    "BoxModelBuilder: time-limited MILP incumbent at d=%d failed "
                    "the model post-check; continuing with the next dimension.",
                    d,
                )
                continue
            if status == "timeout":
                raise BoxModelNotFound(
                    f"BoxModelBuilder inconclusive: MILP solver reached "
                    f"time_limit={self.time_limit}s at d={d} (d_max={self.d_max}); "
                    f"no theorem guarantees existence of a box model when the "
                    f"Helly check passes."
                )

        raise BoxModelNotFound(
            f"BoxModelBuilder inconclusive: MILP infeasible for all dimensions "
            f"d in {schedule} up to d_max={self.d_max} within "
            f"time_limit={self.time_limit}s; no theorem guarantees existence "
            f"of a box model when the Helly check passes."
        )

    def _passes_post_check(self, model: BoxModel) -> bool:
        """Return ``True`` iff ``model`` passes the same checks as :meth:`_verify_model`."""
        checker = model.check()
        if not checker.is_model(self.cm.axioms):
            return False
        if not checker.is_nondegenerate(self.cm):
            return False
        if self.faithful == "strong" and not checker.is_strongly_faithful_on_names(
            self.cm
        ):
            return False
        return True

    def _verify_model(self, model: BoxModel) -> None:
        """Run ModelChecker post-checks on ``model``; raise :class:`InternalConsistencyError` on failure."""
        checker = model.check()
        violated = checker.violations(self.cm.axioms)
        if violated or not checker.is_model(self.cm.axioms):
            raise InternalConsistencyError(
                f"Constructed BoxModel failed ModelChecker.is_model on "
                f"{len(violated)} axiom(s): {violated!r}"
            )
        if not checker.is_nondegenerate(self.cm):
            raise InternalConsistencyError(
                "Constructed BoxModel failed ModelChecker.is_nondegenerate."
            )
        if self.faithful == "strong" and not checker.is_strongly_faithful_on_names(
            self.cm
        ):
            raise InternalConsistencyError(
                "Constructed BoxModel with faithful='strong' failed "
                "ModelChecker.is_strongly_faithful_on_names."
            )

    def _solve_for_dimension(
        self, d: int, remaining_seconds: float
    ) -> Tuple[str, Optional[Dict[str, Box]]]:
        """Assemble and solve the MILP for dimension ``d``."""
        sat_names = [name for name in self.cm.names if name != TOP]
        sat_set = set(sat_names) | {TOP}
        M = self.M
        delta = self.delta

        # Variable allocation
        lb_list: List[float] = []
        ub_list: List[float] = []
        integrality_list: List[int] = []
        c_list: List[float] = []

        def add_var(
            lo: float, hi: float, is_binary: bool = False, obj: float = 0.0
        ) -> int:
            idx = len(lb_list)
            lb_list.append(lo)
            ub_list.append(hi)
            integrality_list.append(1 if is_binary else 0)
            c_list.append(obj)
            return idx

        L_idx: Dict[Tuple[str, int], int] = {}
        U_idx: Dict[Tuple[str, int], int] = {}
        w_idx: Dict[Tuple[str, int], int] = {}

        for name in sat_names:
            for i in range(d):
                L_idx[(name, i)] = add_var(0.0, M, is_binary=False, obj=-1.0)
                U_idx[(name, i)] = add_var(0.0, M, is_binary=False, obj=1.0)
                w_idx[(name, i)] = add_var(0.0, M, is_binary=False, obj=0.0)

        # Sparse constraint triplets: row_ind, col_ind, val, and row_ub (a^T x <= rhs)
        row_inds: List[int] = []
        col_inds: List[int] = []
        vals: List[float] = []
        row_ubs: List[float] = []

        def add_le(terms: List[Tuple[int, float]], rhs: float) -> None:
            r = len(row_ubs)
            for col, coeff in terms:
                if coeff != 0.0:
                    row_inds.append(r)
                    col_inds.append(col)
                    vals.append(coeff)
            row_ubs.append(rhs)

        def L_term(name: str, i: int, coeff: float) -> Tuple[List[Tuple[int, float]], float]:
            """Return (variable_terms, constant_value) for coeff * L[name, i]."""
            if name == TOP:
                return [], 0.0
            return [(L_idx[(name, i)], coeff)], 0.0

        def U_term(name: str, i: int, coeff: float) -> Tuple[List[Tuple[int, float]], float]:
            """Return (variable_terms, constant_value) for coeff * U[name, i]."""
            if name == TOP:
                return [], coeff * M
            return [(U_idx[(name, i)], coeff)], 0.0

        def add_expr_le(
            var_terms: List[Tuple[int, float]], const_lhs: float, rhs: float
        ) -> None:
            """Encode sum(var_terms) + const_lhs <= rhs."""
            if not var_terms:
                if const_lhs > rhs + 1e-12:
                    # Infeasible constant constraint: encode 0 <= -1 using a dummy fixed variable
                    v = add_var(0.0, 0.0, is_binary=False, obj=0.0)
                    add_le([(v, 1.0)], -1.0)
                return
            add_le(var_terms, rhs - const_lhs)

        # 1. Non-degeneracy witness constraints: L[A, i] <= w[A, i] <= U[A, i] and L[A, i] <= U[A, i]
        for name in sat_names:
            for i in range(d):
                li = L_idx[(name, i)]
                ui = U_idx[(name, i)]
                wi = w_idx[(name, i)]
                # L - w <= 0
                add_le([(li, 1.0), (wi, -1.0)], 0.0)
                # w - U <= 0
                add_le([(wi, 1.0), (ui, -1.0)], 0.0)
                # L - U <= 0
                add_le([(li, 1.0), (ui, -1.0)], 0.0)

        # 2. Axiom constraints (NF1 and NF2)
        for ax in self.cm.axioms:
            if isinstance(ax, NF1):
                sub, sup = ax.sub, ax.sup
                if sub not in sat_set:
                    # Unsatisfiable sub has empty box: inclusion holds vacuously
                    continue
                if sup == TOP or sub == sup:
                    continue
                if sup not in sat_set:
                    # Satisfiable sub <= unsatisfiable sup cannot occur in a consistent CanonicalModel
                    return "infeasible", None
                # L[sup, i] <= L[sub, i] and U[sub, i] <= U[sup, i]
                for i in range(d):
                    t1, c1 = L_term(sup, i, 1.0)
                    t2, c2 = L_term(sub, i, -1.0)
                    add_expr_le(t1 + t2, c1 + c2, 0.0)

                    t3, c3 = U_term(sub, i, 1.0)
                    t4, c4 = U_term(sup, i, -1.0)
                    add_expr_le(t3 + t4, c3 + c4, 0.0)

            elif isinstance(ax, NF2):
                sub1, sub2, sup = ax.sub1, ax.sub2, ax.sup
                if sub1 not in sat_set or sub2 not in sat_set:
                    # At least one conjunct has an empty box => intersection is empty
                    continue
                if sup == TOP:
                    # Trivial inclusion in universe [0, M]^d
                    continue

                sup_is_unsat = sup == BOTTOM or sup not in sat_set
                if sup_is_unsat:
                    # z fixed to 1
                    z_var = add_var(1.0, 1.0, is_binary=True, obj=0.0)
                else:
                    z_var = add_var(0.0, 1.0, is_binary=True, obj=0.0)

                e_vars: List[int] = []
                for i in range(d):
                    e_i0 = add_var(0.0, 1.0, is_binary=True, obj=0.0)
                    e_i1 = add_var(0.0, 1.0, is_binary=True, obj=0.0)
                    e_vars.extend([e_i0, e_i1])

                    # U[A, i] + delta <= L[B, i] + (M + delta) * (1 - e[i, 0])
                    # <=> U[A, i] - L[B, i] + (M + delta) * e[i, 0] <= M
                    ta, ca = U_term(sub1, i, 1.0)
                    tb, cb = L_term(sub2, i, -1.0)
                    add_expr_le(
                        ta + tb + [(e_i0, M + delta)], ca + cb, M
                    )

                    # U[B, i] + delta <= L[A, i] + (M + delta) * (1 - e[i, 1])
                    # <=> U[B, i] - L[A, i] + (M + delta) * e[i, 1] <= M
                    tb_u, cb_u = U_term(sub2, i, 1.0)
                    ta_l, ca_l = L_term(sub1, i, -1.0)
                    add_expr_le(
                        tb_u + ta_l + [(e_i1, M + delta)], cb_u + ca_l, M
                    )

                # sum_i (e[i, 0] + e[i, 1]) >= z  <=>  z - sum(e) <= 0
                add_le([(z_var, 1.0)] + [(ev, -1.0) for ev in e_vars], 0.0)

                if not sup_is_unsat:
                    for i in range(d):
                        m_i = add_var(0.0, 1.0, is_binary=True, obj=0.0)
                        n_i = add_var(0.0, 1.0, is_binary=True, obj=0.0)

                        # L[C,i] <= L[A,i] + M(1 - m[i]) + M z
                        # <=> L[C,i] - L[A,i] + M m[i] - M z <= M
                        tc_l, cc_l = L_term(sup, i, 1.0)
                        ta_l, ca_l = L_term(sub1, i, -1.0)
                        add_expr_le(
                            tc_l + ta_l + [(m_i, M), (z_var, -M)],
                            cc_l + ca_l,
                            M,
                        )

                        # L[C,i] <= L[B,i] + M m[i] + M z
                        # <=> L[C,i] - L[B,i] - M m[i] - M z <= 0
                        tb_l, cb_l = L_term(sub2, i, -1.0)
                        add_expr_le(
                            tc_l + tb_l + [(m_i, -M), (z_var, -M)],
                            cc_l + cb_l,
                            0.0,
                        )

                        # U[C,i] >= U[A,i] - M(1 - n[i]) - M z
                        # <=> U[A,i] - U[C,i] + M n[i] - M z <= M
                        ta_u, ca_u = U_term(sub1, i, 1.0)
                        tc_u, cc_u = U_term(sup, i, -1.0)
                        add_expr_le(
                            ta_u + tc_u + [(n_i, M), (z_var, -M)],
                            ca_u + cc_u,
                            M,
                        )

                        # U[C,i] >= U[B,i] - M n[i] - M z
                        # <=> U[B,i] - U[C,i] - M n[i] - M z <= 0
                        tb_u, cb_u = U_term(sub2, i, 1.0)
                        add_expr_le(
                            tb_u + tc_u + [(n_i, -M), (z_var, -M)],
                            cb_u + cc_u,
                            0.0,
                        )

        # 3. Strong faithfulness witness constraints on concept names
        if self.faithful == "strong":
            all_sat = list(self.cm.names)
            for a in all_sat:
                for b in all_sat:
                    if a == b or b == TOP:
                        continue
                    if self.cm.entails(a, b):
                        continue

                    w_ab_coords: List[int] = [
                        add_var(0.0, M, is_binary=False, obj=0.0)
                        for _ in range(d)
                    ]
                    o_vars: List[int] = []

                    for i in range(d):
                        w_abi = w_ab_coords[i]
                        # Inside box A: L[A, i] <= w[A, B, i] <= U[A, i]
                        if a != TOP:
                            add_le([(L_idx[(a, i)], 1.0), (w_abi, -1.0)], 0.0)
                            add_le([(w_abi, 1.0), (U_idx[(a, i)], -1.0)], 0.0)

                        # Outside box B on coordinate i if o[i, 0] = 1 or o[i, 1] = 1
                        o_i0 = add_var(0.0, 1.0, is_binary=True, obj=0.0)
                        o_i1 = add_var(0.0, 1.0, is_binary=True, obj=0.0)
                        o_vars.extend([o_i0, o_i1])

                        # w[A,B,i] <= L[B,i] - delta + (M + delta)(1 - o[i,0])
                        # <=> w[A,B,i] - L[B,i] + (M + delta) o[i,0] <= M
                        add_le(
                            [(w_abi, 1.0), (L_idx[(b, i)], -1.0), (o_i0, M + delta)],
                            M,
                        )

                        # w[A,B,i] >= U[B,i] + delta - (M + delta)(1 - o[i,1])
                        # <=> U[B,i] - w[A,B,i] + (M + delta) o[i,1] <= M
                        add_le(
                            [(U_idx[(b, i)], 1.0), (w_abi, -1.0), (o_i1, M + delta)],
                            M,
                        )

                    # sum_i (o[i, 0] + o[i, 1]) >= 1  <=>  -sum(o) <= -1
                    add_le([(ov, -1.0) for ov in o_vars], -1.0)

        n_vars = len(lb_list)
        if n_vars == 0:
            # Only TOP is satisfiable
            return "optimal", {
                TOP: Box(
                    np.zeros(d, dtype=float), np.full(d, M, dtype=float)
                )
            }

        c_arr = np.asarray(c_list, dtype=float)
        integrality_arr = np.asarray(integrality_list, dtype=int)
        bounds = Bounds(
            np.asarray(lb_list, dtype=float), np.asarray(ub_list, dtype=float)
        )

        n_rows = len(row_ubs)
        if n_rows > 0:
            mat = csc_matrix(
                (
                    np.asarray(vals, dtype=float),
                    (np.asarray(row_inds, dtype=int), np.asarray(col_inds, dtype=int)),
                ),
                shape=(n_rows, n_vars),
            )
            constraints = LinearConstraint(
                mat,
                np.full(n_rows, -np.inf, dtype=float),
                np.asarray(row_ubs, dtype=float),
            )
        else:
            constraints = None

        res = _scipy_milp(
            c=c_arr,
            integrality=integrality_arr,
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": float(remaining_seconds), "disp": False},
        )

        status_code = getattr(res, "status", None)
        status_msg = str(getattr(res, "message", "")).lower()
        has_x = getattr(res, "x", None) is not None
        # HiGHS status codes in scipy.optimize.milp:
        # 0 = optimal; 1 = iteration or time limit reached (res.x holds the
        # incumbent if one was found); 2 = infeasible; 3 = unbounded; 4 = other.
        is_limit = (
            status_code == 1 or "time limit" in status_msg or "timed out" in status_msg
        )
        success = bool(getattr(res, "success", False))

        if (success or is_limit) and has_x:
            sol = np.asarray(res.x, dtype=float)
            boxes_out: Dict[str, Box] = {
                TOP: Box(
                    np.zeros(d, dtype=float), np.full(d, M, dtype=float)
                )
            }
            for name in sat_names:
                lo_vec = np.array(
                    [sol[L_idx[(name, i)]] for i in range(d)], dtype=float
                )
                hi_vec = np.array(
                    [sol[U_idx[(name, i)]] for i in range(d)], dtype=float
                )
                lo_vec = np.clip(lo_vec, 0.0, M)
                hi_vec = np.clip(hi_vec, 0.0, M)
                hi_vec = np.maximum(lo_vec, hi_vec)
                boxes_out[name] = Box(lo_vec, hi_vec)
            return ("optimal" if success else "incumbent"), boxes_out

        if is_limit:
            return "timeout", None
        if status_code == 4:
            logger.info(
                "BoxModelBuilder: MILP solver returned status 4 at d=%d (%s); "
                "treating as no model found at this dimension.",
                d,
                getattr(res, "message", ""),
            )
        return "infeasible", None
