"""The convex programme of BoxLitE (Lourenço et al., KR 2026, Section 5 and Appendix E),
written with CVXPY.

Variables (Section 5.1): a position and a bump vector per individual, a box (``L``, ``U``)
per concept name and three boxes (head, tail, bump) per role name. Constraints, all
linear:

* box widths :math:`0 \\le U - L \\le 2 s_\\Omega` (Eq. 2),
* individuals in the universe box (Eq. 3),
* box consistency, one reserved dimension :math:`i_C` per basic concept
  :math:`C \\in N^\\exists_C` (Eq. 1),
* every TBox axiom as inclusions of bounds (Section 5.1).

Objective (Section 5.2): the maximum of the concept and role assertion losses, plus
:math:`\\lambda_1` times the maximum negative-concept loss over the potential concept
assertions :math:`D(a) \\notin \\mathcal{A}`, plus :math:`\\lambda_2, \\lambda_3` times the
box-width regularisers. The losses are signed distances to :math:`\\mathbb{R}^{2d}_-`,
modelled with the partial-minimisation formula of Proposition 4,
:math:`\\mathrm{sdist}(y) = \\min_{z \\ge y} \\|z\\|_2 + \\max_i (y_i - z_i)`, so the problem
is a second-order cone programme (Theorem 5) that any conic solver of CVXPY can solve.

Two variants are available:

``"paper"``
    The programme as printed: boxes are :math:`\\{L + \\epsilon \\le x \\le U - \\epsilon\\}`
    (Definition 2), individuals satisfy :math:`-s_\\Omega + \\epsilon \\le x \\le s_\\Omega -
    \\epsilon`, box consistency is :math:`L_C[i_C] + U_C[i_C] \\le -s_\\Omega` (Eq. 1), the
    objective takes the maximum of per-assertion losses, the width regulariser is
    :math:`\\sum \\|U - L\\|_2` per box (Eq. 7), and the negatives are all pairs
    :math:`(D, a)`, :math:`D \\in N^\\exists_C`, not witnessed by the ABox
    (:func:`negative_concept_assertions`).

``"reference"``
    What the authors' code (AleksVap/BoxLitE, commit 7d53063, ``main.py``) solves, which
    produced Table 2: closed boxes :math:`[L, U]` (no :math:`\\epsilon` in memberships),
    individuals in :math:`[-s_\\Omega, s_\\Omega]`, box consistency
    :math:`L_C[i_C] + U_C[i_C] \\le -s_\\Omega - 2\\epsilon`; one signed distance of the
    stacked residuals of *all* assertions (lower and upper bound residuals separately,
    combined with a maximum) instead of a maximum of per-assertion signed distances; the
    assertion part is :math:`\\max(\\text{concept term}, \\text{role term})` with an absent
    term counted as 0; the width regulariser is CVXPY's ``mixed_norm`` of the width vector
    reshaped (row-major) to ``d`` rows; negatives as in the reference code.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from mowl.models.boxlite.embedding import BoxLitEEmbedding

#: Variants of the programme (see the module docstring).
VARIANTS = ("paper", "reference")
#: Open-source conic solvers tried in this order when no solver is given.
DEFAULT_SOLVERS = ("CLARABEL", "ECOS", "SCS")


def require_cvxpy():
    """Import and return :mod:`cvxpy`, or raise an informative :class:`ImportError`."""
    try:
        import cvxpy
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "BoxLitE needs the optional package 'cvxpy' (pip install cvxpy); it ships the "
            "open-source conic solvers Clarabel and SCS.") from exc
    return cvxpy


def default_solver():
    """The first installed solver of :data:`DEFAULT_SOLVERS`."""
    cp = require_cvxpy()
    installed = set(cp.installed_solvers())
    for name in DEFAULT_SOLVERS:
        if name in installed:
            return name
    raise ImportError(f"None of the conic solvers {DEFAULT_SOLVERS} is installed.")


def negative_concept_assertions(kb, mode="paper"):
    """The potential concept assertions pushed into complement boxes by the negative
    concept regularisation (Eq. 6), as rows ``[basic concept, individual]``.

    ``"paper"``: all :math:`D(a)` with :math:`D \\in N^\\exists_C` (concept names,
    :math:`\\exists R` and :math:`\\exists R^-`) that are not in the ABox, where a role
    assertion :math:`R(a, b)` counts as asserting :math:`\\exists R(a)` and
    :math:`\\exists R^-(b)`. Consequences of the TBox are not excluded (the paper writes
    :math:`D(a) \\notin \\mathcal{A}`, not :math:`\\mathcal{K} \\not\\models D(a)`).

    ``"reference"``: as in the authors' code: for a concept name, every individual without
    an assertion of it; for a role name occurring in the ABox, :math:`\\exists R(a)` for
    every individual that is not the subject of an :math:`R` assertion (no
    :math:`\\exists R^-` terms); for a role name not in the ABox, :math:`\\exists R(a)` and
    :math:`\\exists R^-(a)` for every individual.
    """
    n_ind, nb, nc, nr = len(kb.individuals), kb.n_basic, len(kb.concepts), len(kb.roles)
    asserted = np.zeros((nb, n_ind), dtype=bool)
    if len(kb.concept_assertions):
        asserted[kb.concept_assertions[:, 0], kb.concept_assertions[:, 1]] = True
    ra = kb.role_assertions
    if mode == "paper":
        if len(ra):
            asserted[nc + ra[:, 0], ra[:, 1]] = True
            asserted[nc + nr + ra[:, 0], ra[:, 2]] = True
        rows = np.argwhere(~asserted)
    elif mode == "reference":
        if len(ra):
            asserted[nc + ra[:, 0], ra[:, 1]] = True
        in_abox = set(ra[:, 0].tolist()) if len(ra) else set()
        keep = np.ones(nb, dtype=bool)
        for r in in_abox:
            keep[nc + nr + r] = False
        rows = np.argwhere(~asserted & keep[:, None])
    else:
        raise ValueError(f"Unknown negatives mode {mode!r}.")
    return rows.astype(np.int64).reshape(-1, 2)


@dataclass
class ProgramResult:
    """Outcome of :meth:`BoxLitEProgram.solve`.

    :param status: CVXPY status (``"optimal"``, ``"infeasible"``, ...).
    :param value: Optimal objective value (``None`` unless solved).
    :param solver: Solver name.
    :param lambdas: :math:`(\\lambda_1, \\lambda_2, \\lambda_3)`.
    :param solve_time: Seconds in the solver.
    :param total_time: Seconds for compilation and solving.
    :param embedding: The solved :class:`~mowl.models.boxlite.embedding.BoxLitEEmbedding`.
    """

    status: str
    value: object
    solver: str
    lambdas: tuple
    solve_time: object = None
    total_time: object = None
    embedding: object = None
    info: dict = field(default_factory=dict)

    @property
    def solved(self):
        return self.status in ("optimal", "optimal_inaccurate")


class BoxLitEProgram:
    """The BoxLitE second-order cone programme for one :math:`\\text{DL-Lite}_\\mathcal{H}`
    KB.

    The regularisation weights are CVXPY parameters, so the problem is compiled once and
    re-solved for every :math:`(\\lambda_1, \\lambda_2, \\lambda_3)` (as in Section 6, Q2).

    :param kb: The knowledge base (:class:`mowl.datasets.dllite.DLLiteKB`).
    :param dim: Dimension :math:`d`; at least :math:`|N_C| + 2|N_R|` (one reserved
        dimension per basic concept for box consistency, Eq. 1).
    :param s_omega: :math:`s_\\Omega` (Section 3). The experiments use 1 (Appendix G.3).
    :param epsilon: :math:`\\epsilon`; the experiments use 0.01. The proofs need
        :math:`0 < \\epsilon \\le s_\\Omega / 8` (footnote 3).
    :param variant: ``"paper"`` or ``"reference"`` (module docstring).
    :param negatives: Rows ``[basic, individual]`` of the negative concept assertions, or
        ``None`` for :func:`negative_concept_assertions` with the variant's mode.
    """

    def __init__(self, kb, dim, s_omega=1.0, epsilon=0.01, variant="paper", negatives=None):
        cp = require_cvxpy()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}.")
        if not (s_omega > 0 and epsilon > 0 and epsilon <= s_omega):
            raise ValueError("Need 0 < epsilon <= s_omega (Section 3).")
        nb = kb.n_basic
        if dim < max(nb, 1):
            raise ValueError(
                f"dim={dim} is too small: box consistency reserves one dimension per basic "
                f"concept (Eq. 1), so dim >= |N_C| + 2|N_R| = {nb}.")
        self.kb, self.dim, self.variant = kb, int(dim), variant
        self.s_omega, self.epsilon = float(s_omega), float(epsilon)
        self.margin = self.epsilon if variant == "paper" else 0.0
        if negatives is None:
            negatives = negative_concept_assertions(kb, variant)
        self.negatives = np.asarray(negatives, dtype=np.int64).reshape(-1, 2)

        d, s, m = self.dim, self.s_omega, self.margin
        ni, nc, nr = len(kb.individuals), len(kb.concepts), len(kb.roles)
        self.lambda1 = cp.Parameter(nonneg=True, value=0.0, name="lambda_1")
        self.lambda2 = cp.Parameter(nonneg=True, value=0.0, name="lambda_2")
        self.lambda3 = cp.Parameter(nonneg=True, value=0.0, name="lambda_3")

        def var(n, name):
            return cp.Variable((n, d), name=name) if n else None

        v = self.variables = {
            "pos": var(ni, "pos"), "bump": var(ni, "bump"),
            "concept_lower": var(nc, "concept_lower"), "concept_upper": var(nc, "concept_upper"),
            "head_lower": var(nr, "head_lower"), "head_upper": var(nr, "head_upper"),
            "tail_lower": var(nr, "tail_lower"), "tail_upper": var(nr, "tail_upper"),
            "bump_lower": var(nr, "bump_lower"), "bump_upper": var(nr, "bump_upper")}
        cons = []

        # Widths (Eq. 2) and universe (Eq. 3).
        for lo, up in (("concept_lower", "concept_upper"), ("head_lower", "head_upper"),
                       ("tail_lower", "tail_upper"), ("bump_lower", "bump_upper")):
            if v[lo] is not None:
                cons += [v[up] - v[lo] >= 0, v[up] - v[lo] <= 2 * s]
        if ni:
            bound = s - m
            cons += [v["pos"] <= bound, v["pos"] >= -bound,
                     v["bump"] <= bound, v["bump"] >= -bound]

        # Bounds of all basic concepts: names, ∃R, ∃R⁻ (Section 5.1).
        lower_parts, upper_parts = [], []
        if nc:
            lower_parts.append(v["concept_lower"])
            upper_parts.append(v["concept_upper"])
        if nr:
            lower_parts += [v["head_lower"] - v["bump_upper"], v["tail_lower"] - v["bump_upper"]]
            upper_parts += [v["head_upper"] - v["bump_lower"], v["tail_upper"] - v["bump_lower"]]
        BL = cp.vstack(lower_parts) if lower_parts else None
        BU = cp.vstack(upper_parts) if upper_parts else None
        self._basic = (BL, BU)

        # Box consistency (Eq. 1): dimension i_C = index of C.
        if nb:
            idx = np.arange(nb)
            rhs = -s if variant == "paper" else -s - 2 * self.epsilon
            cons.append(BL[idx, idx] + BU[idx, idx] <= rhs)

        # TBox (Section 5.1).
        ci = kb.concept_inclusions
        if len(ci):
            sub, sup = ci[:, 0], ci[:, 1]
            cons += [BL[sup] <= BL[sub], BU[sub] <= BU[sup]]
        neg = kb.negative_inclusions
        if len(neg):
            sub, other = neg[:, 0], neg[:, 1]
            cons += [-s - BL[other] <= BL[sub], BU[sub] <= s - BU[other]]
        ri = kb.role_inclusions
        if len(ri):
            # Rows 0..nr-1: R, rows nr..2nr-1: R⁻ (head and tail swapped).
            head_l = cp.vstack([v["head_lower"], v["tail_lower"]])
            head_u = cp.vstack([v["head_upper"], v["tail_upper"]])
            tail_l = cp.vstack([v["tail_lower"], v["head_lower"]])
            tail_u = cp.vstack([v["tail_upper"], v["head_upper"]])
            sub = ri[:, 0] + nr * ri[:, 1]
            sup = ri[:, 2] + nr * ri[:, 3]
            cons += [head_l[sup] <= head_l[sub], head_u[sub] <= head_u[sup],
                     tail_l[sup] <= tail_l[sub], tail_u[sub] <= tail_u[sup],
                     v["bump_lower"][ri[:, 2]] <= v["bump_lower"][ri[:, 0]],
                     v["bump_upper"][ri[:, 0]] <= v["bump_upper"][ri[:, 2]]]

        # Losses (Section 5.2).
        def residuals(lower, upper, x):
            return lower + m - x, x - upper + m

        def sdist_rows(blocks):
            """Signed distance of every row of [lower | upper] (Proposition 4)."""
            y = cp.hstack([cp.vstack([lo for lo, _ in blocks]),
                           cp.vstack([up for _, up in blocks])])
            z = cp.Variable(y.shape)
            cons.append(z >= y)
            return cp.norm(z, 2, axis=1) + cp.max(y - z, axis=1)

        def sdist_vector(y):
            """Signed distance of one stacked residual vector (reference code)."""
            y = cp.vec(y, order="C")
            z = cp.Variable(y.shape)
            cons.append(z >= y)
            return cp.norm(z, 2) + cp.max(y - z)

        def loss(blocks):
            if variant == "paper":
                return cp.max(sdist_rows(blocks))
            return cp.maximum(*[sdist_vector(part) for block in blocks for part in block])

        terms = {}
        ca = kb.concept_assertions
        if len(ca):
            terms["concept"] = loss([residuals(BL[ca[:, 0]], BU[ca[:, 0]], v["pos"][ca[:, 1]])])
        ra = kb.role_assertions
        if len(ra):
            r, a, b = ra[:, 0], ra[:, 1], ra[:, 2]
            P, B = v["pos"], v["bump"]
            terms["role"] = loss([
                residuals(v["head_lower"][r], v["head_upper"][r], P[a] + B[b]),
                residuals(v["tail_lower"][r], v["tail_upper"][r], P[b] + B[a]),
                residuals(v["bump_lower"][r], v["bump_upper"][r], B[a]),
                residuals(v["bump_lower"][r], v["bump_upper"][r], B[b])])
        if len(self.negatives) and ni:
            nbq, nia = self.negatives[:, 0], self.negatives[:, 1]
            terms["negative"] = loss([residuals(-s - BL[nbq], s - BU[nbq], v["pos"][nia])])

        if variant == "paper":
            groups = [terms[k] for k in ("concept", "role") if k in terms]
            assertion = (cp.maximum(*groups) if len(groups) > 1 else groups[0]) if groups \
                else cp.Constant(0.0)
        else:
            assertion = cp.maximum(terms.get("concept", cp.Constant(0.0)),
                                   terms.get("role", cp.Constant(0.0)))
        objective = assertion
        if "negative" in terms:
            objective = objective + self.lambda1 * terms["negative"]

        width_ht, width_b = self._width_regularisers(cp)
        if width_ht is not None:
            objective = objective + self.lambda2 * width_ht
        if width_b is not None:
            objective = objective + self.lambda3 * width_b

        self.terms = terms
        self.assertion_term = assertion
        self.width_terms = (width_ht, width_b)
        self.constraints = cons
        self.problem = cp.Problem(cp.Minimize(objective), cons)

    def _width_regularisers(self, cp):
        v, d = self.variables, self.dim
        widths = []
        for lo, up in (("concept_lower", "concept_upper"), ("head_lower", "head_upper"),
                       ("tail_lower", "tail_upper")):
            if v[lo] is not None:
                widths.append(v[up] - v[lo])
        bump = None if v["bump_lower"] is None else v["bump_upper"] - v["bump_lower"]
        if self.variant == "paper":
            ht = sum(cp.sum(cp.norm(w, 2, axis=1)) for w in widths) if widths else None
            b = cp.sum(cp.norm(bump, 2, axis=1)) if bump is not None else None
            return ht, b

        # Reference code: mixed_norm (sum of row 2-norms) of the box-major width vector
        # reshaped row-major to d rows; a zero block of length d replaces absent concepts.
        def mixed(blocks):
            flat = cp.hstack([cp.vec(w, order="C") for w in blocks])
            n = flat.shape[0]
            return cp.sum(cp.norm(cp.reshape(flat, (d, n // d), order="C"), 2, axis=1))

        if not widths:
            return None, (mixed([bump]) if bump is not None else None)
        if v["concept_lower"] is None:
            widths = [cp.Constant(np.zeros((1, d)))] + widths
        return mixed(widths), (mixed([bump]) if bump is not None else None)

    # ----------------------------------------------------------------------------------

    def size(self):
        """Numbers of scalar variables and constraints of the CVXPY problem."""
        sm = self.problem.size_metrics
        return {"variables": int(sm.num_scalar_variables),
                "equality_constraints": int(sm.num_scalar_eq_constr),
                "inequality_constraints": int(sm.num_scalar_leq_constr)}

    def solve(self, lambdas=(0.0, 0.0, 0.0), solver=None, verbose=False, **solver_options):
        """Solve for the given :math:`(\\lambda_1, \\lambda_2, \\lambda_3)`.

        :param solver: A CVXPY solver name (e.g. ``"CLARABEL"``, ``"SCS"``, ``"MOSEK"``);
            default: :func:`default_solver`.
        :rtype: :class:`ProgramResult`
        """
        cp = require_cvxpy()
        l1, l2, l3 = (float(x) for x in lambdas)
        self.lambda1.value, self.lambda2.value, self.lambda3.value = l1, l2, l3
        if solver is None:
            # Default: try the open-source solvers in order (Clarabel stalls with
            # "InsufficientProgress" on the larger Family programmes, ECOS then succeeds).
            installed = set(cp.installed_solvers())
            candidates = [s for s in DEFAULT_SOLVERS if s in installed] or [default_solver()]
        else:
            candidates = [solver]
        t0 = time.time()
        errors = {}
        for solver in candidates:
            try:
                self.problem.solve(solver=solver, verbose=verbose, **solver_options)
                break
            except cp.error.SolverError as exc:
                errors[solver] = str(exc)
        else:
            return ProgramResult(status="solver_error", value=None, solver=solver,
                                 lambdas=(l1, l2, l3), total_time=time.time() - t0,
                                 info={"error": "; ".join(f"{k}: {v}"
                                                          for k, v in errors.items())})
        total = time.time() - t0
        stats = self.problem.solver_stats
        result = ProgramResult(
            status=self.problem.status, value=self.problem.value, solver=solver,
            lambdas=(l1, l2, l3), solve_time=getattr(stats, "solve_time", None),
            total_time=total,
            info={"terms": {k: _value(t) for k, t in self.terms.items()},
                  "failed_solvers": errors})
        if result.solved:
            result.embedding = self.embedding()
        return result

    def embedding(self):
        """The current variable values as a :class:`BoxLitEEmbedding`."""
        d = self.dim

        def val(name):
            x = self.variables[name]
            return np.zeros((0, d)) if x is None else np.array(x.value, dtype=float)

        return BoxLitEEmbedding(
            **{k: val(k) for k in BoxLitEEmbedding._ARRAYS}, s_omega=self.s_omega,
            margin=self.margin)


def _value(expr):
    try:
        return None if expr.value is None else float(expr.value)
    except (TypeError, ValueError):
        return None
