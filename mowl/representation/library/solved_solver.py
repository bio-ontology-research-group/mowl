"""Fitting by a solver: the part that replaces gradient descent.

A method whose parameters are the solution of an optimisation problem (BoxLitE: a
second-order cone programme) is the same combination of parts as a trained one (class model,
relation model, individual model, measure, regularisers), fitted differently. The fit is
split into three small things, so that everything except the call of the solver is ordinary,
testable code:

:class:`Program`
    The problem, written once against an **algebra**: named array variables, weights, the
    constraints (:meth:`Algebra.le`) and the objective it returns. It never mentions a
    solver.
:class:`NumpyAlgebra`, :class:`TorchAlgebra`, :class:`CvxpyAlgebra`
    The operations a program is written in. The numpy algebra *evaluates* the program at
    given values (:func:`evaluate`: objective and violated constraints, no solver needed);
    the torch algebra does the same for tensors, with gradients; the CVXPY algebra builds the
    symbolic problem. The formulas of a part are written once and run in all three.
:class:`Solver`
    Turns a program and weights into values for the variables: :class:`CvxpySolver` (needs
    the optional package ``cvxpy``; conic solvers), :class:`ScipySolver` (needs only
    SciPy; small problems).

The values are loaded into the tables of the parts (:meth:`SolverFit.fit`), after which the
geometry answers every question as for a trained one. The solver call is the only code here
that cannot run without its package; its tests skip with a reason.

Interface summary (for unifying with other constructions)::

    Program.variables() -> {name: shape}
    Program.weights     -> names of the non-negative weights
    Program.build(alg, v, w) -> objective          # constraints through alg.le(lhs, rhs, name)
    GradientFit(steps, lr, penalty).fit(program, values, weights)   # Adam on the torch algebra
    Solver.solve(program, weights, **options) -> SolveResult(status, value, values, ...)
    SolverFit(solver).fit(program, load, weights) -> SolveResult   # load(values) fills the parts
"""

import functools
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Program", "NumpyAlgebra", "TorchAlgebra", "CvxpyAlgebra", "Evaluation", "evaluate",
           "SolveResult", "Solver", "CvxpySolver", "ScipySolver", "SolverFit", "GradientFit",
           "SolverUnavailable", "DEFAULT_CONIC_SOLVERS", "default_solver"]

#: Open-source conic solvers tried in this order when none is named.
DEFAULT_CONIC_SOLVERS = ("CLARABEL", "ECOS", "SCS")


class SolverUnavailable(ImportError):
    """The package of a solver is not installed."""


# -- algebras ---------------------------------------------------------------------------------
class NumpyAlgebra:
    """Operations on NumPy arrays. :meth:`le` records the slack of a constraint
    (``rhs - lhs``, every entry must be non-negative)."""

    name = "numpy"

    def __init__(self):
        self.constraints = []
        #: Named intermediate objective terms a program wants to report.
        self.terms = {}

    # arrays
    def constant(self, x):
        return np.asarray(x, dtype=float)

    def vstack(self, parts):
        return np.vstack(parts)

    def columns(self, parts):
        """Matrices (or arrays) side by side along the last axis."""
        return np.concatenate(parts, axis=-1)

    def join(self, parts):
        """The row-major vectors of the parts, one after the other."""
        return np.concatenate([np.asarray(p).reshape(-1) for p in parts])

    def flatten(self, x):
        """Row-major vector."""
        return np.asarray(x).reshape(-1)

    def reshape(self, x, shape):
        """Row-major reshape."""
        return np.reshape(x, shape)

    def sum(self, x):
        return np.sum(x)

    def maximum(self, *xs):
        return functools.reduce(np.maximum, xs)

    def amax(self, x):
        """Largest entry."""
        return np.max(x)

    def row_norms(self, x):
        """Euclidean norm of every row of a matrix."""
        return np.linalg.norm(x, axis=1)

    def sdist_rows(self, y):
        """Signed distance of every row of ``y`` to the non-positive orthant
        (:math:`\\|y^+\\|_2` if a component is positive, else :math:`\\max_i y_i`)."""
        y = np.asarray(y, dtype=float)
        pos = np.maximum(y, 0.0)
        return np.where((y > 0).any(axis=-1), np.sqrt((pos ** 2).sum(axis=-1)), y.max(axis=-1))

    def sdist_vector(self, y):
        """Signed distance of the row-major vector of a matrix."""
        return self.sdist_rows(np.asarray(y, dtype=float).reshape(1, -1))[0]

    # constraints
    def le(self, lhs, rhs, name=""):
        """Require ``lhs <= rhs`` (entrywise)."""
        self.constraints.append((name, np.asarray(rhs, float) - np.asarray(lhs, float)))


class TorchAlgebra(NumpyAlgebra):
    """The same operations on tensors (differentiable). :meth:`le` records the slack
    tensors (``rhs - lhs``), so a gradient fit can penalise a violation."""

    name = "torch"

    def constant(self, x):
        import torch as th
        return th.as_tensor(x, dtype=th.get_default_dtype())

    def vstack(self, parts):
        import torch as th
        return th.cat(list(parts), dim=0)

    def columns(self, parts):
        import torch as th
        return th.cat(list(parts), dim=-1)

    def join(self, parts):
        import torch as th
        return th.cat([p.reshape(-1) for p in parts])

    def flatten(self, x):
        return x.reshape(-1)

    def reshape(self, x, shape):
        return x.reshape(shape)

    def sum(self, x):
        return x.sum()

    def maximum(self, *xs):
        import torch as th
        return functools.reduce(th.maximum, xs)

    def amax(self, x):
        return x.max()

    def row_norms(self, x):
        import torch as th
        return th.linalg.norm(x, dim=1)

    def sdist_rows(self, y):
        import torch as th
        pos = th.relu(y)
        outside = (y > 0).any(dim=-1)
        norm = th.sqrt((pos ** 2).sum(dim=-1))
        return th.where(outside, norm, y.max(dim=-1).values)

    def sdist_vector(self, y):
        return self.sdist_rows(y.reshape(1, -1))[0]

    def le(self, lhs, rhs, name=""):
        import torch as th
        self.constraints.append((name, th.as_tensor(rhs) - th.as_tensor(lhs)))


class CvxpyAlgebra:
    """Operations that build CVXPY expressions. The signed distance is the partial
    minimisation :math:`\\mathrm{sdist}(y) = \\min_{z \\ge y} \\|z\\|_2 + \\max_i (y_i - z_i)`
    (a second-order cone), whose auxiliary variables and constraints are collected in
    :attr:`constraints` with the others."""

    name = "cvxpy"

    def __init__(self):
        import cvxpy
        self.cp = cvxpy
        self.constraints = []
        self.terms = {}

    def constant(self, x):
        return self.cp.Constant(np.asarray(x, dtype=float))

    def vstack(self, parts):
        return self.cp.vstack(list(parts))

    def columns(self, parts):
        return self.cp.hstack(list(parts))

    def join(self, parts):
        return self.cp.hstack([self.cp.vec(p, order="C") if p.ndim > 1 else p for p in parts])

    def flatten(self, x):
        return self.cp.vec(x, order="C")

    def reshape(self, x, shape):
        return self.cp.reshape(x, shape, order="C")

    def sum(self, x):
        return self.cp.sum(x)

    def maximum(self, *xs):
        return self.cp.maximum(*xs)

    def amax(self, x):
        return self.cp.max(x)

    def row_norms(self, x):
        return self.cp.norm(x, 2, axis=1)

    def sdist_rows(self, y):
        z = self.cp.Variable(y.shape)
        self.constraints.append(z >= y)
        return self.cp.norm(z, 2, axis=1) + self.cp.max(y - z, axis=1)

    def sdist_vector(self, y):
        y = self.cp.vec(y, order="C")
        z = self.cp.Variable(y.shape)
        self.constraints.append(z >= y)
        return self.cp.norm(z, 2) + self.cp.max(y - z)

    def le(self, lhs, rhs, name=""):
        self.constraints.append(lhs <= rhs)


# -- programs ---------------------------------------------------------------------------------
class Program:
    """A convex program over named array variables: the problem a solver fits the parts to.

    Subclasses define :meth:`variables` and :meth:`build`. ``weights`` names the non-negative
    scalar weights of the objective (they are CVXPY parameters, so a problem is compiled once
    and solved for many weights).
    """

    weights = ()

    def variables(self):
        """``{name: shape}`` of the variables."""
        raise NotImplementedError

    def build(self, alg, v, w):
        """Register the constraints (``alg.le(lhs, rhs, name)``) and return the objective.

        :param alg: An algebra.
        :param v: ``{name: array or expression}``; absent boxes are ``None``.
        :param w: ``{name: float or parameter}`` for every name of :attr:`weights`.
        """
        raise NotImplementedError


@dataclass
class Evaluation:
    """A program evaluated at values (no solver).

    :param objective: Value of the objective.
    :param slacks: ``{constraint name: smallest slack}``; negative where violated.
    """

    objective: float
    slacks: dict = field(default_factory=dict)

    @property
    def max_violation(self):
        """The largest violation of a constraint (0 if all hold)."""
        return max([0.0] + [-s for s in self.slacks.values()])

    def feasible(self, tol=1e-6):
        return self.max_violation <= tol


def evaluate(program, values, weights=None):
    """The objective and the constraint slacks of ``program`` at ``values`` (arrays by
    variable name), with NumPy: the checks of a solution, and the tests of a program, that do
    not need a solver."""
    alg = NumpyAlgebra()
    v = {name: (None if values.get(name) is None or np.size(values[name]) == 0
                else np.asarray(values[name], float)) for name in program.variables()}
    w = {name: float((weights or {}).get(name, 0.0)) for name in program.weights}
    objective = program.build(alg, v, w)
    slacks = {}
    for name, slack in alg.constraints:
        if np.size(slack):
            slacks[name] = min(slacks.get(name, np.inf), float(np.min(slack)))
    return Evaluation(float(objective), slacks)


# -- solvers ----------------------------------------------------------------------------------
@dataclass
class SolveResult:
    """Outcome of a solver.

    :param status: ``"optimal"``, ``"optimal_inaccurate"``, ``"infeasible"``,
        ``"solver_error"``, ...
    :param value: Optimal objective value (``None`` unless solved).
    :param solver: Name of the solver that gave the result.
    :param weights: The weights the problem was solved for.
    :param values: ``{variable: array}`` (``None`` unless solved).
    :param solve_time, total_time: Seconds in the solver, and for compiling and solving.
    :param info: Whatever else the solver reports (the value of each term, failed solvers).
    """

    status: str
    value: object
    solver: str
    weights: dict
    values: object = None
    solve_time: object = None
    total_time: object = None
    info: dict = field(default_factory=dict)

    @property
    def solved(self):
        return self.status in ("optimal", "optimal_inaccurate")


class Solver:
    """Base class: ``solve(program, weights, **options) -> SolveResult``."""

    name = ""

    @classmethod
    def available(cls):
        """Whether the package of the solver is installed."""
        return True

    def solve(self, program, weights=None, **options):
        raise NotImplementedError


def default_solver():
    """The first installed conic solver of :data:`DEFAULT_CONIC_SOLVERS`."""
    cp = _require_cvxpy()
    installed = set(cp.installed_solvers())
    for name in DEFAULT_CONIC_SOLVERS:
        if name in installed:
            return name
    raise SolverUnavailable(f"None of the conic solvers {DEFAULT_CONIC_SOLVERS} is installed.")


def _require_cvxpy():
    try:
        import cvxpy
    except ImportError as exc:
        raise SolverUnavailable(
            "This solver needs the optional package 'cvxpy' (pip install cvxpy); it ships "
            "the open-source conic solvers Clarabel and SCS.") from exc
    return cvxpy


class CompiledProblem:
    """A program compiled to CVXPY once (``variables``, ``parameters``, ``constraints``,
    ``objective``, ``problem``), to be solved for many weights."""

    def __init__(self, program):
        cp = _require_cvxpy()
        self.program = program
        self.algebra = CvxpyAlgebra()
        self.variables = {name: (None if not np.prod(shape) else cp.Variable(shape, name=name))
                          for name, shape in program.variables().items()}
        self.parameters = {name: cp.Parameter(nonneg=True, value=0.0, name=name)
                           for name in program.weights}
        self.objective = program.build(self.algebra, self.variables, self.parameters)
        self.terms = self.algebra.terms
        self.constraints = self.algebra.constraints
        self.problem = cp.Problem(cp.Minimize(self.objective), self.constraints)


class CvxpySolver(Solver):
    """Conic solvers through CVXPY (the optional package ``cvxpy``).

    :param solver: A CVXPY solver name (``"CLARABEL"``, ``"ECOS"``, ``"SCS"``, ``"MOSEK"``),
        or ``None`` to try the open-source ones in order until one succeeds.
    :param options: Keyword arguments of ``cvxpy.Problem.solve``.
    """

    name = "cvxpy"

    def __init__(self, solver=None, **options):
        self.solver = solver
        self.options = options

    @classmethod
    def available(cls):
        try:
            import cvxpy  # noqa: F401
        except ImportError:
            return False
        return True

    def compile(self, program):
        return CompiledProblem(program)

    def solve(self, program, weights=None, solver=None, verbose=False, terms=None, **options):
        """Solve ``program`` (or a :class:`CompiledProblem`) for ``weights``.

        :param terms: ``{name: expression}`` of the compiled problem whose values are
            reported in ``info["terms"]``.
        """
        cp = _require_cvxpy()
        compiled = program if isinstance(program, CompiledProblem) else self.compile(program)
        weights = {n: float((weights or {}).get(n, 0.0)) for n in compiled.parameters}
        for n, p in compiled.parameters.items():
            p.value = weights[n]
        options = {**self.options, **options}
        solver = solver or self.solver
        if solver is None:
            installed = set(cp.installed_solvers())
            candidates = [s for s in DEFAULT_CONIC_SOLVERS if s in installed] \
                or [default_solver()]
        else:
            candidates = [solver]
        t0 = time.time()
        errors = {}
        for name in candidates:
            try:
                compiled.problem.solve(solver=name, verbose=verbose, **options)
                break
            except cp.error.SolverError as exc:
                errors[name] = str(exc)
        else:
            return SolveResult("solver_error", None, candidates[-1], weights,
                               total_time=time.time() - t0,
                               info={"error": "; ".join(f"{k}: {v}"
                                                        for k, v in errors.items())})
        stats = compiled.problem.solver_stats
        result = SolveResult(
            compiled.problem.status, compiled.problem.value, name, weights,
            solve_time=getattr(stats, "solve_time", None), total_time=time.time() - t0,
            info={"failed_solvers": errors,
                  "terms": {k: _value(t) for k, t in (terms or {}).items()}})
        if result.solved:
            d = {n: (None if x is None else np.array(x.value, dtype=float))
                 for n, x in compiled.variables.items()}
            result.values = d
        return result


def _value(expr):
    try:
        return None if expr.value is None else float(expr.value)
    except (TypeError, ValueError):
        return None


class ScipySolver(Solver):
    """A small-problem solver that needs only SciPy: sequential least squares programming
    (``SLSQP``) on the NumPy evaluation of the program, the constraints as inequalities. It is
    for tiny programs (tests, examples, a start for a better solver), not for the sizes of
    the benchmarks; the optimum of a non-smooth objective is approximate.

    :param options: ``scipy.optimize.minimize`` options (``maxiter``, ``ftol``).
    :param seed: Seed of the starting point.
    """

    name = "scipy"

    def __init__(self, seed=0, **options):
        self.seed = seed
        self.options = {"maxiter": 300, "ftol": 1e-9, **options}

    @classmethod
    def available(cls):
        try:
            import scipy  # noqa: F401
        except ImportError:
            return False
        return True

    def solve(self, program, weights=None, start=None, **options):
        from scipy.optimize import minimize
        shapes = {n: s for n, s in program.variables().items() if np.prod(s)}
        sizes = {n: int(np.prod(s)) for n, s in shapes.items()}
        offsets, total = {}, 0
        for n in shapes:
            offsets[n] = total
            total += sizes[n]
        w = {n: float((weights or {}).get(n, 0.0)) for n in program.weights}
        t0 = time.time()

        def unpack(x):
            return {n: (np.zeros(program.variables()[n]) if n not in shapes else
                        x[offsets[n]:offsets[n] + sizes[n]].reshape(shapes[n]))
                    for n in program.variables()}

        def run(x):
            alg = NumpyAlgebra()
            objective = program.build(alg, unpack(x), w)
            slack = np.concatenate([np.asarray(s).reshape(-1) for _, s in alg.constraints]) \
                if alg.constraints else np.zeros(0)
            return float(objective), slack

        rng = np.random.default_rng(self.seed)
        x0 = rng.normal(scale=0.1, size=total) if start is None else np.concatenate(
            [np.asarray(start[n], float).reshape(-1) for n in shapes])
        cache = {}

        def cached(x):
            key = x.tobytes()
            if key not in cache:
                cache.clear()
                cache[key] = run(x)
            return cache[key]

        result = minimize(lambda x: cached(x)[0], x0, method="SLSQP",
                          constraints=[{"type": "ineq", "fun": lambda x: cached(x)[1]}],
                          options={**self.options, **options})
        objective, slack = run(result.x)
        feasible = slack.size == 0 or slack.min() >= -1e-5
        status = "optimal" if feasible else "infeasible"
        return SolveResult(status, objective if feasible else None, "SLSQP", w,
                           values=unpack(result.x) if feasible else None,
                           solve_time=time.time() - t0, total_time=time.time() - t0,
                           info={"message": str(result.message), "iterations": int(result.nit)})


class SolverFit:
    """The fitting regime of a combination of parts that is solved, not trained: solve the
    program for the weights, then load the solution into the parts.

    :param solver: A :class:`Solver` (default :class:`CvxpySolver`).
    """

    key = "solver-fit"

    def __init__(self, solver=None):
        self.solver = solver if solver is not None else CvxpySolver()

    def fit(self, program, load, weights=None, **options):
        """Solve and, if solved, call ``load(values)``; return the :class:`SolveResult`."""
        result = self.solver.solve(program, weights, **options)
        if result.solved:
            load(result.values)
        return result


class GradientFit:
    """The other way to fit the same parts: gradient descent (Adam) on the objective of the
    program, with the constraints as a quadratic penalty. The result is approximate (the
    constraints hold only up to the penalty), but needs no solver; it is what a solved
    method becomes when it is trained like the others.

    :param steps: Number of steps.
    :param lr: Learning rate.
    :param penalty: Weight of the squared violations.

    It is an inner solver, not a training loop: it minimises the program it is given (the
    same contract as the convex solver it replaces, ``fit(program, tables)`` returning a
    :class:`SolveResult`), with a fixed number of steps of its own, and the model that holds
    the program calls it once per solve. A training regime has nothing to hook into here
    (no epochs, data batches, negatives or validation), so it does not run in the shared
    training loop (:mod:`mowl.training.loop`).
    """

    key = "gradient-fit"

    def __init__(self, steps=200, lr=0.01, penalty=10.0):
        self.steps, self.lr, self.penalty = steps, lr, penalty

    def fit(self, program, tables, weights=None):
        """Train the tensors ``tables`` (``{variable: Parameter or None}``) in place and
        return a :class:`SolveResult` (``status="gradient"``, ``value`` the final objective
        and ``info["violation"]`` the largest violated constraint)."""
        import torch as th
        t0 = time.time()
        w = {n: float((weights or {}).get(n, 0.0)) for n in program.weights}
        params = [t for t in tables.values() if t is not None]
        optimiser = th.optim.Adam(params, lr=self.lr)
        objective, violation = None, 0.0
        for _ in range(self.steps):
            alg = TorchAlgebra()
            objective = program.build(alg, tables, w)
            slack = [th.relu(-s).reshape(-1) for _, s in alg.constraints]
            squares = sum((s ** 2).sum() for s in slack) if slack else 0.0
            loss = objective + self.penalty * squares
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            violation = max([0.0] + [float(s.max()) for s in slack if s.numel()])
        return SolveResult("gradient", float(objective), "adam", w,
                           solve_time=time.time() - t0, total_time=time.time() - t0,
                           info={"violation": violation, "steps": self.steps})
