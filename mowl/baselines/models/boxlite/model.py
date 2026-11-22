"""BoxLitE: box embeddings of :math:`\\text{DL-Lite}_\\mathcal{H}` knowledge bases obtained
by convex optimisation (Lourenço, Morgan, Ozaki, Pavlović, Sallinger, KR 2026)."""

from dataclasses import dataclass

import numpy as np

from mowl.base_models.model import Model
from mowl.datasets.dllite import DLLiteKB, extract_dllite, _translate, _Skip, _Trivial
from mowl.representation.library.solved_boxlite_check import check_model
from mowl.baselines.models.boxlite.embedding import BoxLitEEmbedding
from mowl.baselines.models.boxlite.program import BoxLitEProgram, VARIANTS


def kb_faithful_dimension(kb):
    """:math:`d_{min} = |N_C| + |N_R|(2 + |N_I| + 2|N_R|)` of Corollary 2: in this
    dimension every satisfiable KB has a weakly KB-faithful BoxLitE model."""
    nc, nr, ni = len(kb.concepts), len(kb.roles), len(kb.individuals)
    return max(nc + nr * (2 + ni + 2 * nr), kb.n_basic, 1)


def tbox_faithful_dimension(kb):
    """:math:`d_{min} = |N_C| + 3|N_R|` of Corollary 1 (empty ABox)."""
    return max(len(kb.concepts) + 3 * len(kb.roles), kb.n_basic, 1)


@dataclass
class SatisfiabilityResult:
    """Outcome of :meth:`BoxLitE.decide_satisfiability`.

    :param satisfiable: ``True`` or ``False``; ``None`` if the solver failed.
    :param optimum: Optimal value of programme (8) with :math:`\\lambda = 0`.
    :param dim: Dimension used (Corollary 2).
    :param result: The :class:`~mowl.baselines.models.boxlite.program.ProgramResult`.
    :param check: :class:`~mowl.baselines.models.boxlite.checker.ModelCheck` of the solution (a
        certificate when satisfiable).
    """

    satisfiable: object
    optimum: object
    dim: int
    result: object
    check: object = None


class BoxLitE(Model):
    """BoxLitE (Lourenço et al., KR 2026).

    Every individual has a position and a bump vector, every concept name a box and every
    role name a head, a tail and a bump box (Definition 3). :math:`R(a, b)` holds iff
    :math:`pos(a) + bump(b) \\in Head(R)`, :math:`pos(b) + bump(a) \\in Tail(R)` and
    :math:`bump(a), bump(b) \\in Bump(R)`; :math:`\\exists R` is the head box enlarged by
    the bump box; :math:`\\neg C` is the convex complement box
    :math:`[-s_\\Omega - L_C, s_\\Omega - U_C]` (Definitions 3-4). The embedding is the
    solution of a second-order cone programme (Theorems 4 and 5) whose linear constraints
    encode the TBox, box consistency and the box widths, and whose objective contains the
    ABox losses, a negative-concept regulariser (weight :math:`\\lambda_1`) and box-width
    regularisers (:math:`\\lambda_2, \\lambda_3`), see
    :class:`~mowl.baselines.models.boxlite.program.BoxLitEProgram`. There is no gradient descent:
    :meth:`train` calls a conic solver through CVXPY (optional dependency ``cvxpy``).

    The paper's experiments use MOSEK (``solver="MOSEK"``, needs a licence); mOWL defaults
    to the open-source solvers shipped with CVXPY (Clarabel, then ECOS, then SCS). The
    feasible set and the optimal value do not depend on the solver; if the optimum is not
    unique, different solvers may return different optimal embeddings, and the solutions
    agree only up to the solvers' tolerances.

    Deviations and choices (see also the catalogue card):

    * ``variant="paper"`` (default) implements Sections 3 and 5 as printed;
      ``variant="reference"`` reproduces the objective and constraints of the authors'
      code (``AleksVap/BoxLitE``, commit 7d53063), which differ in where
      :math:`\\epsilon` enters, in how the signed distances of several assertions are
      aggregated, in the width regulariser and in the negatives (see
      :mod:`mowl.baselines.models.boxlite.program`).
    * The paper assumes TBoxes in "named form" (one side a concept name) only to simplify
      the presentation; the constraints of Section 5.1 do not need it, and inclusions such
      as :math:`\\exists R \\sqsubseteq \\exists S` are accepted.
    * Box consistency reserves dimension :math:`i_C = ` index of :math:`C` (concept names,
      then :math:`\\exists R`, then :math:`\\exists R^-`), as the reference code does.

    :param dataset: A mOWL dataset; its training ontology is read with
        :func:`mowl.datasets.dllite.extract_dllite` (axioms outside
        :math:`\\text{DL-Lite}_\\mathcal{H}` are reported in ``self.kb.skipped``).
    :param embed_dim: Dimension :math:`d`; the experiments use 32 (Appendix G.3). Must be
        at least :math:`|N_C| + 2|N_R|`.
    :param s_omega: :math:`s_\\Omega` (1 in the experiments).
    :param epsilon: :math:`\\epsilon` (0.01 in the experiments).
    :param lambda1: Weight of the negative concept regularisation.
    :param lambda2: Weight of the head/tail/concept box width regularisation.
    :param lambda3: Weight of the bump box width regularisation.
    :param variant: ``"paper"`` or ``"reference"``.
    :param negatives: ``None`` (the variant's default) or rows ``[basic concept,
        individual]``.
    :param solver: CVXPY solver name (default: try Clarabel, ECOS, SCS in this order
        until one succeeds).
    :param solver_options: Keyword arguments for ``cvxpy.Problem.solve``.
    :param model_filepath: Where :meth:`save` writes the embedding (``.npz``).
    :param kb: Use this :class:`~mowl.datasets.dllite.DLLiteKB` instead of extracting
        one from ``dataset``.
    """

    axiom_scoring = "method"
    scored_forms = ("class_assertion",)

    def __init__(self, dataset, embed_dim=32, s_omega=1.0, epsilon=0.01, lambda1=0.0,
                 lambda2=0.0, lambda3=0.0, variant="paper", negatives=None, solver=None,
                 solver_options=None, model_filepath=None, kb=None):
        super().__init__(dataset, model_filepath=model_filepath)
        if kb is None:
            kb = extract_dllite(dataset.ontology)
        self._setup(kb, embed_dim, s_omega, epsilon, lambda1, lambda2, lambda3, variant,
                    negatives, solver, solver_options)

    @classmethod
    def from_kb(cls, kb, model_filepath=None, **kwargs):
        """Create a model for a :class:`~mowl.datasets.dllite.DLLiteKB` without a mOWL
        dataset (``self.dataset`` is ``None``)."""
        if not isinstance(kb, DLLiteKB):
            raise TypeError("kb must be a mowl.datasets.dllite.DLLiteKB.")
        obj = cls.__new__(cls)
        obj.dataset = None
        obj._model_filepath = model_filepath
        obj._testing_set = obj._evaluator = obj._evaluation_model = obj._metrics = None
        defaults = dict(embed_dim=32, s_omega=1.0, epsilon=0.01, lambda1=0.0, lambda2=0.0,
                        lambda3=0.0, variant="paper", negatives=None, solver=None,
                        solver_options=None)
        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise TypeError(f"Unknown arguments: {sorted(unknown)}")
        defaults.update(kwargs)
        obj._setup(kb, **defaults)
        return obj

    def _setup(self, kb, embed_dim, s_omega, epsilon, lambda1, lambda2, lambda3, variant,
               negatives, solver, solver_options):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}.")
        self.kb = kb
        self.embed_dim = int(embed_dim)
        self.s_omega, self.epsilon = float(s_omega), float(epsilon)
        self.lambdas = (float(lambda1), float(lambda2), float(lambda3))
        self.variant = variant
        self.negatives = negatives
        self.solver = solver
        self.solver_options = dict(solver_options or {})
        self._program = None
        self.embedding = None
        self.result = None

    # ----------------------------------------------------------------------------------
    # Solving
    # ----------------------------------------------------------------------------------

    @property
    def program(self):
        """The compiled :class:`~mowl.baselines.models.boxlite.program.BoxLitEProgram` (built on
        first use)."""
        if self._program is None:
            self._program = BoxLitEProgram(self.kb, self.embed_dim, self.s_omega,
                                           self.epsilon, self.variant, self.negatives)
        return self._program

    def train(self, lambdas=None, solver=None, verbose=False, **solver_options):
        """Solve the convex programme.

        :param lambdas: :math:`(\\lambda_1, \\lambda_2, \\lambda_3)`; default: the
            constructor's values. Re-solving with other values reuses the programme.
        :param solver: Overrides the constructor's solver.
        :return: :class:`~mowl.baselines.models.boxlite.program.ProgramResult`. On success the
            embedding is stored in ``self.embedding``.
        :raises RuntimeError: If the solver does not return an optimal solution.
        """
        lambdas = self.lambdas if lambdas is None else tuple(float(x) for x in lambdas)
        options = {**self.solver_options, **solver_options}
        result = self.program.solve(lambdas, solver=solver or self.solver, verbose=verbose,
                                    **options)
        self.result = result
        if not result.solved:
            raise RuntimeError(f"BoxLitE: the solver returned status {result.status!r} "
                               f"({result.info.get('error', '')}).")
        self.lambdas = lambdas
        self.embedding = result.embedding
        return result

    def check_model(self, kb=None, tol=1e-6):
        """Check, independently of the loss, that the embedding is a box-consistent model
        of ``kb`` (default: the training KB). See
        :func:`mowl.baselines.models.boxlite.checker.check_model`."""
        self._require_embedding()
        return check_model(self.embedding, self.kb if kb is None else kb, tol=tol)

    @staticmethod
    def decide_satisfiability(kb, s_omega=1.0, epsilon=0.01, solver=None, tol=1e-6,
                              dim=None, **solver_options):
        """Decide satisfiability of a :math:`\\text{DL-Lite}_\\mathcal{H}` KB with the
        convex programme (Theorem 4 iii and iv).

        With :math:`\\lambda = 0` and :math:`d \\ge d_{min}` of Corollary 2, a satisfiable
        KB has an optimal solution with value :math:`\\le 0` (Theorem 4 iv); conversely a
        solution with value :math:`\\le 0` is a box-consistent model, so the KB is
        satisfiable (Theorem 4 iii with Proposition 2). Hence the KB is satisfiable iff the
        optimum is :math:`\\le 0`; numerically, iff it is :math:`\\le` ``tol``. The paper's
        existence proof uses :math:`\\epsilon \\le s_\\Omega / 8` (footnote 3 and
        Theorem 3); ``tol`` must be well below :math:`\\epsilon`.

        :param dim: Override the dimension (a smaller one keeps "satisfiable" sound but
            "unsatisfiable" may then be wrong).
        :rtype: :class:`SatisfiabilityResult`
        """
        dim = kb_faithful_dimension(kb) if dim is None else int(dim)
        program = BoxLitEProgram(kb, dim, s_omega, epsilon, variant="paper",
                                 negatives=np.zeros((0, 2), dtype=np.int64))
        result = program.solve((0.0, 0.0, 0.0), solver=solver, **solver_options)
        if not result.solved:
            return SatisfiabilityResult(None, result.value, dim, result)
        check = check_model(result.embedding, kb, tol=max(tol, 1e-6))
        return SatisfiabilityResult(bool(result.value <= tol), result.value, dim, result,
                                    check)

    # ----------------------------------------------------------------------------------
    # Scores (Section 5.2)
    # ----------------------------------------------------------------------------------

    def _require_embedding(self):
        if self.embedding is None:
            raise RuntimeError("BoxLitE has no embedding yet; call train() first.")

    def _individual(self, a):
        if isinstance(a, (int, np.integer)):
            return int(a)
        return self.kb.individuals.index(str(a))

    def _role(self, r):
        if isinstance(r, (int, np.integer)):
            return int(r)
        return self.kb.roles.index(str(r))

    def score_concept_assertion(self, basic, individual):
        """:math:`s(B, a) = -\\mathcal{L}_{concept}(B, a)` (basic-concept index, individual
        index or IRI)."""
        self._require_embedding()
        a = self._individual(individual)
        return float(self.embedding.concept_scores(int(basic), [a])[0])

    def score_role_assertion(self, role, subject, obj):
        """:math:`s(R, a, b) = -\\mathcal{L}_{role}(R, a, b)` (indices or IRIs)."""
        self._require_embedding()
        r, a, b = self._role(role), self._individual(subject), self._individual(obj)
        return float(-self.embedding.role_loss(r, [a], [b])[0])

    def role_scores(self, role, subject=None, obj=None):
        """Scores of :math:`R(subject, x)` (or :math:`R(x, obj)`) for every individual
        :math:`x`, as a NumPy array indexed like ``self.kb.individuals``."""
        self._require_embedding()
        return self.embedding.role_scores(
            self._role(role), None if subject is None else self._individual(subject),
            None if obj is None else self._individual(obj))

    def axiom_costs(self, gci_name, rows):
        """Cost :math:`\\mathcal{L}_{concept}(A, a)` of every ``class_assertion`` row
        ``[a, A]`` (indices of ``individual_index_dict`` and ``class_index_dict``), lower
        is more plausible; infinite for a class or individual outside the training
        signature."""
        import torch as th
        if gci_name != "class_assertion":
            from mowl.evaluation.model_scoring import UnsupportedAxiomForm
            raise UnsupportedAxiomForm(f"BoxLitE has no score for {gci_name!r} axioms.")
        self._require_embedding()
        concept_of = {i: iri for iri, i in self.class_index_dict.items()}
        individual_of = {i: iri for iri, i in self.individual_index_dict.items()}
        concepts = {iri: k for k, iri in enumerate(self.kb.concepts)}
        individuals = {iri: k for k, iri in enumerate(self.kb.individuals)}
        cpu = rows.cpu().numpy()
        out = np.full(len(cpu), np.inf)
        for c in np.unique(cpu[:, 1]):
            basic = concepts.get(concept_of.get(int(c)))
            if basic is None:
                continue
            where = np.flatnonzero(cpu[:, 1] == c)
            known = [(w, individuals.get(individual_of.get(int(cpu[w, 0])))) for w in where]
            known = [(w, a) for w, a in known if a is not None]
            if known:
                index = np.array([w for w, _ in known])
                out[index] = self.embedding.concept_loss(basic, [a for _, a in known])
        return th.as_tensor(out, dtype=th.float32, device=rows.device)

    def score(self, axiom):
        """Score of an OWL ``ClassAssertion`` (basic concept) or ``ObjectPropertyAssertion``.

        :raises ValueError: For other axioms or entities outside the training signature.
        """
        try:
            items = _translate(axiom)
        except (_Skip, _Trivial) as exc:
            raise ValueError(f"Cannot score {axiom}: {exc}") from exc
        if len(items) != 1 or items[0][0] not in ("cassert", "rassert"):
            raise ValueError("BoxLitE scores concept and role assertions only (Section 5.2).")
        item = items[0]
        if item[0] == "rassert":
            return self.score_role_assertion(item[1], item[2], item[3])
        b = item[1]
        nc, nr = len(self.kb.concepts), len(self.kb.roles)
        index = self.kb.concepts.index(b[1]) if b[0] == "concept" else \
            nc + int(b[2]) * nr + self.kb.roles.index(b[1])
        return self.score_concept_assertion(index, item[2])

    # ----------------------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------------------

    def save(self, path=None):
        """Write the embedding (``.npz``) to ``path`` (default: ``model_filepath``)."""
        self._require_embedding()
        path = path or self.model_filepath
        with open(path, "wb") as f:
            self.embedding.save(f)
        return path

    def load(self, path=None):
        """Read an embedding written by :meth:`save`."""
        self.embedding = BoxLitEEmbedding.load(path or self.model_filepath)
        return self.embedding
