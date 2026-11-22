BoxLitE
=======

BoxLitE [lourenco2026]_ embeds a :math:`\text{DL-Lite}_\mathcal{H}` knowledge base into
axis-parallel boxes by solving a **convex** optimisation problem, a second-order cone
programme (SOCP), instead of running gradient descent. The catalogue key is ``boxlite``;
its guarantees, with sources and the tests that check them, are on the card in the
:doc:`method catalogue <catalogue>`.

The logic
---------

:math:`\text{DL-Lite}_\mathcal{H}` has basic concepts :math:`B ::= A \mid \exists R \mid
\exists R^-`, concept inclusions :math:`B \sqsubseteq B'` and :math:`B \sqsubseteq \neg B'`,
role inclusions between roles and inverse roles, and assertions :math:`B(a)`,
:math:`R(a, b)`. mOWL had no :math:`\text{DL-Lite}` pipeline;
:func:`mowl.datasets.dllite.extract_dllite` reads exactly these axioms from an OWL
ontology (including ``DisjointClasses``, domains and ranges, inverse and symmetric
properties, and ``SubClassOf(B owl:Nothing)``, which is :math:`B \sqsubseteq \neg B`) and
returns integer index arrays (:class:`~mowl.datasets.dllite.DLLiteKB`). Every other
logical axiom is counted in ``kb.skipped`` with the reason, never dropped silently.

Geometry (Definitions 2-4)
--------------------------

* An individual has a position :math:`pos(a)` and a bump :math:`bump(a)` in the universe
  box :math:`\Omega = [-s_\Omega, s_\Omega]^d`.
* A concept name is a box; a role name has a head, a tail and a bump box.
  :math:`R(a, b)` holds iff :math:`pos(a) + bump(b) \in Head(R)`,
  :math:`pos(b) + bump(a) \in Tail(R)` and :math:`bump(a), bump(b) \in Bump(R)`.
  :math:`R^-` swaps head and tail.
* :math:`\exists R` is the head box enlarged by the bump box,
  :math:`[L^H - U^B, U^H - L^B]`.
* :math:`\neg C` is the *complement box* :math:`[-s_\Omega - L_C, s_\Omega - U_C]`, a
  convex stand-in for the set complement. A constraint in one reserved dimension per basic
  concept keeps :math:`C` and its complement box disjoint ("box consistency").

The programme (Section 5)
-------------------------

The TBox, box consistency and the box widths are **linear constraints**, so every feasible
embedding satisfies the TBox. The objective is

.. math::

   \max\Big(\max_{D(a) \in \mathcal{A}} \mathcal{L}_{concept},
            \max_{R(a,b) \in \mathcal{A}} \mathcal{L}_{role}\Big)
   + \lambda_1 \max_{D(a) \notin \mathcal{A}} \mathcal{L}_{concept}(\neg D, a)
   + \lambda_2 \sum \lVert U - L\rVert_2 + \lambda_3 \sum_R \lVert U^B_R - L^B_R\rVert_2,

where the losses are signed distances of the points to the boxes. Negatives are not
sampled: the :math:`\lambda_1` term pushes individuals into the complement boxes of the
concepts (in practice the existential concepts) they are not asserted to belong to. The
score of an assertion is the negative of its loss.

:class:`~mowl.models.boxlite.model.BoxLitE` writes the programme with CVXPY (an optional
dependency: ``pip install cvxpy``) and solves it with an open-source conic solver shipped
with CVXPY (Clarabel by default; if it fails, ECOS and then SCS are tried: on F_v1
Clarabel stalls and ECOS solves it). The paper used MOSEK, which can be
selected with ``solver="MOSEK"`` if it is installed and licensed. The optimal value of a
convex programme does not depend on the solver; when the optimum is unique, the optimal
embeddings of different solvers agree up to their tolerances, otherwise they may differ.

Made of parts
-------------

BoxLitE is a combination of the parts of :mod:`mowl.representation`, *fitted by a solver
instead of by gradient descent*: a box per concept name given by its bounds
(:class:`~mowl.representation.library.solved_boxlite.BoundsBoxes`), a head, a tail and a bump
box per role (:class:`~mowl.representation.library.solved_boxlite.HeadTailBumpBoxes`), a
position and a bump vector per individual, the signed distance as the measure and the
objective of the paper. ``BoxLitE.module`` is the geometry of these parts. The solver is a
part of its own (:mod:`mowl.representation.library.solved_solver`): the problem
(:class:`~mowl.representation.library.solved_boxlite.BoxLitEProgram`) is written once against
an algebra, so the same formulas give the losses of the geometry (tensors), the checks of a
solution (arrays, no solver needed) and the CVXPY problem. ``train`` fits the parts with a
conic solver (default), with another
:class:`~mowl.representation.library.solved_solver.Solver` (``solver=ScipySolver()``, small
problems, SciPy only) or by gradient descent on the same program
(``regime=GradientFit()``, approximate):

.. testcode::

   from mowl.datasets.dllite import DLLiteKB
   from mowl.models.boxlite.model import BoxLitE
   from mowl.representation.library.solved_solver import ScipySolver, evaluate

   kb = DLLiteKB.from_lists(inclusions=[("A", "B")], concept_assertions=[("A", "a")])
   model = BoxLitE.from_kb(kb, embed_dim=kb.n_basic)
   print(model.module.parts["classes"], model.module.parts["relations"])
   model.train(solver=ScipySolver(maxiter=50))
   values = {name: getattr(model.embedding, name) for name in model.embedding._ARRAYS}
   print(evaluate(model.program, values).feasible(1e-4))

.. testoutput::

   box head and tail boxes
   True

Guarantees in practice
----------------------

* Every solution satisfies the TBox and is box consistent (Theorem 4 ii).
* With :math:`\lambda = 0` and :math:`d \ge |N_C| + |N_R|(2 + |N_I| + 2|N_R|)` the optimum
  is :math:`\le 0` iff the KB is satisfiable, and then the solution is a weakly faithful
  model (Theorem 4 iii-iv). :meth:`~mowl.models.boxlite.model.BoxLitE.decide_satisfiability`
  uses this as a decision procedure.
* :meth:`~mowl.models.boxlite.model.BoxLitE.check_model` checks a solution with plain set
  semantics, independently of the loss (:mod:`mowl.representation.library.solved_boxlite_check`).
* The approach is restricted to :math:`\text{DL-Lite}_\mathcal{H}` on purpose:
  conjunction on the left of an inclusion makes the constraints non-convex (Section 7).

Two variants
------------

``variant="paper"`` (default) implements the formulas as printed. ``variant="reference"``
reproduces what the authors' code (``AleksVap/BoxLitE``, commit ``7d53063``) solves, which
produced the published numbers: closed boxes without :math:`\epsilon`, one signed distance
over the stacked residuals of all assertions instead of a maximum per assertion, a
different width regulariser and different negatives. The details are in the docstring of
:mod:`mowl.representation.library.solved_boxlite` and in the card's ``notes``.

Example
-------

The block runs only if ``cvxpy`` is installed; otherwise it prints the same line.

.. testcode::

   import importlib.util

   from mowl.datasets.dllite import DLLiteKB

   kb = DLLiteKB.from_lists(
       inclusions=[("∃hasParent", "Person"), ("∃hasParent-", "Person")],
       negative_inclusions=[("∃hasFather-", "∃hasMother-")],
       role_inclusions=[("hasFather", "hasParent"), ("hasMother", "hasParent"),
                        ("hasChild", "hasParent-")],
       role_assertions=[("hasFather", "ann", "bob"), ("hasMother", "ann", "cat"),
                        ("hasChild", "bob", "dan")])

   if importlib.util.find_spec("cvxpy") is None:
       print("model: True, satisfiable: True")
   else:
       from mowl.models.boxlite.model import BoxLitE

       model = BoxLitE.from_kb(kb, embed_dim=kb.n_basic + 2, lambda1=0.01)
       model.train(lambdas=(0.0, 0.0, 0.0))      # lambda_1 = 0: Theorem 4 (iii) applies
       check = model.check_model()
       sat = BoxLitE.decide_satisfiability(kb)
       print(f"model: {check.is_model}, satisfiable: {sat.satisfiable}")
       scores = model.role_scores("hasParent", subject="ann")   # rank candidate parents

.. testoutput::

   model: True, satisfiable: True

With a mOWL dataset, ``BoxLitE(dataset, embed_dim=32, lambda1=..., lambda2=...,
lambda3=...)`` extracts the :math:`\text{DL-Lite}_\mathcal{H}` part of the training
ontology.

Benchmarks
----------

The paper evaluates on four subsets F_v1 - F_v4 of the Family dataset (filtered MRR and
Hits@k of inferred role assertions, Table 2). They are the benchmarks
``boxlite-family-v1`` ... ``-v4`` (dataset
:class:`~mowl.datasets.builtin.boxlite_family.BoxLitEFamilyDataset`, evaluation
:class:`~mowl.evaluation.assertion_ranking.RoleAssertionRankingRegime`). The benchmark
runner only knows models trained for a number of epochs; register the solver family first,
or call :func:`~mowl.benchmarks.protocols.boxlite.run_family_benchmark` directly:

.. testcode::

   from mowl.benchmarks.protocols.boxlite import (PAPER_LAMBDA_GRID, SolverModelFamily,
                                                  run_family_benchmark)
   from mowl.benchmarks.runner import compare, register_family, run_benchmark

   def run_f_v1(select=False):
       """Downloads the data and solves a programme of about 150,000 variables."""
       if select:   # the paper's grid search on the validation MRR (343 solves)
           return run_family_benchmark("boxlite-family-v1", grid=PAPER_LAMBDA_GRID,
                                       model_kwargs={"embed_dim": 32, "variant": "reference"})
       # or through the runner, with the lambdas of Table 9:
       register_family(SolverModelFamily())
       return run_benchmark("boxlite-family-v1", method="boxlite", model_kwargs={
           "embed_dim": 32, "lambda1": 0.001, "lambda2": 3.0, "lambda3": 0.001,
           "variant": "reference", "solver": "ECOS"})

   # for row in compare(run_f_v1(), method="boxlite"): print(row)

Measured in mOWL on F_v1 (CPU, ECOS, ``variant="reference"``, the lambdas of Table 9, no
grid search, about 4 minutes of solving): filtered MRR 0.660, Hits@1 0.480, Hits@3 0.792,
Hits@10 0.958, against 0.720, 0.545, 0.870 and 0.979 published with MOSEK (Table 2). ECOS
stopped "close to optimal" (relative gap 6.5e-4); the optimal value (0.00185) does not
determine the embedding, so another solver can return a different optimal embedding with
different ranks. With ``variant="paper"`` (about 1,200 per-assertion cones instead of a
few large ones) neither Clarabel nor ECOS reached a solution on F_v1 on this machine; it
works on small KBs (tests) and may need MOSEK at this size.

The default TBox of the benchmark is the one hard-coded in the authors' loader, which
differs from Figure 2 of the paper; ``dataset_kwargs={"tbox": "paper"}`` or
``{"tbox": "closure"}`` select the alternatives (see the dataset docstring).
