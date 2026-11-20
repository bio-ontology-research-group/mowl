"""Recipes of BoxLitE (Lourenço et al., KR 2026), from the paper (Tables 2 and 9) and the
authors' code (AleksVap/BoxLitE, commit 7d53063).

BoxLitE is *solved*, not trained: the recipes set ``epochs=0`` and the benchmark runner
ignores the epochs anyway (:class:`mowl.benchmarks.protocols.boxlite.SolverModelFamily`).
"""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries.boxlite import LOURENCO2026

# version -> (lambda_1, lambda_2, lambda_3): the best validation MRR of the grid search
# of Appendix G.3, Table 9 of the paper.
_LAMBDAS = {1: (0.001, 3.0, 0.001), 2: (0.001, 1.0, 0.003), 3: (0.003, 0.1, 0.003),
            4: (0.001, 3.0, 0.003)}

_LOCATOR = (
    "Table 9 of the paper (best lambdas of the Appendix G.3 grid search); Table 2 for the "
    "numbers themselves; Configurations/*.py of AleksVap/BoxLitE for the variant")

_NOTE = (
    "The published Table 2 numbers were obtained with the commercial conic solver MOSEK "
    "(main.py of the code: solver=cp.MOSEK), which is not available here: the recipe "
    "solves the same programme (same feasible set and objective, variant 'reference' = "
    "the authors' code) with the open-source solvers shipped with CVXPY (Clarabel, then "
    "ECOS, then SCS; pass --param solver=... to choose). Measured with ECOS (status "
    "'optimal_inaccurate' on all four subsets), filtered MRR ours / published: F_v1 "
    "0.659 / 0.720, F_v2 0.339 / 0.549, F_v3 0.367 / 0.433, F_v4 0.222 / 0.444. The "
    "metrics depend strongly on which near-optimal point the solver returns: on F_v1, "
    "SCS reaches status 'optimal' with an objective only 3% above ECOS's (0.001905 vs "
    "0.001851) but gives MRR 0.305 instead of 0.659, and Clarabel stops with "
    "'InsufficientProgress'. What does not transfer without MOSEK is therefore the "
    "embedding itself (and with it every published metric), and the paper's model "
    "selection (7^3 = 343 solves per subset), which the recipes replace by the published "
    "best lambdas. What does transfer: the KB, the programme, the lambdas and the "
    "filtered head/tail ranking with ordinal ties. The ablation rows BoxLitE_1/2/3 of "
    "Table 2 are the same recipe with the lambdas of Tables 10-12.")

RECIPES = tuple(
    Recipe(
        benchmark=f"boxlite-family-v{v}",
        method="boxlite",
        source=LOURENCO2026,
        model_kwargs={"embed_dim": 32, "s_omega": 1.0, "epsilon": 0.01,
                      "lambda1": l1, "lambda2": l2, "lambda3": l3,
                      "variant": "reference"},
        epochs=0,
        locator=_LOCATOR,
        notes=_NOTE,
    )
    for v, (l1, l2, l3) in _LAMBDAS.items()
)
