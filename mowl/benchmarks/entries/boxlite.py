"""Benchmarks of BoxLitE (Lourenço et al., KR 2026, Section 6, Tables 1-2): role-assertion
link prediction on the Family subsets F_v1 - F_v4."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.catalog.card import Reference

LOURENCO2026 = Reference(
    "lourenco2026",
    "Lourenço, B. F., Morgan, H., Ozaki, A., Pavlović, A., Sallinger, E. (2026). BoxLitE: A "
    "Faithful Knowledge Base Embedding Based on Convex Optimization. In Proceedings of KR "
    "2026. arXiv:2605.23937.",
    "https://arxiv.org/abs/2605.23937")

IMENES2023 = Reference(
    "imenes2023",
    "Imenes, A., Guimarães, R., Ozaki, A. (2023). Marrying Query Rewriting and Knowledge "
    "Graph Embeddings. In Rules and Reasoning (RuleML+RR 2023), 126-140. Springer.",
    "https://doi.org/10.1007/978-3-031-45072-3_9")

_SIZES = {1: (300, 110, 440, 155), 2: (500, 209, 837, 233), 3: (1001, 432, 1731, 368),
          4: (3014, 1226, 4908, 895)}

_NOTES = (
    "Data: the authors' release (AleksVap/BoxLitE, commit 7d53063), checksum-verified. "
    "The TBox is not in the data files; the default (tbox='reference') is the one hard-coded "
    "in the reference loader, which produced Table 2. It differs from Figure 2 of the paper: "
    "it adds hasChild ⊑ hasParent⁻ and omits ∃hasFather⁻ ⊑ ¬∃hasMother⁻. Validation and "
    "test are exactly the closure of the training ABox under Figure 2 plus hasChild ≡ "
    "hasParent⁻ (tbox='closure'), so some test assertions are not entailed by the reference "
    "TBox. Model selection in the paper: grid search of lambda_1..3 in {0, .001, .003, .1, "
    ".3, 1, 3} on the validation MRR (Appendix G.3; mowl.benchmarks.protocols.boxlite."
    "select_lambdas). Ties are broken by individual index (ordinal), as the reference "
    "code's stable sort over IRI-sorted individuals. BoxLitE is solved, not trained "
    "(the runner's solver family ignores the epochs); the recipes "
    "(mowl.benchmarks.recipe_sets.lourenco2026) fix the published best lambdas and "
    "solve with the open-source CVXPY solvers, as MOSEK is not available: expect "
    "lower numbers than Table 2 (see the recipe notes).")

BENCHMARKS = [
    Benchmark(
        key=f"boxlite-family-v{v}", name=f"BoxLitE Family F_v{v} link prediction",
        task=Task.LINK_PREDICTION,
        description=(
            f"Family subset F_v{v} (Table 1: {_SIZES[v][0]} train, {_SIZES[v][1]} "
            f"validation, {_SIZES[v][2]} test role assertions, {_SIZES[v][3]} "
            "individuals). Each test assertion R(a, b) is ranked as R(a, ?) and R(?, b) "
            "among all individuals; candidates that form a train, validation or test "
            "assertion are filtered; filtered MRR and Hits@k over both directions "
            "(Section 6, Appendix G.4, Table 2)."),
        dataset="mowl.datasets.builtin.boxlite_family.BoxLitEFamilyDataset",
        dataset_kwargs={"version": v, "tbox": "reference"},
        protocol="mowl.benchmarks.protocols.boxlite.BoxLitEFamilyProtocol",
        metrics=("mrr", "hits@1", "hits@3", "hits@5", "hits@10"),
        higher_is_better={"mrr": True, "hits@1": True, "hits@3": True, "hits@5": True,
                          "hits@10": True},
        references=(LOURENCO2026, IMENES2023), notes=_NOTES)
    for v in (1, 2, 3, 4)
]
