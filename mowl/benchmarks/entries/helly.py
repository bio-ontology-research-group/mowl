"""The Helly family T_k as an exact-model benchmark."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import LEEMHUIS2025

_METRICS = ("axiom_satisfaction", "is_model", "empty_fraction", "exact_model",
            "helly_violation_rate")

BENCHMARK = Benchmark(
    key="helly-tk",
    name="Helly family T_k (exact model)",
    task=Task.EXACT_MODEL,
    description=(
        "Train on T_k (A_i ≡ ⊓_{j≠i} X_j, A_i ⊓ A_j ⊑ ⊥; k = 3 is the Leemhuis-Kutz "
        "counterexample of three pairwise compatible, jointly incompatible concepts) and "
        "check whether the trained regions are an exact model: fraction of normal-form "
        "axioms satisfied (exact checks of mowl.geometric.checker.ModelChecker), whether "
        "every axiom holds, fraction of satisfiable names with an empty region "
        "(degeneracy), whether the embedding is a non-degenerate model, and the fraction "
        "of Helly certificates (mowl.geometric.helly.helly_check) whose jointly "
        "unsatisfiable names share a point in the embedding. Use --dataset-param k=5 "
        "for larger k."),
    dataset="mowl.datasets.builtin.helly.HellyTkDataset",
    dataset_kwargs={"k": 3},
    protocol="mowl.benchmarks.protocols.bindings.ExactModelProtocol",
    metrics=_METRICS,
    higher_is_better={"axiom_satisfaction": True, "is_model": True,
                      "empty_fraction": False, "exact_model": True,
                      "helly_violation_rate": False},
    references=(LEEMHUIS2025,),
    notes=(
        "No published numbers exist for this benchmark. By Helly's theorem for "
        "axis-aligned boxes, no single-box model (ELBE, Box2EL, TransBox) can be a "
        "non-degenerate model of T_k for k >= 3; unions of boxes (MultiBoxEL) can. For "
        "unions of boxes only the bottom axioms and inclusions into a single box are "
        "decided; other inclusions are reported as unchecked and exact_model is then 0."),
)
