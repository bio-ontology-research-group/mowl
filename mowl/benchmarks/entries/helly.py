"""The Helly family T_k as an exact-model benchmark."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import LEEMHUIS2025

_METRICS = ("axiom_satisfaction", "is_model", "empty_fraction", "no_interior_fraction",
            "exact_model", "exact_model_nonempty", "helly_violation_rate")

BENCHMARK = Benchmark(
    key="helly-tk",
    name="Helly family T_k (exact model)",
    task=Task.EXACT_MODEL,
    description=(
        "Train on T_k (A_i ≡ ⊓_{j≠i} X_j, A_i ⊓ A_j ⊑ ⊥; k = 3 is the Leemhuis-Kutz "
        "counterexample of three pairwise compatible, jointly incompatible concepts) and "
        "check, exactly and independently of the training loss, whether the trained "
        "regions are a model (mowl.evaluation.exact_model.ExactModelRegime): fraction of "
        "normal-form axioms satisfied, whether every axiom holds, fraction of satisfiable "
        "names with an empty region and with a region without interior, whether the "
        "embedding is a model in which every satisfiable name has non-empty interior "
        "(exact_model) or at least a non-empty region (exact_model_nonempty), and the "
        "fraction of Helly certificates (mowl.geometric.helly.helly_check) whose jointly "
        "unsatisfiable names share a point in the embedding. Use --dataset-param k=5 for "
        "larger k."),
    dataset="mowl.datasets.builtin.helly.HellyTkDataset",
    dataset_kwargs={"k": 3},
    protocol="mowl.benchmarks.protocols.bindings.ExactModelProtocol",
    metrics=_METRICS,
    higher_is_better={"axiom_satisfaction": True, "is_model": True,
                      "empty_fraction": False, "no_interior_fraction": False,
                      "exact_model": True, "exact_model_nonempty": True,
                      "helly_violation_rate": False},
    references=(LEEMHUIS2025,),
    notes=(
        "No published numbers exist for this benchmark. Every propositional normal form "
        "of T_k (C ⊑ D, C ⊓ D ⊑ E, C ⊓ D ⊑ ⊥) is decided exactly for every region "
        "family mOWL trains: single boxes (ELBE, Box2EL, TransBox, ELK-Em, RegD-ELBE, "
        "BoxEL), balls (ELEm, EmEL++, EmEL(var), RegD-ELEm; lens-in-ball in closed form), "
        "prisms (PrismEL), products of octagons (OctagonEL) and unions of boxes "
        "(MultiBoxEL, MultiELBE, volume MultiBoxEL; box-in-union by exact box "
        "subtraction). By Helly's theorem for axis-aligned boxes no single-box embedding "
        "is a model of T_k (k >= 3) with non-empty names. Balls are no model with "
        "non-empty interiors either (A_i = X_j ⊓ X_l forces a lens to be a ball, i.e. "
        "one disc inside the other, which contradicts the disjointness axioms); three "
        "pairwise tangent balls with point-like A_i are a model with non-empty names, "
        "which is why exact_model asks for non-empty interiors and exact_model_nonempty "
        "is reported separately. Prisms and products of octagons can represent T_3 but not "
        "T_k for k >= 4 (their families are closed under intersection with Helly number "
        "3: convex sets in the plane, times boxes); unions of boxes can represent every "
        "T_k. Role axioms do not occur in T_k."),
)
