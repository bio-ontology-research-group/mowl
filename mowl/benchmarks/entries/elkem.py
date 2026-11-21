"""Benchmarks of ELK-Em (Singh, Jiménez-Ruiz & Weyde, KG-NeSy 2026, Section 5.1, Table 3):
approximating deductive reasoning on the GALEN, GO and Anatomy releases of Box2EL."""

from mowl.benchmarks.entries._refs import JACKERMEIER2024
from mowl.benchmarks.spec import Benchmark, Task
from mowl.catalog.card import Reference

SINGH2026 = Reference(
    "singh2026",
    "Singh, N., Jiménez-Ruiz, E., Weyde, T. (2026). ELK-Em: Closure-Aware Embeddings for "
    "EL+⊥ Ontologies. In Third Workshop on Knowledge Graphs and Neurosymbolic AI "
    "(KG-NeSy 2026), ISWC 2026 Workshops Joint Proceedings, CEUR-WS Vol-4313, paper 14.",
    "https://ceur-ws.org/Vol-4313/")

_P = "mowl.benchmarks.protocols.bindings."
_RANK_METRICS = ("hits@1", "hits@10", "hits@100", "med", "mrr", "mr", "auc")
_HIB = {"hits@1": True, "hits@10": True, "hits@100": True, "med": False, "mrr": True,
        "mr": False, "auc": True}

_DESCRIPTION = (
    "Train on the full normalised ontology (all asserted axioms, role inclusions and role "
    "chains); rank the atomic subsumptions C ⊑ D inferred by an EL++ reasoner but not "
    "asserted (90% test, 10% validation, from the Box2EL reference release). The "
    "superclass D is ranked among all classes except owl:Thing and C by the margin-free "
    "loss ℓ0 (equivalently the score exp(−ℓ0)); tied candidates share their average rank; "
    "ranks are unfiltered; Hits@k counts fractional ranks ≤ k, AUC is mean((n − r)/(n − 1)) "
    "(Section 5.1, Table 3 of the ELK-Em paper).")

_NOTES = (
    "Same data as the 'box2el-*-inference' benchmark of the same ontology (data.zip of "
    "KRR-Oxford/BoxSquaredEL, commit a715080, <ONT>/inferences/); the evaluation of the "
    "two papers differs (owl:Thing removed from the candidates, average-rank ties, "
    "margin-free score instead of ordinal ties over all classes), so ELK-Em's numbers are "
    "a benchmark of their own, not comparable with the Box2EL Table 6 numbers. The "
    "authors' evaluation code masks owl:Thing deliberately although their specification "
    "(§8 D8) keeps it, because the full box of owl:Thing saturates the score and would "
    "dominate every tie block. The published numbers were obtained with the training "
    "regime 'elkem-code-training'.")

BENCHMARKS = [
    Benchmark(
        key=f"elkem-{name}-subsumption",
        name=f"ELK-Em approximating deductive reasoning ({label})",
        task=Task.DEDUCTIVE_INFERENCE,
        evaluation_regime="elkem-subsumption-ranking",
        description=_DESCRIPTION,
        dataset="mowl.datasets.builtin.jackermeier2024_reference."
                "Jackermeier2024InferencesDataset",
        dataset_kwargs={"ontology": onto},
        protocol=_P + "ELKEmSubsumptionProtocol",
        metrics=_RANK_METRICS,
        higher_is_better=_HIB,
        references=(SINGH2026, JACKERMEIER2024),
        notes=_NOTES,
    )
    for name, label, onto in (("galen", "GALEN", "GALEN"), ("go", "GO", "GO"),
                              ("anatomy", "Anatomy", "ANATOMY"))
]
