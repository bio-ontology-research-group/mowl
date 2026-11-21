"""Knowledge graph completion benchmarks (FB15k-237, WN18RR), for the methods that were
published for knowledge graphs (BoxE, ExpressivE, ...)."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import ABBOUD2020, PAVLOVIC2023

_DESC = (
    "{label}: train on the training triples, rank every test triple (h, r, t) against all "
    "entities as tail and as head, remove the corruptions that are training, validation or "
    "test triples (filtered setting), and report MR, MRR and Hits@1/3/10 over both "
    "directions. In mOWL a triple is the axiom h ⊑ ∃r.t between classes "
    "(mowl.datasets.builtin.kg.KnowledgeGraphDataset).")
_NOTES = (
    "Tied candidates share the average rank. A knowledge graph has no TBox: this benchmark "
    "tests the methods on the data they were published for, not on an ontology. Any mOWL "
    "method that scores C ⊑ ∃r.D can be run on it.")

_METRICS = ("f_mrr", "f_mr", "f_hits@1", "f_hits@3", "f_hits@10")

BENCHMARKS = [
    Benchmark(
        key=f"kg-{name.lower()}-link-prediction", name=f"{label} link prediction",
        task=Task.LINK_PREDICTION,
        evaluation_regime="kg-filtered-link-prediction",
        completion_tasks=("existential-prediction",), encodes=("link-prediction",),
        description=_DESC.format(label=label),
        dataset="mowl.datasets.builtin.kg.KnowledgeGraphDataset",
        dataset_kwargs={"name": name},
        protocol="mowl.benchmarks.protocols.bindings.KGLinkPredictionProtocol",
        metrics=_METRICS, subsets=("all", "tail", "head"),
        higher_is_better={m: m != "f_mr" for m in _METRICS},
        references=(ABBOUD2020, PAVLOVIC2023), notes=_NOTES)
    for name, label in (("FB15k237", "FB15k-237"), ("WN18RR", "WN18RR"))
]
