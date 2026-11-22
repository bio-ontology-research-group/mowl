"""Inductive knowledge graph completion benchmarks (the GraIL splits of FB15k-237, WN18RR and
NELL-995, versions v1-v4), for methods that score triples over a graph they were not trained
on (RESHUFFLE)."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import PAVLOVIC2025, TERU2020

_DESC = (
    "{label} {version}, inductive: train on the training graph, then rank every test triple "
    "(h, r, t) of a second graph with different entities (the model receives its facts, "
    "mowl.datasets.builtin.inductive_kg.InductiveKnowledgeGraphDataset.inference_graph) "
    "against 50 sampled corrupted tails and 50 sampled corrupted heads; Hits@10 (also "
    "Hits@1/3, MRR, MR) over both directions. In mOWL a triple is the axiom h ⊑ ∃r.t between "
    "classes.")
_NOTES = (
    "Candidates are 50 sampled entities, so the numbers are not comparable with the full "
    "ranking of the kg-* benchmarks. Only models with inductive_costs(graph, rows) can run "
    "(RESHUFFLE); transductive embeddings are refused. The test graph has other entities "
    "than the training graph and the same relations (up to relations that only occur in the "
    "training graph). The published numbers (Pavlovic et al. 2025, Table 2) are Hits@10, "
    "the mean of three seeds; hyper-parameters were selected on the validation triples of "
    "the training graph.")

_METRICS = ("hits@10", "hits@1", "hits@3", "mrr", "mr")
_LABELS = {"FB15k237": "FB15k-237", "WN18RR": "WN18RR", "NELL995": "NELL-995"}

BENCHMARKS = [
    Benchmark(
        key=f"kg-inductive-{name.lower()}-{version}", name=f"{label} {version} inductive "
        "link prediction",
        task=Task.LINK_PREDICTION,
        evaluation_regime="inductive-sampled-ranking",
        completion_tasks=("existential-prediction",), encodes=("link-prediction",),
        description=_DESC.format(label=label, version=version),
        dataset="mowl.datasets.builtin.inductive_kg.InductiveKnowledgeGraphDataset",
        dataset_kwargs={"name": name, "version": version},
        protocol="mowl.benchmarks.protocols.bindings.InductiveLinkPredictionProtocol",
        metrics=_METRICS, subsets=("all", "tail", "head"),
        higher_is_better={m: m != "mr" for m in _METRICS},
        references=(PAVLOVIC2025, TERU2020), notes=_NOTES,
        requires=("unseen names",))
    for name, label in _LABELS.items() for version in ("v1", "v2", "v3", "v4")
]
