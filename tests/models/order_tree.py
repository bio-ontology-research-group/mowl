"""A small taxonomy (a binary tree of depth 3 plus a second parent), written to a scratch
ontology, for the tests of the hierarchy-embedding models."""

import os
import tempfile

NS = "http://mowl.borg/order#"

# child -> parents (told, direct)
EDGES = [("A", "R"), ("B", "R"),
         ("A1", "A"), ("A2", "A"), ("B1", "B"), ("B2", "B"),
         ("A11", "A1"), ("A12", "A1"), ("A21", "A2"), ("A22", "A2"),
         ("B11", "B1"), ("B12", "B1"), ("B21", "B2"), ("B22", "B2"),
         ("A11", "B2")]  # a second parent (a DAG, not a tree)
CLASSES = sorted({c for e in EDGES for c in e})


def ancestors(name, edges=EDGES):
    out, todo = set(), [name]
    while todo:
        c = todo.pop()
        for child, parent in edges:
            if child == c and parent not in out:
                out.add(parent)
                todo.append(parent)
    return out


def tree_dataset(extra_axioms=(), directory=None):
    """A :class:`mowl.datasets.PathDataset` with the told edges as training ontology."""
    from mowl.datasets import PathDataset
    directory = directory or tempfile.mkdtemp(prefix="mowl_order_", dir=tempfile.gettempdir())
    path = os.path.join(directory, "train.owl")
    lines = [f"SubClassOf(<{NS}{c}> <{NS}{p}>)" for c, p in EDGES] + list(extra_axioms)
    with open(path, "w") as f:
        f.write("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n"
                "Ontology(<http://mowl.borg/order/train>\n")
        for line in lines:
            f.write(line + "\n")
        f.write(")\n")
    return PathDataset(path)


def iri(name):
    return NS + name


def common_ranking_benchmark():
    """The joint subsumption-prediction regime on the tiny benchmark dataset."""
    from mowl.benchmarks import Benchmark, Task
    from mowl.benchmarks.entries._refs import JACKERMEIER2024
    metrics = ("f_mrr", "f_hits@1", "f_hits@10", "f_hits@100", "f_mr", "f_med", "f_auc")
    return Benchmark(
        key="tiny-common-gci0", name="tiny", task=Task.AXIOM_PREDICTION,
        description="tiny", dataset="tests.benchmarks.tiny.tiny_dataset",
        protocol="mowl.benchmarks.protocols.bindings.CommonRankingProtocol",
        protocol_kwargs={"gci_name": "gci0"}, metrics=metrics,
        higher_is_better={m: m not in ("f_mr", "f_med") for m in metrics},
        references=(JACKERMEIER2024,), evaluation_regime="common-filtered-ranking",
        completion_tasks=("subsumption-prediction",))


def train_and_separate(model, epochs, seed=0):
    """Trains ``model`` and returns (mean energy of the told ancestors, mean energy of
    the non-ancestors) over all ordered pairs of the tree's classes."""
    import numpy as np
    import torch as th
    np.random.seed(seed)
    th.manual_seed(seed)
    model.train(epochs=epochs)
    pos, neg = [], []
    for c in CLASSES:
        anc = ancestors(c)
        for d in CLASSES:
            if c != d:
                e = model.energies([iri(c)], [iri(d)]).item()
                (pos if d in anc else neg).append(e)
    return float(np.mean(pos)), float(np.mean(neg))
