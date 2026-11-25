"""Helpers of the order / hierarchy embedding tests (tiny taxonomy, no downloads)."""
import numpy as np
import torch as th

from tests.benchmarks.tiny import NS, scratch_dir, tiny_dataset


def iri(name):
    return NS + name


def train(model_class, epochs, seed=0, **kwargs):
    """Train ``model_class`` on the taxonomy of the tiny ontology and return the model."""
    th.manual_seed(seed)
    np.random.seed(seed)
    model = model_class(tiny_dataset(), model_filepath=scratch_dir() + "/m.pt", **kwargs)
    model.train(epochs=epochs, validate_every=10 ** 6)
    return model


def cost(model, sub, sup):
    return float(model.subsumption_costs([iri(sub)], [iri(sup)])[0])


def total_loss(model, seed=0):
    th.manual_seed(seed)
    np.random.seed(seed)
    with th.no_grad():
        data = model.training_datasets["gci0"][:]
        pos = model.module(data, "gci0")
        neg = model.module(model.generate_negatives("gci0", model.training_datasets["gci0"]),
                           "gci0", neg=True)
        return float(pos.mean() + neg.mean())


STATED = [("A", "B"), ("B", "C"), ("D", "C"), ("E", "F"), ("G", "F"), ("H", "F")]


def check_trained_taxonomy(model):
    """Stated pairs are cheaper than their reverses and than unrelated pairs."""
    for sub, sup in STATED:
        assert cost(model, sub, sup) < cost(model, sup, sub), (sub, sup)
    assert cost(model, "A", "C") < cost(model, "C", "A")      # transitive consequence
    assert cost(model, "A", "B") < cost(model, "A", "F")
    assert cost(model, "E", "F") < cost(model, "E", "C")
    # only the taxonomy is trained
    assert set(model.training_datasets) == {"gci0"}
    assert len(model.training_datasets["gci0"]) == len(STATED)
