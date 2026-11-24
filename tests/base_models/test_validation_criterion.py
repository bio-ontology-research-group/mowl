"""The built-in validation keeps the best state by how well validation axioms are told from
corruptions of them, not by their mean loss alone."""
import io
from contextlib import redirect_stdout

import pytest
import torch as th

from mowl.datasets import Dataset
from mowl.evaluation.validation import corrupted_pair_error
from mowl.owlapi import OWLAPIAdapter


class _Constant(th.nn.Module):
    def forward(self, data, gci_name, neg=False):
        return th.zeros(len(data))


class _Knows(th.nn.Module):
    """Scores a row 0 when its two columns are equal, 1 otherwise."""

    def forward(self, data, gci_name, neg=False):
        return (data[:, 0] != data[:, 1]).float()


def test_constant_scores_are_at_chance():
    data = th.arange(40).repeat(2, 1).T.contiguous()
    assert corrupted_pair_error(_Constant(), data, "gci0", range(40)) == 0.5


def test_a_module_that_separates_is_near_zero_and_seeded():
    data = th.arange(200).repeat(2, 1).T.contiguous()
    first = corrupted_pair_error(_Knows(), data, "gci0", range(200), seed=3)
    assert first == corrupted_pair_error(_Knows(), data, "gci0", range(200), seed=3)
    assert first < 0.01
    assert th.equal(data[:, 0], data[:, 1])            # the rows are not changed


def _datasets():
    adapter = OWLAPIAdapter()
    ns = "http://test.org/selection/"
    classes = [adapter.create_class(f"{ns}C{i}") for i in range(60)]
    train = adapter.create_ontology(ns + "train")
    valid = adapter.create_ontology(ns + "valid")
    manager = adapter.owl_manager
    for i in range(1, 60):
        axiom = adapter.create_subclass_of(classes[i], classes[(i - 1) // 3])
        manager.addAxiom(valid if i % 5 == 0 else train, axiom)
    return Dataset(train, validation=valid)


@pytest.mark.parametrize("criterion", ["separation", "loss"])
def test_the_saved_state_is_the_best_by_the_criterion(tmp_path, criterion):
    from mowl.models import BoxEL
    th.manual_seed(0)
    model = BoxEL(_datasets(), embed_dim=10, batch_size=64, learning_rate=0.05,
                  model_filepath=str(tmp_path / "model.pt"))
    model.eval_gci_name = "gci0"
    model.validation_criterion = criterion
    with redirect_stdout(io.StringIO()) as out:
        model.train(epochs=40, validate_every=1)
    lines = [line for line in out.getvalue().splitlines() if "Valid loss" in line]
    assert len(lines) == 40
    assert all(("Valid pair error" in line) == (criterion == "separation") for line in lines)
    name = "Valid pair error: " if criterion == "separation" else "Valid loss: "
    values = [float(line.split(name)[1].split()[0]) for line in lines]
    assert model._best_validation_loss == min(values)
    # the state on disk is the one of the first epoch that reached the best value
    data = model.validation_datasets["gci0"][:]
    model.load_best_model()
    if criterion == "separation":
        candidates = sorted(model.named_class_index_dict.values())
        again = corrupted_pair_error(model.module, data, "gci0", candidates)
        assert again == min(values) < 0.5
    else:
        with th.no_grad():
            assert float(model.module(data, "gci0").mean()) == pytest.approx(min(values))


def test_unknown_criterion_is_refused(tmp_path):
    from mowl.models import ELEmbeddings
    model = ELEmbeddings(_datasets(), embed_dim=4, batch_size=64,
                         model_filepath=str(tmp_path / "model.pt"))
    model.eval_gci_name = "gci0"
    model.validation_criterion = "accuracy"
    with pytest.raises(ValueError, match="validation_criterion"):
        with redirect_stdout(io.StringIO()):
            model.train(epochs=1, validate_every=1)
