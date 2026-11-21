"""The training design of GroundedEL as shared regimes: the composition reproduces the
model's training exactly, and the parts work on their own and with other models."""

import pytest
import torch as th


from mowl.models import GroundedEL
from mowl.training import (ComposedRegime, FixedColumnCorruption, GroundedELTraining,
                           IdleComponentReseeding, KnownPositiveFilter,
                           LagrangianDualWeights, LinearDecay, TemperatureAnnealing,
                           TrainingRegime, dual_ascent_step)
from tests.training.stubs import StubModel


def _model(**kwargs):
    classes = {f"C{i}": i for i in range(4)}
    roles = {"r": 0, "s": 1}
    individuals = {f"a{i}": i for i in range(6)}
    opa = th.tensor([[0, 0, 1], [1, 0, 2], [2, 1, 3], [3, 1, 4], [4, 0, 5]])
    args = dict(embed_dim=4, n_components=2, epochs=6, seed=3, learning_rate=0.05,
                role_mode="translation")
    args.update(kwargs)
    return GroundedEL.from_tables(
        classes, roles, gci0=th.tensor([[0, 1], [1, 2]]), gci2=th.tensor([[2, 0, 3]]),
        class_assertion=th.tensor([[0, 0], [1, 2]]), object_property_assertion=opa,
        individual_index_dict=individuals, **args)


def _state(model):
    return {k: v.clone() for k, v in model.module.state_dict().items()}


def _same(a, b):
    return all(th.equal(a[k], b[k]) for k in a)


def test_default_training_is_the_composed_regime():
    """Training without a regime and with the equivalent composition assigned by hand
    give identical parameters and loss histories."""
    kwargs = dict(reseed_every=2, constraint_mode="lagrangian", neg_weight=0.5)
    implicit = _model(**kwargs)
    implicit.train(lr_final=0.005)

    explicit = _model(**kwargs)
    explicit.training_regime = GroundedELTraining(
        tau0=explicit.tau0, tau1=explicit.tau1, lr_final=0.005, reseed_every=2,
        constraint_mode="lagrangian")
    explicit.train()

    assert _same(_state(implicit), _state(explicit))
    assert implicit.loss_history == explicit.loss_history
    assert implicit.loss_history[0]["tau"] == implicit.tau0
    assert implicit.loss_history[-1]["tau"] == pytest.approx(implicit.tau1)


def test_parts_can_be_replaced():
    """Another temperature schedule gives another model; hooks of an added regime are
    called."""
    base = _model()
    base.train()

    class Count(TrainingRegime):
        starts = ends = finished = 0

        def on_epoch_start(self, epoch, epochs):
            self.starts += 1

        def on_epoch_end(self, epoch, losses):
            assert {"total", "fact", "axioms", "region"} <= set(losses)
            self.ends += 1

        def should_stop(self, epoch):
            return epoch == 3

        def on_train_end(self):
            self.finished += 1

    count = Count()
    other = _model()
    annealing = TemperatureAnnealing(start=2.0, end=0.5, shape="linear")
    other.training_regime = ComposedRegime(count, annealing, LinearDecay(min_lr=0.01))
    other.train()
    assert (count.starts, count.ends, count.finished) == (4, 4, 1)
    assert len(other.loss_history) == 4
    assert annealing.history[0] == 2.0
    assert [r["tau"] for r in other.loss_history] == annealing.history
    assert not _same(_state(base), _state(other))


def test_negatives_come_from_the_regime_and_can_be_filtered():
    """Corrupted facts can be drawn by a shared negative sampler and filtered against
    the stated facts, which the model's own sampler does not do."""
    seen = {}

    class Record(TrainingRegime):
        def filter_negatives(self, gci_name, gci_dataset, negatives):
            seen["name"], seen["rows"] = gci_name, negatives.clone()
            return negatives

    model = _model(neg_weight=1.0, epochs=2)
    model.training_regime = ComposedRegime(
        FixedColumnCorruption(columns={"object_property_assertion": 2},
                              require_negative_loss=False, seed=0),
        KnownPositiveFilter(seed=0), Record(),
        GroundedELTraining.for_model(model))
    model.train()
    assert seen["name"] == "object_property_assertion"
    known = {tuple(r) for r in model.training_datasets["object_property_assertion"][:].tolist()}
    rows = seen["rows"].tolist()
    assert len(rows) == len(known)
    assert not any(tuple(r) in known for r in rows)
    assert model.loss_history[-1]["neg"] >= 0.0


def test_dual_ascent():
    weights = {"a": 1.0, "b": 1.0, "c": 1.0}
    dual_ascent_step(weights, {"a": 0.5, "b": 0.0, "c": 5000.0}, rho=2.0, max_weight=10.0)
    assert weights == {"a": 2.0, "b": 1.0, "c": 10.0}

    model = _model(epochs=3)
    assert model.constraint_mode == "fixed"
    duals = LagrangianDualWeights(rho=1.0, terms=("axioms",))
    model.training_regime = ComposedRegime(duals, TemperatureAnnealing(1.0, 0.1))
    model.train()
    assert model.constraint_mode == "lagrangian"
    assert model.dual_weights["fact"] == 1.0
    assert model.dual_weights["axioms"] >= 1.0
    assert len(duals.history) == 3


def test_regimes_refuse_models_without_the_protocol():
    stub = StubModel({"gci0": [[0, 1]]})
    for regime in (TemperatureAnnealing(), IdleComponentReseeding(), LagrangianDualWeights()):
        with pytest.raises(TypeError):
            regime.attach(stub)


def test_reseeding_resets_the_optimiser():
    model = _model(epochs=4)
    reseeding = IdleComponentReseeding(every=2)
    model.training_regime = ComposedRegime(TemperatureAnnealing(1.0, 0.1), reseeding)
    model.train()
    assert [e for e, _ in reseeding.history] == [1, 3]
    assert len(model.optimizer.state) == 0     # cleared after the last step


def test_regimes_serve_multiboxel(tmp_path):
    """The schedule and the re-seeding regime drive MultiBoxEL through the generic loop."""
    from mowl.datasets import PathDataset
    from mowl.models import MultiBoxEL
    from tests.geometric.helpers import fixture_path

    model = MultiBoxEL(PathDataset(fixture_path("t3.owl")), embed_dim=2, n_components=2,
                       neg_sampling_gcis=[], model_filepath=str(tmp_path / "m.pt"))
    assert model.reseed_every == 0 and not model.tau_anneal
    annealing = TemperatureAnnealing(start=1.0, end=0.1)
    reseeding = IdleComponentReseeding(every=2, reset_optimizer=False)
    model.training_regime = ComposedRegime(annealing, reseeding)
    model.train(epochs=4, validate_every=10)
    assert annealing.history[0] == 1.0 and annealing.history[-1] == pytest.approx(0.1)
    assert float(model.module.tau) == pytest.approx(0.1)
    assert [e for e, _ in reseeding.history] == [1, 3]
