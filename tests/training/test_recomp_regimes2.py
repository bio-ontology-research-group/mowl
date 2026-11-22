"""The loops of E2R, BCBox, CatE, HiT, OnT and FaithEL are regimes, not code of the model class.

For each method: (a) the default regimes read back (and the options of the class map onto
them), (b) the method trains with another regime, (c) a new regime works with another method.
That the combinations give the numbers of the direct implementations is ``tests/baselines``.
"""
import os

import numpy as np
import pytest
import torch as th

import mowl.datasets  # noqa: F401  (JVM first)
from mowl.nn.lm.encoders import BagOfWordsEncoder
from mowl.training import (AlternatingMinibatches, ComposedRegime, EdgeCorruption, EpochBatches,
                           FullBatch, GroupedMarginLoss, IntervalProjection, KnownRowMask,
                           ModuleProjection, PairwiseMarginRankingLoss, PerNormalFormMiniBatches,
                           Regularization, SelfAdversarialBCE, SumLoss, ValidationPatience)
from mowl.training.loop import RegimeTrained
from mowl.training.schedules import StepDecay, validation_loss
from mowl.training.text import LinearWarmupDecay, SampledLogicalLosses
from tests.benchmarks.tiny import scratch_dir, tiny_dataset
from tests.datasetFactory import FamilyDataset


def keys(regime):
    return [r.key for r in getattr(regime, "regimes", (regime,))]


def path(name):
    return os.path.join(scratch_dir(), name)


def finite(values):
    return all(np.isfinite(values))


def parameters_finite(model):
    return all(th.isfinite(p).all() for p in model.module.parameters())


def bcbox(**options):
    from mowl.models import BinaryCodeBox
    return BinaryCodeBox(FamilyDataset(), embed_dim=4, learning_rate=0.05, num_negs=3,
                         model_filepath=path("bc.pt"), **options)


def cate(**options):
    from mowl.models import CatE
    return CatE(FamilyDataset(), embed_dim=4, learning_rate=0.05, batch_size=8,
                model_filepath=path("cate.pt"), **options)


def e2r(**options):
    from mowl.models import E2R
    from tests.baselines.pairs_logic import e2r_dataset
    return E2R(e2r_dataset(), embed_dim=4, learning_rate=0.05, sep_gap=1.0, **options)


def hit(cls="HiT", **options):
    import mowl.models as models
    options = dict(dict(batch_size=8, warmup_steps=0, learning_rate=0.05, seed=0,
                        centri_weight=0.5), **options)
    return getattr(models, cls)(tiny_dataset(), encoder=BagOfWordsEncoder(dim=8, num_buckets=64),
                                model_filepath=path("hit.pt"), **options)


def faithel(**options):
    from mowl.models import FaithEL
    from tests.models.test_canonical_mixture import _write_family_abox_dataset
    return FaithEL(_write_family_abox_dataset(__import__("pathlib").Path(scratch_dir())),
                   batch_size=2, learning_rate=0.01, seed=0, **options)


# ------------------------------------------------------------------------------- (a) defaults

def test_every_loop_is_the_loop_of_the_regimes():
    from mowl.models import BinaryCodeBox, CatE, E2R, FaithEL, HiT, OnT
    for cls in (BinaryCodeBox, CatE, E2R, FaithEL, HiT, OnT):
        assert issubclass(cls, RegimeTrained)
        assert "train" not in vars(cls)         # the loop is not code of the class
        assert "compute_loss" not in vars(cls) or cls in (BinaryCodeBox, CatE, E2R)


def test_default_regimes_read_back():
    assert keys(bcbox().training_regime) == [
        "epoch-batches", "edge-corruption", "known-row-mask", "self-adversarial-bce"]
    assert keys(cate().training_regime) == [
        "epoch-batches", "edge-corruption", "known-row-mask", "grouped-margin", "cyclic-lr",
        "validation-patience"]
    assert keys(e2r().training_regime) == [
        "per-normal-form-minibatch", "edge-corruption", "edge-corruption", "known-row-mask",
        "loss-sum", "regularization"]
    assert keys(hit().training_regime) == [
        "epoch-batches", "linear-warmup-decay", "validation-patience"]
    assert keys(hit("OnT").training_regime) == [
        "epoch-batches", "sampled-logical-losses", "linear-warmup-decay", "validation-patience"]
    assert keys(faithel().training_regime) == ["alternating-minibatches", "regularization"]


def test_the_options_of_the_class_map_onto_the_regimes():
    parts = {r.key: r for r in bcbox(self_adversarial=False).training_regime.regimes}
    assert parts["edge-corruption"].num_negatives == 3 and parts["edge-corruption"].columns == (1, 0)
    assert not parts["self-adversarial-bce"].self_adversarial
    assert [r.key for r in cate(filter_negatives=False, cyclic_lr=False,
                                interaction_loss=False).training_regime.regimes] == [
        "epoch-batches", "edge-corruption", "grouped-margin", "validation-patience"]
    parts = {r.key: r for r in cate(negative_aggregation="mean").training_regime.regimes}
    assert parts["grouped-margin"].aggregation == "mean"
    assert keys(e2r(num_negatives=0).training_regime) == [
        "per-normal-form-minibatch", "loss-sum", "regularization"]
    assert keys(hit("OnT", use_roles=False).training_regime) == [
        "epoch-batches", "linear-warmup-decay", "validation-patience"]
    assert keys(faithel(training=FullBatch()).training_regime) == ["full-batch", "regularization"]


def test_a_method_without_a_regime_trains_with_the_regimes_of_the_method_again():
    model = bcbox()
    model.training_regime = None
    assert model.training_regime is None
    history = model.train(epochs=2)
    assert finite(history) and keys(model.training_regime)[0] == "epoch-batches"


def test_the_margin_of_e2r_is_refused_not_ignored():
    from mowl.representation import METHODS
    from mowl.representation.library.logic_subspace import SubspaceObjective
    with pytest.raises(ValueError, match="no margin"):
        METHODS["e2r"](3, 1, 2, 4, margin=0.5)
    with pytest.raises(ValueError, match="no margin"):
        SubspaceObjective(margin=0.1)
    assert METHODS["e2r"](3, 1, 2, 4, margin=0.0) is not None


# ------------------------------------------------------------------- (b) another regime

def test_bcbox_with_another_loss_and_the_plain_sum():
    model = bcbox()
    model.training_regime = ComposedRegime(
        EpochBatches(4), EdgeCorruption(2, (1, 0)), KnownRowMask(), GroupedMarginLoss(0.5))
    assert finite(model.train(epochs=2)) and parameters_finite(model)
    model = bcbox()
    parts = [SelfAdversarialBCE(False) if isinstance(r, SelfAdversarialBCE) else r
             for r in model.training_regime.regimes]
    model.training_regime = ComposedRegime(*parts)
    assert finite(model.train(epochs=2))


def test_an_epoch_has_as_many_steps_as_minibatches():
    model = bcbox()
    rows = len(model.training_datasets["gci0"])
    model.training_regime = ComposedRegime(EpochBatches(3), EdgeCorruption(1, (1, 0)),
                                           SelfAdversarialBCE())
    model.train(epochs=2)
    assert model.training_steps == 2 * -(-rows // 3)


def test_cate_with_another_loss_and_without_the_schedule():
    model = cate()
    model.training_regime = ComposedRegime(
        EpochBatches(4), EdgeCorruption(2, (1,)), PairwiseMarginRankingLoss(0.5))
    assert finite(model.train(epochs=2)) and parameters_finite(model)
    # a regime of the BCBox loss on the CatE costs
    model = cate()
    model.training_regime = ComposedRegime(
        EpochBatches(4), EdgeCorruption(2, (1,)), KnownRowMask(), SelfAdversarialBCE())
    assert finite(model.train(epochs=2))


def test_cate_validates_and_stops_with_the_regime():
    from mowl.models import CatE
    model = CatE(tiny_dataset(), embed_dim=4, learning_rate=0.05, batch_size=8,
                 early_stopping_patience=1, model_filepath=path("cv.pt"))
    history = model.train(epochs=6, validate_every=1)
    assert len(history) <= 6 and model.validation_history
    assert [e for e, _ in model.validation_history] == list(range(1, len(model.validation_history) + 1))


def test_e2r_with_another_batching_and_without_the_projection():
    model = e2r()
    model.training_regime = ComposedRegime(PerNormalFormMiniBatches(2, order="random"),
                                           SumLoss(0.5))
    assert finite(model.train(epochs=2)) and parameters_finite(model)
    # with the default regimes the indicators stay in the unit cube
    model = e2r()
    model.train(epochs=3)
    for table in (model.module.c_embedding, model.module.r_embedding):
        assert float(table.weight.min()) >= 0.0 and float(table.weight.max()) <= 1.0


def test_e2r_negatives_are_corruptions_of_the_assertions_that_are_not_stated():
    model = e2r()
    regime = model.training_regime
    th.manual_seed(0)
    data = model.training_datasets
    neg = regime.negatives("class_assertion", data["class_assertion"])
    stated = {tuple(r) for r in data["class_assertion"][:].tolist()}
    assert len(neg) == 3 * len(data["class_assertion"])
    kept = regime.filter_negatives("class_assertion", data["class_assertion"], neg)
    assert kept is None or all(tuple(r) not in stated for r in kept.tolist())
    assert regime.negatives("concept_inclusion", data["concept_inclusion"]) is NotImplemented


def test_hit_and_ont_with_another_schedule_and_other_weights():
    model = hit()
    model.training_regime = ComposedRegime(EpochBatches(8, rng=model.rng), StepDecay([1]))
    assert finite(model.train(epochs=2, validate=False))
    model = hit("OnT")
    model.training_regime = ComposedRegime(
        EpochBatches(8, rng=model.rng), SampledLogicalLosses(0.5, 0.0), LinearWarmupDecay(1))
    assert finite(model.train(epochs=2, validate=False)) and parameters_finite(model)


def test_hit_selects_the_best_epoch_and_the_weight_of_the_score():
    model = hit(centri_weight=None)
    model.train(epochs=2)
    assert model.best_validation is not None and 0 <= model.best_validation["epoch"] < 2
    assert [h["epoch"] for h in model.history] == [0, 1]
    assert "valid_mrr" in model.history[0]
    assert model.centri_weight == model.best_validation["centri_weight"]


def test_faithel_with_another_batching_and_without_the_projection():
    model = faithel()
    model.training_regime = ComposedRegime(FullBatch())
    model.train(epochs=2)
    assert len(model.loss_history) == 2          # one step per epoch
    model = faithel()
    model.training_regime = ComposedRegime(AlternatingMinibatches(3))   # no projection
    model.train(epochs=2)
    assert parameters_finite(model) and len(model.loss_history) == 2


def test_faithel_alternates_the_forms_and_keeps_the_parameters_in_the_balls():
    model = faithel()
    regime = model.training_regime.regimes[0]
    model.train(epochs=2)
    steps = regime.steps_per_epoch
    assert steps == sum(-(-len(v.full) // 2) for v in regime.views.values())
    assert len(model.loss_history) == 2
    m = model.module
    assert float((m.concepts.weight - m.concept_centroid).norm(dim=1).max()) <= model.radius + 1e-5


# ------------------------------------------------- (c) a new regime used with another method

def test_edge_batches_corruptions_and_margin_loss_with_an_el_model():
    from mowl.models import ShadowCones
    model = ShadowCones(tiny_dataset(), embed_dim=3, seed=0)
    model.training_regime = ComposedRegime(
        EpochBatches(3), EdgeCorruption(2, (0, 1)), KnownRowMask(), GroupedMarginLoss(0.5))
    model.train(epochs=3)
    assert parameters_finite(model)


def test_the_bce_loss_of_binary_code_boxes_with_an_el_model():
    from mowl.models import OrderEmbeddings
    model = OrderEmbeddings(tiny_dataset(), embed_dim=3)
    model.eval_gci_name = "gci0"
    model.training_regime = ComposedRegime(
        EpochBatches(3), EdgeCorruption(2, (1,)), SelfAdversarialBCE())
    model.train(epochs=2)
    assert parameters_finite(model)


def test_the_warmup_schedule_of_hit_with_an_el_model():
    from mowl.models import ELEmbeddings
    model = ELEmbeddings(tiny_dataset(), embed_dim=4, learning_rate=0.1)
    model.eval_gci_name = "gci0"
    schedule = LinearWarmupDecay(1)
    model.training_regime = schedule
    model.train(epochs=4)
    assert schedule.history == pytest.approx([0.0, 0.1, 0.1 * 2 / 3, 0.1 / 3])


def test_alternating_minibatches_with_an_el_model():
    from mowl.models import ELBE
    model = ELBE(tiny_dataset(), embed_dim=4, learning_rate=0.05)
    model.eval_gci_name = "gci0"
    regime = AlternatingMinibatches(2, forms={"gci0": 2, "gci2": 3})
    model.training_regime = regime
    model.train(epochs=3)
    assert parameters_finite(model) and regime.steps_per_epoch >= 2


def test_validation_with_patience_with_an_el_model():
    from mowl.models import ELBE
    model = ELBE(tiny_dataset(), embed_dim=4, learning_rate=0.05, model_filepath=path("v.pt"))
    model.eval_gci_name = "gci0"
    model.training_regime = ValidationPatience(validation_loss, every=1, patience=2, higher=False)
    model.train(epochs=6, validate_every=1)
    assert model.validation_history and len(model.validation_history) <= 6
    assert parameters_finite(model)


def test_an_interval_projection_of_e2r_with_another_method():
    from mowl.models import ELEmbeddings
    model = ELEmbeddings(tiny_dataset(), embed_dim=4, learning_rate=0.5)
    model.eval_gci_name = "gci0"
    model.training_regime = Regularization(IntervalProjection("class_embed", 0.0, 1.0))
    model.train(epochs=3)
    w = model.module.class_embed.weight
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0
