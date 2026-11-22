"""The training side of the named methods is a regime, not code of the model class.

For ELK-Em, BoxSEL, BoxEL (code regulariser), ShadowCones, RegD and Volume MultiBoxEL:
(a) the method's default regimes read back, (b) the method trains with another regime,
(c) the new regime works with another method; and the regimes reproduce the direct training
(``mowl.baselines``) where the comparison in ``tests/baselines`` does not cover the option.
"""
import os

import pytest
import torch as th

import mowl.datasets  # noqa: F401  (JVM first)
from mowl.training import (AssertionCorruption, FixedConceptRouting, ComposedRegime, ConditionalStatements, ELKEmCodeTraining,
                           ELKEmTraining, GroupedLogSumExpContrastiveLoss, LexicalRegularisation,
                           PairwiseMarginRankingLoss, PearsonContrastiveTerm,
                           PerNormalFormMiniBatches, ShuffledSplits, SumLoss,
                           TemperatureAnnealing, ValidationCheckpoint, BoxBatchRegulariser,
                           Box2ELCodeTraining)
from tests.benchmarks.tiny import scratch_dir, tiny_dataset
from tests.datasetFactory import FamilyDataset

CONDITIONALS = [("http://Mother", "http://Female", 0.6), ("http://Father", "http://Male", 0.4)]


def keys(regime):
    return [r.key for r in getattr(regime, "regimes", (regime,))]


def path(name):
    return os.path.join(scratch_dir(), name)


def with_validation_form(model):
    """The generic loop validates on a form when the dataset has a validation part."""
    model.eval_gci_name = "gci0"
    return model


def finite(model):
    return all(th.isfinite(p).all() for p in model.module.parameters())


# ------------------------------------------------------------------------------- (a) defaults

def test_default_regimes_read_back():
    from mowl.models import (BoxSEL, ELKEm, RegDELBE, RegDELEm, ShadowCones,
                             VolumeMultiBoxEL)
    ds = tiny_dataset()
    elkem = ELKEm(ds, embed_dim=4, model_filepath=path("a.pt"))
    assert isinstance(elkem.training_regime, ELKEmTraining)
    assert keys(elkem.training_regime) == [
        "fixed-concept-routing", "assertion-corruption", "per-normal-form-minibatch",
        "loss-sum", "pearson-contrastive-term", "step-lr-decay", "validation-checkpoint"]
    assert elkem.module.default_regimes       # named by the geometry

    sel = BoxSEL(FamilyDataset(), conditionals=CONDITIONALS, embed_dim=4)
    assert keys(sel.training_regime) == ["conditional-statements"]
    sel = BoxSEL(FamilyDataset(), conditionals=CONDITIONALS, embed_dim=4,
                 temperature_final=10.0)
    assert keys(sel.training_regime) == ["conditional-statements", "temperature-annealing"]

    cones = ShadowCones(ds, embed_dim=3, seed=0)
    assert keys(cones.training_regime) == ["grouped-logsumexp-contrastive"]
    assert type(cones).compute_loss is ShadowCones.__mro__[1].compute_loss

    assert keys(RegDELBE(ds, embed_dim=4).training_regime) == [
        "grouped-logsumexp-contrastive", "mse-to-target"]
    assert keys(RegDELEm(ds, embed_dim=4).training_regime) == ["grouped-logsumexp-contrastive"]

    volume = VolumeMultiBoxEL(ds, embed_dim=2, num_boxes=2, num_points=64, num_batches=2,
                              model_filepath=path("v.pt"))
    assert keys(volume.training_regime) == ["shuffled-splits", "validation-checkpoint"]
    assert volume.training_regime.regimes[0].num_batches == 2


def test_the_options_of_the_named_class_map_onto_the_regime_options():
    from mowl.models import ELKEm
    model = ELKEm(tiny_dataset(), embed_dim=4, k_neg=2, contrastive_factor=1.5, abox_k_neg=1,
                  lr_decay_epoch=None, model_filepath=path("opt.pt"))
    parts = {r.key: r for r in model.training_regime.regimes}
    assert parts["pearson-contrastive-term"].num_negatives == 2
    assert parts["pearson-contrastive-term"].weight == 1.5
    assert parts["assertion-corruption"].num_negatives == 1
    assert "step-lr-decay" not in parts


# ----------------------------------------------------------------- (b) another regime

def test_elkem_with_another_regime():
    from mowl.models import ELKEm
    model = ELKEm(tiny_dataset(), embed_dim=4, batch_size=4, model_filepath=path("b.pt"))
    model.training_regime = ELKEmCodeTraining(batch_size=4, k_neg=2, validate_every=2)
    model.train(epochs=3)
    assert finite(model)
    # without a regime the method trains with its default one again
    model = ELKEm(tiny_dataset(), embed_dim=4, batch_size=4, model_filepath=path("b2.pt"))
    model.training_regime = None
    model.train(epochs=2)
    assert isinstance(model.training_regime, ELKEmTraining) and finite(model)
    model = ELKEm(tiny_dataset(), embed_dim=4, batch_size=4, model_filepath=path("b3.pt"))
    model.training_regime = ComposedRegime(
        FixedConceptRouting(), AssertionCorruption(1), PerNormalFormMiniBatches(batch_size=4),
        SumLoss())
    with_validation_form(model).train(epochs=2)
    assert finite(model)


def test_boxsel_with_another_regime():
    from mowl.models import BoxSEL
    model = BoxSEL(FamilyDataset(), conditionals=CONDITIONALS, embed_dim=4)
    model.training_regime = SumLoss(neg_weight=0.5)      # no conditional term any more
    model.train(epochs=2)
    assert finite(model)


def test_shadow_cones_and_regd_with_another_loss():
    from mowl.models import RegDELBE, ShadowCones
    for model in (ShadowCones(tiny_dataset(), embed_dim=3, seed=0),
                  RegDELBE(tiny_dataset(), embed_dim=4)):
        model.training_regime = PairwiseMarginRankingLoss()
        with_validation_form(model).train(epochs=2)
        assert finite(model)


def test_volume_multiboxel_takes_regimes():
    from mowl.models import VolumeMultiBoxEL
    from mowl.training import unsupported_hooks
    assert frozenset(VolumeMultiBoxEL.training_hooks)           # consults the hooks now
    assert not unsupported_hooks(PairwiseMarginRankingLoss(), VolumeMultiBoxEL)
    model = VolumeMultiBoxEL(tiny_dataset(), embed_dim=2, num_boxes=2, num_points=64,
                             num_batches=2, model_filepath=path("v2.pt"))
    model.training_regime = PerNormalFormMiniBatches(batch_size=4)   # one step per epoch
    with_validation_form(model).train(epochs=2)
    assert finite(model)


# ------------------------------------------------------------ (c) with another method

def test_shuffled_splits_with_another_method():
    from mowl.models import ELBE
    model = ELBE(tiny_dataset(), embed_dim=4)
    regime = ShuffledSplits(num_batches=3, seed=1)
    model.training_regime = regime
    with_validation_form(model).train(epochs=6)                  # a step is one epoch of the generic loop
    assert regime.steps_per_epoch == 3 and len(regime.epoch_losses) == 2
    assert finite(model)


def test_grouped_logsumexp_with_another_method():
    from mowl.models import ELEmbeddings
    model = ELEmbeddings(tiny_dataset(), embed_dim=4)
    model.training_regime = GroupedLogSumExpContrastiveLoss(normal_forms=("gci2",))
    with_validation_form(model).train(epochs=2)
    assert finite(model)


def test_conditional_statements_and_annealing_with_the_composed_model():
    from mowl.models import ComposedEL
    model = ComposedEL(FamilyDataset(), method="boxsel", embed_dim=4)
    assert keys(model.training_regime) == ["conditional-statements"]     # named by the method
    model.training_regime = ConditionalStatements(CONDITIONALS, weight=2.0)
    model.train(epochs=3)
    assert finite(model)
    from mowl.models import BoxSEL
    sel = BoxSEL(FamilyDataset(), conditionals=CONDITIONALS, embed_dim=4, temperature_final=4.0)
    sel.train(epochs=3)                     # the cooling schedule is the regime
    assert float(sel.module.temperature) == pytest.approx(4.0) and finite(sel)
    sel.training_regime = ComposedRegime(ConditionalStatements(None), TemperatureAnnealing(1.0, 2.0))
    sel.train(epochs=3)
    assert float(sel.module.temperature) == pytest.approx(2.0)


def test_box_batch_regulariser_with_boxsel_and_the_code_training():
    from mowl.models import BoxEL, BoxSEL
    model = BoxSEL(FamilyDataset(), conditionals=CONDITIONALS, embed_dim=4)
    model.training_regime = ComposedRegime(
        ConditionalStatements(None, 1.0), PerNormalFormMiniBatches(batch_size=8),
        BoxBatchRegulariser())
    model.train(epochs=2)
    assert finite(model)
    boxel = BoxEL(FamilyDataset(), embed_dim=4)
    boxel.training_regime = Box2ELCodeTraining(batch_size=8, baseline_code=True, seed=0,
                                                select=False)
    boxel.train(epochs=2)
    assert finite(boxel)


def test_pearson_term_works_with_a_geometry_without_a_score_method():
    """The margin-free score is exp(-loss with margin 0), read from the objective of any
    geometry, so the term is not specific to the ELK-Em module."""
    from mowl.models import ELBE
    model = ELBE(tiny_dataset(), embed_dim=4, margin=0.1)
    model.training_regime = ComposedRegime(PerNormalFormMiniBatches(batch_size=4),
                                           PearsonContrastiveTerm(weight=1.0, num_negatives=2))
    with_validation_form(model).train(epochs=2)
    assert finite(model)


# ------------------------------------------- the regimes equal the direct training

def _same(a, b):
    from tests.baselines.pairs_boxrel import ELKEM_TABLES, weights
    x, y = weights(ELKEM_TABLES)(a), weights(ELKEM_TABLES)(b)
    for name in x:
        assert th.allclose(x[name].detach().double(), y[name].detach().double(),
                           atol=1e-5, rtol=1e-4), name


def test_elkem_validation_and_decay_match_the_direct_training():
    from mowl.baselines.models.elkem.model import ELKEm as Direct
    from mowl.models import ELKEm
    def ordered(build):       # (rows have had one order for every build since the fix)
        return build
    ds = tiny_dataset()
    models = []
    for cls in (Direct, ELKEm):
        th.manual_seed(0)
        model = ordered(lambda d, cls=cls: cls(
            d, embed_dim=4, batch_size=4, learning_rate=0.01, lr_decay_epoch=2,
            model_filepath=path(f"lex_{cls.__module__.split('.')[1]}.pt")))(ds)
        models.append(model)
    old, new = models
    from tests.baselines.pairs_boxrel import copy_state
    copy_state(old, new)
    for model in models:
        th.manual_seed(3)
        model.train(epochs=4, validate_every=2)
    _same(old, new)
    assert old.best_validation_median_rank == new.best_validation_median_rank
    assert os.path.exists(old.model_filepath) and os.path.exists(new.model_filepath)


def test_volume_multiboxel_checkpoint_and_callback_match_the_direct_training():
    from mowl.baselines.models.mcmultibox.model import VolumeMultiBoxEL as Direct
    from mowl.models import VolumeMultiBoxEL
    ds = tiny_dataset()
    seen = {}
    for cls in (Direct, VolumeMultiBoxEL):
        model = cls(ds, embed_dim=2, num_boxes=2, num_points=64, num_batches=3,
                    point_chunk_size=64, model_filepath=path(f"vm_{cls.__module__}.pt"))
        calls = []
        th.manual_seed(0)
        model.train(epochs=2, epoch_callback=lambda e, m: calls.append(e))
        seen[cls] = (model, calls)
    (old, old_calls), (new, new_calls) = seen[Direct], seen[VolumeMultiBoxEL]
    assert old_calls == new_calls == [0, 1]
    assert len(old.training_losses) == len(new.training_losses) == 2
    for x, y in zip(old.training_losses, new.training_losses):
        assert x == pytest.approx(y, rel=1e-4)


def test_elkem_lexical_regulariser_and_assertions_match_the_direct_training():
    from mowl.baselines.models.elkem.model import ELKEm as Direct
    from mowl.models import ELKEm
    from tests.baselines.pairs_boxrel import NS, tiny_ontology

    def ordered(build):
        return build
    features = {NS + "a": [1.0, 0.0, 0.5], NS + "b": [0.9, 0.1, 0.4]}
    models = []
    for cls in (Direct, ELKEm):
        th.manual_seed(0)
        models.append(ordered(lambda d, cls=cls: cls(
            d, embed_dim=4, batch_size=64, learning_rate=0.01, lr_decay_epoch=None,
            lexical_features=features, lexical_batch_size=2, abox_k_neg=2, use_contrastive=True,
            model_filepath=path(f"lex2_{cls.__module__.split('.')[1]}.pt")))(tiny_ontology()))
    old, new = models
    from tests.baselines.pairs_boxrel import copy_state
    copy_state(old, new)
    assert "lexical-regularisation" in keys(new.training_regime)
    for model in models:
        th.manual_seed(5)
        model.train(epochs=3)
    _same(old, new)


def test_every_new_regime_has_a_catalogue_card_that_names_it():
    import inspect
    from mowl.catalog.registry import get_regime
    for key in ("fixed-concept-routing", "assertion-corruption", "pearson-contrastive-term",
                "lexical-regularisation", "validation-checkpoint", "elkem-training",
                "conditional-statements", "shuffled-splits", "grouped-logsumexp-contrastive",
                "box-batch-regulariser"):
        card = get_regime(key)
        cls = card.load()
        assert inspect.isclass(cls) and cls.key == key, key
