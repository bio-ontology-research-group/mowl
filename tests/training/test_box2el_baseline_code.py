"""Regimes of the baselines of the Box2EL code: fixed negatives, joint initialisation,
loss variants of ELEm, EmEL++ and BoxEL."""
import os

import pytest
import torch as th

from mowl.datasets.builtin import FamilyDataset
from mowl.models import ELBE, BoxEL, ELEmbeddings, EmELPP
from mowl.training import (Box2ELCodeTraining, FixedNegatives, PerColumnCorruption,
                           JointNormalisedInitialisation, UnboundedNegativeLoss,
                           PerAxiomNormPenalty, Box2ELBaselineLosses,
                           PerNormalFormMiniBatches, ComposedRegime)
from mowl.baselines.nn.el.boxel import losses as boxel_losses
from tests.benchmarks.tiny import scratch_dir, tiny_dataset


@pytest.fixture(scope="module")
def dataset():
    return FamilyDataset()


def _model(cls, dataset, **kwargs):
    return cls(dataset, embed_dim=4,
               model_filepath=os.path.join(scratch_dir(), f"{cls.__name__}.pt"), **kwargs)


def test_fixed_negatives_are_drawn_once(dataset):
    model = _model(ELEmbeddings, dataset)
    batches = PerNormalFormMiniBatches(batch_size=2, seed=0)
    fixed = FixedNegatives(PerColumnCorruption(columns={"gci2": (0, 2)}, seed=0),
                           batch_size=3, seed=0)
    model.training_regime = ComposedRegime(batches, fixed)
    datasets = model.training_regime.training_datasets(model.training_datasets)
    batches.on_epoch_start(0, 1)
    first = fixed.negatives("gci2", datasets["gci2"])
    pool = fixed.pools["gci2"]
    # The pool corrupts every axiom once per side; each step samples from it.
    assert len(pool) == 2 * len(model.training_datasets["gci2"])
    assert first.shape == (3, 3)
    batches.on_epoch_start(1, 1)
    second = fixed.negatives("gci2", datasets["gci2"])
    assert fixed.pools["gci2"] is pool
    rows = {tuple(r) for r in pool.tolist()}
    assert all(tuple(r) in rows for r in first.tolist() + second.tolist())
    assert fixed.negatives("gci0", datasets["gci0"]) is NotImplemented


@pytest.mark.parametrize("cls,size", [(ELEmbeddings, "class_rad"), (ELBE, "class_offset")])
def test_joint_initialisation(dataset, cls, size):
    model = _model(cls, dataset)
    model.training_regime = JointNormalisedInitialisation(seed=1)
    centre = model.module.class_embed.weight
    other = getattr(model.module, size).weight
    joint = th.cat([centre, other], dim=1)
    assert th.allclose(th.linalg.norm(joint, dim=1), th.ones(len(joint)), atol=1e-5)
    assert float(th.linalg.norm(centre, dim=1).max()) < 1.0


def test_joint_initialisation_rejects_unknown_module(dataset):
    model = _model(BoxEL, dataset)
    with pytest.raises(TypeError):
        model.training_regime = JointNormalisedInitialisation()
    before = model.module.min_embedding.weight.detach().clone()
    model.training_regime = JointNormalisedInitialisation(strict=False)
    assert th.equal(before, model.module.min_embedding.weight)


def test_box2el_code_training_baseline_code_on_boxel(dataset):
    model = _model(BoxEL, dataset)
    model.training_regime = Box2ELCodeTraining(batch_size=4, baseline_code=True, seed=0)
    # the regulariser of the batches is a regime of its own, not a mode of the module
    baseline = [r for r in model.training_regime.regimes
                if isinstance(r, Box2ELBaselineLosses)][0]
    assert type(baseline._side).__name__ == "BoxBatchRegulariser"
    assert not hasattr(model.module, "regulariser")


def test_unbounded_negative_loss(dataset):
    model = _model(ELEmbeddings, dataset)
    module = model.module
    with th.no_grad():
        module.class_embed.weight[0] = 0.0
        module.class_embed.weight[1] = 0.0
        module.class_embed.weight[1, 0] = 5.0
        module.class_rad.weight[0] = 0.5
        module.class_rad.weight[1] = 0.5
        module.rel_embed.weight[0] = 0.0
    row = th.tensor([[0, 0, 1]])
    assert float(module(row, "gci2", neg=True)) == pytest.approx(0.0)
    model.training_regime = UnboundedNegativeLoss()
    expected = 0.5 + 0.5 - 5.0 + module.margin
    assert float(module(row, "gci2", neg=True)) == pytest.approx(expected)


def test_per_axiom_norm_penalty(dataset):
    model = _model(ELEmbeddings, dataset)
    module = model.module
    row = th.tensor([[0, 1]])
    plain = module(row, "gci0")
    assert float(module.regularization_loss()) > 0
    model.training_regime = PerAxiomNormPenalty()
    penalty = sum(th.abs(th.linalg.norm(module.class_embed.weight[i]) - 1) for i in (0, 1))
    assert float(module(row, "gci0")) == pytest.approx(float(plain + penalty), abs=1e-6)
    assert float(module.regularization_loss()) == 0.0


def test_baseline_losses_emelpp_drops_radius_term(dataset):
    model = _model(EmELPP, dataset)
    module = model.module
    with th.no_grad():
        module.class_embed.weight[:3] = 0.0
        module.class_embed.weight[:3, 0] = 1.0
        module.class_rad.weight[0] = 1.0
        module.class_rad.weight[1] = 1.0
        module.class_rad.weight[2] = 0.0
    row = th.tensor([[0, 1, 2]])
    before = float(module(row, "gci1"))
    model.training_regime = Box2ELBaselineLosses()
    assert module.negative_loss_form == "linear"
    assert before == pytest.approx(1.0 - module.margin)
    assert float(module(row, "gci1")) == pytest.approx(0.0, abs=1e-6)


def test_baseline_losses_boxel_sum_and_regulariser(dataset):
    model = _model(BoxEL, dataset)
    regime = Box2ELBaselineLosses()
    model.training_regime = regime
    assert type(regime._side).__name__ == "BoxBatchRegulariser"
    pos = th.ones(5, 1)
    assert float(regime.loss("gci0", pos, th.ones(4, 1))) == pytest.approx(9.0)
    model.module.train()
    rows = th.tensor([[0, 1], [1, 2]])
    regime.filter_negatives("gci0", rows, None)         # the loop hands over the batch it scores
    reg = regime.regularization(0)
    boxes = boxel_losses.axiom_boxes("gci0", rows, model.module.min_embedding,
                                     model.module.delta_embedding,
                                     model.module.relation_embedding,
                                     model.module.scaling_embedding)
    expected = sum(boxel_losses.box2el_code_side_regularizer(b) for b in boxes)
    assert float(reg) == pytest.approx(float(expected))
    # Consumed: the next step starts from zero.
    assert float(regime.regularization(0)) == 0.0


def test_boxel_gci3_translated_box_uses_upper_corner(dataset):
    model = _model(BoxEL, dataset)
    m = model.module
    with th.no_grad():
        m.min_embedding.weight[0] = 0.0
        m.delta_embedding.weight[0] = 0.0          # box [0, 1]
        m.min_embedding.weight[1] = -1.0
        m.delta_embedding.weight[1] = float(th.log(th.tensor(3.0)))   # box [-1, 2]
        m.relation_embedding.weight[0] = 0.0
        m.scaling_embedding.weight[0] = 1.0
    # ∃r.C ⊑ D with r the identity and C inside D: zero loss.
    assert float(m(th.tensor([[0, 0, 1]]), "gci3")) == pytest.approx(0.0, abs=1e-5)


def test_box2el_code_training_baseline_options(dataset):
    regime = Box2ELCodeTraining(batch_size=4, margin=0.1, baseline_code=True, seed=0)
    keys = [r.key for r in regime.regimes]
    assert "fixed-negatives" in keys
    assert "joint-normalised-initialisation" in keys
    assert "box2el-baseline-losses" in keys
    model = _model(ELEmbeddings, dataset)
    model.training_regime = regime
    assert model.module.negative_loss_form == "linear"
    assert model.module.norm_penalty == "per_axiom"
    assert float(model.module.class_rad.weight.abs().max()) < 1.0
    plain = Box2ELCodeTraining(batch_size=4)
    assert "fixed-negatives" not in [r.key for r in plain.regimes]


def test_validation_axioms_as_positives():
    from mowl.training import ValidationAxiomsAsPositives
    model = _model(ELEmbeddings, tiny_dataset())
    regime = ValidationAxiomsAsPositives()
    model.training_regime = regime
    out = regime.training_datasets(model.training_datasets)
    n_train = len(model.training_datasets["gci0"])
    n_valid = len(model.validation_datasets["gci0"])
    assert n_valid > 0
    assert len(out["gci0"]) == n_train + n_valid
    assert regime.num_added["gci0"] == n_valid
    # Without a validation ontology nothing is added.
    other = _model(ELEmbeddings, FamilyDataset())
    regime = ValidationAxiomsAsPositives()
    other.training_regime = regime
    assert len(regime.training_datasets(other.training_datasets)["gci0"]) == \
        len(other.training_datasets["gci0"])


def test_mini_batches_sample_small(dataset):
    model = _model(ELEmbeddings, dataset)
    batches = PerNormalFormMiniBatches(batch_size=50, sample_small=True, seed=0)
    model.training_regime = batches
    views = batches.training_datasets(model.training_datasets)
    batches.on_epoch_start(0, 1)
    assert all(len(v) == 50 for v in views.values() if len(v.full) > 0)


@pytest.mark.parametrize("cls,low", [(EmELPP, 0.0), (ELEmbeddings, -1.0)])
def test_emelpp_code_training(cls, low):
    from mowl.training import EmELppCodeTraining
    regime = EmELppCodeTraining(batch_size=8, validate_every=2, seed=0)
    model = _model(cls, tiny_dataset())
    model.training_regime = regime
    module = model.module
    assert module.negative_loss_form == "linear"
    assert regime._init.size_low == low
    assert module.subclass_relation.shape == (4,)
    if cls is EmELPP:
        assert module.radius_form == "relu"
        assert float(module.class_rad.weight.min()) >= 0.0
    else:
        assert module.norm_penalty == "per_axiom"
    # gci0 uses the translation: moving it changes the loss.
    row = th.tensor([[0, 1]])
    before = float(module(row, "gci0"))
    with th.no_grad():
        module.subclass_relation += 3.0
    assert float(module(row, "gci0")) != pytest.approx(before)
    model.train(epochs=3, validate_every=1000)


def test_emelpp_code_objective_squares_row_sums(dataset):
    from mowl.training import EmELppCodeObjective
    model = _model(ELEmbeddings, dataset)
    obj = EmELppCodeObjective(batch_size=2, subclass_translation=False)
    model.training_regime = obj
    obj._top = None
    zero = obj.loss("gci0", th.tensor([[1.0], [2.0]]), th.tensor([[-1.0], [1.0]]))
    assert float(zero) == 0.0
    obj.loss("gci1", th.tensor([[1.0], [0.0]]), None)
    # rows: 1 - 1 + 1 = 1 and 2 + 1 + 0 = 3 -> mean(1, 9) = 5
    assert float(obj.additional_loss(0)) == pytest.approx(5.0)


def test_emelpp_batch_direction_term():
    from mowl.baselines.nn.el.emelpp import losses as EL
    r = th.tensor([[1.0, 0.0], [0.0, 1.0]])
    s = th.tensor([[1.0, 0.0], [0.0, 1.0]])
    # Per row: identical vectors, cosine 1 -> 0. Batch axis: each coordinate column of r
    # and s is the same unit vector -> sum of products 2 -> |1 - 2| = 1 on every row.
    assert th.allclose(EL._direction_term(r, s, "row"), th.zeros(2, 1))
    assert th.allclose(EL._direction_term(r, s, "batch"), th.ones(2, 1))
