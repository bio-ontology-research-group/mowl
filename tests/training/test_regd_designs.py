"""The RegD code's training design: the softmax contrastive loss, the boundary-scored
sampled-F1 evaluation, validation-F1 model selection, and the flipped gci2 of the
prediction task. Hand-computed values on tiny tensors, plus an end-to-end run."""
import math
import os
from unittest import TestCase

import pytest
import torch as th

import mowl.datasets  # noqa: F401  (JVM; mowl.nn must not be imported first)
from mowl.evaluation.sampled_f1 import candidate_dists, pair_scores
from mowl.nn.el.regd.module import RegDELBEModule
from mowl.training import RegDCodeTraining, SoftmaxEnergyContrastiveLoss
from mowl.training.selection import ValidationF1Selection
from tests.benchmarks.tiny import scratch_dir, tiny_dataset


class _FakeModel:
    """Just enough for a loss regime: the module and the recorded negatives."""

    def __init__(self, module):
        self.module = module


def _toy_module():
    """Four classes on the line, radii/offsets chosen by hand; no depth term."""
    m = RegDELBEModule(4, 1, nb_inds=None, embed_dim=1, margin=0, lamda=0.0, p=1,
                       depth_on_containment=False)
    with th.no_grad():
        m.class_embed.weight.zero_()
        m.class_offset.weight[:] = th.tensor([[2.0], [0.5], [3.0], [1.0]])
        m.rel_embed.weight[:] = 5.0
    return m


class TestSoftmaxEnergyContrastiveLoss(TestCase):

    def test_values_match_the_reference_loss_cross(self):
        m = _toy_module()
        regime = SoftmaxEnergyContrastiveLoss(normal_forms=("gci0",))
        regime.attach(_FakeModel(m))
        # positive [0, 1]: child box 0 (offset 2) inside box 1 (offset 0.5) sticks out
        # by 1.5; with no depth term the energy is 1.5, the module's score max(E, 0).
        pos = m(th.tensor([[0, 1]]), "gci0")
        self.assertAlmostEqual(float(pos), 1.5)
        # negative [0, 2]: box 0 (offset 2) inside box 2 (offset 3) is contained; the
        # raw energy is -1, clamped at 0 only for the *positive* term.
        with m.scoring():
            neg_energy = m(th.tensor([[0, 2]]), "gci0")
        self.assertAlmostEqual(float(neg_energy), -1.0)
        # PluginRegD.loss_cross: -log(e^-1.5 / (e^-1.5 + e^-min(-1,0)))
        # = cross_entropy of logits [-1.5, -(-1)] against class 0.
        neg = th.tensor([[0, 2], [0, 2]], dtype=th.long)  # K = 2, aligned
        regime.filter_negatives("gci0", None, neg)
        loss = regime.loss("gci0", th.tensor([[1.5]]), th.zeros(2, 1))
        # two equal negatives: -log(e^-1.5/(e^-1.5 + 2 e)) = log(1 + 2 e^2.5)
        self.assertAlmostEqual(float(loss), math.log(1 + 2 * math.exp(2.5)), places=5)

    def test_rejects_module_without_scoring_context(self):
        from tests.training.stubs import StubModel
        regime = SoftmaxEnergyContrastiveLoss(normal_forms=("gci0",))
        regime.attach(StubModel({"gci0": [[0, 1]]}))
        regime.filter_negatives("gci0", None, th.tensor([[0, 1]]))
        with pytest.raises(TypeError):
            regime.loss("gci0", th.tensor([[1.0]]), th.zeros(1, 1))

    def test_other_normal_forms_fall_through(self):
        regime = SoftmaxEnergyContrastiveLoss(normal_forms=("gci0",))
        regime.attach(_FakeModel(_toy_module()))
        assert regime.loss("gci2", th.tensor([[1.0]]), None) is NotImplemented


class TestReferenceScoring(TestCase):

    def test_candidate_dists_and_pair_scores(self):
        m = _toy_module()
        # query [0, 1]: child = box 0, candidates are all four classes, class 0 itself
        # is filtered to +inf.
        dists = candidate_dists(m, "gci0", th.tensor([[0, 1]]))
        self.assertEqual(dists.shape, (1, 4))
        self.assertEqual(dists[0, 0].item(), float("inf"))   # C ⊑ C filtered
        self.assertAlmostEqual(dists[0, 1].item(), 1.5)      # relu(2 - 0.5)
        self.assertAlmostEqual(dists[0, 2].item(), 0.0)      # contained
        self.assertAlmostEqual(dists[0, 3].item(), 1.0)      # relu(2 - 1)
        self.assertAlmostEqual(pair_scores(m, "gci0", th.tensor([[0, 1]])).item(), 1.5)

    def test_gci2_child_is_the_filler_translated_backwards(self):
        m = _toy_module()
        with th.no_grad():
            m.class_embed.weight[1] = 5.0   # filler centre at 5
            m.class_offset.weight[3] = 0.25  # candidate 3 does not contain the child
        # child = centre 5 - r (r = 5) = 0, offset 0.5; candidate 2 has offset 3:
        # contained -> score 0; candidate 3 (offset 0.25) -> relu(0 + 0.5 - 0.25).
        dists = candidate_dists(m, "gci2", th.tensor([[1, 0, 1]]))
        self.assertAlmostEqual(dists[0, 2].item(), 0.0)
        self.assertAlmostEqual(dists[0, 3].item(), 0.25)


class TestFlipGci2(TestCase):

    def test_flip_reverses_the_inclusion(self):
        kwargs = dict(nb_ont_classes=2, nb_rels=1, nb_inds=None, embed_dim=1, margin=0,
                      lamda=0.0, p=1, depth_on_containment=False)
        standard = RegDELBEModule(**kwargs)
        flipped = RegDELBEModule(flip_gci2=True, **kwargs)
        with th.no_grad():
            for m in (standard, flipped):
                m.class_embed.weight[:] = th.tensor([[0.0], [4.0]])
                m.class_offset.weight[:] = 1.0
                m.rel_embed.weight[:] = 2.0
        row = th.tensor([[0, 0, 1]])  # C ⊑ ∃r.D with C = class 0, D = class 1
        # standard: box(C) + r inside box(D): |0 + 2 - 4| + 1 - 1 = 2
        self.assertAlmostEqual(float(standard(row, "gci2")), 2.0)
        # flipped (the code's nf3Loss_ours): box(D) + r inside box(C): |4 + 2| + 1 - 1 = 6
        self.assertAlmostEqual(float(flipped(row, "gci2")), 6.0)


class TestRegDCodeTrainingEndToEnd:

    def test_training_runs_and_selects(self):
        from mowl.models.regd.model import RegDELBE
        dataset = tiny_dataset()
        model = RegDELBE(dataset, embed_dim=3, depth_on_containment=False,
                         neg_sampling_gcis=["gci0"], learning_rate=0.1,
                         model_filepath=os.path.join(scratch_dir(), "regd.pt"))
        regime = RegDCodeTraining(task="inference", design="regd", batch_size=4,
                                  validate_every=1)
        assert [r.key for r in regime.regimes] == [
            "per-normal-form-minibatch", "softmax-energy-contrastive", "mse-to-target",
            "validation-f1-selection"]
        model.eval_gci_name = "gci0"
        model.training_regime = regime
        model.train(epochs=3)
        selection = regime.regimes[-1]
        assert isinstance(selection, ValidationF1Selection)
        assert selection.best_epoch is not None
        # The regime restored the best checkpoint into the module: re-evaluating the
        # selection metric reproduces the recorded best value.
        from mowl.training.selection import validation_f1
        assert validation_f1(model, gci_name="gci0") == pytest.approx(selection.best_value)

    def test_prediction_design_runs(self):
        from mowl.models.regd.model import RegDELEm
        dataset = tiny_dataset()
        model = RegDELEm(dataset, embed_dim=3, depth_on_containment=False,
                         flip_gci2=True, neg_sampling_gcis=["gci2"],
                         learning_rate=0.1,
                         model_filepath=os.path.join(scratch_dir(), "regd_elem.pt"))
        regime = RegDCodeTraining(task="prediction", design="regd", batch_size=4,
                                  validate_every=1)
        model.eval_gci_name = "gci2"
        model.training_regime = regime
        model.train(epochs=2)
        assert regime.regimes[-1].best_epoch is not None


class TestReferenceThresholdAndSlip(TestCase):

    def test_reference_best_threshold_sweeps_sorted_positions(self):
        from mowl.evaluation.sampled_f1 import reference_best_threshold, _apply_threshold
        # sorted: 0.0(+) 0.1(-) 0.2(+) 0.9(-): F1 at positions 1..4 = 2/3, 1/2, 2/3*?, ...
        t, f1 = reference_best_threshold([0.0, 0.2], [0.1, 0.9])
        # position 1: p=1, r=.5 -> .667; position 3: p=2/3, r=1 -> .8 (the best)
        self.assertAlmostEqual(t, 0.2)
        self.assertAlmostEqual(f1, 0.8)
        f1_t, p, r = _apply_threshold([0.0, 0.2], [0.1, 0.9], t)
        self.assertAlmostEqual(f1_t, 0.8)
        self.assertEqual(reference_best_threshold([], [1.0]), (None, 0.0))

    def test_reference_role_classes_reads_the_class_at_the_role_index(self):
        from mowl.evaluation.sampled_f1 import reference_role_classes

        class _M:
            class_index_dict = {"http://x/A": 0, "http://x/B": 1,
                                "http://y/aux/5": 2}
            object_property_index_dict = {"http://x/r": 0, "http://x/s": 1,
                                          "http://x/t": 2}
        classes = {"<http://x/B>": 0, "<http://x/A>": 1, "<http://z/Q>": 5}
        relations = {"SubClassOf": 0, "<http://x/r>": 1, "<http://x/s>": 5,
                     "<http://x/t>": 9}
        out = reference_role_classes(_M(), "GO", index_files=(classes, relations))
        # r has reference index 1 -> reference class 1 = A -> model class 0
        # s has index 5 -> reference class 5 (unknown IRI) -> aux/5 -> model class 2
        # t has index 9 -> no class -> -1
        self.assertEqual(out.tolist(), [0, 2, -1])

    def test_gci2_reference_needs_the_reference_ontology(self):
        from mowl.evaluation.sampled_f1 import SampledF1Regime, _role_classes_for
        with pytest.raises(ValueError):
            _role_classes_for(object(), "gci2", "reference", None, {})
        assert _role_classes_for(object(), "gci0", "reference", None, {}) is None
        SampledF1Regime(gci_name="gci2", positive_scores="pair")


class TestCodeDesignHooks:

    def test_joint_init_and_raw_energy(self):
        from mowl.models.regd.model import RegDELEm
        from mowl.training.reference import joint_unit_init
        dataset = tiny_dataset()
        model = RegDELEm(dataset, embed_dim=3, depth_on_containment=False,
                         radius="softplus", neg_sampling_gcis=["gci0"],
                         model_filepath=os.path.join(scratch_dir(), "regd_elem2.pt"))
        joint_unit_init(model.module)
        w = th.cat([model.module.class_embed.weight, model.module.class_rad.weight], 1)
        assert th.allclose(th.linalg.norm(w, dim=1), th.ones(len(w)), atol=1e-5)
        regime = RegDCodeTraining(task="inference", design="regd", host_loss="mean",
                                  batch_size=4, validate_every=1)
        assert [r.key for r in regime.regimes][2] == "loss-sum"
        model.eval_gci_name = "gci0"
        model.training_regime = regime
        assert model.module.energy_mode is True
        model.train(epochs=2)
        assert model.module.energy_mode is False
        # softplus radii: the RegD region of a class with raw radius -5 is small, not 5
        with th.no_grad():
            model.module.class_rad.weight[0] = -5.0
        _, r = model.module._region(th.tensor([0]))
        assert float(r) < 0.01


def test_energy_score_adds_the_depth_term():
    from mowl.evaluation.sampled_f1 import candidate_dists, pair_scores
    m = _toy_module()
    # child box 0 (offset 2) in candidate 2 (offset 3): bd = -1, depth = (0 + 1) / (2 * 3)
    d = candidate_dists(m, "gci0", th.tensor([[0, 2]]), score="energy")
    assert float(d[0, 2]) == pytest.approx(-1 + 1 / 6)
    assert float(pair_scores(m, "gci0", th.tensor([[0, 2]]), score="energy")) == \
        pytest.approx(-1 + 1 / 6)
    assert float(candidate_dists(m, "gci0", th.tensor([[0, 2]]))[0, 2]) == 0.0


def test_rectified_negatives_exert_no_force():
    m = _toy_module()
    regime = SoftmaxEnergyContrastiveLoss(normal_forms=("gci0",), rectify_negatives=True)
    regime.attach(_FakeModel(m))
    regime.filter_negatives("gci0", None, th.tensor([[0, 2], [0, 2]], dtype=th.long))
    loss = regime.loss("gci0", th.tensor([[1.5]]), th.zeros(2, 1))
    # contained negatives (E = -1) are rectified to 0: -log(e^-1.5 / (e^-1.5 + 2))
    assert float(loss) == pytest.approx(math.log(1 + 2 * math.exp(1.5)), rel=1e-5)
    assert RegDCodeTraining(rectified=True).raw_energy is False
