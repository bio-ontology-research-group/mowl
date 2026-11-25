"""TransBox with the losses of the authors' code (formulation='reference') and the
training regime transbox-code-training."""
import os

import pytest
import torch as th

from mowl.datasets.builtin import FamilyDataset
from mowl.models import TransBox
from mowl.training import TransBoxCodeTraining
from tests.benchmarks.tiny import scratch_dir, tiny_dataset


@pytest.fixture(scope="module")
def dataset():
    return FamilyDataset()


def _model(dataset, **kwargs):
    return TransBox(dataset, embed_dim=2,
                    model_filepath=os.path.join(scratch_dir(), "transbox-code.pt"), **kwargs)


def _set(module, centres, offsets, rel_centre=None, rel_offset=None):
    with th.no_grad():
        for i, (c, o) in enumerate(zip(centres, offsets)):
            module.class_center_embedding.weight[i] = th.tensor(c)
            module.class_offset_embedding.weight[i] = th.tensor(o)
        if rel_centre is not None:
            module.relation_center_embedding.weight[0] = th.tensor(rel_centre)
            module.relation_offset_embedding.weight[0] = th.tensor(rel_offset)


def test_joint_initialisation(dataset):
    module = _model(dataset, formulation="reference").module
    rows = th.cat([module.class_center_embedding.weight,
                   module.class_offset_embedding.weight], dim=1)
    assert th.allclose(th.linalg.norm(rows, dim=1), th.ones(len(rows)), atol=1e-5)
    # The default initialises centre and offset to unit norm each.
    module = _model(dataset).module
    assert th.allclose(th.linalg.norm(module.class_offset_embedding.weight, dim=1),
                       th.ones(module.nb_ont_classes), atol=1e-5)
    with pytest.raises(ValueError):
        _model(dataset, formulation="paper")


def test_centre_norm_term_inside_every_axiom(dataset):
    module = _model(dataset, formulation="reference").module
    module.margin = 0.0
    # Two identical boxes: inclusion holds, only the centre-norm terms remain.
    _set(module, [[2.0, 0.0], [2.0, 0.0]], [[0.5, 0.5], [0.5, 0.5]])
    loss = module(th.tensor([[0, 1]]), "gci0")
    assert float(loss) == pytest.approx(2.0)          # |2 - 1| + |2 - 1|
    _set(module, [[1.0, 0.0], [1.0, 0.0]], [[0.5, 0.5], [0.5, 0.5]])
    assert float(module(th.tensor([[0, 1]]), "gci0")) == pytest.approx(0.0, abs=1e-6)


def test_existential_minkowski_difference_and_negative(dataset):
    module = _model(dataset, formulation="reference").module
    module.margin = 0.0
    # c - d - r = 0; offsets o_C + o_D - o_r = 0.1 + 0.3 - 0.2 = 0.2 per dimension.
    _set(module, [[0.6, 0.8], [0.6, 0.8]], [[0.1, 0.1], [0.3, 0.3]],
         rel_centre=[0.0, 0.0], rel_offset=[0.2, 0.2])
    row = th.tensor([[0, 0, 1]])
    expected = float(th.linalg.norm(th.tensor([0.2, 0.2])))
    assert float(module(row, "gci2")) == pytest.approx(expected, abs=1e-6)
    # mOWL's default clamps the offset of ∃r_all.D at 0: |c - c_r - c_D| + o_C = 0.1.
    default = _model(dataset).module
    default.margin = 0.0
    _set(default, [[0.6, 0.8], [0.6, 0.8]], [[0.1, 0.1], [0.3, 0.3]],
         rel_centre=[0.0, 0.0], rel_offset=[0.2, 0.2])
    assert float(default(row, "gci2")) == pytest.approx(
        float(th.linalg.norm(th.tensor([0.1, 0.1]))), abs=1e-6)
    # Negative: overlap o_C + o_D + o_r - |c - d - r| = 0.6 per dimension (unsquared).
    assert float(module(row, "gci2", neg=True)) == pytest.approx(
        float(th.linalg.norm(th.tensor([0.6, 0.6]))), abs=1e-6)


def test_conjunction_needs_overlap_only(dataset):
    module = _model(dataset, formulation="reference").module
    module.margin = 0.0
    # C ∩ D = [-0.5, 0.5]^2 (centre 0, half-width 0.5); E is a small box at 0.6 that
    # overlaps the intersection but does not contain it: only the centre-norm terms
    # remain.
    _set(module, [[0.0, 0.0], [0.0, 0.0], [0.6, 0.6]],
         [[0.5, 0.5], [0.5, 0.5], [0.2, 0.2]])
    row = th.tensor([[0, 1, 2]])
    norms = sum(abs(float(th.linalg.norm(module.class_center_embedding.weight[i])) - 1)
                for i in range(3))
    assert float(module(row, "gci1")) == pytest.approx(norms, abs=1e-6)
    default = _model(dataset).module
    default.margin = 0.0
    _set(default, [[0.0, 0.0], [0.0, 0.0], [0.6, 0.6]],
         [[0.5, 0.5], [0.5, 0.5], [0.2, 0.2]])
    assert float(default(row, "gci1")) > 0       # mOWL requires inclusion


def test_code_training_regime():
    with pytest.raises(TypeError):
        _model(tiny_dataset()).training_regime = TransBoxCodeTraining()
    model = _model(tiny_dataset(), formulation="reference")
    regime = TransBoxCodeTraining(batch_size=4, margin=0.0, neg_dist=1.0,
                                  validate_every=2)
    model.training_regime = regime
    assert [r.key for r in regime.regimes] == [
        "per-normal-form-minibatch", "per-column-corruption", "slack-inclusion-margin",
        "mse-to-target", "validation-rank-selection"]
    assert regime.regimes[3].neg_target == 1.0
    model.train(epochs=4)
    assert regime.regimes[-1].best_epoch is not None
    for p in model.module.parameters():
        assert th.isfinite(p).all()


def test_reference_candidate_order_decides_ties():
    from mowl.evaluation.normal_form_ranking import (NormalFormRankingRegime,
                                                     TransBoxNormalFormRankingRegime)
    from mowl.evaluation.axiom_ranking import candidate_indices
    from mowl.models import ELBE
    from tests.benchmarks.tiny import NS
    ds = tiny_dataset()
    model = ELBE(ds, embed_dim=2,
                 model_filepath=os.path.join(scratch_dir(), "elbe-order.pt"))
    with th.no_grad():                       # every candidate ties
        model.module.class_embed.weight.zero_()
    names = ["F", "H", "A", "D", "G", "B", "C", "E", "P1", "P2", "P3"]
    order = [f"{NS}{n}" for n in names]
    idx = candidate_indices(model, order, keep_order=True)
    assert idx.tolist() == [model.class_index_dict[i] for i in order]
    ds.reference_class_order = lambda: list(order)
    regime = TransBoxNormalFormRankingRegime(normal_forms=("gci1",), combined=False)
    assert regime.candidate_order == "reference"
    res = regime.evaluate(model, ds)
    # D ⊓ G ⊑ ?F: F is first in the reference order.
    assert regime.info["candidate_order"] == "reference"
    assert regime.info["num_candidates"] == len(order)
    assert res["gci1"]["hits@1"] == 1.0
    ds.reference_class_order = lambda: list(reversed(order))
    res = regime.evaluate(model, ds)
    # Reversed: F is last; D and G are excluded (score inf, after all ties).
    assert res["gci1"]["med"] == len(order) - 2
    assert NormalFormRankingRegime().candidate_order == "model"
    with pytest.raises(ValueError):
        NormalFormRankingRegime(candidate_order="alphabetical")


def test_elbe_options_of_the_transbox_code(dataset):
    from mowl.models import ELBE
    path = os.path.join(scratch_dir(), "elbe-opts.pt")
    model = ELBE(dataset, embed_dim=2, model_filepath=path, gci3_form="inclusion",
                 init="joint")
    module = model.module
    rows = th.cat([module.class_embed.weight, module.class_offset.weight], dim=1)
    assert th.allclose(th.linalg.norm(rows, dim=1), th.ones(len(rows)), atol=1e-5)
    module.margin = 0.0
    default = ELBE(dataset, embed_dim=2, model_filepath=path).module
    default.margin = 0.0
    for m in (module, default):
        with th.no_grad():
            m.rel_embed.weight[0] = 0.0
            m.class_embed.weight[:2] = 0.0
            m.class_offset.weight[0] = 1.0     # C is larger than D: overlap, no inclusion
            m.class_offset.weight[1] = 0.5
    row = th.tensor([[0, 0, 1]])
    assert float(default(row, "gci3").sum()) == pytest.approx(0.0, abs=1e-6)
    assert float(module(row, "gci3").sum()) > 0
    with pytest.raises(ValueError):
        ELBE(dataset, embed_dim=2, model_filepath=path, gci3_form="volume")


def test_nf2_pairwise_option():
    from mowl.baselines.nn.el.transbox.code_losses import gci1_terms
    model = _model(tiny_dataset(), formulation="reference")
    regime = TransBoxCodeTraining(batch_size=4, margin=0.0, nf2_pairwise=True,
                                  select=False)
    model.training_regime = regime
    regime.training_datasets(model.training_datasets)
    regime.on_epoch_start(0, 1)
    rows = regime._datasets["gci1"][:]
    m = model.module
    p, q = gci1_terms(rows, m.class_center, m.class_offset, m.margin)
    expected = ((p[:, None] + q[None, :]) ** 2).mean()
    assert float(regime.loss("gci1", m(rows, "gci1"), None)) == pytest.approx(
        float(expected))
    assert th.allclose(m(rows, "gci1").reshape(-1), p + q)
