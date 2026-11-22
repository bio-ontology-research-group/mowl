"""The general parts that the box and ball methods use instead of computing in their classes."""
import numpy as np
import torch as th

from mowl.evaluation.region_access import concatenated_rows
from mowl.evaluation.scoring_rules import exp_negative_score
from mowl.evaluation.sel_probability import mean_interval_error
from mowl.representation.library import ballfam_energy as D
from mowl.representation.library.multibox_modules import in_chunks
from mowl.representation.library.structural_regions import pad_slots
from mowl.nn.el.structural import Regions
from mowl.training.batching import epoch_rng
from mowl.training.losses import MSETargetLoss


def test_epoch_rng_is_seed_plus_epoch():
    a = epoch_rng(3, 4).random(5)
    assert np.allclose(a, np.random.default_rng(7).random(5))
    assert not np.allclose(a, epoch_rng(3, 5).random(5))


def test_in_chunks_equals_all_at_once():
    rows = th.arange(10).reshape(5, 2).float()
    f = lambda r: r.sum(dim=1, keepdim=True) ** 2  # noqa: E731
    assert th.equal(in_chunks(f, rows, 2), f(rows))
    assert th.equal(in_chunks(f, rows, None), f(rows))
    assert th.equal(in_chunks(f, rows, 99), f(rows))


def test_pad_slots_adds_inactive_slots():
    r = Regions(cen=th.ones(2, 1, 3), w=th.ones(2, 1, 3), active=th.ones(2, 1, dtype=th.bool))
    p = pad_slots(r, 3)
    assert p.cen.shape == (2, 3, 3) and p.active.tolist() == [[True, False, False]] * 2
    assert (p.w[:, 1:] == -1).all() and (p.cen[:, 1:] == 0).all()
    assert pad_slots(r, 1) is r


def test_exp_negative_score_is_one_for_zero_loss_and_monotone():
    s = exp_negative_score(th.tensor([0.0, 1.0, 2.0]))
    assert s[0] == 1.0 and s[0] > s[1] > s[2] > 0


def test_concatenated_rows_joins_columns():
    out = concatenated_rows([th.zeros(2, 3, requires_grad=True), th.ones(2, 2)])
    assert out.shape == (2, 5) and isinstance(out, np.ndarray)


def test_mean_interval_error():
    assert mean_interval_error([]) == 0.0
    assert mean_interval_error([0.0, 0.5]) == 0.25


def test_energy_tables_cover_both_shapes():
    c, o = th.zeros(2, 3), th.ones(2, 3)
    for shape in ("ball", "box"):
        size = o[:, 0] if shape == "ball" else o
        assert D.BOUNDARY[shape](c, size, c, size).shape == (2,)
        assert D.ENERGY[shape](c, size, c, size).shape == (2,)


def test_mse_target_loss_part_as_elbe_and_as_another_swap():
    pos, neg = th.rand(4, 1), th.rand(8, 1)
    elbe = MSETargetLoss(0.0, 1.0, reduce="mean_first").combine(None, pos, neg)
    assert th.isclose(elbe, pos.mean() ** 2 + (neg.mean() - 1) ** 2)
    other = MSETargetLoss(0.0, 2.0, reduce="per_axiom").combine(None, pos, neg)
    assert th.isfinite(other)
