"""Evaluation regimes read class regions through the common interface
(``mowl.representation.geometry_of``), not by attribute names.

The ``_legacy_*`` functions below are the attribute-reading code that the regimes used
before; they pin that the numbers of ELEmbeddings, ELBE and Box2EL did not change.
"""
import numpy as np
import pytest
import torch as th

import mowl.datasets  # noqa: F401  (JVM; mowl.nn must not be imported first)
from mowl.evaluation import exact_model
from mowl.evaluation.el_scoring import empty_intersection
from mowl.evaluation.inclusion_accuracy import inclusion_mask
from mowl.evaluation.regions import BallRegion, BoxRegion
from mowl.evaluation.sampled_f1 import (_score, candidate_dists, class_regions,
                                        pair_scores)
from tests.datasetFactory import FamilyDataset


def _model(name, seed=0):
    import mowl.models as M
    th.manual_seed(seed)
    model = getattr(M, name)(FamilyDataset(), embed_dim=4)
    with th.no_grad():
        for p in model.module.parameters():
            p.add_(0.5 * th.randn_like(p))
    return model


# -- the code that read attribute names (before) ----------------------------------------
def _centres(module):
    # Box2EL names its centres ``class_center`` (``class_embed`` is None there).
    return (module.class_center if module.class_embed is None else module.class_embed).weight


def _legacy_class_regions(module):
    if hasattr(module, "class_offset"):
        return _centres(module), module.class_offset.weight.abs(), False
    return _centres(module), module.class_rad.weight.abs().squeeze(-1), True


def _legacy_candidates(module, gci_name, rows, score="boundary"):
    centers, sizes, is_ball = _legacy_class_regions(module)
    if gci_name == "gci0":
        child = centers[rows[:, 0]], sizes[rows[:, 0]]
    else:
        child = centers[rows[:, 2]] - module.rel_embed.weight[rows[:, 1]], sizes[rows[:, 2]]
    out = _score(child[0][:, None], child[1][:, None], centers[None], sizes[None], is_ball,
                 score)
    out.scatter_(1, rows[:, 0:1], float("inf"))
    return out


def _legacy_inclusion(module, rows, tol=1e-6):
    c = _centres(module).detach()
    sub, sup = rows[:, 0], rows[:, 1]
    if hasattr(module, "class_rad"):
        r = module.class_rad.weight.detach().abs().reshape(-1)
        dist = th.linalg.norm(c[sub] - c[sup], dim=1)
        return (dist + r[sub] <= r[sup] + tol).numpy()
    o = module.class_offset.weight.detach().abs()
    lo, hi = (c - o).double().numpy(), (c + o).double().numpy()
    s, p = sub.numpy(), sup.numpy()
    return ((lo[p] <= lo[s] + tol) & (hi[s] <= hi[p] + tol)).all(axis=1)


def _legacy_empty(module, rows):
    centre, half, _ = _legacy_class_regions(module)
    if _legacy_class_regions(module)[2]:
        half = half[:, None].expand(-1, centre.shape[1])
    c, d = rows[:, 0], rows[:, 1]
    lower = th.maximum(centre[c] - half[c], centre[d] - half[d])
    upper = th.minimum(centre[c] + half[c], centre[d] + half[d])
    return (lower > upper).any(dim=1)


ROWS0 = th.tensor([[0, 1], [1, 2], [3, 4], [5, 0], [2, 7]])
ROWS2 = th.tensor([[0, 0, 1], [3, 0, 2], [4, 0, 7], [6, 0, 5]])  # one role in the family data
ROWS1 = th.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 0]])


@pytest.mark.parametrize("name", ["ELEmbeddings", "ELBE", "BoxSquaredEL"])
def test_class_regions_equal_the_attribute_readers(name):
    module = _model(name).module
    for new, old in zip(class_regions(module), _legacy_class_regions(module)):
        assert th.equal(th.as_tensor(new), th.as_tensor(old))


# Before, sampled F1 read ``class_embed``, which Box2EL does not have (it failed there), and
# ``rel_embed``, which is not Box2EL's relation: it now works through the geometry, and for
# the methods that worked the scores are pinned here.
@pytest.mark.parametrize("name,forms", [("ELEmbeddings", ("gci0", "gci2")),
                                        ("ELBE", ("gci0", "gci2"))])
@pytest.mark.parametrize("score", ["boundary", "energy"])
def test_sampled_f1_scores_are_unchanged(name, forms, score):
    module = _model(name).module
    for form in forms:
        rows = ROWS0 if form == "gci0" else ROWS2
        with th.no_grad():
            new = candidate_dists(module, form, rows, score=score)
            old = _legacy_candidates(module, form, rows, score=score)
        assert th.equal(new, old.cpu())


@pytest.mark.parametrize("name", ["ELEmbeddings", "ELBE", "BoxSquaredEL"])
def test_inclusion_mask_is_unchanged(name):
    model = _model(name)
    for scale in (0.0, 1.0, 3.0):  # nothing nested / some nested / most nested
        with th.no_grad():
            if hasattr(model.module, "class_rad"):
                model.module.class_rad.weight.mul_(scale + 1)
            else:
                model.module.class_offset.weight.mul_(scale + 1)
        new = inclusion_mask(model, ROWS0)
        assert new.tolist() == _legacy_inclusion(model.module, ROWS0).tolist()


@pytest.mark.parametrize("name", ["ELEmbeddings", "ELBE", "BoxSquaredEL"])
def test_empty_intersection_is_unchanged(name):
    model = _model(name)
    assert th.equal(empty_intersection(model, ROWS1), _legacy_empty(model.module, ROWS1))


def _legacy_exact_regions(module):
    if hasattr(module, "class_rad"):
        c = module.class_embed.weight.detach().double().numpy()
        r = np.abs(module.class_rad.weight.detach().double().numpy()).reshape(-1)
        return [BallRegion(a, b) for a, b in zip(c, r)]
    names = {"BoxSquaredELModule": ("class_center", "class_offset")}
    centre, offset = names.get(type(module).__name__, ("class_embed", "class_offset"))
    c = getattr(module, centre).weight.detach().double().numpy()
    o = np.abs(getattr(module, offset).weight.detach().double().numpy())
    return [BoxRegion(a - b, a + b) for a, b in zip(c, o)]


@pytest.mark.parametrize("name", ["ELEmbeddings", "ELBE", "BoxSquaredEL", "RegDELBE",
                                  "RegDELEm"])
def test_exact_model_regions_are_unchanged(name):
    model = _model(name)
    new = exact_model.extract_regions(model)
    index = model.class_index_dict
    if name.startswith("RegD"):
        with th.no_grad():
            c, s = model.module._region(th.arange(len(index)))
        c = c.double().numpy()
        s = s.double().numpy()
        old = [BallRegion(a, b) for a, b in zip(c, s.reshape(-1))] if name == "RegDELEm" \
            else [BoxRegion(a - b, a + b) for a, b in zip(c, s)]
    else:
        old = _legacy_exact_regions(model.module)
    for iri, i in index.items():
        if name == "RegDELEm" or hasattr(model.module, "class_rad"):
            assert np.array_equal(new[iri].center, old[i].center)
            assert new[iri].radius == old[i].radius
        else:
            assert np.array_equal(new[iri].lo, old[i].lo)
            assert np.array_equal(new[iri].hi, old[i].hi)


# -- methods that did not work before ---------------------------------------------------
def test_sampled_f1_works_for_box2el_relations_and_transbox():
    for name in ("BoxSquaredEL", "TransBox", "BoxSEL"):
        module = _model(name).module
        for form, rows in (("gci0", ROWS0), ("gci2", ROWS2)):
            with th.no_grad():
                dists = candidate_dists(module, form, rows)
                pair = pair_scores(module, form, rows)
            assert dists.shape == (len(rows), 8)
            assert th.isfinite(pair).all()


def test_regions_that_are_not_balls_or_boxes_give_a_clear_error():
    module = _model("MultiELBE").module
    with pytest.raises(TypeError, match="UnionOfBoxes"):
        candidate_dists(module, "gci0", ROWS0)
    with pytest.raises(TypeError, match="UnionOfBoxes"):
        inclusion_mask(_model("MultiELBE"), ROWS0)
    assert empty_intersection(_model("MultiELBE"), ROWS1) is None


def test_modules_without_a_geometry_give_a_clear_error():
    class Bare(th.nn.Module):
        pass
    with pytest.raises(NotImplementedError, match="Bare has none"):
        class_regions(Bare())
