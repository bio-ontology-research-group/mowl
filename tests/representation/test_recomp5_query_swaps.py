"""The general parts that named methods of round 5 (HiT, OnT, EIKE, BoxLitE, the query models)
used to compute with themselves, used in a combination other than the method they came from.
That the methods give the numbers of their direct implementations is ``tests/baselines``."""
import numpy as np
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.evaluation.query_scoring import axiom_logits, axiom_query, dnf_logits  # noqa: E402
from mowl.evaluation.text_ranking import best_on_grid, encode_batched  # noqa: E402
from mowl.models import ELEmbeddings, ComposedEL, ConE, BetaE, AConE, HiT  # noqa: E402
from mowl.representation.library.vector_tables import (load_vector_table,  # noqa: E402
                                                       pca_reduce, save_vector_table)
from mowl.training.losses import PairwiseMarginRankingLoss  # noqa: E402
from tests.datasetFactory import FamilyDataset  # noqa: E402


def _one_epoch(model):
    model.train(epochs=1)
    return model


# -- EIKE's ranking loss on other shape methods ----------------------------------------------
@pytest.mark.parametrize("method", ["octagon", "concept2box"])
def test_the_pairwise_ranking_regime_trains_other_shape_methods(method):
    model = ComposedEL(FamilyDataset(), method=method, embed_dim=4, batch_size=64)
    model.training_regime = PairwiseMarginRankingLoss(margin=0.5)
    losses = model.train(epochs=2)
    assert len(losses) >= 1 and np.isfinite(losses).all()


def test_the_pairwise_ranking_regime_trains_a_ball_method():
    model = ELEmbeddings(FamilyDataset(), embed_dim=4)
    model.training_regime = PairwiseMarginRankingLoss(margin=1.0, neg_scores="module")
    model.train(epochs=1)
    assert all(np.isfinite(v.detach().numpy()).all() for v in model.module.parameters())


# -- text_ranking with an encoder that is not the bag of words ----------------------------------
class CharacterEncoder(th.nn.Module):
    """Maps a text to the counts of a few letters, passed through a learnable linear layer."""
    dim = 8

    def __init__(self):
        super().__init__()
        self.linear = th.nn.Linear(26, 8)

    def forward(self, texts):
        counts = th.zeros(len(texts), 26)
        for i, text in enumerate(texts):
            for ch in text.lower():
                if "a" <= ch <= "z":
                    counts[i, ord(ch) - 97] += 1
        return th.tanh(self.linear(counts / 10))


def test_text_ranking_serves_hit_with_another_encoder():
    model = HiT(FamilyDataset(), encoder=CharacterEncoder(), batch_size=8, num_negatives=1,
                seed=0, centri_weight=0.0)
    n = len(model.class_index_dict)
    rows = th.cartesian_prod(th.arange(n), th.arange(n))
    costs = model.axiom_costs("gci0", rows)
    assert costs.shape == (n * n,) and th.isfinite(costs).all()
    scores = model.score_candidates("gci0", rows[:4], 1, th.arange(n), chunk_size=3)
    assert scores.shape == (4, n)
    mrr, lam = model.validate(rows[:6], grid=(0.0, 0.5))
    assert 0.0 <= mrr <= 1.0 and lam in (0.0, 0.5)


def test_encode_batched_and_the_grid_work_on_any_scorer():
    module = CharacterEncoder()
    out = encode_batched(module, ["a b", "cd", "e"], 2, module, 8, "cpu")
    assert out.shape == (3, 8) and not out.requires_grad
    table = th.eye(3)
    mrr, value = best_on_grid(lambda v, a, b: table[a:b] * (1 + v), th.arange(3), (0.0, 1.0))
    assert mrr == 1.0 and value == 0.0


# -- query_scoring with other region families ------------------------------------------------------
@pytest.mark.parametrize("cls", [BetaE, ConE, AConE])
def test_query_scoring_serves_every_region_family(cls):
    model = cls(FamilyDataset(), embed_dim=8, batch_size=8, negatives=2, seed=0)
    n = model.num_classes
    rows = th.stack([th.arange(5) % n, (th.arange(5) + 1) % n], 1)
    sig, ids, subjects = axiom_query("gci0", rows, n, True)
    model.module.eval()
    with th.no_grad():
        direct = axiom_logits(model.module, sig, ids, subjects, chunk=2)
        again = -model.axiom_costs("gci0", rows)
        node = model._concept(int(rows[0, 1]))
        every = dnf_logits(model.module, [node], model.graph.num_entities, 8, "cpu")
    assert th.allclose(direct, again, atol=1e-5)
    assert every.shape == (1, model.graph.num_entities)
    assert th.isclose(every[0, int(rows[0, 0])], direct[0], atol=1e-5)


# -- vector_tables with another method ---------------------------------------------------------
def test_vector_tables_initialise_another_method(tmp_path):
    model = ELEmbeddings(FamilyDataset(), embed_dim=4)
    rng = np.random.default_rng(0)
    vectors = {iri: rng.normal(size=16) for iri in model.class_index_dict}
    path = tmp_path / "classes.npz"
    save_vector_table(path, pca_reduce(vectors, 4))
    table = load_vector_table(path)
    weight = model.module.class_model.centres.weight
    with th.no_grad():
        for iri, i in model.class_index_dict.items():
            weight[i] = th.tensor(table[iri])
    assert np.allclose(weight[model.class_index_dict[next(iter(vectors))]].detach().numpy(),
                       table[next(iter(vectors))], atol=1e-6)
    _one_epoch(model)
    assert np.isfinite(weight.detach().numpy()).all()
