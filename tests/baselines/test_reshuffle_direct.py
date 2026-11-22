"""The direct RESHUFFLE module (a PyKEEN model): what only it has (the trained matrices and the
inductive scoring), kept with the direct implementation; the maths is tested in
``tests/nn/test_reshuffle_module.py``."""
import torch as th
from pykeen.triples import TriplesFactory

import mowl
mowl.init_jvm("1g")
import mowl.datasets  # noqa: F401,E402
from mowl.baselines.nn.graph.reshuffle import Reshuffle  # noqa: E402



def _factory():
    triples = th.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 0]])
    return TriplesFactory(mapped_triples=triples, entity_to_id={str(i): i for i in range(4)},
                          relation_to_id={"r": 0, "s": 1}, create_inverse_triples=False)


def test_model_scores_and_only_relation_matrices_are_learned():
    model = Reshuffle(triples_factory=_factory(), l=4, k=5, num_layers=2, random_seed=0)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any("entity_representations" in n for n in trainable)
    assert any("relation_representations" in n for n in trainable)
    batch = th.tensor([[0, 0, 1], [1, 1, 2]])
    assert model.score_hrt(batch).shape == (2, 1)
    assert model.score_t(batch[:, :2]).shape == (2, 4)
    assert model.score_h(batch[:, 1:]).shape == (2, 4)
    assert (model.score_hrt(batch) <= 0).all()


def test_inductive_scoring_on_a_new_graph():
    """The model encodes a graph it has not seen (fresh entities, same relations)."""
    model = Reshuffle(triples_factory=_factory(), l=4, k=5, num_layers=2, random_seed=0)
    graph = th.tensor([[0, 0, 1], [1, 1, 2]])
    out = model.score_on_graph(th.tensor([[0, 0, 1], [2, 1, 0]]), graph, num_entities=3)
    assert out.shape == (2,) and (out <= 0).all()
