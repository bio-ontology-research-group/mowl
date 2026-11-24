"""End-to-end tests for the BinaryCodeBox model on projected ontology graphs."""
import os

import mowl
mowl.init_jvm("10g")
import mowl.datasets  # noqa: F401,E402
import pytest  # noqa: E402
import torch as th  # noqa: E402

from mowl.models.bcbox.model import BinaryCodeBox, best_f1  # noqa: E402
from mowl.projection.base import ProjectionModel  # noqa: E402
from mowl.projection.edge import Edge  # noqa: E402
from tests.datasetFactory import FamilyDataset  # noqa: E402

FATHER, PARENT, PERSON, MALE = ("http://Father", "http://Parent", "http://Person",
                                "http://Male")


class _FixedProjector(ProjectionModel):
    """Projector that returns a fixed edge list (a cyclic, non-taxonomic relation)."""

    def __init__(self, edges):
        super().__init__()
        self._fixed = edges

    def project(self, ontology):
        return [Edge(s, r, d) for s, r, d in self._fixed]


@pytest.fixture(scope="module")
def dataset():
    return FamilyDataset()


@pytest.fixture(scope="module")
def trained(dataset, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("bcbox") / "model.pt")
    th.manual_seed(0)
    model = BinaryCodeBox(dataset, embed_dim=8, variant="vbc", num_negs=8,
                          learning_rate=0.05, model_filepath=path)
    history = model.train(epochs=150)
    return model, history


def test_graph_is_the_atomic_subsumption_graph(trained):
    model, _ = trained
    ids = model.graph_node_to_id
    edges = {tuple(e) for e in model.edge_index.tolist()}
    assert len(edges) == 7
    # mOWL edge Father -> Parent (Father ⊑ Parent) is the paper's edge (Parent, Father)
    assert (ids[PARENT], ids[FATHER]) in edges


def test_training_reduces_loss_and_reconstructs(trained):
    model, history = trained
    assert history[-1] < history[0]
    assert all(th.isfinite(th.tensor(history)))
    energies = model.energy_matrix()
    n = model.num_nodes
    adj = th.zeros(n, n, dtype=th.bool)
    for u, v in model.edge_index.tolist():  # paper (u, v) = mOWL v -> u
        adj[v, u] = True
    f1, _ = best_f1(energies, adj)
    assert f1 > 0.8


def test_direction_matches_subsumption(trained):
    model, _ = trained
    down = model.edge_energy([FATHER], [PARENT]).item()
    up = model.edge_energy([PARENT], [FATHER]).item()
    assert down < up
    assert 0.0 < model.score_edges([FATHER], [PERSON]).item() <= 1.0


def test_score_axiom_and_evaluation_adapter(trained, dataset):
    model, _ = trained
    from mowl.owlapi import OWLAPIAdapter
    adapter = OWLAPIAdapter()
    axiom = adapter.create_subclass_of(adapter.create_class(FATHER),
                                       adapter.create_class(PARENT))
    assert model.score(axiom) == pytest.approx(model.score_edges([FATHER], [PARENT]).item())

    cls_idx = {c: i for i, c in enumerate(dataset.classes.as_str)}
    batch = th.tensor([[cls_idx[FATHER], cls_idx[PARENT]]])
    scores = model.evaluation_model(batch, "gci0")
    assert scores.item() == pytest.approx(model.edge_energy([FATHER], [PARENT]).item(), rel=1e-5)
    with pytest.raises(ValueError):
        model.evaluation_model(batch, "gci1")


def test_embeddings_and_pretrained_roundtrip(trained, dataset):
    model, _ = trained
    emb = model.class_embeddings
    assert emb[FATHER].shape == (3 * 8,)  # min, max, vertex code
    assert os.path.exists(model.model_filepath)
    other = BinaryCodeBox(dataset, embed_dim=8, variant="vbc")
    other.from_pretrained(model.model_filepath)
    assert th.allclose(other.module.box_min, model.module.box_min)


@pytest.mark.parametrize("variant", ["gbox", "gbc"])
def test_other_variants_train(dataset, variant):
    th.manual_seed(0)
    model = BinaryCodeBox(dataset, embed_dim=6, variant=variant, num_codes=2, code_reg=1e-3,
                          num_negs=4)
    history = model.train(epochs=20)
    assert history[-1] < history[0]


def test_cyclic_projected_graph(dataset):
    """A projected relation that forms a chordless 3-cycle: VBC-Box reconstructs it,
    G-Box cannot (Corollary 4)."""
    cycle = [(FATHER, "http://r", PARENT), (PARENT, "http://r", MALE),
             (MALE, "http://r", FATHER), (FATHER, "http://other", PERSON)]
    results = {}
    for variant in ("gbox", "vbc"):
        th.manual_seed(0)
        model = BinaryCodeBox(dataset, embed_dim=4, variant=variant, relations=["http://r"],
                              projector=_FixedProjector(cycle), num_negs=16,
                              learning_rate=0.05, self_adversarial=False)
        assert len(model.edge_index) == 3  # "http://other" filtered out
        model.train(epochs=300)
        nodes = [model.graph_node_to_id[x] for x in (FATHER, PARENT, MALE)]
        e = model.energy_matrix()[nodes][:, nodes]
        adj = th.tensor([[False, True, False], [False, False, True], [True, False, False]])
        results[variant] = best_f1(e, adj)[0]
    assert results["gbox"] <= 0.8 + 1e-9
    assert results["vbc"] == 1.0
