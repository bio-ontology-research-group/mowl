"""BoxE, ExpressivE and RESHUFFLE on a projected ontology (through GraphPlusPyKEENModel)."""
import pytest
import torch as th

import mowl
mowl.init_jvm("1g")
from mowl.models import BoxE, ExpressivE, GraphPlusPyKEENModel, Reshuffle  # noqa: E402
from mowl.nn.graph.expressive import ExpressivE as ExpressivEKGE  # noqa: E402
from mowl.nn.graph.reshuffle import Reshuffle as ReshuffleKGE  # noqa: E402
from pykeen.models import BoxE as PyKEENBoxE  # noqa: E402
from tests.models.kg_region_data import abox_dataset  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

CASES = [
    (BoxE, PyKEENBoxE, dict(embed_dim=6)),
    (ExpressivE, ExpressivEKGE, dict(embed_dim=6)),
    (Reshuffle, ReshuffleKGE, dict(l=5, k=6, num_layers=2, num_negs_per_pos=4)),
]


@pytest.fixture(scope="module")
def dataset():
    return abox_dataset()


@pytest.mark.parametrize("cls,kge_cls,kwargs", CASES)
def test_the_model_is_the_shared_pykeen_path(dataset, cls, kge_cls, kwargs):
    model = cls(dataset, batch_size=64, **kwargs)
    assert isinstance(model, GraphPlusPyKEENModel)
    assert isinstance(model.kge_method, kge_cls)
    assert model.axiom_scoring == "method"
    assert {"gci0", "gci2"} <= set(model.scored_forms)


@pytest.mark.parametrize("cls,kge_cls,kwargs", CASES)
def test_the_projection_has_class_level_and_individual_level_triples(dataset, cls,
                                                                     kge_cls, kwargs):
    model = cls(dataset, batch_size=64, **kwargs)
    relations = set(model.graph_relation_to_id)
    assert model.SUBCLASS_RELATION in relations
    assert any(r.endswith("#r") for r in relations)      # C ⊑ ∃r.D and r(a, b)
    assert any(r.endswith("#s") for r in relations)      # a role that occurs in the ABox only
    nodes = set(model.graph_node_to_id)
    assert any(n.endswith("#a1") for n in nodes)         # individuals are entities


@pytest.mark.parametrize("cls,kge_cls,kwargs", CASES)
def test_training_runs_and_scores_axioms(dataset, cls, kge_cls, kwargs):
    th.manual_seed(0)
    model = cls(dataset, batch_size=64, **kwargs)
    model.train(epochs=3)
    idx = model.class_index_dict
    names = {n.rsplit("#", 1)[-1]: i for n, i in idx.items()}
    rows = th.tensor([[names["A"], names["B"]], [names["B"], names["A"]]])
    costs = model.axiom_costs("gci0", rows)
    assert costs.shape == (2,) and th.isfinite(costs).all()
    role = {n.rsplit("#", 1)[-1]: i for n, i in model.object_property_index_dict.items()}
    gci2 = th.tensor([[names["A"], role["r"], names["D"]]])
    assert th.isfinite(model.axiom_costs("gci2", gci2)).all()
    with pytest.raises(Exception):
        model.axiom_costs("gci1", rows)


@pytest.mark.parametrize("cls,kge_cls,kwargs", CASES)
def test_training_separates_the_triples_of_the_graph_from_the_others(dataset, cls, kge_cls,
                                                                      kwargs):
    th.manual_seed(1)
    model = cls(dataset, batch_size=64, **kwargs)
    factory, kge = model.triples_factory, model.kge_method
    n, r = factory.num_entities, factory.num_relations
    everything = th.cartesian_prod(th.arange(n), th.arange(r), th.arange(n))
    known = set(map(tuple, factory.mapped_triples.tolist()))
    unknown = everything[th.tensor([tuple(x) not in known for x in everything.tolist()])]

    def gap():
        with th.no_grad():
            return (kge.score_hrt(factory.mapped_triples).mean()
                    - kge.score_hrt(unknown).mean()).item()

    before = gap()
    model.lr = 0.05
    model.train(epochs=30)
    assert gap() > max(before, 0.0)


def test_reshuffle_defaults_are_those_of_the_authors_best_configuration(dataset):
    model = Reshuffle(dataset, l=4, k=5, num_layers=1)
    assert model.num_negs_per_pos == 100 and model.lr == 5e-3
    kge = model.kge_method
    assert (kge.l, kge.k, kge.num_layers) == (4, 5, 1)


def test_expressive_options_reach_the_interaction(dataset):
    model = ExpressivE(dataset, embed_dim=4, min_denom=0.3, p=1, tanh_map=False)
    interaction = model.kge_method.interaction
    assert (interaction.min_denom, interaction.p, interaction.tanh_map) == (0.3, 1, False)


@pytest.mark.parametrize("method,kwargs,scoring", [
    ("boxe", {"embed_dim": 6}, "model"),
    ("expressive", {"embed_dim": 6}, "model"),
    ("reshuffle", {"l": 5, "k": 6, "num_layers": 2, "num_negs_per_pos": 4}, "model"),
    ("concept2box", {"embed_dim": 6}, "loss"),
])
def test_the_methods_take_part_in_the_joint_ranking_benchmark(method, kwargs, scoring):
    """The common filtered ranking regime ranks C ⊑ ? with each of the four models."""
    from mowl.benchmarks import supports
    from mowl.benchmarks.comparison import compare_methods
    from tests.benchmarks.test_common import _bench
    from tests.benchmarks.tiny import scratch_dir, tiny_dataset
    assert supports(method, "common-go-subsumption-prediction")
    outcome = compare_methods(_bench("gci0"), methods=[method], epochs=2, seeds=(0,),
                              model_kwargs=kwargs, dataset=tiny_dataset(),
                              workdir=scratch_dir(), stop_on_error=True)
    (result,) = outcome.results
    assert 0.0 < result.metrics["all"]["f_mrr"] <= 1.0
    assert result.protocol_info["scoring"] == scoring


def test_boxe_is_pykeens_model_trained_through_the_shared_path(dataset):
    """The BoxE of mOWL is PyKEEN's, trained by GraphPlusPyKEENModel on the projection;
    it scores C ⊑ D and C ⊑ ∃r.D with finite values."""
    th.manual_seed(0)
    model = BoxE(dataset, embed_dim=6, batch_size=64)
    assert type(model.kge_method) is PyKEENBoxE
    assert type(model).__mro__[1].__name__ == "RegionKGEModel"
    assert isinstance(model, GraphPlusPyKEENModel)
    model.train(epochs=3)
    names = {n.rsplit("#", 1)[-1]: i for n, i in model.class_index_dict.items()}
    role = {n.rsplit("#", 1)[-1]: i for n, i in model.object_property_index_dict.items()}
    gci0 = th.tensor([[names["A"], names["B"]], [names["B"], names["C"]],
                      [names["A"], names["E"]]])
    gci2 = th.tensor([[names["A"], role["r"], names["D"]], [names["B"], role["r"], names["E"]]])
    for form, rows in (("gci0", gci0), ("gci2", gci2)):
        costs = model.axiom_costs(form, rows)
        assert costs.shape == (len(rows),) and th.isfinite(costs).all()
