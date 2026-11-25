"""Knowledge graph models on a projection are combinations of parts: BoxE, ExpressivE,
RESHUFFLE, and the projection with any PyKEEN model (``graph_kge``, ``gda_projection``).

(a) the named combination reads back as its parts; (b) every part can be swapped (relation
model, class model, interaction, objective, negative sampler, projection, regulariser, training
regime) and the model trains and scores with finite numbers; (c) the catalogue card and the
parts agree. The comparison with the direct implementations is in ``tests/baselines``."""
import pytest
import torch as th

import mowl
mowl.init_jvm("1g")
pytest.importorskip("pykeen")

from mowl.catalog import get_method  # noqa: E402
from mowl.models import (BoxE, ExpressivE, GraphKGEModel, GraphPlusPyKEENModel,  # noqa: E402
                         Reshuffle)
from mowl.projection import (DL2VecProjector, GDAProjector,  # noqa: E402
                             OWL2VecStarProjector, TaxonomyProjector)
from mowl.representation import METHODS, REPRESENTATIONS, Geometry  # noqa: E402
from mowl.representation.library.graph_projection import (PROJECTIONS, GraphView,  # noqa: E402
                                                          projection)
from mowl.representation.library.other_graph import (  # noqa: E402
    BoxERelations, ExpressivERelations, GraphEncodedPoints, MarginRanking, PointClasses,
    PyKEENGeometry, ReshuffleRelations, SelfAdversarial, TripleGeometry, TripleScore,
    UniformCorruption)
from mowl.representation.library.shapes_classes import Points  # noqa: E402
from mowl.training.regime import TrainingRegime  # noqa: E402
from mowl.training.regularization import NormPenalty  # noqa: E402
from tests.models.kg_region_data import abox_dataset  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

TINY = dict(batch_size=64)
BOXE = dict(embed_dim=6, **TINY)
EXPRESSIVE = dict(embed_dim=6, **TINY)
RESHUFFLE = dict(l=5, k=6, num_layers=2, num_negs_per_pos=4, **TINY)


@pytest.fixture(scope="module")
def dataset():
    return abox_dataset()


def _rows(model):
    names = {n.rsplit("#", 1)[-1]: i for n, i in model.class_index_dict.items()}
    roles = {n.rsplit("#", 1)[-1]: i for n, i in model.object_property_index_dict.items()}
    gci0 = th.tensor([[names["A"], names["B"]], [names["B"], names["C"]]])
    gci2 = th.tensor([[names["A"], roles["r"], names["D"]]])
    return gci0, gci2


def _train_and_score(model, role_edges=True, epochs=1):
    """One epoch on tiny data and the scores of axioms, finite."""
    th.manual_seed(0)
    model.train(epochs=epochs)
    gci0, gci2 = _rows(model)
    costs = model.axiom_costs("gci0", gci0)
    assert costs.shape == (2,) and th.isfinite(costs).all()
    if role_edges:
        assert th.isfinite(model.axiom_costs("gci2", gci2)).all()
    for p in model.module.parameters():
        assert th.isfinite(p).all()


# -- (a) the named combination reads back as its parts ------------------------------------------
@pytest.mark.parametrize("cls,kwargs,expected,types", [
    (BoxE, BOXE,
     {"classes": "point", "relations": "head and tail boxes", "measure": "triple score",
      "objective": "self-adversarial negative sampling",
      "negative sampler": "uniform corruption", "projection": "owl2vec_star"},
     (Points, BoxERelations, SelfAdversarial)),
    (ExpressivE, EXPRESSIVE,
     {"classes": "point", "relations": "region over pairs", "measure": "triple score",
      "objective": "self-adversarial negative sampling",
      "negative sampler": "uniform corruption", "projection": "owl2vec_star"},
     (Points, ExpressivERelations, SelfAdversarial)),
    (Reshuffle, RESHUFFLE,
     {"classes": "point", "relations": "matrix", "measure": "triple score",
      "objective": "margin ranking", "negative sampler": "uniform corruption",
      "projection": "owl2vec_star"},
     (GraphEncodedPoints, ReshuffleRelations, MarginRanking)),
])
def test_the_named_combination_reads_back_as_its_parts(dataset, cls, kwargs, expected, types):
    model = cls(dataset, **kwargs)
    module = model.module
    assert isinstance(module, TripleGeometry) and isinstance(module, Geometry)
    for name, value in expected.items():
        assert module.parts[name] == value, name
    classes, relations, loss = types
    assert isinstance(module.class_model, classes)
    assert isinstance(module.relation_model, relations)
    assert isinstance(module.loss, loss)
    assert isinstance(module.measure, TripleScore)
    assert isinstance(module.sampler, UniformCorruption)
    assert module.sampler.sides == ("h", "t")


def test_the_defaults_of_the_published_methods(dataset):
    boxe = BoxE(dataset, **BOXE).module.loss
    assert (boxe.margin, boxe.temperature, boxe.reduction) == (3.0, 2.0, "sum")
    reshuffle = Reshuffle(dataset, **RESHUFFLE).module
    assert (reshuffle.loss.margin, reshuffle.loss.reduction) == (2.0, "mean")
    assert reshuffle.sampler.num_negs_per_pos == 4
    assert reshuffle.class_model.num_layers == 2


def test_the_registered_builders_are_the_definitions_of_the_methods():
    for key, relations, loss, projection_name in (
            ("boxe", "head and tail boxes", SelfAdversarial, "owl2vec_star"),
            ("expressive", "region over pairs", SelfAdversarial, "owl2vec_star"),
            ("graph_kge", "translation", MarginRanking, "owl2vec_star"),
            ("gda_projection", "translation", MarginRanking, "gda")):
        geometry = METHODS[key](5, 2, 0, embed_dim=4)
        assert isinstance(geometry, TripleGeometry), key
        assert geometry.parts["relations"] == relations, key
        assert isinstance(geometry.loss, loss), key
        assert geometry.parts["projection"] == projection_name, key


def test_gda_projection_is_the_graph_model_with_the_gda_projector(dataset):
    model = GraphKGEModel(dataset, method="gda_projection", embed_dim=6, **TINY)
    assert isinstance(model.projector, GDAProjector)
    assert model.module.parts["projection"] == "gda"
    other = GraphKGEModel(dataset, method="graph_kge", embed_dim=6, **TINY)
    assert isinstance(other.projector, OWL2VecStarProjector)
    assert model.module.parts["relations"] == other.module.parts["relations"]


def test_the_pykeen_adapter_reads_the_parts_of_any_pykeen_model(dataset):
    import pykeen.models
    model = GraphPlusPyKEENModel(dataset)
    model.set_projector(GDAProjector())
    model.set_kge_method(pykeen.models.DistMult, embedding_dim=6, random_seed=0)
    assert isinstance(model.module, PyKEENGeometry) and isinstance(model.module, Geometry)
    parts = model.module.parts
    assert parts["measure"] == "triple score" and parts["relations"] == "model of PyKEEN"
    assert parts["projection"] == "GDAProjector"
    assert parts["negative sampler"] == "uniform corruption"
    assert parts["objective"] == "MarginRankingLoss"


# -- (b) swaps -----------------------------------------------------------------------------------
class _FixedTailRegime(TrainingRegime):
    """A training regime of the model: its own corrupted triples (every tail replaced by the
    next entity), its own loss, a count of steps."""

    def __init__(self):
        super().__init__()
        self.steps = self.epochs = 0

    def negatives(self, gci_name, positives):
        negatives = positives.clone().unsqueeze(1)
        negatives[..., 2] = (negatives[..., 2] + 1) % self.model.triples_factory.num_entities
        return negatives

    def loss(self, gci_name, pos_scores, neg_scores):
        return th.relu(neg_scores - pos_scores + 1.0).mean()

    def after_step(self):
        self.steps += 1

    def on_epoch_end(self, epoch, losses):
        self.epochs += 1


@pytest.mark.parametrize("swap", [
    dict(relations="region over pairs"),
    dict(classes="ball"),
    dict(classes="box"),
    dict(loss=MarginRanking(1.0)),
    dict(loss="margin ranking", margin=2.0),
    dict(sampler=UniformCorruption(3, ("t",))),
    dict(regularizers=(NormPenalty(parameters=("class_embed",), target=1.0),)),
    dict(projector="dl2vec"),
    dict(projector=DL2VecProjector(False)),
    dict(reinitialise=True),
    dict(p=1, tanh_map=False),
], ids=lambda s: ",".join(s))
def test_swapping_a_part_of_boxe(dataset, swap):
    model = BoxE(dataset, **BOXE, **swap)
    _train_and_score(model, role_edges="projector" not in swap)


@pytest.mark.parametrize("swap", [
    dict(relations="head and tail boxes (distance score)"),
    dict(classes="ball"),
    dict(loss="margin ranking"),
    dict(loss=SelfAdversarial(1.0, 1.0, "mean")),
    dict(sampler=UniformCorruption(2)),
    dict(projector="taxonomy"),
    dict(min_denom=0.3, p=1),
], ids=lambda s: ",".join(s))
def test_swapping_a_part_of_expressive(dataset, swap):
    model = ExpressivE(dataset, **EXPRESSIVE, **swap)
    _train_and_score(model, role_edges="projector" not in swap)


@pytest.mark.parametrize("swap", [
    dict(loss=SelfAdversarial(2.0, 1.0, "mean")),
    dict(kge_kwargs={"aggregation": "max_self_noLoop"}),
    dict(kge_kwargs={"row_function": "square"}),
    dict(inverse_triples=False),
    dict(sampler=UniformCorruption(2, ("t",))),
    dict(regularizers=(NormPenalty(parameters=("class_embed",), target=1.0),)),
    dict(projector=TaxonomyProjector(False)),
], ids=lambda s: ",".join(s))
def test_swapping_a_part_of_reshuffle(dataset, swap):
    kwargs = dict(RESHUFFLE)
    kwargs.update(swap)
    model = Reshuffle(dataset, **kwargs)
    _train_and_score(model, role_edges="projector" not in swap)


@pytest.mark.parametrize("cls,kwargs", [(BoxE, BOXE), (ExpressivE, EXPRESSIVE),
                                        (Reshuffle, RESHUFFLE)])
def test_a_training_regime_changes_the_negatives_the_loss_and_the_steps(dataset, cls, kwargs):
    model = cls(dataset, **kwargs)
    regime = _FixedTailRegime()
    model.training_regime = regime
    _train_and_score(model, epochs=2)
    assert regime.epochs == 2 and regime.steps >= 2


def test_a_regime_that_the_loop_does_not_consult_is_refused(dataset):
    class Datasets(TrainingRegime):
        def training_datasets(self, datasets):
            return datasets

    with pytest.raises(TypeError):
        BoxE(dataset, **BOXE).training_regime = Datasets()


def test_reshuffle_early_stopping_is_an_evaluation_regime_on_validation_hits(dataset):
    from mowl.training.triples import HitsEarlyStopping
    model = Reshuffle(dataset, early_stopping=dict(frequency=1, patience=1,
                                                   relative_delta=0.0), **RESHUFFLE)
    assert isinstance(model.training_regime, HitsEarlyStopping)
    # the tiny dataset has no validation ontology: the regime says so instead of ignoring it
    with pytest.raises(ValueError, match="validation"):
        model.train(epochs=2)
    regime = HitsEarlyStopping(frequency=1, patience=1, relative_delta=0.0)
    model.training_regime = regime     # attaching forgets the validation triples
    regime._val = model.triples_factory.mapped_triples[:3]
    hits = regime.hits_at_k()
    assert 0.0 <= hits <= 1.0
    model.train(epochs=3)
    assert 1 <= len(regime.history) <= 3


@pytest.mark.parametrize("kge,projector", [
    ("TransE", OWL2VecStarProjector), ("DistMult", DL2VecProjector),
    ("BoxE", OWL2VecStarProjector), ("TransE", GDAProjector),
    ("TransE", TaxonomyProjector)])
def test_swapping_the_kge_model_or_the_projection_of_graph_plus_pykeen(dataset, kge, projector):
    import pykeen.models
    model = GraphPlusPyKEENModel(dataset)
    model.set_projector(projector())
    model.set_kge_method(getattr(pykeen.models, kge), embedding_dim=6, random_seed=0)
    model.optimizer, model.lr, model.batch_size = th.optim.Adam, 0.01, 64
    model.num_negs_per_pos = 2
    _train_and_score(model, role_edges=projector is not TaxonomyProjector)


def test_the_general_graph_model_with_other_relation_models(dataset):
    for relations in ("translation", "head and tail boxes (distance score)",
                      "region over pairs"):
        model = GraphKGEModel(dataset, method="graph_kge", relations=relations, embed_dim=6,
                              **TINY)
        assert model.module.parts["relations"] in ("translation", "head and tail boxes",
                                                   "region over pairs")
        _train_and_score(model)


def test_projection_is_a_part_with_names_classes_and_instances(dataset):
    assert set(PROJECTIONS) >= {"owl2vec_star", "dl2vec", "taxonomy", "gda"}
    assert isinstance(projection("gda"), GDAProjector)
    assert isinstance(projection(TaxonomyProjector), TaxonomyProjector)
    instance = DL2VecProjector(False)
    assert projection(instance) is instance
    assert isinstance(projection(None), OWL2VecStarProjector)
    with pytest.raises(ValueError):
        projection("nonsense")
    model = BoxE(dataset, projector="taxonomy", **BOXE)
    view = GraphView.of(model)
    assert set(view.relations) == {model.SUBCLASS_RELATION} and len(view) > 0
    # the same graph either way
    other = BoxE(dataset, projector=TaxonomyProjector(), **BOXE)
    assert GraphView.of(other).triples.tolist() == view.triples.tolist()


# -- behaviour of the training and of the model ------------------------------------------------
def test_training_continues_from_the_current_parameters_unless_asked_to_start_again(dataset):
    """PyKEEN's loop draws every parameter again at each call (so the seed of the model never
    decides the result and trained or pretrained parameters are lost); the model continues."""
    model = BoxE(dataset, **BOXE)
    table = model.module.class_model.centres.weight
    with th.no_grad():
        table.fill_(0.25)
    model.lr = 0.0
    model.train(epochs=0)
    assert th.equal(table, th.full_like(table, 0.25))
    model.reinitialise = True
    model.train(epochs=0)
    assert not th.equal(model.module.class_model.centres.weight,
                        th.full_like(table, 0.25))


def test_the_direct_implementation_draws_the_parameters_again_at_each_training(dataset):
    """The evidence for the line above: the PyKEEN loop of the direct implementation."""
    from mowl.baselines.models.boxe import BoxE as Direct
    direct = Direct(dataset, **BOXE)
    table = direct.kge_method.entity_representations[0]._embeddings.weight
    with th.no_grad():
        table.fill_(0.25)
    direct.train(epochs=0)
    assert not th.equal(table, th.full_like(table, 0.25))


def test_the_seed_decides_the_initial_parameters(dataset):
    a = BoxE(dataset, seed=3, **BOXE).module.class_model.centres.weight
    b = BoxE(dataset, seed=3, **BOXE).module.class_model.centres.weight
    c = BoxE(dataset, seed=4, **BOXE).module.class_model.centres.weight
    assert th.equal(a, b) and not th.equal(a, c)


def test_saving_and_loading_the_parameters(dataset, tmp_path):
    path = str(tmp_path / "boxe.pt")
    model = BoxE(dataset, model_filepath=path, **BOXE)
    model.train(epochs=1)
    other = BoxE(dataset, seed=9, **BOXE)
    other.from_pretrained(path)
    assert th.equal(other.module.class_model.centres.weight,
                    model.module.class_model.centres.weight)


def test_added_axioms_keep_the_rows_of_the_nodes_that_were_in_the_graph(dataset):
    from tests.models.kg_region_data import abox_dataset as fresh
    ds = fresh()
    model = BoxE(ds, **BOXE)
    model.train(epochs=1)
    before = {n: model.module.class_model.centres.weight[i].detach().clone()
              for n, i in model.triples_factory.entity_to_id.items()}
    from mowl.owlapi import OWLAPIAdapter
    adapter = OWLAPIAdapter()
    a = adapter.create_class("http://mowl/test#A")
    z = adapter.create_class("http://mowl/test#Zed")
    model.add_axioms(adapter.create_subclass_of(a, z))
    after = model.module.class_model.centres.weight
    for name, row in before.items():
        assert th.equal(after[model.triples_factory.entity_to_id[name]], row)
    assert "http://mowl/test#Zed" in model.triples_factory.entity_to_id
    model.train(epochs=1)


def test_reshuffle_scores_triples_over_a_graph_the_model_was_not_trained_on(dataset):
    from mowl.datasets.builtin.inductive_kg import InductiveGraph
    model = Reshuffle(dataset, **RESHUFFLE)
    model.train(epochs=1)
    relations = list(model.triples_factory.relation_to_id)[:2]
    graph = InductiveGraph(entities=["x", "y", "z"], relations=relations,
                           triples=th.tensor([[0, 0, 1], [1, 1, 2]]))
    rows = th.tensor([[0, 0, 1], [2, 1, 0]])
    for target in ("tail", "head"):
        costs = model.inductive_costs(graph, rows, target=target)
        assert costs.shape == (2,) and th.isfinite(costs).all() and (costs >= 0).all()


# -- the parts on their own ------------------------------------------------------------------
_PYKEEN_SIDE = {"h": "head", "r": "relation", "t": "tail"}


def test_uniform_corruption_draws_what_pykeens_basic_sampler_draws():
    from pykeen.sampling import BasicNegativeSampler
    triples = th.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 0], [4, 0, 2]])
    for k, sides in ((1, ("h", "t")), (4, ("h", "t")), (3, ("t",)), (2, ("h", "r", "t"))):
        th.manual_seed(5)
        ours = UniformCorruption(k, sides)(triples, 6, 3)
        th.manual_seed(5)
        theirs = BasicNegativeSampler(
            mapped_triples=triples, num_entities=6, num_relations=3, num_negs_per_pos=k,
            corruption_scheme=[_PYKEEN_SIDE[side] for side in sides]).sample(triples)[0]
        assert th.equal(ours, theirs), (k, sides)
        # never the triple it came from
        assert (ours != triples.unsqueeze(1)).any(dim=-1).all()


def test_the_losses_are_those_of_the_literature():
    from pykeen.losses import MarginRankingLoss, NSSALoss, SoftMarginRankingLoss
    th.manual_seed(0)
    pos, neg = th.randn(7, 1), th.randn(7, 5)
    for reduction in ("sum", "mean"):
        ours = SelfAdversarial(3.0, 2.0, reduction)(pos, neg)
        theirs = NSSALoss(3.0, 2.0, reduction).process_slcwa_scores(pos, neg)
        assert th.allclose(ours, theirs, atol=1e-6), reduction
        ours = MarginRanking(1.5, reduction)(pos, neg)
        theirs = MarginRankingLoss(1.5, reduction).process_slcwa_scores(pos, neg)
        assert th.allclose(ours, theirs, atol=1e-6), reduction
        ours = MarginRanking(1.5, reduction, "softplus")(pos, neg)
        theirs = SoftMarginRankingLoss(1.5, reduction).process_slcwa_scores(pos, neg)
        assert th.allclose(ours, theirs, atol=1e-6), reduction
    with pytest.raises(ValueError):
        MarginRanking(1.0, "median")


def test_the_encoding_is_computed_once_per_batch_and_is_the_same_with_or_without():
    from mowl.representation.library.other_graph import reshuffle
    triples = th.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3]])
    geometry = reshuffle(4, 2, triples=triples, l=4, k=5, num_layers=2)
    heads, relations, tails = th.tensor([0, 1]), th.tensor([0, 2]), th.tensor([1, 2])
    plain = geometry.class_model(heads).centre
    with geometry.encoding():
        shared = geometry.class_model(heads).centre
    assert th.equal(plain, shared)
    assert geometry.triple_cost(heads, relations, tails).shape == (2, 1)
    with pytest.raises(Exception):
        reshuffle(4, 2)         # defined on a graph


# -- (c) the catalogue card and the parts agree -------------------------------------------------
@pytest.mark.parametrize("key,cls,kwargs", [("boxe", BoxE, BOXE),
                                            ("expressive", ExpressivE, EXPRESSIVE),
                                            ("reshuffle", Reshuffle, RESHUFFLE)])
def test_the_catalogue_card_and_the_parts_agree(dataset, key, cls, kwargs):
    card = get_method(key)
    model = cls(dataset, **kwargs)
    module = model.module
    # tasks and expressivity: what the card says the method scores is what the model scores
    assert "subsumption" in card.tasks and "gci0" in model.scored_forms
    assert card.expressivity == "EL with ABox"
    assert {"class_assertion", "object_property_assertion"} <= set(model.scored_forms)
    assert model.axiom_scoring == "method"
    # training regime named by the card: the loss and the corruption of the parts
    if "logsigmoid-negative-sampling" in card.training:
        assert isinstance(module.loss, SelfAdversarial)
    if "pairwise-margin-ranking" in card.training:
        assert isinstance(module.loss, MarginRanking)
    if "either-side-corruption" in card.training:
        assert set(module.sampler.sides) == {"h", "t"}
    # the representation the library states for the method
    representation = REPRESENTATIONS[key]
    relations = representation.relations
    relations = relations if isinstance(relations, (tuple, list)) else (relations,)
    assert module.parts["relations"] in [str(r.value if hasattr(r, "value") else r)
                                         for r in relations]
    measures = representation.measure
    measures = measures if isinstance(measures, (tuple, list)) else (measures,)
    assert module.parts["measure"] in [str(getattr(m, "value", m)) for m in measures]
    # the implementation the card names is the class built from the parts
    module_name, _, name = card.implementation.rpartition(".")
    import importlib
    assert getattr(importlib.import_module(module_name), name) is cls


@pytest.mark.parametrize("key", ["graph_kge", "gda_projection"])
def test_the_cards_of_the_projection_methods_name_the_general_model(key):
    card = get_method(key)
    module_name, _, name = card.implementation.rpartition(".")
    import importlib
    cls = getattr(importlib.import_module(module_name), name)
    assert cls is GraphPlusPyKEENModel
    assert "gci0" in cls.scored_forms and "subsumption" in (card.tasks or ("subsumption",))
    assert "mowl.projection.GDAProjector" in " ".join(get_method("gda_projection").notes
                                                      if isinstance(
                                                          get_method("gda_projection").notes,
                                                          (tuple, list))
                                                      else [get_method("gda_projection").notes])
