"""Knowledge graph models in the common interface (group ``other``)."""
import pytest
import torch as th

import mowl
mowl.init_jvm("1g")
pytest.importorskip("pykeen")

from mowl.models import BoxE, ComposedEL, ExpressivE, Reshuffle  # noqa: E402
from mowl.representation import (ComposedGeometry, ConstructorUnavailable,  # noqa: E402
                                 Objective, geometry_of)
from mowl.representation.library.other_graph import (  # noqa: E402
    BoxERelations, ExpressivERelations, PointClasses, PyKEENGeometry, TripleGeometry,
    TripleScore)
from tests.datasetFactory import FamilyDataset  # noqa: E402
from tests.models.kg_region_data import abox_dataset  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

CASES = [(BoxE, dict(embed_dim=6)), (ExpressivE, dict(embed_dim=6)),
         (Reshuffle, dict(l=5, k=6, num_layers=2, num_negs_per_pos=4))]


@pytest.fixture(scope="module")
def dataset():
    return abox_dataset()


def _trained(cls, kwargs, dataset):
    th.manual_seed(0)
    model = cls(dataset, batch_size=64, **kwargs)
    model.train(epochs=2)
    return model


def _names(model):
    return {n.rsplit("#", 1)[-1]: i for n, i in model.class_index_dict.items()}


@pytest.mark.parametrize("cls,kwargs", CASES)
def test_costs_are_the_models_own(dataset, cls, kwargs):
    model = _trained(cls, kwargs, dataset)
    geometry = geometry_of(model)
    assert isinstance(geometry, PyKEENGeometry)
    names = _names(model)
    roles = {n.rsplit("#", 1)[-1]: i for n, i in model.object_property_index_dict.items()}
    gci0 = th.tensor([[names["A"], names["B"]], [names["B"], names["A"]],
                      [names["E"], names["C"]]])
    got = geometry.inside(geometry.classes(gci0[:, 0]), geometry.classes(gci0[:, 1]))
    assert got.shape == (3, 1)
    assert th.allclose(got[:, 0], model.axiom_costs("gci0", gci0), atol=1e-5)
    gci2 = th.tensor([[names["A"], roles["r"], names["D"]],
                      [names["B"], roles["r"], names["E"]]])
    target = geometry.exists(gci2[:, 1], geometry.classes(gci2[:, 2]))
    got = geometry.inside(geometry.classes(gci2[:, 0]), target)
    assert th.allclose(got[:, 0], model.axiom_costs("gci2", gci2), atol=1e-5)


@pytest.mark.parametrize("cls,kwargs", CASES)
def test_membership_and_unavailable_constructors(dataset, cls, kwargs):
    model = _trained(cls, kwargs, dataset)
    geometry = geometry_of(model)
    individuals = {n.rsplit("#", 1)[-1]: i for n, i in model.individual_index_dict.items()}
    names = _names(model)
    ids = th.tensor([individuals["a1"], individuals["d1"]])
    cost = geometry.member(ids, geometry.classes(th.tensor([names["A"], names["A"]])))
    assert cost.shape == (2, 1) and th.isfinite(cost).all()
    roles = {n.rsplit("#", 1)[-1]: i for n, i in model.object_property_index_dict.items()}
    target = geometry.exists(th.tensor([roles["r"]] * 2),
                             geometry.individuals(th.tensor([individuals["d1"]] * 2)))
    assert th.isfinite(geometry.member(ids[:2], target)).all()
    a = geometry.classes(th.tensor([names["A"]]))
    with pytest.raises(ConstructorUnavailable):
        geometry.conjunction(a, a)
    with pytest.raises(ConstructorUnavailable):
        geometry.negation(a)
    with pytest.raises(ConstructorUnavailable):
        geometry.apart(a, a)
    assert geometry.num_classes == len(model.class_index_dict)
    assert geometry.parts["measure"] == "triple score"


@pytest.mark.parametrize("cls,kwargs", CASES)
def test_parameters_are_shared(dataset, cls, kwargs):
    model = _trained(cls, kwargs, dataset)
    geometry = geometry_of(model)
    names = _names(model)
    a = geometry.classes(th.tensor([names["A"], names["B"]]))
    table = model.kge_method.entity_representations[0]
    assert a.vectors.requires_grad or cls is Reshuffle     # Reshuffle: fixed features
    model.kge_method.zero_grad()
    geometry.inside(a, geometry.classes(th.tensor([names["C"], names["C"]]))).sum().backward()
    params = [p for p in table.parameters() if p.grad is not None]
    assert params or cls is Reshuffle      # Reshuffle: features are fixed, relations learn
    if cls is Reshuffle:
        rel = model.kge_method.relation_representations[0]
        assert any(p.grad is not None for p in rel.parameters())


@pytest.mark.parametrize("cls,kwargs", CASES[:2])
def test_the_parts_equal_the_pykeen_model(dataset, cls, kwargs):
    model = _trained(cls, kwargs, dataset)
    geometry = geometry_of(model)
    parts = geometry.as_parts()
    assert isinstance(parts, TripleGeometry)
    triples = model.triples_factory.mapped_triples
    sub = geometry._subclass
    edges = triples[triples[:, 1] == sub]
    expected = -model.kge_method.score_hrt(edges).reshape(-1, 1)
    got = parts(edges[:, [0, 2]], "gci0")
    assert th.allclose(got, expected, atol=1e-5)
    other = triples[triples[:, 1] != sub][:4]
    expected = -model.kge_method.score_hrt(other).reshape(-1, 1)
    assert th.allclose(parts(other[:, [0, 1, 2]], "gci2"), expected, atol=1e-5)
    # the parts share the tables: a change of the model shows in the parts
    with th.no_grad():
        for p in model.kge_method.parameters():
            p.add_(0.1)
    expected = -model.kge_method.score_hrt(other).reshape(-1, 1)
    assert th.allclose(parts(other, "gci2"), expected, atol=1e-5)


def test_reshuffle_has_no_parts(dataset):
    model = _trained(Reshuffle, CASES[2][1], dataset)
    with pytest.raises(NotImplementedError):
        geometry_of(model).as_parts()


@pytest.mark.parametrize("name", ["boxe", "expressive", "graph_kge"])
@pytest.mark.parametrize("classes", ["point", "ball", "box"])
def test_parts_combine_with_other_class_models_and_train(name, classes):
    """BoxE and ExpressivE relations over balls and boxes (their centres), TransE too."""
    dataset = FamilyDataset()
    th.manual_seed(0)
    model = ComposedEL(dataset, method=name, classes=classes, embed_dim=6, margin=1.0,
                       learning_rate=0.01)
    assert isinstance(model.module, TripleGeometry)
    datasets = model.training_datasets          # a triple score reads gci0 and gci2 only
    for form in [f for f in datasets if f not in model.module.scored_forms]:
        del datasets[form]
    forms = [f for f in datasets if len(datasets[f])]
    assert set(forms) == {"gci0", "gci2"}

    def total():
        return sum(float(model.module(datasets[f][:], f).mean())
                   for f in forms)
    before = total()
    model.train(epochs=60)
    assert total() < before
    with pytest.raises(ConstructorUnavailable):
        model.module(th.zeros((1, 3), dtype=th.long), "gci3")


def test_boxe_and_expressive_regions_are_relation_models_of_points():
    boxe = BoxERelations(2, 4)
    expressive = ExpressivERelations(2, 4)
    classes = PointClasses(5, 4, bump=True)
    a, b = classes(th.tensor([0, 1])), classes(th.tensor([2, 3]))
    ids = th.tensor([0, 1])
    for relations in (boxe, expressive):
        assert relations.cost(ids, a, b).shape == (2, 1)
        with pytest.raises(ConstructorUnavailable):
            relations.image(ids, a)
        measure = TripleScore(relations)
        inside = measure.inside(a, relations.preimage(ids, b))
        assert th.allclose(inside, relations.cost(ids, a, b))
        assert (measure.apart(a, relations.preimage(ids, b), 5.0) >= 0).all()
    assert ComposedGeometry is not None and Objective is not None
