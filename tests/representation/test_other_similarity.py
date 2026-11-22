"""Walk and sentence methods: only similarity is defined."""
import pytest
import torch as th

import mowl
mowl.init_jvm("1g")
pytest.importorskip("gensim")

from mowl.models import RandomWalkPlusW2VModel, SyntacticPlusW2VModel  # noqa: E402
from mowl.projection import TaxonomyProjector  # noqa: E402
from mowl.representation import ConstructorUnavailable, geometry_of  # noqa: E402
from mowl.representation.library.other_similarity import SimilarityGeometry  # noqa: E402
from mowl.walking import DeepWalk  # noqa: E402
from tests.datasetFactory import FamilyDataset  # noqa: E402


@pytest.fixture(scope="module")
def dataset():
    return FamilyDataset()


def _walk(dataset):
    model = RandomWalkPlusW2VModel(dataset)
    model.set_projector(TaxonomyProjector())
    model.set_walker(DeepWalk(5, 5, 0.1, workers=1))
    model.set_w2v_model(vector_size=8, min_count=1, workers=1, epochs=1, seed=0)
    model.train(epochs=1)
    return model


def _corpus(dataset):
    model = SyntacticPlusW2VModel(dataset)
    model.set_w2v_model(vector_size=8, min_count=1, workers=1, epochs=1, seed=0)
    model.generate_corpus(save=True, with_annotations=False)
    model.train(epochs=1)
    return model


@pytest.mark.parametrize("build", [_walk, _corpus])
def test_geometry_gives_the_models_own_score(dataset, build):
    model = build(dataset)
    geometry = geometry_of(model)
    assert isinstance(geometry, SimilarityGeometry)
    rows = th.tensor([[0, 1], [2, 3], [1, 0]])
    got = geometry.inside(geometry.classes(rows[:, 0]), geometry.classes(rows[:, 1]))
    expected = model.evaluation_model(rows)
    known = th.isfinite(got[:, 0])
    assert known.any()
    assert th.allclose(got[known, 0].float(), expected[known].float().detach(), atol=1e-5)
    # similarity, not inclusion: symmetric
    back = geometry.inside(geometry.classes(rows[:, 1]), geometry.classes(rows[:, 0]))
    assert th.allclose(got[known], back[known])
    a, b = geometry.classes(rows[:, 0]), geometry.classes(rows[:, 1])
    assert th.allclose(geometry.apart(a, b)[known] + got[known], th.ones_like(got[known]))
    assert geometry.parts["measure"] == "similarity"


@pytest.mark.parametrize("build", [_walk, _corpus])
def test_vectors_are_shared_and_constructors_unavailable(dataset, build):
    model = build(dataset)
    geometry = geometry_of(model)
    assert geometry.vectors.data_ptr() == model.w2v_model.wv.vectors.__array_interface__[
        "data"][0]
    a = geometry.classes(th.tensor([0]))
    for call in (lambda: geometry.exists(th.tensor([0]), a),
                 lambda: geometry.conjunction(a, a), lambda: geometry.negation(a),
                 lambda: geometry.disjunction(a, a)):
        with pytest.raises(ConstructorUnavailable):
            call()
    assert geometry.num_classes == len(model.class_index_dict)
    ind = geometry.individuals(th.tensor([0]))      # the family ontology has none
    assert len(ind) == 1 and not bool(ind.known[0])
    assert th.isinf(geometry.member(th.tensor([0]), a)).all()


def test_individuals_have_vectors():
    from tests.models.kg_region_data import abox_dataset
    model = SyntacticPlusW2VModel(abox_dataset())
    model.set_w2v_model(vector_size=8, min_count=1, workers=1, epochs=1, seed=0)
    model.generate_corpus(save=True, with_annotations=False)
    model.train(epochs=1)
    geometry = geometry_of(model)
    ids = th.arange(len(model.individual_index_dict))
    cost = geometry.member(ids, geometry.classes(th.zeros_like(ids)))
    assert cost.shape == (len(ids), 1) and th.isfinite(cost).any()
