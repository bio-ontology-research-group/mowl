"""``GraphPlusPyKEENModel.axiom_costs``: every edge form of the projection is scored."""
import pytest
import torch as th
from java.util import HashSet
from pykeen.models import TransE

import mowl
mowl.init_jvm("2g")

from mowl.datasets import Dataset  # noqa: E402
from mowl.evaluation.model_scoring import UnsupportedAxiomForm  # noqa: E402
from mowl.models import GraphPlusPyKEENModel  # noqa: E402
from mowl.owlapi.adapter import OWLAPIAdapter  # noqa: E402
from mowl.projection import OWL2VecStarProjector  # noqa: E402

NS = "http://x#"


@pytest.fixture(scope="module")
def model():
    ad = OWLAPIAdapter()
    onto = ad.create_ontology("http://x/o")
    a_, b_, c_ = (ad.create_class(NS + n) for n in "ABC")
    r = ad.create_object_property(NS + "r")
    s = ad.create_object_property(NS + "s")
    i, j, k = (ad.create_individual(NS + n) for n in "ijk")
    axioms = HashSet()
    axioms.add(ad.create_subclass_of(a_, b_))
    axioms.add(ad.create_subclass_of(a_, ad.create_object_some_values_from(r, c_)))
    axioms.add(ad.create_class_assertion(a_, i))
    axioms.add(ad.create_class_assertion(b_, j))
    axioms.add(ad.create_object_property_assertion(r, i, j))
    axioms.add(ad.create_object_property_assertion(s, j, k))
    ad.owl_manager.addAxioms(onto, axioms)
    dataset = Dataset(onto, validation=onto, testing=onto)
    m = GraphPlusPyKEENModel(dataset)
    m.set_projector(OWL2VecStarProjector())
    m.set_kge_method(TransE, embedding_dim=8, random_seed=0)
    m.optimizer = th.optim.Adam
    m.lr = 0.01
    m.batch_size = 8
    m._kge_method.eval()
    return m


def _cost(model, node_h, relation, node_t):
    factory = model.triples_factory
    triple = th.tensor([[factory.entity_to_id[node_h], factory.relation_to_id[relation],
                         factory.entity_to_id[node_t]]])
    return float(-model._kge_method.score_hrt(triple).detach().reshape(-1))


def _row(model, *names):
    classes, roles = model.class_index_dict, model.object_property_index_dict
    inds = model.individual_index_dict
    out = []
    for n in names:
        key = NS + n
        out.append(classes.get(key, roles.get(key, inds.get(key))))
    return th.tensor([out])


def test_forms_are_declared(model):
    assert {"gci0", "gci2", "class_assertion", "object_property_assertion"} <= set(
        model.scored_forms)


def test_class_assertion_is_the_type_edge(model):
    cost = model.axiom_costs("class_assertion", th.tensor([[model.individual_index_dict[NS + "i"],
                                                            model.class_index_dict[NS + "A"]]]))
    assert float(cost) == pytest.approx(_cost(model, NS + "i", "http://type", NS + "A"),
                                        abs=1e-5)


def test_role_assertion_is_the_role_edge(model):
    ind, role = model.individual_index_dict, model.object_property_index_dict
    rows = th.tensor([[ind[NS + "i"], role[NS + "r"], ind[NS + "j"]],
                      [ind[NS + "j"], role[NS + "s"], ind[NS + "k"]]])
    cost = model.axiom_costs("object_property_assertion", rows)
    assert cost[0].item() == pytest.approx(_cost(model, NS + "i", NS + "r", NS + "j"), abs=1e-5)
    assert cost[1].item() == pytest.approx(_cost(model, NS + "j", NS + "s", NS + "k"), abs=1e-5)


def test_tbox_forms_are_unchanged(model):
    cls, role = model.class_index_dict, model.object_property_index_dict
    gci0 = model.axiom_costs("gci0", th.tensor([[cls[NS + "A"], cls[NS + "B"]]]))
    assert float(gci0) == pytest.approx(_cost(model, NS + "A", "http://subclassof", NS + "B"),
                                        abs=1e-5)
    gci2 = model.axiom_costs("gci2", th.tensor([[cls[NS + "A"], role[NS + "r"],
                                                 cls[NS + "C"]]]))
    assert float(gci2) == pytest.approx(_cost(model, NS + "A", NS + "r", NS + "C"), abs=1e-5)


def test_a_form_without_edges_is_infinite_and_others_are_refused(model):
    # class assertion of a pair that is not an edge is still scored (finite)
    ind, cls = model.individual_index_dict, model.class_index_dict
    assert th.isfinite(model.axiom_costs(
        "class_assertion", th.tensor([[ind[NS + "k"], cls[NS + "C"]]]))).all()
    with pytest.raises(UnsupportedAxiomForm):
        model.axiom_costs("gci1", th.zeros((1, 3), dtype=th.long))


def test_projection_without_type_edges_gives_infinite_class_assertion_costs():
    from mowl.projection import TaxonomyProjector
    ad = OWLAPIAdapter()
    onto = ad.create_ontology("http://x/o2")
    axioms = HashSet()
    axioms.add(ad.create_subclass_of(ad.create_class(NS + "A"), ad.create_class(NS + "B")))
    axioms.add(ad.create_class_assertion(ad.create_class(NS + "A"), ad.create_individual(NS + "i")))
    ad.owl_manager.addAxioms(onto, axioms)
    m = GraphPlusPyKEENModel(Dataset(onto, validation=onto, testing=onto))
    m.set_projector(TaxonomyProjector())
    m.set_kge_method(TransE, embedding_dim=4, random_seed=0)
    cost = m.axiom_costs("class_assertion", th.tensor([[0, 0]]))
    assert th.isinf(cost).all()
