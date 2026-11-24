"""``add_axioms`` on models whose modules are built from parts: the tables are named after the
parts, not listed in ``entity_embedding_names``, and their rows must follow the entities."""
import warnings

import pytest
import torch as th

from mowl.datasets import Dataset
from mowl.owlapi import OWLAPIAdapter

NAMES = ["ELEmbeddings", "ELBE", "BoxEL", "BoxSEL", "BoxSquaredEL", "TransBox", "EIKE",
         "PrismEL", "OctagonEL", "MultiBoxEL", "MultiELBE", "EmELPP", "EmELVar", "RegDELBE",
         "RegDELEm", "Concept2Box", "BoxLattice", "GumbelBox"]


def _dataset(tag):
    adapter = OWLAPIAdapter()
    ns = f"http://test.org/{tag}/"
    dataset = Dataset(adapter.create_ontology(ns + "ontology"))
    a, b, c, d = [adapter.create_class(ns + name) for name in "ABCD"]
    r = adapter.create_object_property(ns + "r")
    dataset.add_axioms(adapter.create_subclass_of(a, b),
                       adapter.create_subclass_of(b, adapter.create_object_some_values_from(r, c)))
    return adapter, dataset, ns, (a, b, c, d), r


@pytest.mark.parametrize("name", NAMES)
def test_rows_follow_the_entities(name):
    import mowl.models as models
    cls = getattr(models, name, None)
    if cls is None:
        pytest.skip(f"{name} is not exported")
    adapter, dataset, ns, (a, b, c, d), r = _dataset(name)
    model = cls(dataset, embed_dim=4, batch_size=4)
    old_index = dict(model.class_index_dict)
    old_state = {k: v.clone() for k, v in model.module.state_dict().items()}
    s = adapter.create_object_property(ns + "s")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.add_axioms(
            adapter.create_subclass_of(c, d),
            adapter.create_subclass_of(d, adapter.create_object_some_values_from(s, a)))
    new_index = model.class_index_dict
    assert ns + "D" in new_index and ns + "D" not in old_index
    new_state = model.module.state_dict()
    old_ids = th.tensor([old_index[iri] for iri in old_index])
    new_ids = th.tensor([new_index[iri] for iri in old_index])
    checked = 0
    for key, old in old_state.items():
        new = new_state[key]
        if old.dim() == 0 or old.shape[0] != len(old_index) or new.shape[0] != len(new_index):
            continue
        assert th.equal(new[new_ids], old[old_ids]), key
        checked += 1
    assert checked, "no class-indexed table found"
    gci0 = model.training_datasets["gci0"]
    assert th.isfinite(model.module(gci0.data, "gci0")).all()
