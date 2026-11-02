"""Tests that EmbeddingELModel passes individual_index_dict consistently across splits."""

from org.semanticweb.owlapi.model import IRI
import torch as th
from mowl.owlapi import OWLAPIAdapter
from mowl.datasets import PathDataset
from mowl.datasets.el import ELDataset
from mowl.models import ELBE


def test_held_out_individual_index_consistent_across_splits(tmp_path):
    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    factory = adapter.data_factory
    ns = "http://test.org/ind_split#"

    cls_A = adapter.create_class(ns + "A")
    cls_B = adapter.create_class(ns + "B")
    ind_a = adapter.create_individual(ns + "a")
    ind_b = adapter.create_individual(ns + "b")
    ind_c = adapter.create_individual(ns + "c")

    train_ont = adapter.create_ontology("http://test.org/ind_split/train")
    manager.addAxiom(train_ont, adapter.create_subclass_of(cls_A, cls_B))
    manager.addAxiom(train_ont, adapter.create_class_assertion(cls_A, ind_a))
    manager.addAxiom(train_ont, adapter.create_class_assertion(cls_B, ind_b))

    valid_ont = adapter.create_ontology("http://test.org/ind_split/valid")
    manager.addAxiom(valid_ont, factory.getOWLDeclarationAxiom(ind_c))
    manager.addAxiom(valid_ont, adapter.create_class_assertion(cls_A, ind_c))

    test_ont = adapter.create_ontology("http://test.org/ind_split/test")
    ind_d = adapter.create_individual(ns + "d")
    rel_r = adapter.create_object_property(ns + "r")
    manager.addAxiom(test_ont, factory.getOWLDeclarationAxiom(ind_d))
    manager.addAxiom(test_ont, adapter.create_class_assertion(cls_B, ind_d))
    manager.addAxiom(test_ont, adapter.create_object_property_assertion(rel_r, ind_c, ind_d))

    train_path = tmp_path / "train.owl"
    valid_path = tmp_path / "valid.owl"
    test_path = tmp_path / "test.owl"
    manager.saveOntology(train_ont, IRI.create("file:" + str(train_path)))
    manager.saveOntology(valid_ont, IRI.create("file:" + str(valid_path)))
    manager.saveOntology(test_ont, IRI.create("file:" + str(test_path)))

    dataset = PathDataset(
        str(train_path),
        validation_path=str(valid_path),
        testing_path=str(test_path),
    )
    model = ELBE(dataset, embed_dim=8, batch_size=4)

    c_iri = ns + "c"
    d_iri = ns + "d"
    assert c_iri in model.individual_index_dict
    assert d_iri in model.individual_index_dict
    c_idx = model.individual_index_dict[c_iri]
    d_idx = model.individual_index_dict[d_iri]

    # Without passing individual_index_dict, the training ontology has only 2 individuals (a, b)
    # and a standalone validation ELDataset assigns c local index 0 (colliding with training individual a).
    n_train_inds = len(dataset.ontology.getIndividualsInSignature())
    assert n_train_inds == 2
    unfixed_valid_ca = ELDataset(dataset.validation, extended=False).get_gci_datasets()[
        "class_assertion"
    ][:]
    assert int(unfixed_valid_ca[0, 0].item()) == 0 < n_train_inds

    # With the fix, the model's individual table has 4 rows (> n_train_inds) and c, d sit
    # beyond the training-only table size (indices >= n_train_inds).
    assert model.module.ind_embed.weight.shape[0] == len(model.individual_index_dict) == 4
    assert model.module.ind_embed.weight.shape[0] > n_train_inds
    assert n_train_inds <= c_idx < model.module.ind_embed.weight.shape[0]
    assert n_train_inds <= d_idx < model.module.ind_embed.weight.shape[0]

    # Validation class_assertion tensor indexes c with model.individual_index_dict[c]
    # and looks up the exact c_idx row of ind_embed (not training row 0).
    val_ca = model.validation_datasets["class_assertion"][:]
    assert val_ca.shape == (1, 2)
    assert int(val_ca[0, 0].item()) == c_idx
    assert th.equal(
        model.module.ind_embed(val_ca[:, 0])[0],
        model.module.ind_embed.weight[c_idx],
    )

    # Testing class_assertion and object_property_assertion tensors index c and d consistently
    test_ca = model.testing_datasets["class_assertion"][:]
    assert test_ca.shape == (1, 2)
    assert int(test_ca[0, 0].item()) == d_idx
    assert th.equal(
        model.module.ind_embed(test_ca[:, 0])[0],
        model.module.ind_embed.weight[d_idx],
    )

    test_opa = model.testing_datasets["object_property_assertion"][:]
    assert test_opa.shape == (1, 3)
    assert int(test_opa[0, 0].item()) == c_idx
    assert int(test_opa[0, 2].item()) == d_idx
