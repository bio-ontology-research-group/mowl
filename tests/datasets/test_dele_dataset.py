"""Candidate classes of the DELE benchmarks (DELEDataset.evaluation_classes)."""

from org.semanticweb.owlapi.model import IRI

from mowl.datasets.builtin.mashkova2024 import DELEDataset
from mowl.owlapi import OWLAPIAdapter


def _files(tmp_path):
    ad = OWLAPIAdapter()
    prot = ad.create_class("http://4932.YAL001C")
    go = ad.create_class("http://purl.obolibrary.org/obo/GO_0000001")
    aux = ad.create_class("urn:absolute:yeast_ontology#A01lu")
    label = ad.create_class("http://mowl/label/ec_number/1")
    hf = ad.create_object_property("http://has_function")
    paths = []
    for split in ("train", "valid", "test"):
        ont = ad.create_ontology(f"http://test.org/dele_hf/{split}")
        ad.owl_manager.addAxiom(ont, ad.create_subclass_of(
            prot, ad.create_object_some_values_from(hf, go)))
        if split == "train":
            ad.owl_manager.addAxiom(ont, ad.create_subclass_of(go, aux))
            ad.owl_manager.addAxiom(ont, ad.create_subclass_of(label, go))
            ad.owl_manager.addAxiom(ont, ad.create_subclass_of(go, ad.create_class(
                "http://www.w3.org/2002/07/owl#Thing")))
        path = tmp_path / f"{split}.owl"
        ad.owl_manager.saveOntology(ont, IRI.create("file:" + str(path)))
        paths.append(str(path))
    return paths


def test_function_tails_are_all_non_protein_classes(tmp_path):
    """As the reference code: the candidate functions are every class that is not a
    protein, owl:Thing/Nothing or a 'label' class -- including the classes of the
    authors' normalisation (A01lu, ...)."""
    ds = DELEDataset("yeast_function", paths=_files(tmp_path))
    heads, tails = ds.evaluation_classes
    assert heads.as_str == ["http://4932.YAL001C"]
    assert sorted(tails.as_str) == ["http://purl.obolibrary.org/obo/GO_0000001",
                                    "urn:absolute:yeast_ontology#A01lu"]
