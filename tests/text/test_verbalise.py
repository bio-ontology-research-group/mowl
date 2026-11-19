"""Labels and verbalisation of EL class expressions (mowl.text.verbalise)."""

import pytest

import mowl.datasets  # noqa: F401
from mowl.owlapi import OWLAPIAdapter
from mowl.text.verbalise import (Verbaliser, entity_labels, fallback_label, iri_fragment,
                                 split_camel_case)

P = "http://example.org/onto#"


def _label(adapter, iri, text, lang=""):
    from org.semanticweb.owlapi.model import IRI
    df = adapter.data_factory
    literal = df.getOWLLiteral(text, lang) if lang else df.getOWLLiteral(text)
    return df.getOWLAnnotationAssertionAxiom(df.getRDFSLabel(), IRI.create(iri), literal)


@pytest.fixture(scope="module")
def onto():
    ad = OWLAPIAdapter()
    ont = ad.create_ontology("http://example.org/onto")
    labels = {"Person": "person", "Student": "student", "Class": "class",
              "teach": "teaches", "isParentOf": "is parent of", "Father": "father"}
    axioms = [_label(ad, P + k, v) for k, v in labels.items()]
    axioms.append(_label(ad, P + "Father", "Vater", "de"))
    axioms.append(ad.create_subclass_of(ad.create_class(P + "Father"),
                                        ad.create_class(P + "Person")))
    axioms.append(ad.create_subclass_of(ad.create_class(P + "SuperiorRectalArtery"),
                                        ad.create_class(P + "Person")))
    for ax in axioms:
        ad.owl_manager.addAxiom(ont, ax)
    return ad, ont


def _c(ad, name):
    return ad.create_class(P + name)


def _r(ad, name):
    return ad.create_object_property(P + name)


def test_iri_helpers():
    assert iri_fragment("http://www.co-ode.org/ontologies/galen#RectalArtery") == "RectalArtery"
    assert iri_fragment("http://purl.obolibrary.org/obo/GO_0008150") == "GO_0008150"
    assert iri_fragment("<http://a/b/c>") == "c"
    assert split_camel_case("SuperiorRectalArtery") == "superior rectal artery"
    assert split_camel_case("ABOBloodGroupTest") == "ABO blood group test"
    assert split_camel_case("has_part") == "has part"
    assert fallback_label(P + "HeartValve", "camel_case") == "heart valve"
    with pytest.raises(KeyError):
        fallback_label(P + "X", None)


def test_labels_language_and_fallback(onto):
    ad, ont = onto
    labels = entity_labels(ont)
    assert labels[P + "Father"] == "father"          # English/untagged preferred
    assert labels[P + "SuperiorRectalArtery"] == "SuperiorRectalArtery"
    assert entity_labels(ont, lang="de")[P + "Father"] == "Vater"
    camel = entity_labels(ont, entities=[P + "SuperiorRectalArtery"], fallback="camel_case")
    assert camel == {P + "SuperiorRectalArtery": "superior rectal artery"}


def test_paper_rules(onto):
    """OnT, Section 4.1: V(C ⊓ D) = "V(C) and V(D)", V(∃r.C) = "something that V(r) some
    V(C)"; the paper's examples."""
    ad, ont = onto
    v = Verbaliser(ont)
    assert v(_c(ad, "Person")) == "person"
    conj = ad.create_object_intersection_of(_c(ad, "Person"), _c(ad, "Student"))
    assert v(conj) == "person and student"
    ex = ad.create_object_some_values_from(_r(ad, "isParentOf"), _c(ad, "Person"))
    assert v(ex) == "something that is parent of some person"
    nested = ad.create_object_some_values_from(
        _r(ad, "teach"), ad.create_object_intersection_of(_c(ad, "Class"), _c(ad, "Student")))
    assert v(nested) in ("something that teaches some class and student",
                         "something that teaches some student and class")
    assert v(ad.create_class("http://www.w3.org/2002/07/owl#Thing")) == "thing"


def test_release_and_deeponto_styles(onto):
    ad, ont = onto
    ex = ad.create_object_some_values_from(_r(ad, "teach"), _c(ad, "Class"))
    conj = ad.create_object_intersection_of(_c(ad, "Person"), ex)
    assert Verbaliser(ont, style="release")(ex) == "teaches some class"
    assert Verbaliser(ont, style="release")(conj) == "person and teaches some class"
    deep = Verbaliser(ont, style="deeponto")
    assert deep(conj) == "person that teaches some class"
    assert deep(ex) == "something that teaches some class"
    two = ad.create_object_intersection_of(
        ex, ad.create_object_some_values_from(_r(ad, "teach"), _c(ad, "Student")))
    assert deep(two) in ("something that teaches some class and student",
                         "something that teaches some student and class")


def test_axiom_sides_and_labels_override(onto):
    ad, ont = onto
    v = Verbaliser(ont, labels={P + "Person": "human"})
    ax = ad.create_subclass_of(_c(ad, "Father"), _c(ad, "Person"))
    assert v.axiom_sides(ax) == [("father", "human")]
    eq = ad.create_equivalent_classes(_c(ad, "Father"), _c(ad, "Person"))
    assert sorted(v.axiom_sides(eq)) == [("father", "human"), ("human", "father")]


def test_non_el_constructors_raise(onto):
    ad, ont = onto
    v = Verbaliser(ont)
    union = ad.create_object_union_of(_c(ad, "Person"), _c(ad, "Student"))
    with pytest.raises(NotImplementedError):
        v(union)
    with pytest.raises(ValueError):
        Verbaliser(ont, style="other")
