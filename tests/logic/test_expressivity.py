"""mowl.logic.expressivity: description logic notation and the ontology check."""

import warnings

import pytest

from mowl.logic import Construct as C, Expressivity, NAMED, check_ontology, \
    ontology_expressivity


@pytest.mark.parametrize("name", ["EL", "EL⊥", "ELH", "ELHO(∘)⊥", "ALC", "ALCH", "SHOIN(D)",
                                  "SROIQ", "ELO⊥"])
def test_names_round_trip(name):
    assert Expressivity.parse(name).name == name


def test_named_logics_and_abox_suffix():
    assert Expressivity.parse("EL++").name == "EL++"
    assert Expressivity.parse("EL++").dl_name == "ELHO(∘)⊥(D)"
    assert Expressivity.parse("DL-Lite_H").name == "DL-Lite_H"
    with_abox = Expressivity.parse("EL⊥ with ABox")
    assert with_abox.has_abox and str(with_abox) == "EL⊥ with ABox"
    assert with_abox.tbox == Expressivity.parse("EL⊥")
    assert Expressivity.parse("EL_bot") == Expressivity.parse("EL⊥")
    with pytest.raises(ValueError):
        Expressivity.parse("XYZ")


def test_order_and_missing():
    el, alc = Expressivity.parse("EL"), Expressivity.parse("ALC")
    assert el <= alc and el < alc and not alc <= NAMED["EL++"]
    assert Expressivity.parse("ELH⊥").missing(alc) == [C.DISJUNCTION, C.NEGATION, C.UNIVERSAL]
    assert C.NOMINAL in NAMED["EL++"] and C.NEGATION not in NAMED["EL++"]
    assert Expressivity.parse("S") == Expressivity.parse("ALC") | {C.TRANSITIVITY}


def _ontology(name):
    from mowl.owlapi import OWLAPIAdapter
    ad = OWLAPIAdapter()
    ns = "http://mowl.borg/expressivity#"
    ont = ad.create_ontology("http://mowl.borg/expressivity_" + name)
    a, b, c = (ad.create_class(ns + x) for x in "ABC")
    r = ad.create_object_property(ns + "r")
    return ad, ont, (a, b, c), r


def test_ontology_expressivity_and_check():
    ad, ont, (a, b, c), r = _ontology("check")
    manager = ad.owl_manager
    for axiom in (ad.create_subclass_of(a, b),
                  ad.create_subclass_of(a, ad.create_object_some_values_from(r, b)),
                  ad.create_subclass_of(ad.create_object_intersection_of(a, b), c)):
        manager.addAxiom(ont, axiom)
    assert ontology_expressivity(ont).name == "EL"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_ontology(ont, "EL").ok          # no warning

    manager.addAxiom(ont, ad.create_subclass_of(a, ad.create_object_all_values_from(r, b)))
    manager.addAxiom(ont, ad.create_subclass_of(a, ad.create_complement_of(b)))
    assert ontology_expressivity(ont).name == "ALC"
    with pytest.warns(UserWarning, match="2 of 5 logical axioms go beyond EL⊥"):
        report = check_ontology(ont, "EL⊥")
    assert not report.ok and report.unsupported_axioms == 2
    assert report.by_axiom_type == {"SubClassOf": 2}
    assert report.by_construct == {C.UNIVERSAL: 1, C.NEGATION: 1}
    assert check_ontology(ont, "ALC", warn=False).ok


def test_disjointness_needs_bottom():
    ad, ont, (a, b, c), r = _ontology("disjoint")
    from java.util import HashSet
    classes = HashSet()
    classes.add(a)
    classes.add(b)
    factory = ad.data_factory
    ad.owl_manager.addAxiom(ont, factory.getOWLDisjointClassesAxiom(classes))
    assert C.BOTTOM in ontology_expressivity(ont)
    assert not check_ontology(ont, "EL", warn=False).ok
    assert check_ontology(ont, "EL⊥", warn=False).ok
