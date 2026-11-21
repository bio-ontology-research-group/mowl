"""Reasoner backends and the subsumption oracle."""

import pytest

from mowl.logic import LogicalStatus
from mowl.owlapi import OWLAPIAdapter
from mowl.reasoning import (SubsumptionOracle, create_reasoner, list_reasoners,
                            register_reasoner, select_reasoner)


def _ontology(name, universal=False):
    from org.semanticweb.owlapi.model import IRI
    ad = OWLAPIAdapter()
    ns = "http://mowl.borg/oracle#"
    ont = ad.create_ontology("http://mowl.borg/oracle_" + name)
    a, b, c, d = (ad.create_class(ns + x) for x in "ABCD")
    r = ad.create_object_property(ns + "r")
    factory = ad.data_factory
    ind = factory.getOWLNamedIndividual(IRI.create(ns + "a"))
    axioms = [ad.create_subclass_of(a, b),
              ad.create_subclass_of(b, ad.create_object_some_values_from(r, c)),
              ad.create_subclass_of(ad.create_object_intersection_of(c, d),
                                    factory.getOWLNothing()),
              factory.getOWLClassAssertionAxiom(a, ind),
              factory.getOWLClassAssertionAxiom(c, ind)]
    if universal:
        axioms.append(ad.create_subclass_of(a, ad.create_object_all_values_from(r, d)))
    for axiom in axioms:
        ad.owl_manager.addAxiom(ont, axiom)
    return ad, ont, (a, b, c, d), r, ind


def test_selection_by_expressivity():
    assert [b.name for b in list_reasoners()] == ["structural", "elk", "km", "hermit"]
    assert select_reasoner("taxonomy").name == "structural"
    assert select_reasoner("EL⊥").name == "elk"
    assert select_reasoner("ELH(∘)⊥ with ABox").name == "elk"
    assert select_reasoner("ALC").name == "km"
    assert select_reasoner("ELO").name == "km"           # ELK has no nominals


def test_auto_picks_the_cheapest_complete_reasoner_and_warns_otherwise():
    _, el, *_ = _ontology("auto_el")
    assert create_reasoner(el)[1].name == "elk"
    _, alc, *_ = _ontology("auto_alc", universal=True)
    assert create_reasoner(alc)[1].name == "km"
    with pytest.warns(UserWarning, match="inferences that depend on"):
        create_reasoner(alc, "elk")


@pytest.mark.parametrize("backend", ["elk", "hermit", "km"])
def test_oracle_reduces_tasks_to_subsumption(backend):
    from mowl.reasoning import get_backend
    if not get_backend(backend).is_available():
        pytest.skip(f"{backend} is not available")
    ad, ont, (a, b, c, d), r, ind = _ontology("oracle_" + backend)
    before = ont.getAxiomCount()
    oracle = SubsumptionOracle(ont, backend)
    some_r_c = ad.create_object_some_values_from(r, c)
    assert oracle.entails_subsumption(a, some_r_c)            # A ⊑ B ⊑ ∃r.C
    assert not oracle.entails_subsumption(some_r_c, a)
    assert not oracle.is_satisfiable(ad.create_object_intersection_of(c, d))
    assert oracle.entails_instance(ind, some_r_c)             # {a} ⊑ ∃r.C
    assert oracle.status(a, b) is LogicalStatus.ENTAILED
    assert oracle.status(a, c) is LogicalStatus.UNDETERMINED
    # a is an A and a C, and C and D are disjoint, so A ⊑ D cannot be added.
    assert oracle.status(a, d) is LogicalStatus.CONTRADICTORY
    assert ont.getAxiomCount() == before                      # definitions were removed
    oracle.dispose()


def test_register_reasoner():
    calls = []

    def factory(ontology):
        calls.append(ontology)
        from org.semanticweb.owlapi.reasoner.structural import StructuralReasonerFactory
        return StructuralReasonerFactory().createReasoner(ontology)

    backend = register_reasoner("test-told", factory, "taxonomy", cost=99, replace=True)
    assert backend in list_reasoners()
    _, ont, *_ = _ontology("register")
    with pytest.warns(UserWarning):
        create_reasoner(ont, "test-told")
    assert calls
    with pytest.raises(ValueError):
        register_reasoner("test-told", factory, "taxonomy")


def test_km_is_packaged_and_preferred_to_hermit():
    """mOWL packages the KM reasoner; it is cheaper than HermiT and covers OWL 2 DL."""
    from mowl.reasoning import get_backend
    km = get_backend("km")
    assert km.is_available()
    assert str(km.expressivity).startswith("SROIQ(D)")
    assert not km.instance_queries
    assert select_reasoner("ALC").name == "km"
