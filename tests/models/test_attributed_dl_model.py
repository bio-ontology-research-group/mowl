"""Tests of the convex geometric models of ground (temporally) attributed ontologies
(mowl.geometric.attributed).

Source: Bourgaux, Ozaki & Pan, DL 2021 (arXiv:2108.12239): Definitions 1, 2, 5, Theorem 4,
Theorem 6 (grounding and convex model) and Theorem 12 / Example 10 (temporal attributes).
"""
import pytest
from java.util import HashSet
from org.semanticweb.owlapi.model import IRI

from mowl.geometric.attributed import (
    ANY, AttributedConvexModel, AttributedOntology, Basic, ConceptFact, ConceptInc, RoleFact,
    RoleInc, Specifier, from_owl_ontology)
from mowl.geometric.errors import InconsistentOntology, UnsupportedAxiom
from mowl.owlapi import OWLAPIAdapter

S, T = Specifier.closed(source="s"), Specifier.closed(source="t")


def Tm(j):
    return Specifier.closed(time=j)


def _model(*axioms, **kw):
    return AttributedConvexModel(AttributedOntology(axioms, **kw))


def test_annotated_inclusion_applies_only_under_its_annotation():
    m = _model(ConceptFact("A", "a", S), ConceptFact("A", "b", T),
               ConceptInc((Basic("A", S),), Basic("B", S)))
    assert m.is_model()
    assert m.holds_concept("a", "B", S) and not m.holds_concept("b", "B", S)
    assert not m.holds_concept("a", "B", T)
    # inclusion of regions is the geometric reading of the ground inclusion (Definition 2)
    assert m.included("A", S, "B", S)
    assert not m.included("A", T, "B", S)


def test_open_specifiers_follow_the_implication_of_the_paper():
    full = Specifier.closed(source="x", time=1)
    m = _model(ConceptFact("A", "a", full),
               ConceptInc((Basic("A", Specifier.at_least(source="x")),), Basic("B")))
    # [source: x, time: 1] => ⌊source: x⌋ => ⌊⌋
    assert full.implies(Specifier.at_least(source="x")) and full.implies(ANY)
    assert not full.implies(Specifier.closed(source="x"))
    assert m.holds_concept("a", "A", Specifier.at_least(source="x"))
    assert m.holds_concept("a", "A")            # plain A: any annotation
    assert m.holds_concept("a", "B")            # derived through the open specifier
    assert not m.holds_concept("a", "A", Specifier.closed(source="x"))


def test_roles_with_specifiers_and_role_inclusions():
    m = _model(RoleFact("r", "a", "b", S), RoleFact("r", "c", "d", T),
               RoleInc("r", S, "q", S),
               ConceptInc((Basic("r", S, exists=True),), Basic("C", S)),
               ConceptInc((Basic("C", S),), Basic("p", S, exists=True)))
    assert m.is_model()
    assert m.holds_role("a", "r", "b", S) and not m.holds_role("a", "r", "b", T)
    assert m.holds_role("a", "q", "b", S) and not m.holds_role("c", "q", "d", S)
    assert m.holds_concept("a", "C", S) and not m.holds_concept("c", "C", S)
    assert m.holds_role("a", "r", "b", ANY) and m.holds_role("c", "r", "d", ANY)


def test_negative_inclusion_separates_annotations_and_detects_inconsistency():
    clash = ConceptInc((Basic("A", S), Basic("B", S)), None)
    ok = _model(ConceptFact("A", "a", S), ConceptFact("B", "a", T), clash)
    assert ok.is_model() and ok.holds_concept("a", "A", S) and ok.holds_concept("a", "B", T)
    assert not ok.holds_concept("a", "B", S)
    with pytest.raises(InconsistentOntology):
        _model(ConceptFact("A", "a", S), ConceptFact("B", "a", S), clash)


def test_conjunction_of_more_than_two_concepts_and_existential_head():
    m = _model(ConceptFact("A", "a"), ConceptFact("B", "a"), ConceptFact("C", "a"),
               ConceptFact("A", "b"), ConceptFact("B", "b"),
               ConceptInc((Basic("A"), Basic("B"), Basic("C")), Basic("D")),
               ConceptInc((Basic("D"),), Basic("r", exists=True)))
    assert m.is_model()
    assert m.holds_concept("a", "D") and not m.holds_concept("b", "D")
    assert m.is_satisfiable("D")


def test_geometry_is_strongly_faithful_to_the_grounded_ontology_and_convex():
    """phi(eta) = M: exactly the entailed assertions hold in the geometry (Proposition 3 of
    Gutierrez-Basulto & Schockaert, used in the proof of Theorem 6)."""
    specs = [ANY, S, T, Specifier.at_least(source="s")]
    m = _model(ConceptFact("A", "a", S), ConceptFact("A", "b", T),
               ConceptInc((Basic("A", S),), Basic("B", S)),
               ConceptInc((Basic("B", Specifier.at_least(source="s")),), Basic("C", T)))
    for ind in ("a", "b"):
        for concept in "ABC":
            for spec in specs:
                held = m.holds_concept(ind, concept, spec)
                assert held == m.entails_instance(ind, concept, spec), (ind, concept, str(spec))
    # convex model: a region contains the segment between two of its points
    region = m.concept_region("A", ANY)
    p, q = m.point("a"), m.point("b")
    assert region.contains(p) and region.contains(q) and region.contains(0.5 * p + 0.5 * q)
    assert m.dimension == len(m.model.names) - 1


def test_temporal_during_is_the_conjunction_of_time_points():
    during = Specifier.closed(during=(1, 3))
    m = _model(ConceptFact("A", "a", Specifier.closed(time=1)),
               ConceptFact("A", "a", Specifier.closed(time=2)),
               ConceptFact("A", "a", Specifier.closed(time=3)),
               ConceptFact("A", "b", Specifier.closed(time=1)),
               ConceptFact("A", "b", Specifier.closed(time=2)),
               ConceptFact("A", "c", during),
               ConceptInc((Basic("A", during),), Basic("B", during)))
    assert m.is_model()
    assert m.holds_concept("a", "A", during)            # time 1, 2, 3 give during (1, 3)
    assert not m.holds_concept("b", "A", during)        # time 3 is missing
    assert m.holds_concept("c", "A", Specifier.closed(time=2))   # during implies each point
    assert m.holds_concept("c", "B", during) and m.holds_concept("a", "B", during)
    assert not m.holds_concept("b", "B", during)


def test_temporal_since_and_until():
    m = _model(ConceptFact("A", "a", Specifier.closed(since=2)),
               ConceptFact("A", "b", Specifier.closed(until=2)),
               ConceptInc((Basic("A", Tm(4)),), Basic("B", Tm(4))),
               ConceptInc((Basic("A", Tm(0)),), Basic("B", Tm(0))),
               ConceptInc((Basic("A", Tm(1)),), Basic("B", Tm(1))))
    assert m.holds_concept("a", "B", Specifier.closed(time=4))       # since 2 reaches 4
    assert not m.holds_concept("a", "B", Specifier.closed(time=1))   # but not 1
    assert m.holds_concept("b", "B", Specifier.closed(time=0))       # until 2 covers 0 and 1
    assert m.holds_concept("b", "B", Specifier.closed(time=1))
    assert not m.holds_concept("b", "B", Specifier.closed(time=4))


def test_conditions_of_theorem_12_are_enforced():
    with pytest.raises(UnsupportedAxiom):       # before / after / between express disjunction
        _model(ConceptFact("A", "a", Specifier.closed(before=3)))
    with pytest.raises(UnsupportedAxiom):       # Example 10: temporal specifier on a role
        _model(RoleFact("R", "a", "a", Specifier.closed(time=1)))
    with pytest.raises(UnsupportedAxiom):       # temporal attributes only in closed specifiers
        _model(ConceptInc((Basic("A", Specifier.at_least(time=1)),), Basic("B")))
    with pytest.raises(ValueError):
        _model(ConceptFact("A", "a", Specifier.closed(time=-1)))


def test_owl_axiom_annotations_are_read_and_modelled():
    ns, attr = "http://mowl.borg/attributed_test#", "http://mowl.borg/attribute#"
    ad = OWLAPIAdapter()
    df = ad.data_factory
    ont = ad.create_ontology("http://mowl.borg/attributed_test")

    def annotated(**pairs):
        s = HashSet()
        for k, v in pairs.items():
            s.add(df.getOWLAnnotation(df.getOWLAnnotationProperty(IRI.create(attr + k)),
                                      df.getOWLLiteral(str(v))))
        return s

    A, B = ad.create_class(ns + "A"), ad.create_class(ns + "B")
    a, b = ad.create_individual(ns + "a"), ad.create_individual(ns + "b")
    r = ad.create_object_property(ns + "r")
    for axiom in (
            df.getOWLClassAssertionAxiom(A, a, annotated(source="s")),
            df.getOWLClassAssertionAxiom(A, b, annotated(source="t")),
            df.getOWLSubClassOfAxiom(A, B, annotated(source="s")),
            df.getOWLObjectPropertyAssertionAxiom(r, a, b, annotated(source="u")),
            df.getOWLClassAssertionAxiom(B, b),
            df.getOWLSubClassOfAxiom(A, df.getOWLObjectAllValuesFrom(r, B))):
        ad.owl_manager.addAxiom(ont, axiom)
    onto, skipped = from_owl_ontology(ont)
    assert len(onto.axioms) == 5 and len(skipped) == 1       # the ∀ axiom is skipped
    m = AttributedConvexModel(onto)
    assert m.is_model()
    assert m.holds_concept(ns + "a", ns + "B", Specifier.closed(source="s"))
    assert not m.holds_concept(ns + "b", ns + "B", Specifier.closed(source="s"))
    assert m.holds_concept(ns + "b", ns + "B")                # plain assertion, no annotation
    assert m.holds_role(ns + "a", ns + "r", ns + "b", ANY)
    assert m.holds_role(ns + "a", ns + "r", ns + "b", Specifier.closed(source="u"))
