"""Tests of the DL-Lite_H extractor (mowl.datasets.dllite)."""

import os

import numpy as np
import pytest

import mowl.datasets  # noqa: F401  (import order: datasets before nn)
from mowl.datasets.dllite import DLLiteKB, extract_dllite
from mowl.owlapi import OWLAPIAdapter

NS = "http://mowl.borg/dllite#"
FIXTURES = os.path.join(os.path.dirname(__file__), "..", "geometric", "fixtures")


def _build():
    """A hand-built ontology with every supported axiom type and several unsupported
    ones."""
    ad = OWLAPIAdapter()
    mgr, df = ad.owl_manager, ad.data_factory
    ont = ad.create_ontology("http://mowl.borg/dllite/test")
    A, B, C, D, E = (ad.create_class(NS + n) for n in "ABCDE")
    r, s, t, u = (ad.create_object_property(NS + n) for n in ("r", "s", "t", "u"))
    a, b, c = (ad.create_individual(NS + n) for n in "abc")
    thing, nothing = df.getOWLThing(), df.getOWLNothing()

    def some(p, filler=thing):
        return df.getOWLObjectSomeValuesFrom(p, filler)

    def inv(p):
        return df.getOWLObjectInverseOf(p)

    axioms = [
        ad.create_subclass_of(A, B),                                     # pos
        ad.create_subclass_of(A, some(r)),                               # pos
        ad.create_subclass_of(some(inv(r)), C),                          # pos
        ad.create_subclass_of(B, df.getOWLObjectComplementOf(some(s))),  # neg
        ad.create_subclass_of(E, nothing),                               # neg E ⊑ ¬E
        ad.create_subclass_of(C, df.getOWLObjectIntersectionOf(D, some(inv(t)))),  # 2 pos
        ad.create_disjoint_classes(A, C, D),                             # 3 neg
        ad.create_equivalent_classes(D, some(u)),                        # 2 pos
        df.getOWLObjectPropertyDomainAxiom(s, B),                        # ∃s ⊑ B
        df.getOWLObjectPropertyRangeAxiom(s, C),                         # ∃s⁻ ⊑ C
        df.getOWLSubObjectPropertyOfAxiom(r, s),                         # role
        df.getOWLSubObjectPropertyOfAxiom(r, inv(t)),                    # role
        df.getOWLInverseObjectPropertiesAxiom(t, u),                     # 2 roles
        df.getOWLSymmetricObjectPropertyAxiom(s),                        # role
        ad.create_class_assertion(A, a),                                 # cassert
        ad.create_class_assertion(some(inv(r)), b),                      # cassert
        ad.create_object_property_assertion(r, a, b),                    # rassert
        df.getOWLObjectPropertyAssertionAxiom(inv(s), c, a),             # s(a, c)
        # trivial
        ad.create_subclass_of(A, thing),
        ad.create_subclass_of(nothing, A),
        # unsupported
        ad.create_subclass_of(df.getOWLObjectIntersectionOf(A, B), C),
        ad.create_subclass_of(A, some(r, B)),
        ad.create_subclass_of(some(r, B), A),
        ad.create_subclass_of(A, df.getOWLObjectAllValuesFrom(r, B)),
        ad.create_subclass_of(A, df.getOWLObjectUnionOf(B, C)),
        ad.create_class_assertion(df.getOWLObjectComplementOf(A), c),
        df.getOWLTransitiveObjectPropertyAxiom(r),
    ]
    for ax in axioms:
        mgr.addAxiom(ont, ax)
    return ont


@pytest.fixture(scope="module")
def kb():
    return extract_dllite(_build())


def _named(kb):
    names = [kb.basic_concept_name(i) for i in range(kb.n_basic)]
    return {"pos": {(names[x], names[y]) for x, y in kb.concept_inclusions},
            "neg": {(names[x], names[y]) for x, y in kb.negative_inclusions}}


def test_signature_and_basic_concepts(kb):
    assert [c.rsplit("#")[1] for c in kb.concepts] == ["A", "B", "C", "D", "E"]
    assert [r.rsplit("#")[1] for r in kb.roles] == ["r", "s", "t", "u"]
    assert [i.rsplit("#")[1] for i in kb.individuals] == ["a", "b", "c"]
    assert kb.n_basic == 5 + 2 * 4
    assert kb.basic_concept_name(kb.exists_index(1)) == "∃s"
    assert kb.basic_concept_name(kb.exists_index(0, inverse=True)) == "∃r⁻"


def test_concept_inclusions(kb):
    pos = _named(kb)["pos"]
    assert pos == {("A", "B"), ("A", "∃r"), ("∃r⁻", "C"), ("C", "D"), ("C", "∃t⁻"),
                   ("D", "∃u"), ("∃u", "D"), ("∃s", "B"), ("∃s⁻", "C")}


def test_negative_inclusions(kb):
    neg = _named(kb)["neg"]
    assert neg == {("B", "∃s"), ("E", "E"), ("A", "C"), ("A", "D"), ("C", "D")}


def test_role_inclusions(kb):
    names = {(kb.roles[r].rsplit("#")[1] + "⁻" * ri, kb.roles[t].rsplit("#")[1] + "⁻" * ti)
             for r, ri, t, ti in kb.role_inclusions}
    assert names == {("r", "s"), ("r", "t⁻"), ("t", "u⁻"), ("u", "t⁻"), ("s", "s⁻")}


def test_assertions(kb):
    ind = [i.rsplit("#")[1] for i in kb.individuals]
    ca = {(kb.basic_concept_name(b), ind[a]) for b, a in kb.concept_assertions}
    assert ca == {("A", "a"), ("∃r⁻", "b")}
    ra = {(kb.roles[r].rsplit("#")[1], ind[a], ind[b]) for r, a, b in kb.role_assertions}
    assert ra == {("r", "a", "b"), ("s", "a", "c")}  # ObjectInverseOf(s)(c, a) is s(a, c)


def test_nothing_is_dropped_silently(kb):
    assert sum(kb.trivial.values()) == 2
    assert sum(kb.skipped.values()) == 7
    reasons = " ".join(kb.skipped)
    for text in ("ObjectIntersectionOf on the left", "qualified existential",
                 "ObjectAllValuesFrom", "ObjectUnionOf", "negative concept assertion",
                 "TransitiveObjectProperty"):
        assert text in reasons, text
    assert kb.ignored.get("Declaration", 0) == 0  # none were added
    summary = kb.summary()
    assert summary["skipped"] == 7 and summary["trivial"] == 2


def test_index_arrays_and_tensors(kb):
    t = kb.as_tensors()
    assert t["concept_inclusions"].shape == (9, 2)
    assert t["role_assertions"].shape == (2, 3)
    assert kb.concept_inclusions.dtype == np.int64


def test_signature_reuse_counts_unknown_entities(kb):
    ad = OWLAPIAdapter()
    ont = ad.create_ontology("http://mowl.borg/dllite/test2")
    r = ad.create_object_property(NS + "r")
    a, b, z = (ad.create_individual(NS + n) for n in "abz")
    ad.owl_manager.addAxiom(ont, ad.create_object_property_assertion(r, b, a))
    ad.owl_manager.addAxiom(ont, ad.create_object_property_assertion(r, a, z))
    test = extract_dllite(ont, signature=kb)
    assert test.individuals == kb.individuals
    assert test.role_assertions.tolist() == [[0, 1, 0]]
    assert sum(test.skipped.values()) == 1
    assert "entity not in signature" in next(iter(test.skipped))


def test_t3_helly_obstruction_is_outside_dllite():
    """Leemhuis & Kutz's T3 (conjunctions defining A, B, C and ternary-style
    disjointness A ⊓ B ⊑ ⊥ ...) uses conjunction on the left: not DL-Lite_H."""
    from mowl.owlapi import OWLAPIAdapter as Adapter
    from org.semanticweb.owlapi.model import IRI
    path = os.path.abspath(os.path.join(FIXTURES, "t3.owl"))
    ont = Adapter().owl_manager.loadOntologyFromOntologyDocument(IRI.create("file:" + path))
    kb = extract_dllite(ont)
    # A ≡ X1 ⊓ X2 contributes A ⊑ X1, A ⊑ X2 (right-hand side), the other direction and
    # the disjointness of conjunctions are skipped.
    assert len(kb.negative_inclusions) == 0
    assert any("ObjectIntersectionOf on the left" in k for k in kb.skipped)


def test_from_lists():
    kb = DLLiteKB.from_lists(inclusions=[("A", "∃r"), ("some r-", "B")],
                             negative_inclusions=[("A", "B")],
                             role_inclusions=[("r", "s-")],
                             concept_assertions=[("A", "x")],
                             role_assertions=[("s-", "x", "y")])
    assert kb.concepts == ["A", "B"] and kb.roles == ["r", "s"]
    assert kb.role_inclusions.tolist() == [[0, 0, 1, 1]]
    assert kb.role_assertions.tolist() == [[1, 1, 0]]  # s⁻(x, y) is s(y, x)
    assert kb.basic_concept_name(kb.concept_inclusions[1, 0]) == "∃r⁻"
    with pytest.raises(ValueError):
        DLLiteKB(["A"], [], [], concept_inclusions=[[0, 3]])
