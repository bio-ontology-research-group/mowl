"""Lattice construction and saturation of CatE (Zhapa-Camacho & Hoehndorf,
arXiv:2305.07163v3, Sections 3 and 3.1) on hand-written ALC axioms."""

import os

import pytest

import mowl
mowl.init_jvm("2g")

from mowl.owlapi import OWLAPIAdapter  # noqa: E402
from mowl.owlapi.defaults import BOT, TOP  # noqa: E402
from mowl.projection.categorical.edge import Node  # noqa: E402
from mowl.projection.categorical.lattice import (CatELattice, UnseenConceptError,  # noqa: E402
                                                 node_name)
from mowl.projection.categorical import saturation as sat  # noqa: E402

NS = "http://mowl.borg/cate#"
ad = OWLAPIAdapter()
df = ad.data_factory


def cls(name):
    return ad.create_class(NS + name)


def ind(name):
    return df.getOWLNamedIndividual(ad.create_class(NS + name).getIRI())


R = ad.create_object_property(NS + "r")
TOPC, BOTC = ad.create_class(TOP), ad.create_class(BOT)
AND = ad.create_object_intersection_of
OR = ad.create_object_union_of
NOT = ad.create_complement_of
SOME = ad.create_object_some_values_from
ALL = ad.create_object_all_values_from
SUB = ad.create_subclass_of


_COUNTER = [0]


def ontology(*axioms):
    _COUNTER[0] += 1
    ont = ad.create_ontology(f"{NS}sat{_COUNTER[0]}")
    for ax in axioms:
        ad.owl_manager.addAxiom(ont, ax)
    return ont


def has(lat, src, dst):
    """Morphism between two class expressions (or Nodes)."""
    s = src if isinstance(src, Node) else Node(owl_class=src)
    d = dst if isinstance(dst, Node) else Node(owl_class=dst)
    assert s in lat.node_to_id, f"{s} is not an element"
    assert d in lat.node_to_id, f"{d} is not an element"
    return lat.has_edge(lat.node_to_id[s], lat.node_to_id[d])


A, B, C, D = cls("A"), cls("B"), cls("C"), cls("D")


# ---------------------------------------------------------------------------------------
# Construction (Section 3)
# ---------------------------------------------------------------------------------------

def test_intersection_morphisms():
    lat = CatELattice(ontology(SUB(AND(A, B), C)))
    assert has(lat, AND(A, B), A) and has(lat, AND(A, B), B)
    assert has(lat, AND(A, B), C)
    # Subsumption axiom: ⊤ ⊑ ¬(A ⊓ B) ⊔ C.
    assert has(lat, TOPC, OR(NOT(AND(A, B)), C))


def test_union_morphisms():
    lat = CatELattice(ontology(SUB(A, OR(B, C))))
    assert has(lat, B, OR(B, C)) and has(lat, C, OR(B, C))
    assert has(lat, A, OR(B, C))
    assert has(lat, TOPC, OR(NOT(A), OR(B, C)))


def test_negation_morphisms_named_operand():
    lat = CatELattice(ontology(SUB(A, NOT(B))), complete_negation=False)
    assert has(lat, A, NOT(B))
    # Fig. 2c: B ⊓ ¬B ⊑ ⊥, ⊤ ⊑ B ⊔ ¬B with projections and injections.
    meet, join = AND(B, NOT(B)), OR(B, NOT(B))
    assert has(lat, meet, BOTC) and has(lat, TOPC, join)
    assert has(lat, meet, B) and has(lat, meet, NOT(B))
    assert has(lat, B, join) and has(lat, NOT(B), join)


def test_negation_morphisms_complex_operand_completed():
    c = AND(B, C)
    meet, join = AND(c, NOT(c)), OR(c, NOT(c))
    raw = CatELattice(ontology(SUB(A, NOT(c))), complete_negation=False)
    assert has(raw, meet, BOTC) and has(raw, TOPC, join)
    # The projector leaves the diagram of Fig. 2c incomplete for complex operands ...
    assert not has(raw, meet, c) and not has(raw, c, join)
    full = CatELattice(ontology(SUB(A, NOT(c))), complete_negation=True)
    # ... and mOWL completes it.
    assert has(full, meet, c) and has(full, meet, NOT(c))
    assert has(full, c, join) and has(full, NOT(c), join)


def test_existential_morphisms():
    ex = SOME(R, B)
    lat = CatELattice(ontology(SUB(A, ex)))
    role = Node(owl_class=ex, relation=R)
    dom = Node(owl_class=ex, relation=R, domain=True)
    cod = Node(owl_class=ex, relation=R, codomain=True)
    assert has(lat, A, ex)
    assert has(lat, role, Node(relation=R))           # r_{∃r.B} ⊑ r
    assert has(lat, cod, B)                           # codomain(r_{∃r.B}) ⊑ B
    assert has(lat, dom, ex) and has(lat, ex, dom)    # domain(r_{∃r.B}) ≡ ∃r.B
    assert lat.existential_id(NS + "r", lat.node_id(B)) == lat.node_id(ex)
    assert node_name(role) == f"Role({NS}r|{ex.toString()})"


def test_universal_morphisms():
    univ = ALL(R, B)
    lat = CatELattice(ontology(SUB(A, univ)))
    dual = NOT(SOME(R, NOT(B)))
    assert has(lat, A, univ)
    assert has(lat, univ, dual) and has(lat, dual, univ)   # ∀r.B ≡ ¬∃r.¬B
    assert SOME(R, NOT(B)) in [n.owl_class for n in lat.nodes if n.relation is None]


def test_top_and_bottom_morphisms():
    lat = CatELattice(ontology(SUB(A, OR(B, NOT(C))), SUB(AND(B, C), SOME(R, D))))
    for i, node in enumerate(lat.nodes):
        if i in (lat.bot_id, lat.top_id):
            continue
        assert lat.has_edge(lat.bot_id, i), node
        assert lat.has_edge(i, lat.top_id), node


def test_assertions():
    a, b = ind("a"), ind("b")
    lat = CatELattice(ontology(df.getOWLClassAssertionAxiom(A, a),
                               df.getOWLObjectPropertyAssertionAxiom(R, a, b)))
    ia = lat.individual_id(NS + "a")
    assert lat.has_edge(ia, lat.node_id(A))                         # {a} ⊑ A
    assert lat.has_edge(ia, lat.node_id(SOME(R, cls("b"))))         # {a} ⊑ ∃r.{b}
    assert lat.has_edge(lat.bot_id, ia) and lat.has_edge(ia, lat.top_id)
    with pytest.raises(UnseenConceptError):
        lat.individual_id(NS + "nobody")


def test_disjoint_and_equivalent_classes():
    lat = CatELattice(ontology(ad.create_equivalent_classes(A, AND(B, C)),
                               ad.create_disjoint_classes(A, D)))
    assert has(lat, A, AND(B, C)) and has(lat, AND(B, C), A)
    assert has(lat, A, NOT(D)) and has(lat, D, NOT(A))


def test_added_classes_and_existentials():
    lat = CatELattice(ontology(SUB(A, B)), classes=[NS + "Z"],
                      existentials={NS + "r": [NS + "A", NS + "Z"]})
    z = lat.class_to_id[NS + "Z"]
    assert lat.has_edge(lat.bot_id, z) and lat.has_edge(z, lat.top_id)
    ex = lat.existential_id(NS + "r", z)
    assert lat.nodes[ex] == Node(owl_class=SOME(R, cls("Z")))
    assert has(lat, Node(owl_class=SOME(R, cls("Z")), relation=R, codomain=True), cls("Z"))
    with pytest.raises(UnseenConceptError):
        lat.existential_id(NS + "r", lat.node_id(B))
    with pytest.raises(UnseenConceptError):
        lat.node_id(AND(A, cls("Z")))


def test_names_are_unique_and_unambiguous():
    # ¬(B ⊓ C) and (¬B) ⊓ C render identically in the projector's string output.
    lat = CatELattice(ontology(SUB(A, NOT(AND(B, C))), SUB(A, AND(NOT(B), C))))
    i, j = lat.node_id(NOT(AND(B, C))), lat.node_id(AND(NOT(B), C))
    assert i != j and lat.names[i] != lat.names[j]
    assert len(set(lat.names)) == len(lat.names)


# ---------------------------------------------------------------------------------------
# Saturation (Section 3.1)
# ---------------------------------------------------------------------------------------

def test_rule1_contraposition():
    lat = CatELattice(ontology(SUB(A, NOT(B))), saturation_steps=1,
                      rules=("contraposition",))
    assert has(lat, B, NOT(A))


def test_rule2_disjointness():
    lat0 = CatELattice(ontology(SUB(AND(A, B), BOTC)))
    assert not has(lat0, A, NOT(B))
    lat = CatELattice(ontology(SUB(AND(A, B), BOTC)), saturation_steps=1,
                      rules=("disjointness",))
    assert has(lat, A, NOT(B)) and has(lat, B, NOT(A))


def test_rule2_nary_splits():
    lat = CatELattice(ontology(SUB(AND(A, B, C), BOTC)), saturation_steps=1,
                      rules=("disjointness",))
    assert has(lat, A, NOT(AND(B, C)))
    assert has(lat, AND(A, B), NOT(C))


def test_rule3_excluded_middle():
    # Premise read as ⊤ ⊑ C ⊔ D (printed as ⊥ ⊑ C ⊔ D).
    lat = CatELattice(ontology(SUB(TOPC, OR(A, B))), saturation_steps=1,
                      rules=("excluded_middle",))
    assert has(lat, NOT(A), B) and has(lat, NOT(B), A)


def test_rule3_does_not_fire_without_top():
    lat = CatELattice(ontology(SUB(C, OR(A, B))), saturation_steps=1,
                      rules=("excluded_middle",))
    assert not has(lat, NOT(A), B)
    # ... but the subsumption axiom's ⊤ ⊑ ¬C ⊔ A ⊔ B gives C ⊑ A ⊔ B back.
    assert has(lat, C, OR(A, B))


def test_rule4_transitivity():
    ont = ontology(SUB(A, B), SUB(B, C))
    assert not has(CatELattice(ont), A, C)
    assert has(CatELattice(ont, transitive_closure=True), A, C)


def test_rule5_restricted_existential_monotonicity():
    ont = ontology(SUB(A, B), SUB(C, SOME(R, A)), SUB(D, SOME(R, B)))
    assert not has(CatELattice(ont), SOME(R, A), SOME(R, B))
    lat = CatELattice(ont, saturation_steps=1, rules=("existential_monotonicity",))
    assert has(lat, SOME(R, A), SOME(R, B))
    # Restricted form: no new existential elements.
    assert SOME(R, C) not in [n.owl_class for n in lat.nodes]


def test_s1_chain_and_termination():
    # A ⊓ B ⊑ ⊥ gives A ⊑ ¬B (rule 2); then rule 1 gives B ⊑ ¬A.
    lat = CatELattice(ontology(SUB(AND(A, B), BOTC), SUB(TOPC, OR(C, D)),
                               SUB(A, NOT(C))),
                      saturation_steps=None)
    stats = lat.stats["saturation"]
    assert stats["fixed_point"]
    assert stats["added"][-1] == 0
    assert has(lat, A, NOT(B)) and has(lat, B, NOT(A))
    assert has(lat, NOT(C), D) and has(lat, C, NOT(A))
    again = sat.saturate(lat.graph, rules=sat.S1 + sat.TR, max_iterations=None)
    assert again["fixed_point"]


def test_unknown_rule():
    with pytest.raises(ValueError):
        CatELattice(ontology(SUB(A, B)), saturation_steps=1, rules=("modus_ponens",))


# ---------------------------------------------------------------------------------------
# T3 Helly pattern (Leemhuis & Kutz): what the lattice contains
# ---------------------------------------------------------------------------------------

T3_PATH = os.path.join(os.path.dirname(__file__), "..", "geometric", "fixtures", "t3.owl")
GEO = "http://mowl.borg/geo#"


def t3_lattice(**kwargs):
    from mowl.datasets import PathDataset
    ds = PathDataset(T3_PATH)
    return CatELattice(ds.ontology, classes=ds.classes.as_str, **kwargs)


def test_t3_lattice_contents():
    g = {n: ad.create_class(GEO + n) for n in ("A", "B", "C", "X1", "X2", "X3", "Z")}
    lat = t3_lattice()
    for name, (x, y) in {"A": ("X1", "X2"), "B": ("X2", "X3"), "C": ("X1", "X3")}.items():
        meet = AND(g[x], g[y])
        assert has(lat, g[name], meet) and has(lat, meet, g[name])
        assert has(lat, meet, g[x]) and has(lat, meet, g[y])
    for x, y in (("A", "B"), ("B", "C"), ("A", "C")):
        assert has(lat, AND(g[x], g[y]), BOTC)
        # Disjointness becomes A ⊑ ¬B only through saturation rule (2).
        assert not has(lat, g[x], NOT(g[y]))
    # The Helly witness X1 ⊓ X2 ⊓ X3 (entailed to be unsatisfiable) is not an element.
    with pytest.raises(UnseenConceptError):
        lat.node_id(AND(g["X1"], g["X2"], g["X3"]))
    # Z is isolated apart from ⊤ and its own negation diagram (Z ⊑ Z ⊔ ¬Z).
    z = lat.class_to_id[GEO + "Z"]
    join = lat.node_id(OR(g["Z"], NOT(g["Z"])))
    assert lat.successors(z) == sorted({z, join, lat.top_id})

    sat_lat = t3_lattice(saturation_steps=None)
    assert sat_lat.stats["saturation"]["fixed_point"]
    for x, y in (("A", "B"), ("B", "C"), ("A", "C")):
        assert has(sat_lat, g[x], NOT(g[y])) and has(sat_lat, g[y], NOT(g[x]))
