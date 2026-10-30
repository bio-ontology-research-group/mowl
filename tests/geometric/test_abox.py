"""Tests for ABox support in mowl.geometric (individuals as nominal names)."""
from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

import pytest

import mowl
mowl.init_jvm("2g")

from org.semanticweb.elk.owlapi import ElkReasonerFactory
from org.semanticweb.owlapi.model import OWLOntology

import mowl.geometric as geo
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.normal_forms import BOTTOM, TOP, normalise
from mowl.owlapi import OWLAPIAdapter

NS = "http://mowl.borg/geo#"
IND = "http://mowl.borg/geo/ind#"


@pytest.fixture
def abox_kb() -> OWLOntology:
    """Build a small EL KB with TBox + ABox covering class and role assertions."""
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology("http://mowl.borg/geo/abox_test")

    top = adapter.create_class(TOP)
    bot = adapter.create_class(BOTTOM)
    father = adapter.create_class(NS + "Father")
    mother = adapter.create_class(NS + "Mother")
    parent = adapter.create_class(NS + "Parent")
    male = adapter.create_class(NS + "Male")
    female = adapter.create_class(NS + "Female")
    has_child = adapter.create_object_property(NS + "hasChild")

    bob = adapter.create_individual(IND + "bob")
    ann = adapter.create_individual(IND + "ann")
    carl = adapter.create_individual(IND + "carl")

    # TBox:
    # Father <= Parent, Mother <= Parent
    mgr.addAxiom(ont, adapter.create_subclass_of(father, parent))
    mgr.addAxiom(ont, adapter.create_subclass_of(mother, parent))
    # exists hasChild.TOP <= Parent
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_some_values_from(has_child, top), parent
        ),
    )
    # Male and Female <= BOTTOM
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(male, female), bot
        ),
    )

    # ABox:
    # Father(bob), Male(bob), hasChild(bob, ann), hasChild(carl, ann)
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(father, bob))
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(male, bob))
    mgr.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(has_child, bob, ann))
    mgr.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(has_child, carl, ann))

    return ont


def _build_random_abox_kb(seed: int) -> Tuple[OWLOntology, List[str], List[str]]:
    """Build a seeded random small EL KB with 3-6 individuals."""
    rng = random.Random(seed)
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology(f"http://mowl.borg/geo/random_abox_{seed}")

    top = adapter.create_class(TOP)
    bot = adapter.create_class(BOTTOM)
    class_iris = [f"{NS}C{i}" for i in range(6)]
    role_iris = [f"{NS}r0", f"{NS}r1"]
    n_inds = rng.randint(3, 6)
    ind_iris = [f"{IND}a{i}" for i in range(n_inds)]

    classes = {iri: adapter.create_class(iri) for iri in class_iris}
    roles = {iri: adapter.create_object_property(iri) for iri in role_iris}
    inds = {iri: adapter.create_individual(iri) for iri in ind_iris}

    for c_obj in classes.values():
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(c_obj))
    for r_obj in roles.values():
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(r_obj))
    for i_obj in inds.values():
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(i_obj))

    # Disjoint pair C4 and C5 <= BOTTOM (assertions draw from C0..C4 so KB stays consistent)
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(
                classes[class_iris[4]], classes[class_iris[5]]
            ),
            bot,
        ),
    )

    safe_classes = class_iris[:5]
    # Random TBox axioms over safe_classes
    for _ in range(rng.randint(5, 8)):
        kind = rng.choice(["nf1", "nf2", "nf3", "nf4", "nf6"])
        if kind == "nf1":
            c1, c2 = rng.sample(safe_classes, 2)
            mgr.addAxiom(ont, adapter.create_subclass_of(classes[c1], classes[c2]))
        elif kind == "nf2":
            c1, c2, c3 = rng.sample(safe_classes, 3)
            mgr.addAxiom(
                ont,
                adapter.create_subclass_of(
                    adapter.create_object_intersection_of(classes[c1], classes[c2]),
                    classes[c3],
                ),
            )
        elif kind == "nf3":
            c1, c2 = rng.sample(safe_classes, 2)
            r = rng.choice(role_iris)
            mgr.addAxiom(
                ont,
                adapter.create_subclass_of(
                    classes[c1],
                    adapter.create_object_some_values_from(roles[r], classes[c2]),
                ),
            )
        elif kind == "nf4":
            c1, c2 = rng.sample(safe_classes, 2)
            r = rng.choice(role_iris)
            filler = top if rng.random() < 0.3 else classes[c1]
            mgr.addAxiom(
                ont,
                adapter.create_subclass_of(
                    adapter.create_object_some_values_from(roles[r], filler),
                    classes[c2],
                ),
            )
        else:
            r1, r2 = rng.sample(role_iris, 2)
            mgr.addAxiom(ont, df.getOWLSubObjectPropertyOfAxiom(roles[r1], roles[r2]))

    # Random ABox class assertions and role assertions
    for _ in range(rng.randint(3, 6)):
        a = rng.choice(ind_iris)
        c = rng.choice(safe_classes)
        mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(classes[c], inds[a]))

    for _ in range(rng.randint(2, 5)):
        a = rng.choice(ind_iris)
        b = rng.choice(ind_iris)
        r = rng.choice(role_iris)
        mgr.addAxiom(
            ont, df.getOWLObjectPropertyAssertionAxiom(roles[r], inds[a], inds[b])
        )

    return ont, ind_iris, class_iris


@pytest.fixture
def abox_kb_random() -> List[Tuple[int, OWLOntology, List[str], List[str]]]:
    """Return 20 seeded random small KBs (seeds 0..19) with 3-6 individuals each."""
    return [(seed, *_build_random_abox_kb(seed)) for seed in range(20)]


def test_class_assertion_entailed_via_hierarchy(abox_kb):       # Father ⊑ Parent, Father(bob)
    cm = CanonicalModel.from_ontology(abox_kb)                   # use the existing constructor path
    assert cm.instance_of(IND+"bob", NS+"Parent")
    assert not cm.instance_of(IND+"bob", NS+"Mother")


def test_role_assertion_propagates_existential(abox_kb):        # hasChild(bob, ann), ∃hasChild.⊤ ⊑ Parent
    cm = CanonicalModel.from_ontology(abox_kb)
    assert cm.instance_of(IND+"bob", NS+"Parent")
    assert cm.instance_of(IND+"carl", NS+"Parent")
    assert not cm.instance_of(IND+"ann", NS+"Parent")
    assert (IND+"bob", IND+"ann") in cm.role_instances(NS+"hasChild")
    assert (IND+"carl", IND+"ann") in cm.role_instances(NS+"hasChild")


def test_assertion_contradicted(abox_kb):                        # Male ⊓ Female ⊑ ⊥, Male(bob)
    cm = CanonicalModel.from_ontology(abox_kb)
    assert cm.assertion_contradicted(IND+"bob", NS+"Female")
    assert not cm.assertion_contradicted(IND+"bob", NS+"Parent")


def test_individual_with_only_role_assertions(abox_kb):          # hasChild(carl, ann) only
    cm = CanonicalModel.from_ontology(abox_kb)
    assert IND+"carl" in cm.individuals and cm.instance_of(IND+"carl", TOP)


def test_concept_names_exclude_nominals(abox_kb):
    from mowl.geometric.normal_forms import individual_of_name, is_nominal_name

    cm = CanonicalModel.from_ontology(abox_kb)
    assert set(cm.individuals.keys()) == {IND + "bob", IND + "ann", IND + "carl"}
    for ind_iri, nom_name in cm.individuals.items():
        assert is_nominal_name(nom_name)
        assert nom_name.startswith(geo.NOMINAL_PREFIX)
        assert individual_of_name(nom_name) == ind_iri
        assert nom_name in cm.names
        assert nom_name not in cm.concept_names

    assert not any(geo.is_nominal_name(c) for c in cm.concept_names)
    assert {
        NS + "Father",
        NS + "Mother",
        NS + "Parent",
        NS + "Male",
        NS + "Female",
        TOP,
        BOTTOM,
    }.issubset(cm.concept_names)


def test_complex_class_assertion_raises():                       # ClassAssertion(∃r.C, a) → UnsupportedAxiomError
    from mowl.geometric.errors import UnsupportedAxiomError

    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology("http://mowl.borg/geo/complex_ca")

    r = adapter.create_object_property(NS + "r")
    c = adapter.create_class(NS + "C")
    a = adapter.create_individual(IND + "a")
    complex_ce = adapter.create_object_some_values_from(r, c)
    ax = df.getOWLClassAssertionAxiom(complex_ce, a)
    mgr.addAxiom(ont, ax)

    with pytest.raises(UnsupportedAxiomError) as exc_info:
        normalise(ont)
    assert str(ax) in str(exc_info.value)


def test_matches_elk_instance_checking(abox_kb_random):           # 20 random small KBs: cm.instance_of == ELK getTypes (helpers' ELK access as in test_completion.py)
    adapter = OWLAPIAdapter()
    assert len(abox_kb_random) == 20

    for seed, ont, ind_iris, class_iris in abox_kb_random:
        assert 3 <= len(ind_iris) <= 6
        cm = CanonicalModel.from_ontology(ont)

        reasoner = ElkReasonerFactory().createReasoner(ont)
        try:
            reasoner.precomputeInferences()
            assert bool(reasoner.isConsistent()), f"Seed {seed} unexpectedly inconsistent"
            for ind_iri in ind_iris:
                ind_obj = adapter.create_individual(ind_iri)
                elk_types: Set[str] = {
                    str(cls.toStringID())
                    for cls in reasoner.getTypes(ind_obj, False).getFlattened()
                }
                for cls_iri in class_iris + [TOP]:
                    expected = cls_iri in elk_types
                    actual = cm.instance_of(ind_iri, cls_iri)
                    assert actual == expected, (
                        f"Instance mismatch on seed {seed} for ({ind_iri}, {cls_iri}): "
                        f"cm={actual}, ELK={expected}"
                    )
                assert not cm.instance_of(ind_iri, BOTTOM)
        finally:
            reasoner.dispose()
