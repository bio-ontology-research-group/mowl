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
    assert cm.role_instances(NS+"hasChild") == {
        (IND+"bob", IND+"ann"),
        (IND+"carl", IND+"ann"),
    }
    assert cm.role_instances(NS+"unknownRole") == set()


def test_assertion_contradicted(abox_kb):                        # Male ⊓ Female ⊑ ⊥, Male(bob)
    cm = CanonicalModel.from_ontology(abox_kb)
    assert cm.assertion_contradicted(IND+"bob", NS+"Female")
    assert not cm.assertion_contradicted(IND+"bob", NS+"Parent")
    assert not cm.assertion_contradicted(IND+"unknown", TOP)
    assert not cm.assertion_contradicted(IND+"unknown", BOTTOM)
    assert not cm.assertion_contradicted(IND+"bob", NS+"UnknownClass")
    assert not cm.instance_of(IND+"unknown", TOP)
    assert not cm.instance_of(IND+"bob", NS+"UnknownClass")


def test_individual_with_only_role_assertions(abox_kb):          # hasChild(carl, ann) only
    cm = CanonicalModel.from_ontology(abox_kb)
    assert IND+"carl" in cm.individuals and cm.instance_of(IND+"carl", TOP)


def test_element_of(abox_kb):
    cm = CanonicalModel.from_ontology(abox_kb)
    for cname in cm.names:
        pos = cm.element_of(cname)
        assert cm.names[pos] == cname
        assert cm.domain[pos] == f"d_{cname}"
    for ind_iri, nom_name in cm.individuals.items():
        pos_ind = cm.element_of(ind_iri)
        pos_nom = cm.element_of(nom_name)
        assert pos_ind == pos_nom
        assert cm.names[pos_ind] == nom_name
    with pytest.raises(KeyError):
        cm.element_of(IND + "unknown_individual")
    with pytest.raises(KeyError):
        cm.element_of(BOTTOM)


def test_interval_model_on_abox_kb(abox_kb):
    from mowl.geometric.intervals import IntervalUnionModel

    cm = CanonicalModel.from_ontology(abox_kb)
    ivl = IntervalUnionModel.from_canonical_model(cm)
    assert ivl.is_model()
    assert ivl.is_nondegenerate()
    assert ivl.is_strongly_faithful_on_names()
    for ind_iri, nom_name in cm.individuals.items():
        assert nom_name in ivl.class_regions, f"Missing {nom_name} for {ind_iri}"
        assert not ivl.class_region(nom_name).is_empty
        assert not ivl.class_regions[nom_name].is_empty


def test_polyhedral_model_on_abox_kb(abox_kb):
    from mowl.geometric.polyhedral import ConvexPolyhedralModel

    cm = CanonicalModel.from_ontology(abox_kb)
    poly = ConvexPolyhedralModel.from_canonical_model(cm)
    assert poly.is_model()
    assert poly.is_nondegenerate()
    assert poly.is_strongly_faithful_on_names()
    for ind_iri, nom_name in cm.individuals.items():
        assert nom_name in poly.class_regions, f"Missing {nom_name} for {ind_iri}"
        assert not poly.class_region(nom_name).is_empty
        assert not poly.class_regions[nom_name].is_empty


def test_box_model_on_abox_kb(abox_kb):
    from mowl.geometric.boxes import BoxModelBuilder
    from mowl.geometric.normal_forms import NF1, NF2, NormalisedOntology

    no = normalise(abox_kb)
    prop_axioms = [ax for ax in no.axioms if isinstance(ax, (NF1, NF2))]
    prop_no = NormalisedOntology(
        axioms=prop_axioms,
        concept_names=no.concept_names,
        role_names=set(),
        dropped_axioms=0,
    )
    cm = CanonicalModel.from_normalised(prop_no)
    box_model = BoxModelBuilder(cm, faithful="strong").build()
    assert box_model.is_model()
    assert box_model.is_nondegenerate()
    assert box_model.is_strongly_faithful_on_names()
    for ind_iri, nom_name in cm.individuals.items():
        assert nom_name in box_model.class_regions, f"Missing {nom_name} for {ind_iri}"
        assert not box_model.class_region(nom_name).is_empty
        assert not box_model.class_regions[nom_name].is_empty


def test_unsatisfiable_nominal_raises_inconsistent_ontology():
    from mowl.geometric.errors import InconsistentOntology

    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory

    # 1. Direct clash on bob: Male ⊓ Female ⊑ ⊥, Male(bob), Female(bob)
    ont1 = adapter.create_ontology("http://mowl.borg/geo/inconsistent_abox_direct")
    male = adapter.create_class(NS + "Male")
    female = adapter.create_class(NS + "Female")
    bot = adapter.create_class(BOTTOM)
    bob = adapter.create_individual(IND + "bob")
    mgr.addAxiom(
        ont1,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(male, female), bot
        ),
    )
    mgr.addAxiom(ont1, df.getOWLClassAssertionAxiom(male, bob))
    mgr.addAxiom(ont1, df.getOWLClassAssertionAxiom(female, bob))
    with pytest.raises(InconsistentOntology) as exc1:
        CanonicalModel.from_ontology(ont1)
    assert (IND + "bob") in str(exc1.value)

    # 2. Propagated clash via role assertion: A ⊓ B ⊑ ⊥, B(a), r(a,b), B(b), ∃r.B ⊑ A
    ont2 = adapter.create_ontology("http://mowl.borg/geo/inconsistent_abox_role")
    cls_a = adapter.create_class(NS + "A")
    cls_b = adapter.create_class(NS + "B")
    role_r = adapter.create_object_property(NS + "r")
    ind_a = adapter.create_individual(IND + "a")
    ind_b = adapter.create_individual(IND + "b")
    mgr.addAxiom(
        ont2,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(cls_a, cls_b), bot
        ),
    )
    mgr.addAxiom(
        ont2,
        adapter.create_subclass_of(
            adapter.create_object_some_values_from(role_r, cls_b), cls_a
        ),
    )
    mgr.addAxiom(ont2, df.getOWLClassAssertionAxiom(cls_b, ind_a))
    mgr.addAxiom(ont2, df.getOWLClassAssertionAxiom(cls_b, ind_b))
    mgr.addAxiom(ont2, df.getOWLObjectPropertyAssertionAxiom(role_r, ind_a, ind_b))
    with pytest.raises(InconsistentOntology) as exc2:
        CanonicalModel.from_ontology(ont2)
    assert (IND + "a") in str(exc2.value)


def test_concept_names_exclude_nominals(abox_kb):
    from mowl.geometric.normal_forms import (
        individual_of_name,
        is_nominal_name,
        nominal_name,
        to_owl,
    )

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

    # Special characters round-trip and to_owl round-trip
    special_iris = [
        "http://example.org/ind#bob smith%20?x=1&y=2",
        "http://example.org/ind/ümlaut_λ",
        "urn:uuid:12345678-1234-5678-1234-567812345678",
    ]
    for s_iri in special_iris:
        enc = nominal_name(s_iri)
        assert is_nominal_name(enc)
        assert individual_of_name(enc) == s_iri

    rt_cm = CanonicalModel.from_ontology(to_owl(normalise(abox_kb)))
    assert rt_cm.individuals == cm.individuals
    assert rt_cm.instance_of(IND + "bob", NS + "Parent")


def test_complex_class_assertion_raises():                       # ClassAssertion(∃r.C, a) → UnsupportedAxiomError
    from java.util import HashSet
    from mowl.geometric.errors import UnsupportedAxiomError

    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology("http://mowl.borg/geo/complex_ca")

    r = adapter.create_object_property(NS + "r")
    c = adapter.create_class(NS + "C")
    a = adapter.create_individual(IND + "a")
    b = adapter.create_individual(IND + "b")
    anon = df.getOWLAnonymousIndividual()
    complex_ce = adapter.create_object_some_values_from(r, c)
    ax = df.getOWLClassAssertionAxiom(complex_ce, a)
    mgr.addAxiom(ont, ax)

    with pytest.raises(UnsupportedAxiomError) as exc_info:
        normalise(ont)
    assert str(ax) in str(exc_info.value)

    # Also verify SameIndividual, DifferentIndividuals, inverse ObjectPropertyAssertion,
    # and assertions on anonymous individuals raise UnsupportedAxiomError with axiom text
    inds_set = HashSet()
    inds_set.add(a)
    inds_set.add(b)
    unsupported_axioms = [
        df.getOWLSameIndividualAxiom(inds_set),
        df.getOWLDifferentIndividualsAxiom(inds_set),
        df.getOWLObjectPropertyAssertionAxiom(df.getOWLObjectInverseOf(r), a, b),
        df.getOWLClassAssertionAxiom(c, anon),
        df.getOWLObjectPropertyAssertionAxiom(r, a, anon),
    ]
    for idx, bad_ax in enumerate(unsupported_axioms):
        ont_bad = adapter.create_ontology(f"http://mowl.borg/geo/unsupported_abox_{idx}")
        mgr.addAxiom(ont_bad, bad_ax)
        with pytest.raises(UnsupportedAxiomError) as bad_exc:
            normalise(ont_bad)
        assert str(bad_ax) in str(bad_exc.value)


def test_matches_elk_instance_checking(abox_kb_random):           # 20 random small KBs: cm.instance_of == ELK getTypes (helpers' ELK access as in test_completion.py)
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    assert len(abox_kb_random) == 20

    total_contradicted = 0
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

        # Compare cm.assertion_contradicted(ind, cls) with ELK (KB + cls(ind) inconsistent)
        for ind_iri in ind_iris:
            ind_obj = adapter.create_individual(ind_iri)
            for cls_iri in class_iris:
                cls_obj = adapter.create_class(cls_iri)
                probe_ax = df.getOWLClassAssertionAxiom(cls_obj, ind_obj)
                already_present = bool(ont.containsAxiom(probe_ax))
                if not already_present:
                    mgr.addAxiom(ont, probe_ax)
                probe_reasoner = ElkReasonerFactory().createReasoner(ont)
                try:
                    elk_contradicted = not bool(probe_reasoner.isConsistent())
                finally:
                    probe_reasoner.dispose()
                    if not already_present:
                        mgr.removeAxiom(ont, probe_ax)
                actual_contradicted = cm.assertion_contradicted(ind_iri, cls_iri)
                assert actual_contradicted == elk_contradicted, (
                    f"Contradiction mismatch on seed {seed} for ({ind_iri}, {cls_iri}): "
                    f"cm={actual_contradicted}, ELK={elk_contradicted}"
                )
                if actual_contradicted:
                    total_contradicted += 1

    assert total_contradicted > 0
