"""Generate test fixture OWL ontologies for mowl.geometric.

Fixtures generated in ``tests/geometric/fixtures/`` (namespace ``http://mowl.borg/geo#``):
- ``t3.owl``: Leemhuis & Kutz (NeSy 2025) T3 counterexample to box representability,
  plus an isolated declared class ``Z``.
- ``helly_ok.owl``: Propositional EL_bot ontology where every inclusion-minimal
  jointly-unsatisfiable set of concept names has size <= 2 (verified via ELK).
  Expected minimal jointly-unsatisfiable sets of concept names:
  {A, C}, {A, E}, {B, E}, {C, D}, {C, E}, {D, E}, {D, F}.
- ``roles.owl``: EL++ ontology with existential restrictions, role hierarchy
  (r <= s), and role chain (r o s <= t).
- ``top_lhs.owl``: Ontology with owl:Thing on the left-hand side of an
  existential restriction (top <= exists r.B).
"""
from __future__ import annotations

import itertools
import os
from typing import Dict, List, Set

import mowl
mowl.init_jvm("2g")

from java.util import ArrayList
from org.semanticweb.elk.owlapi import ElkReasonerFactory
from org.semanticweb.owlapi.model import IRI, OWLOntology

from mowl.owlapi import OWLAPIAdapter
from mowl.owlapi.defaults import BOT, TOP

NS = "http://mowl.borg/geo#"
ONT_BASE = "http://mowl.borg/geo"
FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


def _save(adapter: OWLAPIAdapter, ont: OWLOntology, filename: str) -> str:
    path = os.path.join(FIXTURES_DIR, filename)
    adapter.owl_manager.saveOntology(ont, IRI.create("file:" + os.path.abspath(path)))
    return path


def build_t3(adapter: OWLAPIAdapter) -> OWLOntology:
    """Build t3.owl:
    A ≡ X1 ⊓ X2, B ≡ X2 ⊓ X3, C ≡ X1 ⊓ X3,
    A ⊓ B ⊑ ⊥, B ⊓ C ⊑ ⊥, A ⊓ C ⊑ ⊥, and declared isolated class Z.
    """
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology(f"{ONT_BASE}/t3")
    bot = adapter.create_class(BOT)
    cls = {n: adapter.create_class(NS + n) for n in ["A", "B", "C", "X1", "X2", "X3", "Z"]}

    mgr.addAxiom(
        ont,
        adapter.create_equivalent_classes(
            cls["A"], adapter.create_object_intersection_of(cls["X1"], cls["X2"])
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_equivalent_classes(
            cls["B"], adapter.create_object_intersection_of(cls["X2"], cls["X3"])
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_equivalent_classes(
            cls["C"], adapter.create_object_intersection_of(cls["X1"], cls["X3"])
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(cls["A"], cls["B"]), bot
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(cls["B"], cls["C"]), bot
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(cls["A"], cls["C"]), bot
        ),
    )
    mgr.addAxiom(ont, df.getOWLDeclarationAxiom(cls["Z"]))
    return ont


def build_helly_ok(adapter: OWLAPIAdapter) -> OWLOntology:
    """Build helly_ok.owl:
    A ⊑ B, C ⊑ B, A ⊓ C ⊑ ⊥, D ⊑ A, E ⊓ B ⊑ ⊥, F ⊓ D ⊑ E.

    Expected inclusion-minimal jointly-unsatisfiable sets of concept names
    (all of size <= 2):
    {A, C}, {A, E}, {B, E}, {C, D}, {C, E}, {D, E}, {D, F}.
    """
    mgr = adapter.owl_manager
    ont = adapter.create_ontology(f"{ONT_BASE}/helly_ok")
    bot = adapter.create_class(BOT)
    cls = {n: adapter.create_class(NS + n) for n in ["A", "B", "C", "D", "E", "F"]}

    mgr.addAxiom(ont, adapter.create_subclass_of(cls["A"], cls["B"]))
    mgr.addAxiom(ont, adapter.create_subclass_of(cls["C"], cls["B"]))
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(cls["A"], cls["C"]), bot
        ),
    )
    mgr.addAxiom(ont, adapter.create_subclass_of(cls["D"], cls["A"]))
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(cls["E"], cls["B"]), bot
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(cls["F"], cls["D"]), cls["E"]
        ),
    )
    return ont


def verify_helly_ok_with_elk(adapter: OWLAPIAdapter, ont: OWLOntology) -> List[List[str]]:
    """Verify with ELK that every minimal jointly-unsatisfiable set of concept names
    in helly_ok has size <= 2."""
    mgr = adapter.owl_manager
    names = ["A", "B", "C", "D", "E", "F"]
    cls = {n: adapter.create_class(NS + n) for n in names}

    probe_axioms = []
    z_map: Dict[tuple, object] = {}
    for r in range(1, len(names) + 1):
        for comb in itertools.combinations(names, r):
            z = adapter.create_class(NS + "_probe_" + "_".join(comb))
            z_map[comb] = z
            if len(comb) == 1:
                ax = adapter.create_subclass_of(z, cls[comb[0]])
            else:
                ax = adapter.create_subclass_of(
                    z, adapter.create_object_intersection_of(*[cls[x] for x in comb])
                )
            mgr.addAxiom(ont, ax)
            probe_axioms.append(ax)

    reasoner = ElkReasonerFactory().createReasoner(ont)
    reasoner.precomputeInferences()
    try:
        unsat: List[Set[str]] = []
        for comb, z in z_map.items():
            if not reasoner.isSatisfiable(z):
                unsat.append(set(comb))
    finally:
        reasoner.dispose()
        for ax in probe_axioms:
            mgr.removeAxiom(ont, ax)

    minimal_unsat = [
        sorted(s) for s in unsat if not any(m < s for m in unsat)
    ]
    minimal_unsat.sort()
    expected = [["A", "C"], ["A", "E"], ["B", "E"], ["C", "D"], ["C", "E"], ["D", "E"], ["D", "F"]]
    assert all(len(s) <= 2 for s in minimal_unsat), f"Found minimal unsat of size > 2: {minimal_unsat}"
    assert minimal_unsat == expected, f"Expected {expected}, got {minimal_unsat}"
    return minimal_unsat


def build_roles(adapter: OWLAPIAdapter) -> OWLOntology:
    """Build roles.owl:
    A ⊑ ∃r.B, ∃s.B ⊑ C, r ⊑ s, r ∘ s ⊑ t, ∃t.B ⊑ D, B ⊑ ∃r.B, E ⊓ C ⊑ ⊥, E ⊑ ∃r.B.
    """
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology(f"{ONT_BASE}/roles")
    bot = adapter.create_class(BOT)
    cls = {n: adapter.create_class(NS + n) for n in ["A", "B", "C", "D", "E"]}
    roles = {n: adapter.create_object_property(NS + n) for n in ["r", "s", "t"]}

    # A ⊑ ∃r.B
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            cls["A"], adapter.create_object_some_values_from(roles["r"], cls["B"])
        ),
    )
    # ∃s.B ⊑ C
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_some_values_from(roles["s"], cls["B"]), cls["C"]
        ),
    )
    # r ⊑ s
    mgr.addAxiom(ont, df.getOWLSubObjectPropertyOfAxiom(roles["r"], roles["s"]))
    # r ∘ s ⊑ t
    chain = ArrayList()
    chain.add(roles["r"])
    chain.add(roles["s"])
    mgr.addAxiom(ont, df.getOWLSubPropertyChainOfAxiom(chain, roles["t"]))
    # ∃t.B ⊑ D
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_some_values_from(roles["t"], cls["B"]), cls["D"]
        ),
    )
    # B ⊑ ∃r.B
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            cls["B"], adapter.create_object_some_values_from(roles["r"], cls["B"])
        ),
    )
    # E ⊓ C ⊑ ⊥
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(cls["E"], cls["C"]), bot
        ),
    )
    # E ⊑ ∃r.B
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            cls["E"], adapter.create_object_some_values_from(roles["r"], cls["B"])
        ),
    )
    return ont


def build_top_lhs(adapter: OWLAPIAdapter) -> OWLOntology:
    """Build top_lhs.owl:
    ⊤ ⊑ ∃r.B, A ⊑ ∃r.C, C ⊑ B.
    """
    mgr = adapter.owl_manager
    ont = adapter.create_ontology(f"{ONT_BASE}/top_lhs")
    top = adapter.create_class(TOP)
    cls = {n: adapter.create_class(NS + n) for n in ["A", "B", "C"]}
    r = adapter.create_object_property(NS + "r")

    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(top, adapter.create_object_some_values_from(r, cls["B"])),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(cls["A"], adapter.create_object_some_values_from(r, cls["C"])),
    )
    mgr.addAxiom(ont, adapter.create_subclass_of(cls["C"], cls["B"]))
    return ont


def main() -> None:
    adapter = OWLAPIAdapter()

    t3 = build_t3(adapter)
    _save(adapter, t3, "t3.owl")

    helly_ok = build_helly_ok(adapter)
    verify_helly_ok_with_elk(adapter, helly_ok)
    _save(adapter, helly_ok, "helly_ok.owl")

    roles = build_roles(adapter)
    _save(adapter, roles, "roles.owl")

    top_lhs = build_top_lhs(adapter)
    _save(adapter, top_lhs, "top_lhs.owl")


if __name__ == "__main__":
    main()
