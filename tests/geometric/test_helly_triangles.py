"""Tests for scalable Helly certificate triangle enumeration in mowl.geometric.helly."""
from __future__ import annotations

import random
from typing import FrozenSet, Set

import mowl
mowl.init_jvm("2g")

from mowl.geometric import certificate_triangles
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.errors import InconsistentOntology
from mowl.geometric.helly import helly_check
from mowl.geometric.normal_forms import BOTTOM, NF1, NF2, TOP, NormalisedOntology
from tests.geometric.helpers import (
    NS,
    helly_ok_cm,
    helly_ok_dataset,
    random_el_tbox,
    t3_cm,
    t3_dataset,
)

__all__ = [
    "helly_ok_cm",
    "helly_ok_dataset",
    "t3_cm",
    "t3_dataset",
]


def _direct_nf2_defined_pairs(cm: CanonicalModel) -> Set[FrozenSet[str]]:
    """Return pairs {A, B} that have a direct NF2(A, B, N) definition in cm.axioms."""
    pairs: Set[FrozenSet[str]] = set()
    for ax in cm.axioms:
        if not isinstance(ax, NF2):
            continue
        a, b, n = ax.sub1, ax.sub2, ax.sup
        if a == b or BOTTOM in (a, b, n) or TOP in (a, b, n):
            continue
        if (
            cm.is_satisfiable(a)
            and cm.is_satisfiable(b)
            and cm.is_satisfiable(n)
            and cm.entails(n, a)
            and cm.entails(n, b)
        ):
            pairs.add(frozenset((a, b)))
    return pairs


def _augment_with_named_conjunctions(
    no: NormalisedOntology, rng: random.Random
) -> NormalisedOntology:
    """Extend a 6-name random propositional TBox with <= 4 named binary conjunctions and disjointness (<= 12 names total)."""
    base_names = sorted(c for c in no.concept_names if c not in (TOP, BOTTOM))
    axioms = list(no.axioms)
    concept_names = set(no.concept_names)

    # Choose a triple and 1 extra pair to name as binary conjunctions P0..P3 (6 + 4 = 10 <= 12 names)
    triple = rng.sample(base_names, 3)
    pairs = [
        (triple[0], triple[1]),
        (triple[0], triple[2]),
        (triple[1], triple[2]),
    ]
    extra_pair = tuple(rng.sample(base_names, 2))
    if frozenset(extra_pair) not in {frozenset(p) for p in pairs}:
        pairs.append((extra_pair[0], extra_pair[1]))

    p_names = []
    for idx, (u, v) in enumerate(pairs):
        p_name = f"{NS}P{idx}"
        p_names.append(p_name)
        concept_names.add(p_name)
        axioms.append(NF2(u, v, p_name))
        axioms.append(NF1(p_name, u))
        axioms.append(NF1(p_name, v))

    # Add random disjointness between some named conjunctions
    for i in range(len(p_names)):
        for j in range(i + 1, len(p_names)):
            if rng.random() < 0.6:
                axioms.append(NF2(p_names[i], p_names[j], BOTTOM))

    assert len(concept_names - {TOP, BOTTOM}) <= 12
    return NormalisedOntology(
        axioms=axioms,
        concept_names=concept_names,
        role_names=set(),
        dropped_axioms=0,
    )


def test_t3_triangle_matches_helly_check(t3_cm: CanonicalModel) -> None:
    tri = certificate_triangles(t3_cm)
    assert len(tri) == 1
    c = tri[0]
    assert (c.c1, c.c2, c.c3) == ((NS + "X1",), (NS + "X2",), (NS + "X3",))
    assert set(c.pairwise_names) == {NS + "A", NS + "B", NS + "C"}
    assert c.pairwise_names == (NS + "A", NS + "C", NS + "B")
    rep = helly_check(t3_cm)
    assert {(x.c1, x.c2, x.c3) for x in rep.certificates_nondegenerate} == {
        (c.c1, c.c2, c.c3)
    }


def test_helly_ok_has_no_triangles(helly_ok_cm: CanonicalModel) -> None:
    assert certificate_triangles(helly_ok_cm) == []


def test_random_propositional_subset_of_helly_check() -> None:
    # 30 random propositional TBoxes (seeded 0..29), <= 12 names, with named binary
    # conjunctions and random disjointness; build via tests/geometric/helpers (existing
    # random TBox builder used by test_completion) — every triangle must be a
    # nondegenerate certificate of helly_check (as a set of three singleton names), and
    # every helly_check nondegenerate certificate whose three pairwise names are direct
    # NF2 definitions must be returned.
    total_triangles = 0
    for seed in range(30):
        rng = random.Random(seed)
        base_no = random_el_tbox(rng, allow_roles=False)
        aug_no = _augment_with_named_conjunctions(base_no, rng)

        for no in (base_no, aug_no):
            assert len(no.concept_names - {TOP, BOTTOM}) <= 12
            try:
                cm = CanonicalModel.from_normalised(no)
            except InconsistentOntology:
                continue

            tri = certificate_triangles(cm)
            total_triangles += len(tri)

            # Verify sorted order and lexicographical X1 < X2 < X3
            tri_keys = [(c.c1, c.c2, c.c3) for c in tri]
            assert tri_keys == sorted(tri_keys)
            for c in tri:
                assert len(c.c1) == 1 and len(c.c2) == 1 and len(c.c3) == 1
                assert c.c1[0] < c.c2[0] < c.c3[0]
                assert all(p is not None for p in c.pairwise_names)

            rep = helly_check(cm, k_max=12, max_certificates=1000)
            assert rep.complete

            helly_nondeg_triples = {
                (x.c1, x.c2, x.c3)
                for x in rep.certificates_nondegenerate
                if len(x.c1) == 1 and len(x.c2) == 1 and len(x.c3) == 1
            }
            tri_triples = set(tri_keys)

            # Every triangle must be a nondegenerate certificate of helly_check
            assert tri_triples.issubset(helly_nondeg_triples)

            # Every helly_check nondegenerate singleton certificate whose three pairwise
            # conjunctions are direct NF2 definitions must be returned by certificate_triangles
            direct_pairs = _direct_nf2_defined_pairs(cm)
            direct_helly_triples = {
                (x.c1, x.c2, x.c3)
                for x in rep.certificates_nondegenerate
                if len(x.c1) == 1
                and len(x.c2) == 1
                and len(x.c3) == 1
                and frozenset((x.c1[0], x.c2[0])) in direct_pairs
                and frozenset((x.c1[0], x.c3[0])) in direct_pairs
                and frozenset((x.c2[0], x.c3[0])) in direct_pairs
            }
            assert direct_helly_triples == tri_triples

    assert total_triangles > 0


def test_max_triangles_truncates(t3_cm: CanonicalModel) -> None:
    assert certificate_triangles(t3_cm, max_triangles=0) == []
    assert len(certificate_triangles(t3_cm, max_triangles=1)) == 1
