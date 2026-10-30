"""Tests for CanonicalMixtureModule in mowl.nn.el.canonical_mixture."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import pytest
import torch

import mowl
mowl.init_jvm("2g")

from org.semanticweb.owlapi.model import OWLOntology

from mowl.geometric.completion import CanonicalModel
from mowl.geometric.membership import MembershipStructure
from mowl.geometric.normal_forms import BOTTOM, TOP
from mowl.nn.el.canonical_mixture import CanonicalMixtureModule
from mowl.owlapi import OWLAPIAdapter
from tests.geometric.test_abox import IND, NS, abox_kb  # noqa: F401
from tests.geometric.test_membership import _dicts_for_abox_kb


def _build_module_for_abox_kb(
    abox_kb: OWLOntology,
    rank: int = 16,
    entropy_reg: float = 0.0,
    init_self_mass: float = 0.9,
) -> Tuple[CanonicalMixtureModule, CanonicalModel, MembershipStructure]:
    cm = CanonicalModel.from_ontology(abox_kb)
    class_index_dict, individual_index_dict, relation_index_dict = _dicts_for_abox_kb()
    ms = MembershipStructure.from_canonical_model(
        cm, class_index_dict, individual_index_dict, relation_index_dict
    )
    # Training assertions on bob: Father(bob), Male(bob)
    bob_idx = individual_index_dict[IND + "bob"]
    training_ca = torch.tensor(
        [
            [bob_idx, class_index_dict[NS + "Father"]],
            [bob_idx, class_index_dict[NS + "Male"]],
        ],
        dtype=torch.long,
    )
    module = CanonicalMixtureModule(
        ms,
        rank=rank,
        entropy_reg=entropy_reg,
        init_self_mass=init_self_mass,
        training_class_assertions=training_ca,
    )
    return module, cm, ms


def test_entailed_and_contradicted_scores_at_init(abox_kb: OWLOntology) -> None:
    """Entailed training assertion scores == 0 (prob 1) at init; contradicted == inf (prob 0)."""
    torch.manual_seed(0)
    module, _, ms = _build_module_for_abox_kb(abox_kb)
    cls_dict = ms.class_index_dict
    ind_dict = ms.individual_index_dict

    bob = ind_dict[IND + "bob"]
    father = cls_dict[NS + "Father"]
    male = cls_dict[NS + "Male"]
    parent = cls_dict[NS + "Parent"]
    female = cls_dict[NS + "Female"]

    # Asserted and entailed for bob -> score == 0
    entailed_data = torch.tensor(
        [[bob, father], [bob, male], [bob, parent]], dtype=torch.long
    )
    entailed_scores = module.class_assertion_score(entailed_data)
    assert entailed_scores.shape == (3,)
    assert torch.allclose(entailed_scores, torch.zeros_like(entailed_scores), atol=1e-6)

    # Contradicted for bob (Male & Female <= BOTTOM) -> score == inf
    contradicted_data = torch.tensor([[bob, female]], dtype=torch.long)
    contradicted_scores = module.class_assertion_score(contradicted_data)
    assert contradicted_scores.shape == (1,)
    assert torch.isinf(contradicted_scores).all() and (contradicted_scores > 0).all()

    # Check rank_classes agrees with class_assertion_score
    all_scores = module.rank_classes(torch.tensor([bob], dtype=torch.long))
    assert all_scores.shape == (1, len(cls_dict))
    assert float(all_scores[0, father].item()) == pytest.approx(0.0, abs=1e-6)
    assert float(all_scores[0, parent].item()) == pytest.approx(0.0, abs=1e-6)
    assert math.isinf(float(all_scores[0, female].item()))


def test_hierarchy_violation_rate_zero_on_500_random_triples() -> None:
    """Hierarchy violation rate is 0 on 500 random (a, C <= D) triples under arbitrary weights."""
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology("http://mowl.borg/geo/hvr_kb")

    bot = adapter.create_class(BOTTOM)
    n_cls = 10
    n_ind = 8
    c_iris = [f"{NS}C{i}" for i in range(n_cls)]
    ind_iris = [f"{IND}ind{i}" for i in range(n_ind)]
    classes = {iri: adapter.create_class(iri) for iri in c_iris}
    inds = {iri: adapter.create_individual(iri) for iri in ind_iris}

    # Build a non-trivial hierarchy: C0 <= C1 <= C2 <= C3, C4 <= C5 <= C6, C7 & C8 <= C9, C3 & C6 <= bot
    chains = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (8, 9)]
    for i, j in chains:
        mgr.addAxiom(ont, adapter.create_subclass_of(classes[c_iris[i]], classes[c_iris[j]]))
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(classes[c_iris[7]], classes[c_iris[8]]),
            classes[c_iris[9]],
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(classes[c_iris[3]], classes[c_iris[6]]),
            bot,
        ),
    )

    # Give individuals various consistent assertions
    assertions = [
        (0, 0),
        (1, 1),
        (2, 4),
        (3, 7),
        (4, 8),
        (5, 2),
        (6, 5),
        (7, 9),
    ]
    for ind_i, cls_i in assertions:
        mgr.addAxiom(
            ont, df.getOWLClassAssertionAxiom(classes[c_iris[cls_i]], inds[ind_iris[ind_i]])
        )

    cm = CanonicalModel.from_ontology(ont)
    cls_dict = {iri: i for i, iri in enumerate(c_iris)}
    ind_dict = {iri: i for i, iri in enumerate(ind_iris)}
    ms = MembershipStructure.from_canonical_model(cm, cls_dict, ind_dict)

    train_ca = torch.tensor(assertions, dtype=torch.long)
    torch.manual_seed(123)
    module = CanonicalMixtureModule(ms, rank=16, training_class_assertions=train_ca)
    # Randomize parameters heavily so p_a spreads non-trivially over S_a
    with torch.no_grad():
        module.U.normal_(0.0, 2.0)
        module.V.normal_(0.0, 2.0)
        module.b.normal_(0.0, 1.0)

    # Collect all entailed pairs (C, D) with C, D in c_iris
    entailed_pairs: List[Tuple[int, int]] = [
        (cls_dict[c1], cls_dict[c2])
        for c1 in c_iris
        for c2 in c_iris
        if cm.entails(c1, c2)
    ]
    assert len(entailed_pairs) >= 15

    rng = random.Random(7)
    triples = [
        (rng.randrange(n_ind), *rng.choice(entailed_pairs)) for _ in range(500)
    ]
    ind_t = torch.tensor([t[0] for t in triples], dtype=torch.long)
    sub_t = torch.tensor([t[1] for t in triples], dtype=torch.long)
    sup_t = torch.tensor([t[2] for t in triples], dtype=torch.long)

    score_sub = module.class_assertion_score(torch.stack([ind_t, sub_t], dim=1))
    score_sup = module.class_assertion_score(torch.stack([ind_t, sup_t], dim=1))

    # Since C <= D, P(a : C) <= P(a : D), so score(a, D) = -log P(a : D) <= score(a, C) + 1e-6.
    # A violation is score(a, C) < score(a, D) - 1e-6 (i.e. sub strictly more plausible than sup).
    violations = (score_sub < score_sup - 1e-6).sum().item()
    assert violations == 0


def test_toy_two_prototype_kb_recovers_held_out_membership() -> None:
    """Toy two-prototype KB recovers a held-out membership (rank of true class 1 after 200 steps)."""
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology("http://mowl.borg/geo/two_proto")

    bot = adapter.create_class(BOTTOM)
    # Two prototypes P1 and P2:
    # P1 <= A, P1 <= B, P1 <= Target1
    # P2 <= C, P2 <= D, P2 <= Target2
    # Distractor classes X1, X2, X3
    c_names = ["P1", "P2", "A", "B", "Target1", "C", "D", "Target2", "X1", "X2", "X3"]
    c_iris = [NS + name for name in c_names]
    cls_objs = {name: adapter.create_class(NS + name) for name in c_names}
    for obj in cls_objs.values():
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(obj))

    for sup in ["A", "B", "Target1"]:
        mgr.addAxiom(ont, adapter.create_subclass_of(cls_objs["P1"], cls_objs[sup]))
    for sup in ["C", "D", "Target2"]:
        mgr.addAxiom(ont, adapter.create_subclass_of(cls_objs["P2"], cls_objs[sup]))
    # Make P1 and P2 disjoint so they don't mix
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(cls_objs["P1"], cls_objs["P2"]), bot
        ),
    )

    # Individual a1 has training assertion A(a1) only (in the ontology),
    # and we train on B(a1) as an observed assertion while holding out Target1(a1)!
    # Notice: if S_{a1} is built from A(a1), then S_{a1} = {d_A, d_{P1}, d_{N_{a1}}}.
    # Wait: among S_{a1} = {d_A, d_{P1}, d_{N_{a1}}}, ONLY d_{P1} belongs to B!
    # So training on B(a1) pushes all mass of a1 onto d_{P1}, which is also in Target1!
    # Therefore P(a1 : Target1) approaches 1, making Target1 rank 1 among all non-entailed classes!
    ind_a1 = adapter.create_individual(IND + "a1")
    ind_a2 = adapter.create_individual(IND + "a2")
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(cls_objs["A"], ind_a1))
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(cls_objs["C"], ind_a2))

    cm = CanonicalModel.from_ontology(ont)
    cls_dict = {NS + name: i for i, name in enumerate(c_names)}
    ind_dict = {IND + "a1": 0, IND + "a2": 1}
    ms = MembershipStructure.from_canonical_model(cm, cls_dict, ind_dict)

    # Admissible support S_a is restricted by A(a1) and C(a2), while training assertions
    # include B(a1) and D(a2) so gradient descent concentrates mass on d_{P1} and d_{P2}.
    support_ca = torch.tensor(
        [[0, cls_dict[NS + "A"]], [1, cls_dict[NS + "C"]]], dtype=torch.long
    )
    train_pos = torch.tensor(
        [
            [0, cls_dict[NS + "A"]],
            [0, cls_dict[NS + "B"]],
            [1, cls_dict[NS + "C"]],
            [1, cls_dict[NS + "D"]],
        ],
        dtype=torch.long,
    )

    torch.manual_seed(42)
    module = CanonicalMixtureModule(
        ms, rank=8, init_self_mass=0.5, training_class_assertions=support_ca
    )
    opt = torch.optim.Adam(module.parameters(), lr=0.05)

    for _ in range(200):
        opt.zero_grad()
        pos_loss = module.class_assertion_loss(train_pos, neg=False).mean()
        neg_data = module.sample_class_negatives(train_pos)
        neg_loss = module.class_assertion_loss(neg_data, neg=True).mean()
        loss = pos_loss + neg_loss + module.regularization_loss()
        loss.backward()
        opt.step()

    # Evaluate rank of Target1 for a1 among non-training-asserted classes {Target1, P1, Target2, P2, C, D, X1, X2, X3}
    # Specifically among the non-entailed held-out candidate targets {Target1, Target2, X1, X2, X3}
    # (or even all classes not in {A, B}): d_{P1} is in P1, A, B, Target1, so Target1 has the minimum score!
    scores_a1 = module.rank_classes(torch.tensor([0], dtype=torch.long)).squeeze(0)
    target1_idx = cls_dict[NS + "Target1"]
    # Filter out training-observed classes A, B (and P1 which is co-extensional with Target1 on S_{a1})
    candidates = [
        cls_dict[NS + name]
        for name in ["Target1", "Target2", "C", "D", "X1", "X2", "X3"]
    ]
    cand_scores = scores_a1[candidates]
    best_cand = candidates[int(torch.argmin(cand_scores).item())]
    assert best_cand == target1_idx
    # Strict rank 1: Target1 score is finite (< 1.0) and strictly smaller than all other candidates
    other_scores = [
        float(scores_a1[c].item()) for c in candidates if c != target1_idx
    ]
    assert float(scores_a1[target1_idx].item()) < 1.0
    assert all(float(scores_a1[target1_idx].item()) < s for s in other_scores)


def test_singleton_support_and_init_self_mass() -> None:
    """Singleton support S_a has log_p == 0 on own element; multi-element S_a has p(own) == init_self_mass at U=0, b=0."""
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology("http://mowl.borg/geo/singleton_kb")

    father = adapter.create_class(NS + "Father")
    male = adapter.create_class(NS + "Male")
    bob = adapter.create_individual(IND + "bob")
    ann = adapter.create_individual(IND + "ann")
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(father, bob))
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(male, bob))
    mgr.addAxiom(ont, df.getOWLDeclarationAxiom(ann))

    cm = CanonicalModel.from_ontology(ont)
    cls_dict = {NS + "Father": 0, NS + "Male": 1}
    ind_dict = {IND + "bob": 0, IND + "ann": 1}
    ms = MembershipStructure.from_canonical_model(cm, cls_dict, ind_dict)

    # bob has both Father and Male -> only bob's own element is in S_bob (singleton support!)
    # ann has no types -> all 4 elements of S are in S_ann
    train_ca = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
    module = CanonicalMixtureModule(
        ms, rank=8, init_self_mass=0.9, training_class_assertions=train_ca
    )
    with torch.no_grad():
        module.U.zero_()
        module.b.zero_()

    log_p = module.log_p(torch.tensor([0, 1], dtype=torch.long))
    p = torch.exp(log_p)
    bob_elem = int(ms.individual_element[0].item())
    ann_elem = int(ms.individual_element[1].item())

    # Singleton S_bob: p[0, bob_elem] == 1.0, others == 0.0
    assert float(p[0, bob_elem].item()) == pytest.approx(1.0, abs=1e-6)
    assert float(p[0].sum().item()) == pytest.approx(1.0, abs=1e-6)

    # Multi-element S_ann (|S_ann| = 4): with U=0, b=0, own logit has offset log(0.9 * (4 - 1) / 0.1) or log(9 * |S_a|)
    # Check p[1, ann_elem] >= 0.85 and p[1].sum() == 1.0
    assert float(p[1, ann_elem].item()) == pytest.approx(0.9, abs=0.05)
    assert float(p[1].sum().item()) == pytest.approx(1.0, abs=1e-6)


def test_sparse_memory_and_role_score(abox_kb: OWLOntology) -> None:
    """Module stores sparse M and roles without densifying; role score matches p^T E_r q."""
    module, _, ms = _build_module_for_abox_kb(abox_kb, entropy_reg=0.1)
    assert module.structure.M.is_sparse
    for r_mat in module.structure.roles.values():
        assert r_mat.is_sparse

    ind_dict = ms.individual_index_dict
    bob = ind_dict[IND + "bob"]
    ann = ind_dict[IND + "ann"]
    carl = ind_dict[IND + "carl"]

    # Entailed role assertion hasChild(bob, ann): at init bob has singleton support {d_bob}
    # and ann has support S; let's check role score against explicit p_a^T E_r p_b
    role_data = torch.tensor([[bob, 0, ann], [carl, 0, ann], [ann, 0, bob]], dtype=torch.long)
    scores = module.object_property_assertion_score(role_data)
    assert scores.shape == (3,)

    p_all = torch.exp(module.log_p(torch.tensor([bob, ann, carl], dtype=torch.long)))
    e_r = ms.roles[0].to_dense().float()
    expected_bob_ann = -torch.log((p_all[0] @ e_r @ p_all[1]).clamp_min(1e-30))
    assert float(scores[0].item()) == pytest.approx(float(expected_bob_ann.item()), rel=1e-5)

    # Test object_property_assertion_loss (pos and neg) and regularization_loss
    pos_loss = module.object_property_assertion_loss(role_data[:2], neg=False)
    neg_loss = module.object_property_assertion_loss(role_data[2:], neg=True)
    reg_loss = module.regularization_loss()
    assert pos_loss.shape == (2,)
    assert neg_loss.shape == (1,)
    assert reg_loss.ndim == 0
    assert torch.isfinite(reg_loss)
