"""Tests for CanonicalMixtureModule in mowl.nn.el.canonical_mixture."""
from __future__ import annotations

import math
import random
import time
from typing import List, Tuple

import numpy as np
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
    """Entailed assertions (incl. role-entailed) score == 0.0 exactly at init; contradicted == +inf."""
    torch.manual_seed(0)
    module, _, ms = _build_module_for_abox_kb(abox_kb)
    cls_dict = ms.class_index_dict
    ind_dict = ms.individual_index_dict

    bob = ind_dict[IND + "bob"]
    carl = ind_dict[IND + "carl"]
    father = cls_dict[NS + "Father"]
    male = cls_dict[NS + "Male"]
    parent = cls_dict[NS + "Parent"]
    female = cls_dict[NS + "Female"]

    # Asserted and entailed for bob, plus role-entailed Parent(carl) via hasChild(carl, ann) -> score == 0 exactly
    entailed_data = torch.tensor(
        [[bob, father], [bob, male], [bob, parent], [carl, parent]], dtype=torch.long
    )
    entailed_scores = module.class_assertion_score(entailed_data)
    assert entailed_scores.shape == (4,)
    assert torch.all(entailed_scores == 0.0)

    # Contradicted for bob (Male & Female <= BOTTOM) -> score == +inf
    contradicted_data = torch.tensor([[bob, female]], dtype=torch.long)
    contradicted_scores = module.class_assertion_score(contradicted_data)
    assert contradicted_scores.shape == (1,)
    assert torch.isinf(contradicted_scores).all() and (contradicted_scores > 0).all()

    # Check rank_classes agrees with exact 0.0 for entailed and +inf for contradicted
    all_scores = module.rank_classes(torch.tensor([bob, carl], dtype=torch.long))
    assert all_scores.shape == (2, len(cls_dict))
    assert float(all_scores[0, father].item()) == 0.0
    assert float(all_scores[0, male].item()) == 0.0
    assert float(all_scores[0, parent].item()) == 0.0
    assert float(all_scores[1, parent].item()) == 0.0
    assert math.isinf(float(all_scores[0, female].item()))


def test_role_entailed_type_scores_exact_zero() -> None:
    """A type entailed only through a role (exists r.B <= P, r(a,b), B(b)) scores == 0.0 exactly for a."""
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology("http://mowl.borg/geo/role_entailed_kb")

    c_a = adapter.create_class(NS + "A")
    c_b = adapter.create_class(NS + "B")
    c_p = adapter.create_class(NS + "P")
    c_q = adapter.create_class(NS + "Q")
    role_r = adapter.create_object_property(NS + "r")
    ind_a = adapter.create_individual(IND + "a")
    ind_b = adapter.create_individual(IND + "b")

    # exists r.B <= P; A(a); B(b); r(a, b)
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_some_values_from(role_r, c_b), c_p
        ),
    )
    mgr.addAxiom(ont, df.getOWLDeclarationAxiom(c_q))
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(c_a, ind_a))
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(c_b, ind_b))
    mgr.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(role_r, ind_a, ind_b))

    cm = CanonicalModel.from_ontology(ont)
    cls_dict = {NS + "A": 0, NS + "B": 1, NS + "P": 2, NS + "Q": 3}
    ind_dict = {IND + "a": 0, IND + "b": 1}
    rel_dict = {NS + "r": 0}
    ms = MembershipStructure.from_canonical_model(cm, cls_dict, ind_dict, rel_dict)

    # Pass ONLY the asserted class assertions A(a) and B(b); P(a) is entailed only via r(a, b) & B(b)
    train_ca = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    module = CanonicalMixtureModule(ms, rank=8, training_class_assertions=train_ca)

    score_pa = module.class_assertion_score(torch.tensor([[0, 2]], dtype=torch.long))
    assert float(score_pa.item()) == 0.0

    # Even after arbitrary parameter perturbations, P(a) still scores 0.0 exactly
    with torch.no_grad():
        module.U.normal_(0.0, 5.0)
        module.V.normal_(0.0, 5.0)
        module.b.normal_(0.0, 5.0)
    score_pa_perturbed = module.class_assertion_score(
        torch.tensor([[0, 2]], dtype=torch.long)
    )
    assert float(score_pa_perturbed.item()) == 0.0


def test_extreme_parameters_preserve_exact_guarantees_without_nan(
    abox_kb: OWLOntology,
) -> None:
    """With b = -2e9 and U @ V^T of magnitude 1e6: entailed -> 0.0, contradicted -> +inf, no NaN."""
    torch.manual_seed(11)
    module, _, ms = _build_module_for_abox_kb(abox_kb, rank=16)
    cls_dict = ms.class_index_dict
    ind_dict = ms.individual_index_dict

    bob = ind_dict[IND + "bob"]
    carl = ind_dict[IND + "carl"]
    ann = ind_dict[IND + "ann"]
    father = cls_dict[NS + "Father"]
    male = cls_dict[NS + "Male"]
    parent = cls_dict[NS + "Parent"]
    female = cls_dict[NS + "Female"]

    with torch.no_grad():
        module.b.fill_(-2e9)
        # Make U @ V^T have magnitude ~1e6
        module.U.uniform_(-1e3, 1e3)
        module.V.uniform_(-1e3 / math.sqrt(16), 1e3 / math.sqrt(16))

    uv = module.U @ module.V.t()
    assert float(uv.abs().max().item()) >= 1e5

    all_inds = torch.tensor([bob, ann, carl], dtype=torch.long)
    lp = module.log_p(all_inds)
    assert not bool(torch.isnan(lp).any().item())

    entailed_data = torch.tensor(
        [[bob, father], [bob, male], [bob, parent], [carl, parent]], dtype=torch.long
    )
    entailed_scores = module.class_assertion_score(entailed_data)
    assert not bool(torch.isnan(entailed_scores).any().item())
    assert torch.all(entailed_scores == 0.0)

    contradicted_data = torch.tensor([[bob, female]], dtype=torch.long)
    contradicted_scores = module.class_assertion_score(contradicted_data)
    assert not bool(torch.isnan(contradicted_scores).any().item())
    assert torch.isinf(contradicted_scores).all() and (contradicted_scores > 0).all()

    ranked = module.rank_classes(all_inds)
    assert not bool(torch.isnan(ranked).any().item())
    assert float(ranked[0, father].item()) == 0.0
    assert float(ranked[2, parent].item()) == 0.0
    assert math.isinf(float(ranked[0, female].item()))


def test_non_entailed_training_assertion_raises_value_error(
    abox_kb: OWLOntology,
) -> None:
    """Passing a training class assertion not entailed by M raises ValueError instead of forcing x_a into S_a."""
    cm = CanonicalModel.from_ontology(abox_kb)
    class_index_dict, individual_index_dict, relation_index_dict = _dicts_for_abox_kb()
    ms = MembershipStructure.from_canonical_model(
        cm, class_index_dict, individual_index_dict, relation_index_dict
    )
    carl = individual_index_dict[IND + "carl"]
    father = class_index_dict[NS + "Father"]
    bad_ca = torch.tensor([[carl, father]], dtype=torch.long)
    with pytest.raises(ValueError, match="not entailed by M"):
        CanonicalMixtureModule(ms, rank=8, training_class_assertions=bad_ca)


def test_csr_supports_and_no_dense_ind_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supports are stored CSR-style and no (n_ind x |S|) or (n_ind x n_classes) tensor is ever allocated."""
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology("http://mowl.borg/geo/csr_mem_kb")

    n_cls = 15
    n_ind = 11
    c_iris = [f"{NS}C{i}" for i in range(n_cls)]
    ind_iris = [f"{IND}ind{i}" for i in range(n_ind)]
    classes = {iri: adapter.create_class(iri) for iri in c_iris}
    inds = {iri: adapter.create_individual(iri) for iri in ind_iris}
    for c_obj in classes.values():
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(c_obj))
    for i, ind_iri in enumerate(ind_iris):
        cls_iri = c_iris[i % n_cls]
        mgr.addAxiom(
            ont, df.getOWLClassAssertionAxiom(classes[cls_iri], inds[ind_iri])
        )

    cm = CanonicalModel.from_ontology(ont)
    cls_dict = {iri: i for i, iri in enumerate(c_iris)}
    ind_dict = {iri: i for i, iri in enumerate(ind_iris)}
    ms = MembershipStructure.from_canonical_model(cm, cls_dict, ind_dict)
    n_elements = ms.M.shape[0]
    assert n_elements == n_cls + n_ind  # 26 elements

    forbidden_shapes = {
        (n_ind, n_elements),
        (n_elements, n_ind),
        (n_ind, n_cls),
        (n_cls, n_ind),
        (n_elements, n_elements),
        (n_elements, n_cls),
        (n_cls, n_elements),
    }

    def _wrap_alloc(fn):
        def _checked(*args, **kwargs):
            out = fn(*args, **kwargs)
            if isinstance(out, torch.Tensor) and not out.is_sparse:
                assert tuple(out.shape) not in forbidden_shapes, (
                    f"Forbidden dense allocation of shape {tuple(out.shape)}"
                )
            return out

        return _checked

    for fn_name in ("zeros", "empty", "full", "ones"):
        monkeypatch.setattr(torch, fn_name, _wrap_alloc(getattr(torch, fn_name)))

    train_ca = torch.tensor([[i, i % n_cls] for i in range(n_ind)], dtype=torch.long)
    module = CanonicalMixtureModule(
        ms, rank=8, entropy_reg=0.05, training_class_assertions=train_ca
    )

    # Check CSR support buffer shapes and absence of dense per-individual tables
    assert not hasattr(module, "_non_entailed_by_ind")
    assert module.support_offsets.ndim == 1
    assert module.support_offsets.shape == (n_ind + 1,)
    assert module.support_indices.ndim == 1
    assert module.support_indices.numel() == int(module.support_lengths.sum().item())
    assert module.support_indices.numel() == 2 * n_ind  # each ind has 2 elements in S_a

    for name, buf in module.named_buffers():
        if not buf.is_sparse:
            assert tuple(buf.shape) not in forbidden_shapes, (
                f"Buffer {name} has forbidden shape {tuple(buf.shape)}"
            )

    # Run mini-batch operations (B = 4 < n_ind) under the allocation guard
    batch_ca = train_ca[:4]
    negs = module.sample_class_negatives(batch_ca)
    assert negs.shape == (4, 2)
    # Verify sampled negatives are never entailed for the individual
    dense_m = ms.M.to_dense()
    for row in negs:
        ind_i = int(row[0].item())
        neg_c = int(row[1].item())
        own_el = int(ms.individual_element[ind_i].item())
        assert not bool(dense_m[own_el, neg_c].item())

    loss = (
        module.class_assertion_loss(batch_ca, neg=False).mean()
        + module.class_assertion_loss(negs, neg=True).mean()
        + module.regularization_loss(batch_ca[:, 0])
    )
    loss.backward()


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

    chains = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (8, 9)]
    for i, j in chains:
        mgr.addAxiom(
            ont, adapter.create_subclass_of(classes[c_iris[i]], classes[c_iris[j]])
        )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(
                classes[c_iris[7]], classes[c_iris[8]]
            ),
            classes[c_iris[9]],
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(
                classes[c_iris[3]], classes[c_iris[6]]
            ),
            bot,
        ),
    )

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
            ont,
            df.getOWLClassAssertionAxiom(
                classes[c_iris[cls_i]], inds[ind_iris[ind_i]]
            ),
        )

    cm = CanonicalModel.from_ontology(ont)
    cls_dict = {iri: i for i, iri in enumerate(c_iris)}
    ind_dict = {iri: i for i, iri in enumerate(ind_iris)}
    ms = MembershipStructure.from_canonical_model(cm, cls_dict, ind_dict)

    train_ca = torch.tensor(assertions, dtype=torch.long)
    torch.manual_seed(123)
    module = CanonicalMixtureModule(ms, rank=16, training_class_assertions=train_ca)
    with torch.no_grad():
        module.U.normal_(0.0, 2.0)
        module.V.normal_(0.0, 2.0)
        module.b.normal_(0.0, 1.0)

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

    violations = (score_sub < score_sup - 1e-6).sum().item()
    assert violations == 0


def test_toy_two_prototype_kb_recovers_held_out_membership() -> None:
    """At init a distractor class has strictly lower (better) score than held-out Target1; after 200 steps on ontology assertions Target1 ranks 1."""
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology("http://mowl.borg/geo/two_proto")

    bot = adapter.create_class(BOTTOM)
    # Classes in the ontology:
    # A is the asserted class of a1.
    # Prototype P1 <= A, P1 <= Target1, P1 <= exists r.W.
    # Distractor prototypes P2 <= A, P2 <= Distractor and P3 <= A, P3 <= Distractor
    # (so S_{a1} contains 2 elements in Distractor and only 1 element in Target1 ->
    # at init Distractor has ~2x the probability mass of Target1 and strictly lower score).
    # Distractor also has superclasses D1..D4 (not subclasses of A) so negative sampling
    # penalizes mass on P2/P3 while role training on r(a1, b1) rewards mass on P1.
    c_names = [
        "A",
        "P1",
        "P2",
        "P3",
        "Target1",
        "Distractor",
        "W",
        "D1",
        "D2",
        "D3",
        "D4",
    ]
    cls_objs = {name: adapter.create_class(NS + name) for name in c_names}
    for obj in cls_objs.values():
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(obj))
    role_r = adapter.create_object_property(NS + "r")
    mgr.addAxiom(ont, df.getOWLDeclarationAxiom(role_r))

    mgr.addAxiom(ont, adapter.create_subclass_of(cls_objs["P1"], cls_objs["A"]))
    mgr.addAxiom(ont, adapter.create_subclass_of(cls_objs["P1"], cls_objs["Target1"]))
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            cls_objs["P1"],
            adapter.create_object_some_values_from(role_r, cls_objs["W"]),
        ),
    )

    for proto in ("P2", "P3"):
        mgr.addAxiom(ont, adapter.create_subclass_of(cls_objs[proto], cls_objs["A"]))
        mgr.addAxiom(
            ont, adapter.create_subclass_of(cls_objs[proto], cls_objs["Distractor"])
        )

    for d_sup in ("D1", "D2", "D3", "D4"):
        mgr.addAxiom(
            ont, adapter.create_subclass_of(cls_objs["Distractor"], cls_objs[d_sup])
        )

    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(
                cls_objs["Target1"], cls_objs["Distractor"]
            ),
            bot,
        ),
    )

    # Assertions present in the ontology: A(a1), W(b1), r(a1, b1)
    ind_a1 = adapter.create_individual(IND + "a1")
    ind_b1 = adapter.create_individual(IND + "b1")
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(cls_objs["A"], ind_a1))
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(cls_objs["W"], ind_b1))
    mgr.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(role_r, ind_a1, ind_b1))

    cm = CanonicalModel.from_ontology(ont)
    cls_dict = {NS + name: i for i, name in enumerate(c_names)}
    ind_dict = {IND + "a1": 0, IND + "b1": 1}
    rel_dict = {NS + "r": 0}
    ms = MembershipStructure.from_canonical_model(cm, cls_dict, ind_dict, rel_dict)

    # Train ONLY on assertions present in the ontology
    train_ca = torch.tensor(
        [[0, cls_dict[NS + "A"]], [1, cls_dict[NS + "W"]]], dtype=torch.long
    )
    train_opa = torch.tensor([[0, 0, 1]], dtype=torch.long)

    torch.manual_seed(42)
    module = CanonicalMixtureModule(
        ms, rank=8, init_self_mass=0.5, training_class_assertions=train_ca
    )

    target1_idx = cls_dict[NS + "Target1"]
    distractor_idx = cls_dict[NS + "Distractor"]
    candidates = [
        cls_dict[NS + name]
        for name in ["Target1", "Distractor", "W", "D1", "D2", "D3", "D4"]
    ]

    # Check at init: Distractor has a strictly lower (better) finite score than Target1
    init_scores = module.rank_classes(torch.tensor([0], dtype=torch.long)).squeeze(0)
    init_target_score = float(init_scores[target1_idx].item())
    init_distractor_score = float(init_scores[distractor_idx].item())
    assert math.isfinite(init_target_score)
    assert math.isfinite(init_distractor_score)
    assert init_distractor_score < init_target_score
    init_best_cand = candidates[int(torch.argmin(init_scores[candidates]).item())]
    assert init_best_cand != target1_idx

    opt = torch.optim.Adam(module.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        pos_ca_loss = module.class_assertion_loss(train_ca, neg=False).mean()
        neg_ca = module.sample_class_negatives(train_ca)
        neg_ca_loss = module.class_assertion_loss(neg_ca, neg=True).mean()
        pos_opa_loss = module.object_property_assertion_loss(
            train_opa, neg=False
        ).mean()
        neg_opa = module.sample_role_negatives(train_opa)
        neg_opa_loss = module.object_property_assertion_loss(neg_opa, neg=True).mean()
        loss = pos_ca_loss + neg_ca_loss + pos_opa_loss + neg_opa_loss
        loss.backward()
        opt.step()

    # After training: Target1 ranks 1 among all non-entailed / finite candidates
    post_scores = module.rank_classes(torch.tensor([0], dtype=torch.long)).squeeze(0)
    cand_scores = post_scores[candidates]
    best_cand = candidates[int(torch.argmin(cand_scores).item())]
    assert best_cand == target1_idx
    other_scores = [
        float(post_scores[c].item()) for c in candidates if c != target1_idx
    ]
    assert all(float(post_scores[target1_idx].item()) < s for s in other_scores)


def test_singleton_support_and_init_self_mass() -> None:
    """Singleton support S_a has p(own) == 1; multi-element S_a has p(own) == 0.9 within 1e-5 at init."""
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

    train_ca = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
    torch.manual_seed(0)
    module = CanonicalMixtureModule(
        ms, rank=8, init_self_mass=0.9, training_class_assertions=train_ca
    )

    bob_elem = int(ms.individual_element[0].item())
    ann_elem = int(ms.individual_element[1].item())

    # Check right at random init (before zeroing U)
    p_init = torch.exp(module.log_p(torch.tensor([0, 1], dtype=torch.long)))
    assert float(p_init[0, bob_elem].item()) == pytest.approx(1.0, abs=1e-6)
    assert float(p_init[1, ann_elem].item()) == pytest.approx(0.9, abs=1e-5)

    # Check at exact U=0, b=0
    with torch.no_grad():
        module.U.zero_()
        module.b.zero_()

    log_p = module.log_p(torch.tensor([0, 1], dtype=torch.long))
    p = torch.exp(log_p)
    assert float(p[0, bob_elem].item()) == pytest.approx(1.0, abs=1e-6)
    assert float(p[0].sum().item()) == pytest.approx(1.0, abs=1e-6)
    assert float(p[1, ann_elem].item()) == pytest.approx(0.9, abs=1e-5)
    assert float(p[1].sum().item()) == pytest.approx(1.0, abs=1e-6)


def test_sparse_memory_and_role_score(abox_kb: OWLOntology) -> None:
    """Module stores sparse M and roles without densifying; role score via sparse.mm matches p^T E_r q."""
    module, _, ms = _build_module_for_abox_kb(abox_kb, entropy_reg=0.1)
    assert module.structure.M.is_sparse
    for r_mat in module.structure.roles.values():
        assert r_mat.is_sparse

    ind_dict = ms.individual_index_dict
    bob = ind_dict[IND + "bob"]
    ann = ind_dict[IND + "ann"]
    carl = ind_dict[IND + "carl"]

    role_data = torch.tensor(
        [[bob, 0, ann], [carl, 0, ann], [ann, 0, bob]], dtype=torch.long
    )
    scores = module.object_property_assertion_score(role_data)
    assert scores.shape == (3,)

    p_all = torch.exp(module.log_p(torch.tensor([bob, ann, carl], dtype=torch.long)))
    e_r = ms.roles[0].to_dense().float()
    expected_bob_ann = -torch.log((p_all[0] @ e_r @ p_all[1]).clamp_min(1e-30))
    assert float(scores[0].item()) == pytest.approx(
        float(expected_bob_ann.item()), rel=1e-5
    )

    pos_loss = module.object_property_assertion_loss(role_data[:2], neg=False)
    neg_loss = module.object_property_assertion_loss(role_data[2:], neg=True)
    reg_loss = module.regularization_loss(
        torch.tensor([bob, ann, carl], dtype=torch.long)
    )
    assert pos_loss.shape == (2,)
    assert neg_loss.shape == (1,)
    assert reg_loss.ndim == 0
    assert torch.isfinite(reg_loss)


def test_regularization_loss_requires_batch(abox_kb: OWLOntology) -> None:
    """Calling regularization_loss without a batch argument raises TypeError."""
    module, _, _ = _build_module_for_abox_kb(abox_kb, entropy_reg=0.1)
    with pytest.raises(TypeError):
        module.regularization_loss()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        module.regularization_loss(None)  # type: ignore[arg-type]


def test_support_construction_matches_reference_on_fixture_kb(
    abox_kb: OWLOntology,
) -> None:
    """CSC minimal-type intersection produces results identical to the old type_mask reference on abox_kb."""
    module, _, ms = _build_module_for_abox_kb(abox_kb)
    bob_idx = ms.individual_index_dict[IND + "bob"]
    training_ca = torch.tensor(
        [
            [bob_idx, ms.class_index_dict[NS + "Father"]],
            [bob_idx, ms.class_index_dict[NS + "Male"]],
        ],
        dtype=torch.long,
    )
    ref_offsets, ref_indices, ref_own_offsets = (
        module._build_admissible_supports_reference(training_ca)
    )
    assert torch.equal(module.support_offsets, ref_offsets)
    assert torch.equal(module.support_indices, ref_indices)
    assert torch.equal(module.own_logit_offset, ref_own_offsets)


@pytest.mark.slow
def test_synthetic_large_support_construction_timing() -> None:
    """Synthetic KB with n_ind = 6,000, |S| = 50,000, ~15 types per element builds supports in < 60 s on CPU."""
    n_ind = 6_000
    n_elements = 50_000
    n_classes = n_elements - n_ind  # 44,000
    rng = np.random.default_rng(0)

    # Build a transitively closed 15-level DAG hierarchy on n_classes so each element has ~15 types
    ancestors: List[np.ndarray] = [None] * n_classes  # type: ignore[list-item]
    layer_size = 250
    for c in range(layer_size):
        ancestors[c] = np.array([c], dtype=np.int64)
    for layer in range(1, 14):
        start = layer * layer_size
        prev_start = (layer - 1) * layer_size
        for c in range(start, start + layer_size):
            p = int(rng.integers(prev_start, start))
            ancestors[c] = np.append(ancestors[p], c)
    start_14 = 14 * layer_size
    prev_start = 13 * layer_size
    for c in range(start_14, n_classes):
        p = int(rng.integers(prev_start, start_14))
        ancestors[c] = np.append(ancestors[p], c)

    rows_list: List[np.ndarray] = []
    cols_list: List[np.ndarray] = []
    for c in range(n_classes):
        a = ancestors[c]
        rows_list.append(np.full(a.size, c, dtype=np.int64))
        cols_list.append(a)

    for i in range(n_ind):
        el = n_classes + i
        c1 = int(rng.integers(start_14, n_classes))
        c2 = int(rng.integers(prev_start, start_14))
        a = np.union1d(ancestors[c1], ancestors[c2])
        rows_list.append(np.full(a.size, el, dtype=np.int64))
        cols_list.append(a)

    rows_np = np.concatenate(rows_list)
    cols_np = np.concatenate(cols_list)
    avg_types = rows_np.size / n_elements
    assert 13.0 <= avg_types <= 17.0

    indices = torch.stack(
        [torch.from_numpy(rows_np), torch.from_numpy(cols_np)], dim=0
    )
    values = torch.ones((rows_np.size,), dtype=torch.bool)
    m_sparse = torch.sparse_coo_tensor(
        indices, values, size=(n_elements, n_classes), dtype=torch.bool
    ).coalesce()

    element_names = [f"{NS}C{c}" for c in range(n_classes)] + [
        f"{IND}ind{i}" for i in range(n_ind)
    ]
    element_index = {name: idx for idx, name in enumerate(element_names)}
    class_index_dict = {f"{NS}C{c}": c for c in range(n_classes)}
    individual_index_dict = {f"{IND}ind{i}": i for i in range(n_ind)}
    ind_elem = torch.arange(n_classes, n_elements, dtype=torch.long)

    ms = MembershipStructure(
        element_names=element_names,
        element_index=element_index,
        M=m_sparse,
        roles={},
        individual_element=ind_elem,
        class_index_dict=class_index_dict,
        individual_index_dict=individual_index_dict,
        role_index_dict={},
    )

    t0 = time.perf_counter()
    module = CanonicalMixtureModule(ms, rank=16)
    elapsed = time.perf_counter() - t0

    assert elapsed < 60.0, f"Construction took {elapsed:.2f}s (>= 60s)"
    assert module.support_offsets.shape == (n_ind + 1,)
    assert int(module.support_lengths.min().item()) >= 1

