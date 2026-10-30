"""Tests for PointSupervision in mowl.datasets.el.points."""
from __future__ import annotations

import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import mowl
mowl.init_jvm("2g")

from org.semanticweb.owlapi.model import OWLOntology

from mowl.datasets import PathDataset
from mowl.datasets.el.points import (
    MAX_EXACT_CHECKS_FACTOR,
    POINTS_CODE_TAG,
    PointSupervision,
    build_or_load_point_supervision,
    ontology_sha256,
)
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.normal_forms import BOTTOM, TOP
from mowl.nn import MultiBoxELModule, MultiELBEModule
from mowl.owlapi import OWLAPIAdapter
from tests.geometric.helpers import IND, NS, _build_random_abox_kb, fixture_path


@pytest.fixture
def t3_kb() -> OWLOntology:
    """Return the OWL ontology for ``t3.owl``."""
    return PathDataset(fixture_path("t3.owl")).ontology


def _class_dict_from_ontology(ont: OWLOntology) -> Dict[str, int]:
    classes = sorted(
        str(c.toStringID())
        for c in ont.getClassesInSignature()
        if str(c.toStringID()) not in (TOP, BOTTOM)
    )
    return {iri: idx for idx, iri in enumerate(classes)}


def test_point_pairs_consistent_with_cm(t3_kb: OWLOntology) -> None:
    """Every pos pair is entailed; no hard neg is entailed; unsat names are absent."""
    # Add an unsatisfiable class U <= BOTTOM and TOP/BOTTOM to class_index_dict
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    ont = adapter.create_ontology("http://mowl.borg/geo/t3_with_unsat")
    for ax in t3_kb.getAxioms():
        mgr.addAxiom(ont, ax)
    unsat_iri = NS + "UnsatClass"
    c_unsat = adapter.create_class(unsat_iri)
    c_bot = adapter.create_class(BOTTOM)
    mgr.addAxiom(ont, adapter.create_subclass_of(c_unsat, c_bot))

    cm = CanonicalModel.from_ontology(ont)
    assert not cm.is_satisfiable(unsat_iri)

    cls_dict = _class_dict_from_ontology(t3_kb)
    cls_dict[unsat_iri] = len(cls_dict)
    cls_dict[TOP] = len(cls_dict)
    cls_dict[BOTTOM] = len(cls_dict)
    inv_cls = {v: k for k, v in cls_dict.items()}

    ps = PointSupervision.from_canonical_model(cm, cls_dict, {})

    # Unsatisfiable class, TOP, and BOTTOM are absent from elements and pairs
    assert unsat_iri not in ps.element_names
    assert TOP not in ps.element_names
    assert BOTTOM not in ps.element_names
    assert ps.n_elements == 7
    assert not bool(ps.element_is_individual.any().item())

    forbidden_cls_indices = {cls_dict[unsat_iri], cls_dict[TOP], cls_dict[BOTTOM]}
    for c_idx in ps.pos_pairs[:, 1].tolist():
        assert c_idx not in forbidden_cls_indices
    for c_idx in ps.hard_neg_pairs[:, 1].tolist():
        assert c_idx not in forbidden_cls_indices

    # Every positive pair (x_D, C) is entailed in cm, and all entailed pairs are present
    pos_set = set()
    for elem_i, cls_j in ps.pos_pairs.tolist():
        d_name = ps.element_names[elem_i]
        c_iri = inv_cls[cls_j]
        assert cm.entails(d_name, c_iri), f"Pos pair ({d_name}, {c_iri}) not entailed"
        pos_set.add((elem_i, cls_j))

    for elem_i, d_name in enumerate(ps.element_names):
        for c_iri, cls_j in cls_dict.items():
            if c_iri in (unsat_iri, TOP, BOTTOM):
                continue
            if cm.entails(d_name, c_iri):
                assert (elem_i, cls_j) in pos_set

    # Every hard negative (x_D, C) is jointly unsatisfiable and not entailed,
    # and the set equals the exact brute-force {!cm.joint_satisfiable({D, C})} set:
    # on t3.owl there are 12 directed hard negative pairs (6 from the 3 direct
    # disjointness axioms plus 6 two-step pairs (A,X3),(B,X1),(C,X2) and reverses).
    assert ps.hard_neg_capped is False
    assert ps.hard_neg_pairs.shape == (12, 2)
    hard_named = {
        (ps.element_names[e].split("#")[-1], inv_cls[c].split("#")[-1])
        for e, c in ps.hard_neg_pairs.tolist()
    }
    expected_t3_hard = {
        ("A", "B"), ("B", "A"),
        ("A", "C"), ("C", "A"),
        ("B", "C"), ("C", "B"),
        ("A", "X3"), ("X3", "A"),
        ("B", "X1"), ("X1", "B"),
        ("C", "X2"), ("X2", "C"),
    }
    assert hard_named == expected_t3_hard
    brute_hard = {
        (elem_i, cls_j)
        for elem_i, d_name in enumerate(ps.element_names)
        for c_iri, cls_j in cls_dict.items()
        if c_iri not in (unsat_iri, TOP, BOTTOM)
        and not cm.joint_satisfiable({d_name, c_iri})
    }
    assert {tuple(row) for row in ps.hard_neg_pairs.tolist()} == brute_hard
    for elem_i, cls_j in ps.hard_neg_pairs.tolist():
        d_name = ps.element_names[elem_i]
        c_iri = inv_cls[cls_j]
        assert not cm.entails(d_name, c_iri)
        assert not cm.joint_satisfiable({d_name, c_iri})

    # No per-element lists/dicts are stored on the PointSupervision instance
    assert not hasattr(ps, "_non_entailed_by_elem")
    assert not hasattr(ps, "_hard_neg_by_elem")

    # Sampled batch respects total budget <= n, 1 rand per pos, 1 hard per pos when element has any
    sampled = ps.sample(10, rng=42)
    assert set(sampled.keys()) == {"pos", "hard", "rand", "roles"}
    total_pairs = sum(t.shape[0] for t in sampled.values())
    assert total_pairs <= 10
    assert sampled["pos"].shape[0] >= 1
    assert sampled["rand"].shape == sampled["pos"].shape
    assert sampled["hard"].shape[0] <= sampled["pos"].shape[0]
    for elem_i, cls_j in sampled["rand"].tolist():
        d_name = ps.element_names[elem_i]
        c_iri = inv_cls[cls_j]
        assert c_iri not in (unsat_iri, TOP, BOTTOM)
        assert not cm.entails(d_name, c_iri)
    hard_set = {tuple(row) for row in ps.hard_neg_pairs.tolist()}
    for elem_i, cls_j in sampled["hard"].tolist():
        assert (elem_i, cls_j) in hard_set


def test_points_include_individuals(abox_kb: OWLOntology) -> None:
    """Individual elements are flagged and alias ind_embed rows in module lookup."""
    cm = CanonicalModel.from_ontology(abox_kb)
    class_iris = [
        NS + "Father",
        NS + "Mother",
        NS + "Parent",
        NS + "Male",
        NS + "Female",
    ]
    ind_iris = [IND + "bob", IND + "ann", IND + "carl"]
    role_iris = [NS + "hasChild"]
    cls_dict = {iri: idx for idx, iri in enumerate(class_iris)}
    ind_dict = {iri: idx for idx, iri in enumerate(ind_iris)}
    rel_dict = {iri: idx for idx, iri in enumerate(role_iris)}

    ps = PointSupervision.from_canonical_model(
        cm, cls_dict, ind_dict, rel_dict, rng_seed=0
    )

    assert ps.n_elements == 8
    assert int((~ps.element_is_individual).sum().item()) == 5
    assert int(ps.element_is_individual.sum().item()) == 3

    ind_elem_indices = torch.nonzero(ps.element_is_individual, as_tuple=False).reshape(-1)
    cls_elem_indices = torch.nonzero(~ps.element_is_individual, as_tuple=False).reshape(-1)
    assert ps.element_class_index[ind_elem_indices].tolist() == [0, 1, 2]
    assert ps.element_class_index[cls_elem_indices].tolist() == [0, 1, 2, 3, 4]

    # Role pairs from hasChild(bob, ann) and hasChild(carl, ann)
    assert ps.role_pairs.shape == (2, 3)

    # Check aliasing with ind_embed on both MultiBoxELModule and MultiELBEModule
    for mod in (
        MultiBoxELModule(nb_ont_classes=5, nb_rels=1, nb_inds=3, embed_dim=4),
        MultiELBEModule(nb_ont_classes=5, nb_rels=1, nb_inds=3, embed_dim=4),
    ):
        mod.set_point_supervision(ps)
        assert mod.elem_embed is not None
        assert mod.elem_embed.num_embeddings == ps.n_elements
        assert mod.ind_embed is not None

        with torch.no_grad():
            mod.ind_embed.weight.fill_(7.5)
            mod.elem_embed.weight.fill_(-3.0)

        all_idx = torch.arange(ps.n_elements, dtype=torch.long)
        pts = mod.element_embeddings(all_idx)
        assert torch.allclose(pts[ind_elem_indices], mod.ind_embed.weight)
        assert torch.allclose(pts[cls_elem_indices], mod.elem_embed.weight[cls_elem_indices])

        # Backward through individual element points routes gradient to ind_embed.weight only
        mod.zero_grad()
        loss_ind = mod.element_embeddings(ind_elem_indices).sum()
        loss_ind.backward()
        assert mod.ind_embed.weight.grad is not None
        assert torch.all(mod.ind_embed.weight.grad == 1.0)
        assert mod.elem_embed.weight.grad is None or torch.all(
            mod.elem_embed.weight.grad[ind_elem_indices] == 0.0
        )

    # MultiBoxELModule.ind_embed is a read-only property aliasing ind_center
    mb_mod = MultiBoxELModule(nb_ont_classes=5, nb_rels=1, nb_inds=3, embed_dim=4)
    assert mb_mod.ind_embed is mb_mod.ind_center
    with pytest.raises(AttributeError, match="read-only"):
        mb_mod.ind_embed = torch.nn.Embedding(3, 4)


def test_cache_roundtrip(t3_kb: OWLOntology, abox_kb: OWLOntology, tmp_path: Path) -> None:
    """save/load round-trips PointSupervision and validates POINTS_CODE_TAG and ontology sha256."""
    cm = CanonicalModel.from_ontology(t3_kb)
    cls_dict = _class_dict_from_ontology(t3_kb)
    sha = ontology_sha256(t3_kb, cls_dict, {})
    sha_other = ontology_sha256(abox_kb, _class_dict_from_ontology(abox_kb), {})
    assert sha != sha_other
    ps = PointSupervision.from_canonical_model(
        cm, cls_dict, {}, rng_seed=7, ontology_sha256=sha
    )

    cache_file = tmp_path / "points_cache.pt"
    ps.save(cache_file)

    loaded = PointSupervision.load(cache_file, expected_sha256=sha)
    assert loaded.n_elements == ps.n_elements
    assert loaded.element_names == ps.element_names
    assert torch.equal(loaded.element_class_index, ps.element_class_index)
    assert torch.equal(loaded.element_is_individual, ps.element_is_individual)
    assert torch.equal(loaded.pos_pairs, ps.pos_pairs)
    assert torch.equal(loaded.hard_neg_pairs, ps.hard_neg_pairs)
    assert torch.equal(loaded.role_pairs, ps.role_pairs)
    assert loaded.hard_neg_capped == ps.hard_neg_capped

    # Wrong expected_sha256 raises ValueError
    with pytest.raises(ValueError, match="sha256"):
        PointSupervision.load(cache_file, expected_sha256="deadbeef")

    # Tampered code_tag raises ValueError
    raw = torch.load(cache_file, weights_only=False)
    assert raw["code_tag"] == POINTS_CODE_TAG
    raw["code_tag"] = "invalid-tag"
    bad_file = tmp_path / "points_bad_tag.pt"
    torch.save(raw, bad_file)
    with pytest.raises(ValueError, match="code_tag"):
        PointSupervision.load(bad_file)

    # Helper build_or_load_point_supervision writes and reuses cache file
    ds = PathDataset(fixture_path("t3.owl"))
    cache_dir = tmp_path / "points_dir"
    ps1 = build_or_load_point_supervision(
        ds, cls_dict, {}, cache_dir=cache_dir, rng_seed=0
    )
    cached_files = list(cache_dir.glob(f"*{POINTS_CODE_TAG}*.pt"))
    assert len(cached_files) == 1
    ps2 = build_or_load_point_supervision(
        ds, cls_dict, {}, cache_dir=cache_dir, rng_seed=0
    )
    assert torch.equal(ps1.pos_pairs, ps2.pos_pairs)


def _build_3step_chain_kb() -> OWLOntology:
    """Build a 3-step conjunction chain KB where E3 & G3 <= bot requires 3 NF2 steps."""
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    ont = adapter.create_ontology("http://mowl.borg/geo/chain3")
    c = {
        name: adapter.create_class(NS + name)
        for name in ("A", "B", "Cc", "D", "E1", "E2", "E3", "G1", "G2", "G3")
    }
    bot = adapter.create_class(BOTTOM)
    # D <= A, D <= G1, D <= G2, D <= G3; Cc <= B
    mgr.addAxiom(ont, adapter.create_subclass_of(c["D"], c["A"]))
    mgr.addAxiom(ont, adapter.create_subclass_of(c["D"], c["G1"]))
    mgr.addAxiom(ont, adapter.create_subclass_of(c["D"], c["G2"]))
    mgr.addAxiom(ont, adapter.create_subclass_of(c["D"], c["G3"]))
    mgr.addAxiom(ont, adapter.create_subclass_of(c["Cc"], c["B"]))
    # A & B <= E1; E1 & G1 <= E2; E2 & G2 <= E3; E3 & G3 <= bot
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(c["A"], c["B"]), c["E1"]
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(c["E1"], c["G1"]), c["E2"]
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(c["E2"], c["G2"]), c["E3"]
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(c["E3"], c["G3"]), bot
        ),
    )
    return ont


def _assert_supervision_exact_against_cm(ont: OWLOntology) -> PointSupervision:
    """Assert pos_pairs and hard_neg_pairs match brute-force cm.entails / !cm.joint_satisfiable."""
    cm = CanonicalModel.from_ontology(ont)
    cls_dict = _class_dict_from_ontology(ont)
    inds = sorted(str(i.toStringID()) for i in ont.getIndividualsInSignature())
    ind_dict = {iri: idx for idx, iri in enumerate(inds)}
    rels = sorted(str(r.toStringID()) for r in ont.getObjectPropertiesInSignature())
    rel_dict = {iri: idx for idx, iri in enumerate(rels)}
    inv_cls = {idx: iri for iri, idx in cls_dict.items()}

    ps = PointSupervision.from_canonical_model(cm, cls_dict, ind_dict, rel_dict)
    assert ps.hard_neg_capped is False

    cand_cls = ps.candidate_class_indices.tolist()
    brute_pos = {
        (e, c)
        for e, d in enumerate(ps.element_names)
        for c in cand_cls
        if cm.entails(d, inv_cls[c])
    }
    brute_hard = {
        (e, c)
        for e, d in enumerate(ps.element_names)
        for c in cand_cls
        if not cm.joint_satisfiable({d, inv_cls[c]})
    }
    pos_set = {tuple(row) for row in ps.pos_pairs.tolist()}
    hard_set = {tuple(row) for row in ps.hard_neg_pairs.tolist()}
    assert pos_set == brute_pos
    assert hard_set == brute_hard
    return ps


def test_hard_neg_pairs_exact_against_brute_force(
    t3_kb: OWLOntology, abox_kb: OWLOntology
) -> None:
    """hard_neg_pairs equals brute-force {(x_D, C) : not cm.joint_satisfiable({D, C})} on every fixture."""
    # 1. t3.owl: 12 pairs including two-step (A,X3),(B,X1),(C,X2) and reverses
    ps_t3 = _assert_supervision_exact_against_cm(t3_kb)
    assert ps_t3.hard_neg_pairs.shape == (12, 2)

    # 2. helly_ok.owl: 14 pairs including two-step (D,F) and (F,D)
    helly_ont = PathDataset(fixture_path("helly_ok.owl")).ontology
    ps_helly = _assert_supervision_exact_against_cm(helly_ont)
    assert ps_helly.hard_neg_pairs.shape == (14, 2)
    helly_cls = _class_dict_from_ontology(helly_ont)
    helly_inv = {v: k.split("#")[-1] for k, v in helly_cls.items()}
    helly_named = {
        (ps_helly.element_names[e].split("#")[-1], helly_inv[c])
        for e, c in ps_helly.hard_neg_pairs.tolist()
    }
    assert ("D", "F") in helly_named and ("F", "D") in helly_named

    # 3. roles.owl and top_lhs.owl: 0 pairs
    for fname in ("roles.owl", "top_lhs.owl"):
        ps_f = _assert_supervision_exact_against_cm(
            PathDataset(fixture_path(fname)).ontology
        )
        assert ps_f.hard_neg_pairs.shape == (0, 2)

    # 4. abox_kb: 3 pairs (including individual element bob vs Female)
    ps_abox = _assert_supervision_exact_against_cm(abox_kb)
    assert ps_abox.hard_neg_pairs.shape == (3, 2)

    # 5. All 20 random ABox KBs from tests/geometric/test_abox.py
    for seed in range(20):
        ont_seed, _, _ = _build_random_abox_kb(seed)
        _assert_supervision_exact_against_cm(ont_seed)

    # 6. 3-step conjunction chain KB: 12 exact pairs (vs 4 under one-step closure)
    chain_ont = _build_3step_chain_kb()
    ps_chain = _assert_supervision_exact_against_cm(chain_ont)
    assert ps_chain.hard_neg_pairs.shape == (12, 2)
    chain_cls = _class_dict_from_ontology(chain_ont)
    chain_inv = {v: k.split("#")[-1] for k, v in chain_cls.items()}
    chain_named = {
        (ps_chain.element_names[e].split("#")[-1], chain_inv[c])
        for e, c in ps_chain.hard_neg_pairs.tolist()
    }
    for multi_step_pair in (
        ("D", "Cc"), ("Cc", "D"),
        ("D", "B"), ("B", "D"),
        ("D", "E1"), ("E1", "D"),
        ("D", "E2"), ("E2", "D"),
    ):
        assert multi_step_pair in chain_named

    # 7. When hard_neg_cap is hit, hard_neg_capped is True and output is capped
    cm_t3 = CanonicalModel.from_ontology(t3_kb)
    ps_capped = PointSupervision.from_canonical_model(
        cm_t3, _class_dict_from_ontology(t3_kb), {}, rng_seed=0, hard_neg_cap=5
    )
    assert ps_capped.hard_neg_capped is True
    assert ps_capped.hard_neg_pairs.shape == (5, 2)


def test_t3_tiny_hard_neg_cap(t3_kb: OWLOntology) -> None:
    """With a tiny cap (10) on t3, returns 10 distinct valid hard negatives and hard_neg_capped=True."""
    cm = CanonicalModel.from_ontology(t3_kb)
    cls_dict = _class_dict_from_ontology(t3_kb)
    inv_cls = {idx: iri for iri, idx in cls_dict.items()}
    ps_full = PointSupervision.from_canonical_model(cm, cls_dict, {})
    exact_hard_set = {tuple(row) for row in ps_full.hard_neg_pairs.tolist()}
    assert len(exact_hard_set) == 12

    for seed in range(5):
        ps_cap10 = PointSupervision.from_canonical_model(
            cm, cls_dict, {}, rng_seed=seed, hard_neg_cap=10
        )
        assert ps_cap10.hard_neg_capped is True
        assert ps_cap10.hard_neg_pairs.shape == (10, 2)
        rows = [tuple(r) for r in ps_cap10.hard_neg_pairs.tolist()]
        assert len(set(rows)) == 10
        for elem_i, cls_j in rows:
            assert (elem_i, cls_j) in exact_hard_set
            d_name = ps_cap10.element_names[elem_i]
            c_iri = inv_cls[cls_j]
            assert not cm.entails(d_name, c_iri)
            assert not cm.joint_satisfiable({d_name, c_iri})


def test_sample_permutes_all_elements_coverage() -> None:
    """sample() permutes positive and role index arrays so every element is sampled when n_pos_take < n_pos."""
    n_elem = 700
    n_cls = 10
    pos = torch.stack(
        [
            torch.arange(n_elem, dtype=torch.long),
            torch.zeros(n_elem, dtype=torch.long),
        ],
        dim=1,
    )
    hard = torch.stack(
        [
            torch.arange(n_elem, dtype=torch.long),
            torch.ones(n_elem, dtype=torch.long),
        ],
        dim=1,
    )
    roles = torch.stack(
        [
            torch.zeros(n_elem, dtype=torch.long),
            torch.arange(n_elem, dtype=torch.long),
            torch.arange(n_elem, dtype=torch.long),
        ],
        dim=1,
    )
    ps = PointSupervision(
        n_elements=n_elem,
        element_names=[f"e{i}" for i in range(n_elem)],
        element_class_index=torch.arange(n_elem, dtype=torch.long) % n_cls,
        element_is_individual=torch.zeros(n_elem, dtype=torch.bool),
        pos_pairs=pos,
        hard_neg_pairs=hard,
        role_pairs=roles,
        candidate_class_indices=torch.arange(n_cls, dtype=torch.long),
    )

    seen_pos = set()
    seen_roles = set()
    for seed in range(20):
        b = ps.sample(1500, rng=seed)
        seen_pos.update(b["pos"][:, 0].tolist())
        seen_roles.update(b["roles"][:, 1].tolist())
    assert len(seen_pos) == n_elem
    assert max(seen_pos) == n_elem - 1
    assert len(seen_roles) == n_elem


def test_nf2_exact_check_bound_and_seeded_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NF2 exact checks are bounded by 4 * hard_neg_cap (setting hard_neg_capped) and cache keys include seed and cap."""
    import mowl.datasets.el.points as points_mod

    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    ont = adapter.create_ontology("http://mowl.borg/geo/nf2_bound_test")

    c_a = adapter.create_class(NS + "A")
    c_b = adapter.create_class(NS + "B")
    c_e = adapter.create_class(NS + "E")
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(c_a, c_b), c_e
        ),
    )
    sub_a = [adapter.create_class(f"{NS}A_{i}") for i in range(6)]
    sub_b = [adapter.create_class(f"{NS}B_{i}") for i in range(6)]
    for sa in sub_a:
        mgr.addAxiom(ont, adapter.create_subclass_of(sa, c_a))
    for sb in sub_b:
        mgr.addAxiom(ont, adapter.create_subclass_of(sb, c_b))

    cm = CanonicalModel.from_ontology(ont)
    names = sorted(cm.concept_names - {TOP, BOTTOM})
    cls_dict = {iri: idx for idx, iri in enumerate(names)}

    orig_check = points_mod._pair_joint_satisfiable
    calls = {"count": 0}

    def _counting_check(*args, **kwargs):
        calls["count"] += 1
        return orig_check(*args, **kwargs)

    monkeypatch.setattr(points_mod, "_pair_joint_satisfiable", _counting_check)

    tiny_cap = 2
    max_checks = MAX_EXACT_CHECKS_FACTOR * tiny_cap
    ps = PointSupervision.from_canonical_model(
        cm, cls_dict, rng_seed=3, hard_neg_cap=tiny_cap
    )
    assert calls["count"] == max_checks
    assert ps.hard_neg_capped is True

    # Verify build_or_load_point_supervision keys by rng_seed, hard_neg_cap, and POINTS_CODE_TAG
    class _DummyDataset:
        def __init__(self, ontology, data_root: Path) -> None:
            self.ontology = ontology
            self.data_root = str(data_root)

    ds = _DummyDataset(ont, tmp_path)
    ps_s1 = build_or_load_point_supervision(
        ds, cls_dict, rng_seed=1, hard_neg_cap=2, cache_dir=tmp_path
    )
    ps_s2 = build_or_load_point_supervision(
        ds, cls_dict, rng_seed=2, hard_neg_cap=2, cache_dir=tmp_path
    )
    ps_cap5 = build_or_load_point_supervision(
        ds, cls_dict, rng_seed=1, hard_neg_cap=5, cache_dir=tmp_path
    )
    assert ps_s1.rng_seed == 1
    assert ps_s2.rng_seed == 2
    assert ps_cap5.rng_seed == 1
    assert ps_s1.ontology_sha256 != ps_s2.ontology_sha256
    assert ps_s1.ontology_sha256 != ps_cap5.ontology_sha256

    cache_files = sorted(p.name for p in tmp_path.glob(f"points_{POINTS_CODE_TAG}_*.pt"))
    assert len(cache_files) == 3
    assert any("seed1_cap2" in name for name in cache_files)
    assert any("seed2_cap2" in name for name in cache_files)
    assert any("seed1_cap5" in name for name in cache_files)

    # Reload from cache and verify stored rng_seed is preserved
    calls_before = calls["count"]
    ps_s2_reload = build_or_load_point_supervision(
        ds, cls_dict, rng_seed=2, hard_neg_cap=2, cache_dir=tmp_path
    )
    assert calls["count"] == calls_before
    assert ps_s2_reload.rng_seed == 2
    assert ps_s2_reload.hard_neg_capped is True


def _run_synthetic_scale_benchmark() -> Tuple[float, float, int, int, int]:
    """Run the 50k x 50k synthetic scale benchmark and return (elapsed_s, max_rss_gb, n_pos, n_hard, n_sampled)."""
    n_elements = 50_000
    n_classes = 50_000
    ancestors_per_elem = 15
    n_bot_axioms = 300

    rng = np.random.default_rng(12345)
    diag_rows = np.arange(n_elements, dtype=np.int64)
    diag_cols = np.arange(n_classes, dtype=np.int64)
    extra_rows = np.repeat(np.arange(n_elements, dtype=np.int64), ancestors_per_elem - 1)
    extra_cols = rng.integers(0, n_classes, size=extra_rows.shape[0], dtype=np.int64)

    rows = np.concatenate([diag_rows, extra_rows])
    cols = np.concatenate([diag_cols, extra_cols])
    data = np.ones(rows.shape[0], dtype=bool)
    membership = sp.csr_matrix((data, (rows, cols)), shape=(n_elements, n_classes), dtype=bool)

    dis_a = rng.integers(0, n_classes, size=n_bot_axioms, dtype=np.int64)
    dis_b = (dis_a + rng.integers(1, n_classes, size=n_bot_axioms, dtype=np.int64)) % n_classes
    disjoint_pairs = list(zip(dis_a.tolist(), dis_b.tolist()))

    element_names = [f"http://example.org/C_{i}" for i in range(n_elements)]
    element_class_index = torch.arange(n_elements, dtype=torch.long)
    element_is_individual = torch.zeros(n_elements, dtype=torch.bool)

    t0 = time.perf_counter()
    ps = PointSupervision.from_sparse_membership(
        M=membership,
        element_names=element_names,
        element_class_index=element_class_index,
        element_is_individual=element_is_individual,
        disjoint_pairs=disjoint_pairs,
        rng_seed=0,
    )
    assert isinstance(ps.hard_neg_capped, bool)
    batch = ps.sample(100_000, rng=0)
    elapsed = time.perf_counter() - t0
    max_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)

    for elem_i, cls_j in batch["rand"][:200].tolist():
        assert not bool(membership[elem_i, cls_j])

    n_sampled = sum(int(t.shape[0]) for t in batch.values())
    return (
        elapsed,
        max_rss_gb,
        int(ps.pos_pairs.shape[0]),
        int(ps.hard_neg_pairs.shape[0]),
        n_sampled,
    )


@pytest.mark.slow
def test_point_supervision_synthetic_scale() -> None:
    """Synthetic scale test: 50,000 elements x 50,000 classes (~15 ancestors, 300 bot axioms) finishes in <60s and <2GB."""
    # Earlier tests in the session may have already increased the process/JVM
    # high-water mark (ru_maxrss), so we check the RSS increase across the
    # benchmark and also check total ru_maxrss when the baseline is < 1.5 GB.
    rss_before_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    elapsed, rss_after_gb, n_pos, n_hard, n_sampled = _run_synthetic_scale_benchmark()
    rss_delta_gb = max(0.0, rss_after_gb - rss_before_gb)
    print(
        f"[scale_test] elapsed={elapsed:.3f}s, rss_before={rss_before_gb:.3f}GB, "
        f"rss_after={rss_after_gb:.3f}GB, rss_delta={rss_delta_gb:.3f}GB, "
        f"pos={n_pos}, hard={n_hard}, sampled={n_sampled}"
    )
    assert elapsed < 60.0, f"Scale test took {elapsed:.2f}s (>= 60s limit)"
    assert rss_delta_gb < 2.0, f"RSS increase was {rss_delta_gb:.3f} GB (>= 2.0 GB limit)"
    if rss_before_gb < 1.5:
        assert rss_after_gb < 2.0, f"Peak RSS (ru_maxrss) was {rss_after_gb:.3f} GB (>= 2.0 GB limit)"
    assert n_pos >= 50_000
    assert 0 < n_hard <= 2_000_000
    assert n_sampled <= 100_000


_GO_LIKE_SUBPROCESS_SCRIPT = """
import json
import resource
import time
import numpy as np
import scipy.sparse as sp
import torch
import mowl
mowl.init_jvm("1g")
from mowl.datasets.el.points import PointSupervision

n_classes = 50_000
n_inds = 23_000
n_elements = n_classes + n_inds  # 73,000 elements
rng = np.random.default_rng(42)

# Three root classes 0, 1, 2 with 30k, 12k, 8k class descendants
root_for_cls = np.empty(n_classes, dtype=np.int64)
root_for_cls[:30_000] = 0
root_for_cls[30_000:42_000] = 1
root_for_cls[42_000:] = 2

elem_cls = np.concatenate([
    np.arange(n_classes, dtype=np.int64),
    rng.integers(0, n_classes, size=n_inds, dtype=np.int64),
])
elem_root = root_for_cls[elem_cls]

rows_list = [np.arange(n_elements, dtype=np.int64), np.arange(n_elements, dtype=np.int64)]
cols_list = [elem_cls, elem_root]
extra_rows = np.repeat(np.arange(n_elements, dtype=np.int64), 13)
extra_cols = rng.integers(3, n_classes, size=extra_rows.shape[0], dtype=np.int64)
rows = np.concatenate(rows_list + [extra_rows])
cols = np.concatenate(cols_list + [extra_cols])
membership = sp.csr_matrix(
    (np.ones(rows.shape[0], dtype=bool), (rows, cols)),
    shape=(n_elements, n_classes),
    dtype=bool,
)

dis_a = rng.integers(3, n_classes, size=300, dtype=np.int64)
dis_b = (dis_a + rng.integers(1, n_classes - 3, size=300, dtype=np.int64)) % n_classes
disjoint_pairs = [(0, 1), (0, 2), (1, 2)] + list(zip(dis_a.tolist(), dis_b.tolist()))

element_names = [f"http://example.org/E_{i}" for i in range(n_elements)]
element_class_index = torch.cat([
    torch.arange(n_classes, dtype=torch.long),
    torch.arange(n_inds, dtype=torch.long),
])
element_is_individual = torch.cat([
    torch.zeros(n_classes, dtype=torch.bool),
    torch.ones(n_inds, dtype=torch.bool),
])

t0 = time.perf_counter()
ps = PointSupervision.from_sparse_membership(
    M=membership,
    element_names=element_names,
    element_class_index=element_class_index,
    element_is_individual=element_is_individual,
    candidate_class_indices=torch.arange(n_classes, dtype=torch.long),
    class_element_rows=torch.arange(n_classes, dtype=torch.long),
    disjoint_pairs=disjoint_pairs,
    rng_seed=0,
)
batch = ps.sample(100_000, rng=0)
elapsed = time.perf_counter() - t0
peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)

for elem_i, cls_j in ps.hard_neg_pairs[:200].tolist():
    assert not bool(membership[elem_i, cls_j])
for elem_i, cls_j in batch["rand"][:200].tolist():
    assert not bool(membership[elem_i, cls_j])

print("GO_SCALE_RESULT:" + json.dumps({
    "elapsed": elapsed,
    "peak_rss_gb": peak_rss_gb,
    "n_pos": int(ps.pos_pairs.shape[0]),
    "n_hard": int(ps.hard_neg_pairs.shape[0]),
    "hard_neg_capped": bool(ps.hard_neg_capped),
    "n_sampled": int(sum(t.shape[0] for t in batch.values())),
}))
"""


@pytest.mark.slow
def test_point_supervision_go_like_scale() -> None:
    """GO-like scale test (73k elements, 50k classes, 3 pairwise-disjoint 30k/12k/8k roots + 300 bot axioms) finishes in <60s and <2GB child peak RSS."""
    proc = subprocess.run(
        [sys.executable, "-c", _GO_LIKE_SUBPROCESS_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
    )
    result_line = next(
        line for line in proc.stdout.splitlines() if line.startswith("GO_SCALE_RESULT:")
    )
    metrics = json.loads(result_line[len("GO_SCALE_RESULT:") :])
    elapsed = float(metrics["elapsed"])
    peak_rss_gb = float(metrics["peak_rss_gb"])
    n_pos = int(metrics["n_pos"])
    n_hard = int(metrics["n_hard"])
    capped = bool(metrics["hard_neg_capped"])
    n_sampled = int(metrics["n_sampled"])
    print(
        f"[go_scale_test] elapsed={elapsed:.3f}s, peak_rss={peak_rss_gb:.3f}GB, "
        f"pos={n_pos}, hard={n_hard}, capped={capped}, sampled={n_sampled}"
    )
    assert elapsed < 60.0, f"GO-like scale test took {elapsed:.2f}s (>= 60s limit)"
    assert peak_rss_gb < 2.0, f"Child peak RSS was {peak_rss_gb:.3f} GB (>= 2.0 GB limit)"
    assert capped is True
    assert n_hard == 2_000_000
    assert n_pos >= 73_000
    assert n_sampled <= 100_000
