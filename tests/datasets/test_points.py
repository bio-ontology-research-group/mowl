"""Tests for PointSupervision in mowl.datasets.el.points."""
from __future__ import annotations

from pathlib import Path
import resource
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
    POINTS_CODE_TAG,
    PointSupervision,
    build_or_load_point_supervision,
    ontology_sha256,
)
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.normal_forms import BOTTOM, TOP
from mowl.nn import MultiBoxELModule, MultiELBEModule
from mowl.owlapi import OWLAPIAdapter
from tests.geometric.helpers import NS, fixture_path
from tests.geometric.test_abox import IND, abox_kb  # noqa: F401


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

    # Every hard negative (x_D, C) is jointly unsatisfiable and not entailed
    # On t3.owl there are 3 disjointness axioms (A & B <= bot, A & C <= bot, B & C <= bot),
    # yielding 6 directed disjoint pairs independent of has_bottom_axiom.
    assert ps.hard_neg_pairs.shape == (6, 2)
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
    # Note: tests/conftest.py preloads 9 builtin datasets into a 10 GB JVM at
    # session start (~5.8 GB baseline ru_maxrss), so we measure ru_maxrss
    # before and after construction (and also check total ru_maxrss when the
    # baseline is < 1.5 GB).
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
