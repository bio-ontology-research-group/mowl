"""Tests for PointSupervision in mowl.datasets.el.points."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
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
    assert ps.hard_neg_pairs.shape[0] > 0
    for elem_i, cls_j in ps.hard_neg_pairs.tolist():
        d_name = ps.element_names[elem_i]
        c_iri = inv_cls[cls_j]
        assert not cm.entails(d_name, c_iri)
        assert not cm.joint_satisfiable({d_name, c_iri})

    # Sampled random negatives are never entailed
    sampled = ps.sample(5, rng=42)
    assert set(sampled.keys()) == {"pos", "hard", "rand", "roles"}
    assert sampled["pos"].shape == (5, 2)
    assert sampled["rand"].shape == (5, 2)
    for elem_i, cls_j in sampled["rand"].tolist():
        d_name = ps.element_names[elem_i]
        c_iri = inv_cls[cls_j]
        assert c_iri not in (unsat_iri, TOP, BOTTOM)
        assert not cm.entails(d_name, c_iri)


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


def test_cache_roundtrip(t3_kb: OWLOntology, tmp_path: Path) -> None:
    """save/load round-trips PointSupervision and validates POINTS_CODE_TAG and ontology sha256."""
    cm = CanonicalModel.from_ontology(t3_kb)
    cls_dict = _class_dict_from_ontology(t3_kb)
    sha = ontology_sha256(t3_kb, cls_dict, {})
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
