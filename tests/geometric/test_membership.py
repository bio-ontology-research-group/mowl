"""Tests for MembershipStructure in mowl.geometric.membership."""
from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

import pytest
import torch

import mowl
mowl.init_jvm("2g")

from org.semanticweb.owlapi.model import OWLOntology

from mowl.geometric.completion import CanonicalModel
from mowl.geometric.membership import MEMBERSHIP_CODE_TAG, MembershipStructure
from mowl.geometric.normal_forms import BOTTOM, TOP
from tests.geometric.test_abox import IND, NS, _build_random_abox_kb, abox_kb  # noqa: F401


def _dicts_for_abox_kb() -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    class_iris = [
        NS + "Father",
        NS + "Mother",
        NS + "Parent",
        NS + "Male",
        NS + "Female",
    ]
    ind_iris = [
        IND + "bob",
        IND + "ann",
        IND + "carl",
    ]
    role_iris = [NS + "hasChild"]
    class_index_dict = {iri: idx for idx, iri in enumerate(class_iris)}
    individual_index_dict = {iri: idx for idx, iri in enumerate(ind_iris)}
    relation_index_dict = {iri: idx for idx, iri in enumerate(role_iris)}
    return class_index_dict, individual_index_dict, relation_index_dict


def test_toy_family_kb_expected_matrix(abox_kb: OWLOntology) -> None:
    """Toy family KB (from WP-A Task 1's fixture) gives the expected matrix and role edges."""
    cm = CanonicalModel.from_ontology(abox_kb)
    class_index_dict, individual_index_dict, _ = _dicts_for_abox_kb()

    ms = MembershipStructure.from_canonical_model(
        cm, class_index_dict, individual_index_dict
    )

    # Support S has 5 satisfiable named concepts + 3 individuals = 8 elements
    assert len(ms.element_names) == 8
    assert set(ms.element_index.keys()) == set(ms.element_names)
    for idx, name in enumerate(ms.element_names):
        assert ms.element_index[name] == idx
    assert TOP not in ms.element_index
    assert BOTTOM not in ms.element_index

    assert ms.M.is_sparse
    assert ms.M.dtype == torch.bool
    assert ms.M.shape == (8, 5)
    assert ms.individual_element.dtype == torch.long
    assert ms.individual_element.shape == (3,)

    # Expected memberships per element in the toy family KB:
    # Father <= Parent; Mother <= Parent; exists hasChild.TOP <= Parent; Male and Female <= BOTTOM
    # Father(bob), Male(bob), hasChild(bob, ann), hasChild(carl, ann)
    expected_classes_by_element: Dict[str, Set[str]] = {
        NS + "Father": {NS + "Father", NS + "Parent"},
        NS + "Mother": {NS + "Mother", NS + "Parent"},
        NS + "Parent": {NS + "Parent"},
        NS + "Male": {NS + "Male"},
        NS + "Female": {NS + "Female"},
        cm.individuals[IND + "bob"]: {NS + "Father", NS + "Parent", NS + "Male"},
        cm.individuals[IND + "ann"]: set(),
        cm.individuals[IND + "carl"]: {NS + "Parent"},
    }

    dense_m = ms.M.to_dense()
    for elem_name, expected_sups in expected_classes_by_element.items():
        row_idx = ms.element_index[elem_name]
        for cls_iri, col_idx in class_index_dict.items():
            assert bool(dense_m[row_idx, col_idx]) == (cls_iri in expected_sups), (
                f"Mismatch at ({elem_name}, {cls_iri})"
            )

    # Check individual_element maps each individual index to its nominal's row in S
    for ind_iri, ind_idx in individual_index_dict.items():
        nom_name = cm.individuals[ind_iri]
        assert int(ms.individual_element[ind_idx].item()) == ms.element_index[nom_name]

    # Canonical role edges for hasChild (role index 0): (bob, ann) and (carl, ann)
    assert 0 in ms.roles
    r_mat = ms.roles[0]
    assert r_mat.is_sparse
    assert r_mat.dtype == torch.bool
    assert r_mat.shape == (8, 8)
    bob_el = int(ms.individual_element[individual_index_dict[IND + "bob"]].item())
    ann_el = int(ms.individual_element[individual_index_dict[IND + "ann"]].item())
    carl_el = int(ms.individual_element[individual_index_dict[IND + "carl"]].item())
    expected_edges = {(bob_el, ann_el), (carl_el, ann_el)}
    r_indices = r_mat.coalesce().indices()
    r_values = r_mat.coalesce().values()
    actual_edges = {
        (int(r_indices[0, k].item()), int(r_indices[1, k].item()))
        for k in range(r_values.numel())
        if bool(r_values[k].item())
    }
    assert actual_edges == expected_edges


def test_m_agrees_with_cm_entails_on_200_random_pairs() -> None:
    """M agrees with cm.entails on 200 random pairs across random ABox KBs."""
    rng = random.Random(42)
    pairs_checked = 0

    for seed in range(10):
        ont, ind_iris, class_iris = _build_random_abox_kb(seed)
        cm = CanonicalModel.from_ontology(ont)
        class_index_dict = {iri: idx for idx, iri in enumerate(class_iris)}
        individual_index_dict = {iri: idx for idx, iri in enumerate(ind_iris)}

        ms = MembershipStructure.from_canonical_model(
            cm, class_index_dict, individual_index_dict
        )
        dense_m = ms.M.to_dense()

        for _ in range(20):
            x_idx = rng.randrange(len(ms.element_names))
            c_iri = rng.choice(class_iris)
            c_idx = class_index_dict[c_iri]
            name_of_x = ms.element_names[x_idx]

            assert bool(dense_m[x_idx, c_idx].item()) == cm.entails(name_of_x, c_iri)
            pairs_checked += 1

    assert pairs_checked == 200


def test_type_mask_of_individual_contains_own_element(abox_kb: OWLOntology) -> None:
    """type_mask of an individual's types contains its own element (and matches conjunction of M columns)."""
    cm = CanonicalModel.from_ontology(abox_kb)
    class_index_dict, individual_index_dict, _ = _dicts_for_abox_kb()
    ms = MembershipStructure.from_canonical_model(
        cm, class_index_dict, individual_index_dict
    )

    dense_m = ms.M.to_dense()
    for ind_iri, ind_idx in individual_index_dict.items():
        own_elem = int(ms.individual_element[ind_idx].item())
        ind_types = torch.nonzero(dense_m[own_elem], as_tuple=False).reshape(-1).long()
        mask = ms.type_mask(ind_types)
        assert mask.dtype == torch.bool
        assert mask.shape == (len(ms.element_names),)
        assert bool(mask[own_elem].item()) is True

        # Check conjunction semantics for bob (Father & Male -> only bob's element has both)
        if ind_iri == IND + "bob":
            bob_asserted = torch.tensor(
                [class_index_dict[NS + "Father"], class_index_dict[NS + "Male"]],
                dtype=torch.long,
            )
            bob_mask = ms.type_mask(bob_asserted)
            assert bool(bob_mask[own_elem].item()) is True
            # Father element does not have Male, Male element does not have Father
            assert bool(bob_mask[ms.element_index[NS + "Father"]].item()) is False
            assert bool(bob_mask[ms.element_index[NS + "Male"]].item()) is False


def test_empty_type_individual_returns_all_of_s(abox_kb: OWLOntology) -> None:
    """type_mask with an empty tensor returns all-True over S."""
    cm = CanonicalModel.from_ontology(abox_kb)
    class_index_dict, individual_index_dict, _ = _dicts_for_abox_kb()
    ms = MembershipStructure.from_canonical_model(
        cm, class_index_dict, individual_index_dict
    )

    empty_types = torch.empty((0,), dtype=torch.long)
    mask = ms.type_mask(empty_types)
    assert mask.dtype == torch.bool
    assert mask.shape == (len(ms.element_names),)
    assert bool(mask.all().item()) is True


def test_absent_class_and_absent_individual_handling(abox_kb: OWLOntology) -> None:
    """A class absent from cm gets an all-zero column; an absent individual raises ValueError."""
    cm = CanonicalModel.from_ontology(abox_kb)
    class_index_dict, individual_index_dict, _ = _dicts_for_abox_kb()

    # Add a class to class_index_dict that is not in the canonical model
    class_with_extra = dict(class_index_dict)
    extra_cls_idx = len(class_with_extra)
    class_with_extra[NS + "UnseenClass"] = extra_cls_idx

    ms = MembershipStructure.from_canonical_model(
        cm, class_with_extra, individual_index_dict
    )
    assert ms.M.shape == (len(ms.element_names), len(class_with_extra))
    dense_m = ms.M.to_dense()
    assert not bool(dense_m[:, extra_cls_idx].any().item())

    # Auxiliary concept in cm (not in class_index_dict) is excluded from S
    partial_classes = {NS + "Father": 0, NS + "Parent": 1}
    ms_partial = MembershipStructure.from_canonical_model(
        cm, partial_classes, individual_index_dict
    )
    assert NS + "Mother" not in ms_partial.element_index
    assert NS + "Father" in ms_partial.element_index

    # Absent individual raises ValueError
    ind_with_missing = dict(individual_index_dict)
    ind_with_missing[IND + "nonexistent"] = len(ind_with_missing)
    with pytest.raises(ValueError, match="nonexistent"):
        MembershipStructure.from_canonical_model(
            cm, class_index_dict, ind_with_missing
        )


def test_save_and_load_roundtrip(abox_kb: OWLOntology, tmp_path) -> None:
    """save/load round-trips MembershipStructure and validates MEMBERSHIP_CODE_TAG."""
    cm = CanonicalModel.from_ontology(abox_kb)
    class_index_dict, individual_index_dict, _ = _dicts_for_abox_kb()
    ms = MembershipStructure.from_canonical_model(
        cm, class_index_dict, individual_index_dict
    )

    out_file = tmp_path / "membership.pt"
    ms.save(out_file)

    raw = torch.load(out_file, weights_only=False)
    assert isinstance(raw, dict)
    assert raw["code_tag"] == MEMBERSHIP_CODE_TAG

    loaded = MembershipStructure.load(out_file)
    assert loaded.element_names == ms.element_names
    assert loaded.element_index == ms.element_index
    assert torch.equal(loaded.individual_element, ms.individual_element)
    assert torch.equal(loaded.M.to_dense(), ms.M.to_dense())
    assert set(loaded.roles.keys()) == set(ms.roles.keys())
    for k in ms.roles:
        assert torch.equal(loaded.roles[k].to_dense(), ms.roles[k].to_dense())

    # Tampered code_tag raises ValueError on load
    raw["code_tag"] = "stale-tag"
    stale_file = tmp_path / "membership_stale.pt"
    torch.save(raw, stale_file)
    with pytest.raises(ValueError, match="code_tag"):
        MembershipStructure.load(stale_file)
