"""Tests for CanonicalMixture model in mowl.models.canonical_mixture."""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Iterator

import pytest
import torch

import mowl
mowl.init_jvm("2g")

from org.semanticweb.owlapi.model import IRI

from mowl.datasets import PathDataset
from mowl.geometric.normal_forms import BOTTOM, TOP
from mowl.models import CanonicalMixture
from mowl.owlapi import OWLAPIAdapter
from tests.datasetFactory import FamilyDataset

NS = "http://mowl.borg/test/cm#"
IND = "http://mowl.borg/test/cm/ind#"


@pytest.fixture
def local_scratch_dir() -> Iterator[Path]:
    """Workspace-local scratch directory avoiding /tmp."""
    scratch = Path(__file__).resolve().parent / "_scratch_canonical_mixture"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        yield scratch
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def _write_family_abox_dataset(tmp_path: Path) -> PathDataset:
    """Create a small Family-style OWL dataset with individuals (train + valid) on disk."""
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory

    ont = adapter.create_ontology("http://mowl.borg/test/cm/train")
    top = adapter.create_class(TOP)
    bot = adapter.create_class(BOTTOM)
    father = adapter.create_class(NS + "Father")
    mother = adapter.create_class(NS + "Mother")
    parent = adapter.create_class(NS + "Parent")
    male = adapter.create_class(NS + "Male")
    female = adapter.create_class(NS + "Female")
    person = adapter.create_class(NS + "Person")
    has_child = adapter.create_object_property(NS + "hasChild")

    bob = adapter.create_individual(IND + "bob")
    alice = adapter.create_individual(IND + "alice")
    carl = adapter.create_individual(IND + "carl")
    diana = adapter.create_individual(IND + "diana")

    for cls_obj in [father, mother, parent, male, female, person]:
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(cls_obj))
    mgr.addAxiom(ont, df.getOWLDeclarationAxiom(has_child))
    for ind_obj in [bob, alice, carl, diana]:
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(ind_obj))

    # TBox:
    # Father <= Parent,Father <= Male, Mother <= Parent, Mother <= Female
    # Parent <= Person, Male <= Person, Female <= Person
    # exists hasChild.TOP <= Parent
    # Male and Female <= BOTTOM
    mgr.addAxiom(ont, adapter.create_subclass_of(father, parent))
    mgr.addAxiom(ont, adapter.create_subclass_of(father, male))
    mgr.addAxiom(ont, adapter.create_subclass_of(mother, parent))
    mgr.addAxiom(ont, adapter.create_subclass_of(mother, female))
    mgr.addAxiom(ont, adapter.create_subclass_of(parent, person))
    mgr.addAxiom(ont, adapter.create_subclass_of(male, person))
    mgr.addAxiom(ont, adapter.create_subclass_of(female, person))
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_some_values_from(has_child, top), parent
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(male, female), bot
        ),
    )

    # ABox:
    # Father(bob), Mother(alice), Male(carl), Person(diana)
    # hasChild(bob, carl), hasChild(alice, diana)
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(father, bob))
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(mother, alice))
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(male, carl))
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(person, diana))
    mgr.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(has_child, bob, carl))
    mgr.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(has_child, alice, diana))

    train_path = tmp_path / "family_abox_train.owl"
    mgr.saveOntology(ont, IRI.create(train_path.as_uri()))

    val_ont = adapter.create_ontology("http://mowl.borg/test/cm/valid")
    mgr.addAxiom(val_ont, df.getOWLClassAssertionAxiom(parent, bob))
    mgr.addAxiom(val_ont, df.getOWLClassAssertionAxiom(person, carl))
    valid_path = tmp_path / "family_abox_valid.owl"
    mgr.saveOntology(val_ont, IRI.create(valid_path.as_uri()))

    return PathDataset(str(train_path), validation_path=str(valid_path))


def test_tbox_only_dataset_raises_value_error(family_dataset: FamilyDataset) -> None:
    """CanonicalMixture raises ValueError('CanonicalMixture needs individuals') on TBox-only FamilyDataset."""
    with pytest.raises(ValueError, match="CanonicalMixture needs individuals"):
        CanonicalMixture(family_dataset, rank=8)


def test_family_abox_dataset_trains_and_preserves_faithfulness(
    local_scratch_dir: Path,
) -> None:
    """Family-with-individuals dataset trains for 3 epochs; entailed scores == 0 and contradicted == inf before and after."""
    ds = _write_family_abox_dataset(local_scratch_dir)
    ckpt = local_scratch_dir / "cm_best.pt"
    model = CanonicalMixture(
        ds,
        rank=8,
        lr=0.01,
        batch_size=2,
        entropy_reg=0.01,
        model_filepath=str(ckpt),
    )

    bob_idx = model.individual_index_dict[IND + "bob"]
    alice_idx = model.individual_index_dict[IND + "alice"]
    father_idx = model.class_index_dict[NS + "Father"]
    parent_idx = model.class_index_dict[NS + "Parent"]
    person_idx = model.class_index_dict[NS + "Person"]
    female_idx = model.class_index_dict[NS + "Female"]
    male_idx = model.class_index_dict[NS + "Male"]

    entailed_pairs = torch.tensor(
        [
            [bob_idx, father_idx],
            [bob_idx, parent_idx],
            [bob_idx, person_idx],
            [alice_idx, parent_idx],
        ],
        dtype=torch.long,
    )
    contradicted_pairs = torch.tensor(
        [
            [bob_idx, female_idx],
            [alice_idx, male_idx],
        ],
        dtype=torch.long,
    )

    # Check at init
    init_entailed = model.module.class_assertion_score(entailed_pairs)
    init_contra = model.module.class_assertion_score(contradicted_pairs)
    assert torch.allclose(init_entailed, torch.zeros_like(init_entailed), atol=1e-6)
    assert torch.isinf(init_contra).all() and (init_contra > 0).all()

    # Train 3 epochs with validation checkpointing
    epochs_seen = []
    model.train(
        epochs=3,
        validate_every=1,
        epoch_callback=lambda ep, m: epochs_seen.append(ep),
    )
    assert epochs_seen == [0, 1, 2]
    assert ckpt.exists()

    # Check after 3 epochs
    post_entailed = model.module.class_assertion_score(entailed_pairs)
    post_contra = model.module.class_assertion_score(contradicted_pairs)
    assert torch.allclose(post_entailed, torch.zeros_like(post_entailed), atol=1e-6)
    assert torch.isinf(post_contra).all() and (post_contra > 0).all()

    # Check structure cache file was written next to the dataset ontology_path
    cache_files = list(local_scratch_dir.glob("*membership*.pt"))
    assert len(cache_files) == 1

    # Re-instantiating CanonicalMixture loads the cached MembershipStructure
    model2 = CanonicalMixture(
        ds,
        rank=8,
        lr=0.01,
        batch_size=2,
        model_filepath=str(local_scratch_dir / "cm_best2.pt"),
    )
    assert model2.structure.element_names == model.structure.element_names
