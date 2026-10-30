"""Tests for CanonicalMixture model in mowl.models.canonical_mixture."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import mowl
mowl.init_jvm("2g")

from org.semanticweb.owlapi.model import IRI

import mowl.geometric.membership as membership_mod
from mowl.datasets import PathDataset
from mowl.geometric.normal_forms import BOTTOM, TOP
from mowl.models import CanonicalMixture
from mowl.owlapi import OWLAPIAdapter
from tests.datasetFactory import FamilyDataset

NS = "http://mowl.borg/test/cm#"
IND = "http://mowl.borg/test/cm/ind#"


def _write_family_abox_dataset(
    tmp_path: Path, include_unseen_val_ind: bool = False
) -> PathDataset:
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
    # Father <= Parent, Father <= Male, Mother <= Parent, Mother <= Female
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
    # hasChild(bob, carl), hasChild(alice, diana), hasChild(carl, diana)
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(father, bob))
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(mother, alice))
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(male, carl))
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(person, diana))
    mgr.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(has_child, bob, carl))
    mgr.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(has_child, alice, diana))
    mgr.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(has_child, carl, diana))

    train_path = tmp_path / "family_abox_train.owl"
    mgr.saveOntology(ont, IRI.create(train_path.as_uri()))

    val_ont = adapter.create_ontology("http://mowl.borg/test/cm/valid")
    mgr.addAxiom(val_ont, df.getOWLClassAssertionAxiom(female, diana))
    if include_unseen_val_ind:
        eve = adapter.create_individual(IND + "eve")
        mgr.addAxiom(val_ont, df.getOWLClassAssertionAxiom(person, eve))
    valid_path = tmp_path / "family_abox_valid.owl"
    mgr.saveOntology(val_ont, IRI.create(valid_path.as_uri()))

    return PathDataset(str(train_path), validation_path=str(valid_path))


def test_tbox_only_dataset_raises_value_error(family_dataset: FamilyDataset) -> None:
    """CanonicalMixture raises ValueError('CanonicalMixture needs individuals') on TBox-only FamilyDataset."""
    with pytest.raises(ValueError, match="CanonicalMixture needs individuals"):
        CanonicalMixture(family_dataset, rank=8)


def test_family_abox_dataset_trains_and_preserves_faithfulness(
    tmp_path: Path,
) -> None:
    """Family-with-individuals dataset trains for 3 epochs; entailed scores == 0.0 and contradicted == +inf before and after."""
    ds = _write_family_abox_dataset(tmp_path)
    ckpt = tmp_path / "cm_best.pt"
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
    carl_idx = model.individual_index_dict[IND + "carl"]
    father_idx = model.class_index_dict[NS + "Father"]
    parent_idx = model.class_index_dict[NS + "Parent"]
    person_idx = model.class_index_dict[NS + "Person"]
    female_idx = model.class_index_dict[NS + "Female"]
    male_idx = model.class_index_dict[NS + "Male"]

    # Includes Parent(carl), which is entailed only through hasChild(carl, diana)
    entailed_pairs = torch.tensor(
        [
            [bob_idx, father_idx],
            [bob_idx, parent_idx],
            [bob_idx, person_idx],
            [alice_idx, parent_idx],
            [carl_idx, parent_idx],
        ],
        dtype=torch.long,
    )
    contradicted_pairs = torch.tensor(
        [
            [bob_idx, female_idx],
            [alice_idx, male_idx],
            [carl_idx, female_idx],
        ],
        dtype=torch.long,
    )

    # Check at init (exact 0.0, no atol)
    init_entailed = model.module.class_assertion_score(entailed_pairs)
    init_contra = model.module.class_assertion_score(contradicted_pairs)
    assert torch.all(init_entailed == 0.0)
    assert torch.isinf(init_contra).all() and (init_contra > 0).all()

    # Train 3 epochs with validation checkpointing; on final epoch perturb U so validation
    # loss worsens, and verify train() reloads the best checkpoint at the end.
    epochs_seen = []

    def _cb(ep: int, m: CanonicalMixture) -> None:
        epochs_seen.append(ep)
        if ep == 2:
            with torch.no_grad():
                m.module.b.fill_(100.0)

    model.train(
        epochs=3,
        validate_every=1,
        epoch_callback=_cb,
    )
    assert epochs_seen == [0, 1, 2]
    assert ckpt.exists()
    saved_state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert torch.equal(model.module.b.cpu(), saved_state["b"].cpu())

    # Check after 3 epochs (exact 0.0, no atol)
    post_entailed = model.module.class_assertion_score(entailed_pairs)
    post_contra = model.module.class_assertion_score(contradicted_pairs)
    assert torch.all(post_entailed == 0.0)
    assert torch.isinf(post_contra).all() and (post_contra > 0).all()

    # Check structure cache file was written next to the dataset ontology_path
    cache_files = list(tmp_path.glob("*membership*.pt"))
    assert len(cache_files) == 1

    # Re-instantiating CanonicalMixture loads the cached MembershipStructure
    model2 = CanonicalMixture(
        ds,
        rank=8,
        lr=0.01,
        batch_size=2,
        model_filepath=str(tmp_path / "cm_best2.pt"),
    )
    assert model2.structure.element_names == model.structure.element_names


def test_validation_only_individual_gets_full_support_and_cache_hashes_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Individuals occurring only in validation get support = all of S; cache key hashes completion.py and membership.py."""
    ds = _write_family_abox_dataset(tmp_path, include_unseen_val_ind=True)
    ckpt = tmp_path / "cm_unseen.pt"
    model = CanonicalMixture(
        ds,
        rank=8,
        lr=0.01,
        batch_size=2,
        model_filepath=str(ckpt),
    )

    eve_idx = model.individual_index_dict[IND + "eve"]
    assert int(model.structure.individual_element[eve_idx].item()) == -1
    assert int(model.module.support_lengths[eve_idx].item()) == model.module.n_elements

    eve_scores = model.module.rank_classes(torch.tensor([eve_idx], dtype=torch.long))
    person_idx = model.class_index_dict[NS + "Person"]
    assert torch.isfinite(eve_scores[0, person_idx])

    # Verify _cache_path changes if membership.py source changes
    orig_cache_path = model._cache_path()
    fake_src = tmp_path / "fake_membership.py"
    fake_src.write_bytes(b"# modified membership source\n")
    monkeypatch.setattr(membership_mod, "__file__", str(fake_src))
    new_cache_path = model._cache_path()
    assert orig_cache_path != new_cache_path


@pytest.mark.slow
def test_real_helis_one_epoch_cuda_peak_memory(tmp_path: Path) -> None:
    """One epoch on real helis (/home/leechuck/tmp/gfe-data/helis/ontology.owl) at batch_size=1024 uses < 3 GB CUDA memory."""
    helis_owl = Path("/home/leechuck/tmp/gfe-data/helis/ontology.owl")
    if not helis_owl.exists():
        pytest.skip(f"Missing real helis ontology at {helis_owl}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real helis CUDA peak memory test")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    ds = PathDataset(str(helis_owl))
    ckpt = tmp_path / "helis_cm.pt"
    model = CanonicalMixture(
        ds,
        rank=50,
        lr=0.001,
        batch_size=1024,
        model_filepath=str(ckpt),
        device="cuda",
    )
    model.train(epochs=1, validate_every=0)
    peak_bytes = int(torch.cuda.max_memory_allocated())
    assert peak_bytes < int(3.0 * 1e9), (
        f"helis 1-epoch CUDA max_memory_allocated {peak_bytes / 1e9:.3f} GB >= 3.0 GB"
    )

