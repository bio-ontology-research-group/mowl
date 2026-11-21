"""Conversion of the OnT data release (mowl.datasets.builtin.yang2025ont)."""

import hashlib
import json
import os
import zipfile

import pytest

import mowl.datasets  # noqa: F401
from mowl.datasets.builtin import yang2025ont as Y

CONCEPTS = {"0": "dog", "1": "animal", "2": "person", "3": "pet owner", "4": "wheel",
            "5": "owns some dog", "6": "car", "7": "dog"}   # 7 duplicates the name of 0
ROLES = {"0": "owns", "1": "has part"}


def _fake_release(path, folder="GALEN_OnT"):
    train = [{"child": "dog", "parent": "animal", "negative": ["car"]},
             {"child": "person and owns some dog", "parent": "pet owner",
              "negative": ["wheel"]},
             {"child": "car", "parent": "has part some wheel", "negative": ["has part some dog"]},
             {"child": "owns some dog", "parent": "pet owner", "negative": ["car"]},
             {"child": "pet owner", "parent": "owns some dog", "negative": ["car"]},
             {"child": "dog", "parent": "animal", "negative": ["wheel"]}]  # duplicate line
    conj = [{"Concept": "person and owns some dog", "con1": "person", "con2": "owns some dog"}]
    exist = [{"Concept": "has part some wheel", "role": "has part", "con": "wheel"},
             {"Concept": "owns some dog", "role": "owns", "con": "dog"}]
    test = {"query_sentences": {
        "nf1": [{"name": "dog"}],
        "nf2": [{"name": "person and animal", "con1": "person", "con2": "animal"}],
        "nf3": [{"name": "has part some wheel", "role": "has part", "con": "wheel"}],
        "nf4": [{"name": "owns some animal", "role": "owns", "con": "animal"}]},
        "answer_ids": {"nf1": [1], "nf2": [3], "nf3": [6], "nf4": [3]}}
    val = {"query_sentences": {"nf1": [{"name": "car"}]}, "answer_ids": {"nf1": [1]}}
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{folder}/concept_names.json", json.dumps(CONCEPTS))
        zf.writestr(f"{folder}/role_names.json", json.dumps(ROLES))
        zf.writestr(f"{folder}/train.jsonl", "\n".join(json.dumps(x) for x in train))
        zf.writestr(f"{folder}/train_conj.jsonl", "\n".join(json.dumps(x) for x in conj))
        zf.writestr(f"{folder}/train_exist.jsonl", "\n".join(json.dumps(x) for x in exist))
        zf.writestr(f"{folder}/test.json", json.dumps(test))
        zf.writestr(f"{folder}/val.json", json.dumps(val))
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_release_reader(tmp_path):
    path = str(tmp_path / "fake.zip")
    _fake_release(path)
    with zipfile.ZipFile(path) as zf:
        rel = Y._Release(zf, "GALEN_OnT")
        assert rel.concept_id["dog"] == 0      # duplicate names map to the smallest id
        train = rel.training_axioms()
        assert train == [("nf1", 0, 1), ("nf2", 2, 5, 3), ("nf3", 6, 1, 4), ("nf4", 0, 0, 3),
                         ("nf3", 3, 0, 0), ("nf1", 0, 1)]
        assert rel.evaluation_axioms("test") == [("nf1", 0, 1), ("nf2", 2, 1, 3),
                                                 ("nf3", 6, 1, 4), ("nf4", 0, 1, 3)]


def test_checksum_is_enforced(tmp_path):
    path = str(tmp_path / "fake.zip")
    _fake_release(path)
    with pytest.raises(ValueError, match="SHA-256"):
        Y.fetch_ont_release("GALEN_predict", zip_path=path)
    with pytest.raises(ValueError):
        Y.fetch_ont_release("NOPE_predict", zip_path=path)


def test_dataset_conversion(tmp_path, monkeypatch):
    from mowl.text.verbalise import Verbaliser
    from org.semanticweb.owlapi.model import AxiomType
    path = str(tmp_path / "fake.zip")
    digest = _fake_release(path)
    monkeypatch.setitem(Y.ONT_DATA_SHA256, "GALEN_predict", digest)
    ds = Y.Yang2025OnTDataset("GALEN", "prediction", data_root=str(tmp_path / "root"),
                              zip_path=path)
    assert os.path.exists(os.path.join(ds.dataset_dir, "ontology.owl"))
    assert len(ds.candidate_classes) == len(CONCEPTS)
    assert ds.candidate_classes[3] == ds.base_iri + "C3"
    subs = list(ds.ontology.getAxioms(AxiomType.SUBCLASS_OF))
    assert len(subs) == 5                                  # the duplicate line collapses
    v = Verbaliser(ds.ontology, style="release")
    texts = sorted(v.axiom_sides(ax)[0] for ax in subs)
    assert ("car", "has part some wheel") in texts
    assert ("dog", "animal") in texts
    assert ("owns some dog", "pet owner") in texts
    assert ("pet owner", "owns some dog") in texts
    assert any(t in texts for t in (("person and owns some dog", "pet owner"),
                                    ("owns some dog and person", "pet owner")))
    assert ds.testing.getAxiomCount() == 4
    assert ds.validation.getAxiomCount() == 1
    # every concept is declared and labelled, including those used only in tests
    assert v.label(ds.base_iri + "C7") == "dog"
    with pytest.raises(ValueError):
        Y.Yang2025OnTDataset("FOO")


@pytest.mark.slow
def test_galen_release():
    """Downloads the GALEN prediction release (13 MB) and checks the counts of Table 1
    that the release reproduces."""
    from mowl.datasets.el import ELDataset
    ds = Y.Yang2025OnTDataset("GALEN", "prediction")
    assert len(ds.candidate_classes) == 35501
    classes = {c: i for i, c in enumerate(ds.classes.as_str)}
    roles = {r: i for i, r in enumerate(ds.object_properties.as_str)}
    ds_el = ELDataset(ds.testing, classes, roles, extended=True)
    counts = {k: len(v) for k, v in ds_el.get_gci_datasets().items()}
    # Table 1 test column: 3,203 / 1,462 / 3,165 / 788; four NF2 test lines of the release
    # are duplicates and collapse in OWL.
    assert counts["gci0"] == 3203 and counts["gci2"] == 3165 and counts["gci3"] == 788
    assert counts["gci1"] == 1458
