import mowl
import pytest
from mowl.datasets.builtin.jackermeier2024 import GALENJackermeier2024Dataset, \
    GOJackermeier2024Dataset, AnatomyJackermeier2024Dataset
from unittest import TestCase


class TestJackermeier2024Datasets(TestCase):

    def test_instance_of_dataset(self):
        from mowl.datasets import RemoteDataset
        self.assertTrue(issubclass(GALENJackermeier2024Dataset, RemoteDataset))
        self.assertTrue(issubclass(GOJackermeier2024Dataset, RemoteDataset))
        self.assertTrue(issubclass(AnatomyJackermeier2024Dataset, RemoteDataset))

    def test_urls(self):
        from mowl.datasets.builtin.jackermeier2024 import DATA_GALEN_URL, DATA_GO_URL, \
            DATA_ANATOMY_URL
        for url in (DATA_GALEN_URL, DATA_GO_URL, DATA_ANATOMY_URL):
            self.assertIsInstance(url, str)
            self.assertTrue(url.startswith("https://"))
            self.assertTrue(url.endswith(".tar.gz"))


# Expected normal form counts per split, taken from the reference benchmark
# (KRR-Oxford/BoxSquaredEL, data/GALEN|GO|ANATOMY/prediction).
# Order of the split tuples: (gci0, gci1, gci2, gci3, gci0_bot)
EXPECTED_COUNTS = {
    GALENJackermeier2024Dataset: {
        "classes": 23144,
        "object_properties": 950,
        "train": (22299, 10876, 22494, 10877, 0),
        "val": (2759, 1329, 2803, 1337, 0),
        "test": (2756, 1346, 2804, 1341, 0),
        "disjoint": 0,
        "role_inclusion": 958,
        "role_chain": 58,
    },
    GOJackermeier2024Dataset: {
        "classes": 45897,
        "object_properties": 8,
        "train": (68376, 9704, 16259, 9703, 0),
        "val": (7604, 1177, 1968, 1176, 0),
        "test": (7601, 1178, 1987, 1181, 0),
        "disjoint": 30,
        "role_inclusion": 3,
        "role_chain": 6,
    },
    AnatomyJackermeier2024Dataset: {
        "classes": 106363,
        "object_properties": 187,
        "train": (97616, 1696, 121831, 1714, 1),
        "val": (9423, 210, 14807, 212, 1),
        "test": (9439, 211, 14833, 213, 0),
        "disjoint": 184,
        "role_inclusion": 89,
        "role_chain": 31,
    },
}

SPLIT_GCIS = ("gci0", "gci1", "gci2", "gci3", "gci0_bot")


def counts_of(el_datasets, gci_names):
    return tuple(len(el_datasets[gci_name]) for gci_name in gci_names)


@pytest.mark.parametrize("dataset_class, expected", list(EXPECTED_COUNTS.items()),
                         ids=lambda value: getattr(value, "__name__", ""))
@pytest.mark.slow
def test_jackermeier2024_counts(dataset_class, expected):
    """The normal forms of each split must round-trip through ELDataset with exactly the
    counts of the reference benchmark (the release as published, ``repaired=False``; the
    default separates the normalisation helpers, which adds one class per helper)."""
    from mowl.datasets import ELDataset
    dataset = dataset_class(repaired=False)

    assert len(dataset.classes.as_dict) == expected["classes"]
    assert len(dataset.object_properties.as_dict) == expected["object_properties"]

    splits = [("train", dataset.ontology),
              ("val", dataset.validation),
              ("test", dataset.testing)]

    for split_name, ontology in splits:
        el_datasets = ELDataset(ontology, extended=True, load_normalized=False,
                                load_role_axioms=True).get_gci_datasets()

        counts = counts_of(el_datasets, SPLIT_GCIS)
        assert counts == expected[split_name], f"{split_name}: {counts}"

        if split_name == "train":
            # The disjointness axioms are in the gci1_bot normal form, and the role axioms
            # are only present in the training ontology.
            assert len(el_datasets["gci1_bot"]) == expected["disjoint"]
            assert len(el_datasets["role_inclusion"]) == expected["role_inclusion"]
            assert len(el_datasets["role_chain"]) == expected["role_chain"]
        else:
            # Asked for with load_role_axioms=True, but these splits hold no role axioms.
            assert "role_inclusion" not in el_datasets
            assert "role_chain" not in el_datasets


@pytest.mark.slow
def test_normalized_ontologies_can_skip_normalization():
    """These ontologies are shipped already in normal form, so ``load_normalized=True``
    (which reads the axioms as they are instead of running jcel) must yield exactly the
    same normal forms."""
    from mowl.datasets import ELDataset
    dataset = GALENJackermeier2024Dataset()

    normalized = ELDataset(dataset.ontology, extended=True, load_normalized=True,
                           load_role_axioms=True).get_gci_datasets()
    from_scratch = ELDataset(dataset.ontology, extended=True, load_normalized=False,
                             load_role_axioms=True).get_gci_datasets()

    assert sorted(normalized) == sorted(from_scratch)
    for gci_name in from_scratch:
        assert len(normalized[gci_name]) == len(from_scratch[gci_name]), gci_name


# --- issue 178: normalisation helpers share IRIs with real classes -----------------------

def _ontology(lines):
    import os
    import tempfile
    import java.io
    from org.semanticweb.owlapi.apibinding import OWLManager
    directory = tempfile.mkdtemp(prefix="mowl_helpers_", dir=tempfile.gettempdir())
    path = os.path.join(directory, "o.ofn")
    with open(path, "w") as f:
        f.write("Prefix(:=<http://ex.org/>)\nPrefix(owl:=<http://www.w3.org/2002/07/owl#>)\n"
                "Ontology(<http://ex.org/o>\n" + "\n".join(lines) + "\n)\n")
    return OWLManager.createOWLOntologyManager().loadOntologyFromOntologyDocument(
        java.io.File(path))


def _unsatisfiable(ontology):
    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    reasoner = ElkReasonerFactory().createReasoner(ontology)
    try:
        return {str(c.toStringID()) for c in
                reasoner.getUnsatisfiableClasses().getEntitiesMinusBottom()}
    finally:
        reasoner.dispose()


def test_helpers_are_separated_from_the_classes_whose_names_they_carry():
    from mowl.datasets.builtin.jackermeier2024_reference import (
        HELPER_PREFIX, find_normalisation_helpers, separate_normalisation_helpers)
    # D ≡ B ⊓ ∃r.C was normalised with a helper that got the name of the real class X.
    # X is also a real class: X ⊑ Y, and Y is disjoint from B. Read literally, the two
    # helper axioms then say ∃r.C ⊑ X ⊑ Y, which makes E (⊑ B and ⊑ ∃r.C) unsatisfiable.
    train = _ontology([
        "SubClassOf(ObjectSomeValuesFrom(:r :C) :X)",
        "SubClassOf(:X :Y)",
        "SubClassOf(ObjectIntersectionOf(:B :Y) owl:Nothing)",
        "SubClassOf(:E :B)",
        "SubClassOf(:E ObjectSomeValuesFrom(:r :C))",
        # a class used twice on the right of ∃r.C ⊑ . is a real class, not a helper
        "SubClassOf(ObjectSomeValuesFrom(:r :C) :Real)",
        "SubClassOf(ObjectSomeValuesFrom(:s :C) :Real)",
        "SubClassOf(ObjectIntersectionOf(:B :Real) :F)",
    ])
    test = _ontology(["SubClassOf(ObjectIntersectionOf(:B :X) :D)"])   # other split
    assert "http://ex.org/E" in _unsatisfiable(train)

    helpers, _ = find_normalisation_helpers([train, None, test])
    assert helpers == {"http://ex.org/X"}
    before = (train.getAxiomCount(), test.getAxiomCount())
    assert separate_normalisation_helpers([train, None, test]) == helpers
    assert (train.getAxiomCount(), test.getAxiomCount()) == before
    assert not _unsatisfiable(train)

    def classes(ontology):
        return {str(c.toStringID()) for c in ontology.getClassesInSignature()}
    helper = HELPER_PREFIX + "http://ex.org/X"
    assert helper in classes(train) and helper in classes(test)
    assert "http://ex.org/X" in classes(train)          # the real class keeps X ⊑ Y
    assert "http://ex.org/X" not in classes(test)
    assert HELPER_PREFIX + "http://ex.org/Real" not in classes(train)
    # Running it again changes nothing.
    assert separate_normalisation_helpers([train, None, test]) == set()


@pytest.mark.slow
def test_repaired_galen_keeps_the_axioms_and_the_split():
    from mowl.datasets import ELDataset
    from mowl.datasets.builtin.jackermeier2024_reference import HELPER_PREFIX
    release = GALENJackermeier2024Dataset(repaired=False)
    repaired = GALENJackermeier2024Dataset()
    assert repaired.repaired and "repaired" in repaired.ontology_path
    for a, b in ((release.ontology, repaired.ontology),
                 (release.validation, repaired.validation),
                 (release.testing, repaired.testing)):
        assert a.getLogicalAxiomCount() == b.getLogicalAxiomCount()
        x = ELDataset(a, extended=True, load_normalized=True).get_gci_datasets()
        y = ELDataset(b, extended=True, load_normalized=True).get_gci_datasets()
        assert {k: len(v) for k, v in x.items()} == {k: len(v) for k, v in y.items()}
    helpers = [c for c in repaired.classes.as_str if c.startswith(HELPER_PREFIX)]
    assert len(helpers) > 10000
    assert not any(c.startswith(HELPER_PREFIX) for c in release.classes.as_str)


def test_incomplete_axiom_lines_of_the_release_are_dropped(tmp_path):
    """The human PPI training file of the release ends in the middle of an axiom."""
    from mowl.datasets.builtin.jackermeier2024_reference import _write_ontology
    from mowl.datasets.base import PathDataset
    path = str(tmp_path / "ontology.owl")
    lines = ["SubClassOf(<http://a> ObjectSomeValuesFrom(<http://interacts> <http://b>))\n",
             "\n",
             "SubClassOf(<http://b> ObjectSomeValuesFrom(<http://interacts> <h"]
    assert _write_ontology(path, "http://example.org/t", lines) == 1
    ontology = PathDataset(path).ontology
    assert ontology.getAxiomCount() == 1
