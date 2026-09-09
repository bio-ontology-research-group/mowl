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
    counts of the reference benchmark."""
    from mowl.datasets import ELDataset
    dataset = dataset_class()

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
