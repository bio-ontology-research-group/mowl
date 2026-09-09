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
# Order: (gci0, gci1, gci2, gci3, gci0_bot)
EXPECTED_COUNTS = {
    "GALENJackermeier2024Dataset": {
        "classes": 23142,
        "train": (22299, 10876, 22494, 10877, 0),
        "val": (2759, 1329, 2803, 1337, 0),
        "test": (2756, 1346, 2804, 1341, 0),
        "disjoint": 0,
        "role_inclusion": 958,
        "role_chain": 58,
    },
    "GOJackermeier2024Dataset": {
        "classes": 45895,
        "train": (68376, 9704, 16259, 9703, 0),
        "val": (7604, 1177, 1968, 1176, 0),
        "test": (7601, 1178, 1987, 1181, 0),
        "disjoint": 30,
        "role_inclusion": 3,
        "role_chain": 6,
    },
    "AnatomyJackermeier2024Dataset": {
        "classes": 106363,
        "train": (97616, 1696, 121831, 1714, 1),
        "val": (9423, 210, 14807, 212, 1),
        "test": (9439, 211, 14833, 213, 0),
        "disjoint": 184,
        "role_inclusion": 89,
        "role_chain": 31,
    },
}


@pytest.mark.parametrize("dataset_name, expected", list(EXPECTED_COUNTS.items()))
@pytest.mark.slow
def test_jackermeier2024_counts(dataset_name, expected):
    from mowl.datasets import ELDataset
    dataset_class = {
        "GALENJackermeier2024Dataset": GALENJackermeier2024Dataset,
        "GOJackermeier2024Dataset": GOJackermeier2024Dataset,
        "AnatomyJackermeier2024Dataset": AnatomyJackermeier2024Dataset,
    }[dataset_name]
    dataset = dataset_class()

    # The ontology signature contains all the benchmark classes (including owl:Thing)
    self_classes = len(dataset.classes.as_dict)
    assert self_classes >= expected["classes"]

    for split_name, ontology in [("train", dataset.ontology),
                                 ("val", dataset.validation),
                                 ("test", dataset.testing)]:
        el_dataset = ELDataset(ontology, extended=True, load_normalized=False)
        datasets = el_dataset.get_gci_datasets()
        counts = (len(datasets["gci0"]), len(datasets["gci1"]),
                  len(datasets["gci2"]), len(datasets["gci3"]),
                  len(datasets["gci0_bot"]))
        assert counts == expected[split_name], f"{dataset_name} {split_name}: {counts}"

    # The disjoint axioms are stored in the gci1_bot normal form (training only)
    train = ELDataset(dataset.ontology, extended=True, load_normalized=False)
    assert len(train.get_gci_datasets()["gci1_bot"]) == expected["disjoint"]
    if expected["role_inclusion"] > 0:
        assert len(train.get_gci_datasets()["role_inclusion"]) == expected["role_inclusion"]
    if expected["role_chain"] > 0:
        assert len(train.get_gci_datasets()["role_chain"]) == expected["role_chain"]
