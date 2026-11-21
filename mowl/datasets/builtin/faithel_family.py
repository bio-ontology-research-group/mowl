"""The family knowledge base of the FaithEL code, with a random split of its ABox."""

import os
import urllib.request

import numpy as np

from mowl.datasets.base import PathDataset, default_data_root
from mowl.datasets.builtin._abox import entailed_assertions, tbox_axioms, write_ontology

#: The commit of https://github.com/victorlacerdab/FaithEL_method the ontology is read from.
COMMIT = "0f642bdf9eb50cdb90399620806101db516e3b62"
URL = ("https://raw.githubusercontent.com/victorlacerdab/FaithEL_method/"
       f"{COMMIT}/family_ontology.owl")
IRI_PREFIX = "http://mowl.borg/faithel/"


class FaithELFamilyDataset(PathDataset):
    """The family knowledge base that the code of FaithEL (Lacerda, Ozaki, Guimarães,
    RuleML+RR 2024) trains and evaluates on, with the ABox split at random.

    The release is one OWL file (``family_ontology.owl``: 10 classes, 7 object properties,
    469 individuals taken from Wikidata, 9 subclass axioms, 1,110 object property
    assertions and no class assertions). The authors' repository
    (https://github.com/victorlacerdab/FaithEL_method) states **no licence**; the file is
    downloaded at run time from the pinned commit and not redistributed with mOWL. The
    Wikidata facts it derives from are CC0.

    As in the code, the class assertions are those that *follow from the knowledge base*
    (the code reads the indirect types of every individual after classifying with
    HermiT through owlready2; here HermiT through the OWL API gives the named classes of
    the entailed types) and the role assertions are the entailed fillers of every object
    property. They are split 70% / 30% separately, at random with a fixed seed, into the
    assertions of the training ontology and of the test ontology; the TBox (everything that
    is not a declaration or an assertion) stays in the training ontology and the individuals
    and names are declared in both. The code splits with PyTorch's ``random_split``;
    the split here is NumPy's and differs from it.

    :param split_seed: Seed of the split.
    :param train_fraction: Fraction of the assertions that is training data (code: 0.7).
    :param data_root: Where to cache the files (default: :func:`default_data_root`).
    :param url: Override the download location (for example a local mirror).
    """

    def __init__(self, split_seed=0, train_fraction=0.7, data_root=None, url=None):
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1).")
        self.split_seed, self.train_fraction = int(split_seed), float(train_fraction)
        self.url = url or URL
        root = default_data_root() if data_root is None else data_root
        self.directory = os.path.join(root, "faithel_family")
        tag = f"s{self.split_seed}_f{self.train_fraction:g}"
        paths = [os.path.join(self.directory, f"{part}_{tag}.ofn")
                 for part in ("ontology", "test")]
        if not all(os.path.exists(p) for p in paths):
            self._build(paths)
        super().__init__(paths[0], None, paths[1])

    def _raw(self):
        raw = os.path.join(self.directory, "raw")
        os.makedirs(raw, exist_ok=True)
        path = os.path.join(raw, "family_ontology.owl")
        if not os.path.exists(path):
            partial = f"{path}.{os.getpid()}.part"
            urllib.request.urlretrieve(self.url, partial)
            os.replace(partial, path)
        return path

    def _build(self, paths):
        from mowl.owlapi import OWLAPIAdapter
        adapter = OWLAPIAdapter()
        source = PathDataset(self._raw()).ontology
        classes = sorted(str(c.getIRI()) for c in source.getClassesInSignature()
                         if not c.isOWLThing() and not c.isOWLNothing())
        roles = sorted(str(r.getIRI()) for r in source.getObjectPropertiesInSignature())
        individuals = sorted(str(i.getIRI()) for i in source.getIndividualsInSignature())
        concept_facts, role_facts = entailed_assertions(source)
        rng = np.random.default_rng(self.split_seed)

        def split(facts):
            order = rng.permutation(len(facts))
            cut = int(self.train_fraction * len(facts))
            return ([facts[i] for i in sorted(order[:cut])],
                    [facts[i] for i in sorted(order[cut:])])

        train_c, test_c = split(concept_facts)
        train_r, test_r = split(role_facts)
        names = {"classes": classes, "roles": roles, "individuals": individuals}
        write_ontology(adapter, f"{IRI_PREFIX}train/{os.path.basename(paths[0])}",
                       tbox_axioms(source), class_assertions=train_c, role_assertions=train_r,
                       path=paths[0], **names)
        write_ontology(adapter, f"{IRI_PREFIX}test/{os.path.basename(paths[1])}", [],
                       class_assertions=test_c, role_assertions=test_r, path=paths[1],
                       **names)
