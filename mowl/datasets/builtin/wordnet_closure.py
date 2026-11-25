"""The WordNet noun hierarchy with the splits of Ganea et al. (2018)."""

import os
import urllib.request

from mowl.datasets.base import PathDataset, default_data_root

#: The commit of https://github.com/dalab/hyperbolic_cones (Apache-2.0) the files are read from.
COMMIT = "fbf1558e9f455b68f5a0069968341c3ce4b78390"
_BASE = f"https://raw.githubusercontent.com/dalab/hyperbolic_cones/{COMMIT}/data/maxn/"
_PREFIX = "noun_closure.tsv"
IRI_PREFIX = "http://mowl.borg/wordnet/"
PERCENTAGES = (0, 10, 25, 50, 90)


def synset_iri(name):
    """The class IRI of a WordNet synset name such as ``dog.n.01``."""
    from urllib.parse import quote
    return IRI_PREFIX + quote(name, safe="._-")


class WordNetNounClosureDataset(PathDataset):
    """The hypernymy hierarchy of the WordNet nouns (82,114 synsets) as a taxonomy, with
    the training, validation and test splits of [ganea2018]_ (Section 5), which
    [suzuki2019]_ and [yu2023shadow]_ also use.

    The transitive closure of the hierarchy has 743,086 edges. The *basic* edges (those
    of the transitive reduction, 84,363) are always in the training data; of the
    remaining, non-basic edges, 5% are validation positives, 5% test positives, and a
    given percentage (``percent``: 0, 10, 25, 50 or 90) is added to the training data.
    Every validation and test positive comes with 10 fixed negatives (corruptions of one
    side), read from the release and available as :attr:`validation_negatives` and
    :attr:`testing_negatives`.

    An edge of the release is ``parent<TAB>child``; it becomes the axiom
    ``SubClassOf(child parent)``. The files are downloaded from the authors' repository
    (https://github.com/dalab/hyperbolic_cones, Apache-2.0) at a fixed commit and converted
    once.

    :param percent: Percentage of non-basic edges in the training data.
    :param data_root: Where to cache the files (default: :func:`default_data_root`).
    :param reversed_order: Reverse every edge (the "WordNet nouns reversed" setting of
        [suzuki2019]_).
    """

    def __init__(self, percent=0, data_root=None, reversed_order=False):
        if percent not in PERCENTAGES:
            raise ValueError(f"percent must be one of {PERCENTAGES}.")
        self.percent, self.reversed_order = percent, bool(reversed_order)
        root = default_data_root() if data_root is None else data_root
        self.directory = os.path.join(root, "wordnet_noun_closure")
        suffix = f"{percent}{'r' if self.reversed_order else ''}"
        paths = [os.path.join(self.directory, f"{part}_{suffix}.ofn")
                 for part in ("ontology", "valid", "test")]
        if not all(os.path.exists(p) for p in paths):
            self._build(paths)
        self._negatives = {}
        super().__init__(*paths)

    # -- construction -------------------------------------------------------------------
    def _raw(self, name):
        raw = os.path.join(self.directory, "raw")
        os.makedirs(raw, exist_ok=True)
        path = os.path.join(raw, name)
        if not os.path.exists(path):
            partial = f"{path}.{os.getpid()}.part"
            urllib.request.urlretrieve(_BASE + name, partial)
            os.replace(partial, path)
        return path

    def _vocabulary(self):
        names = {}
        with open(self._raw(_PREFIX + ".vocab"), encoding="utf-8") as f:
            for line in f:
                index, name = line.rstrip("\n").split("\t")
                names[index] = synset_iri(name)
        return names

    def _edges(self, name, names):
        """``(child IRI, parent IRI)`` pairs of a release file."""
        out = []
        with open(self._raw(name), encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                parent, child = names[parts[0]], names[parts[1]]
                out.append((parent, child) if self.reversed_order else (child, parent))
        return out

    def _build(self, paths):
        names = self._vocabulary()
        files = (f"{_PREFIX}.train_{self.percent}percent", f"{_PREFIX}.valid",
                 f"{_PREFIX}.test")
        for path, name, declare in zip(paths, files, (True, False, False)):
            edges = self._edges(name, names)
            partial = f"{path}.{os.getpid()}.part"
            with open(partial, "w", encoding="utf-8") as out:
                out.write("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n")
                out.write(f"Ontology(<{IRI_PREFIX}{os.path.basename(path)}>\n")
                if declare:
                    for iri in names.values():
                        out.write(f"Declaration(Class(<{iri}>))\n")
                for child, parent in edges:
                    out.write(f"SubClassOf(<{child}> <{parent}>)\n")
                out.write(")\n")
            os.replace(partial, path)

    # -- the fixed negatives ------------------------------------------------------------
    def _negative_pairs(self, split):
        if split not in self._negatives:
            self._negatives[split] = self._edges(f"{_PREFIX}.{split}_neg", self._vocabulary())
        return self._negatives[split]

    @property
    def validation_negatives(self):
        """The release's validation negatives, as ``(subclass IRI, superclass IRI)`` pairs."""
        return self._negative_pairs("valid")

    @property
    def testing_negatives(self):
        """The release's test negatives, as ``(subclass IRI, superclass IRI)`` pairs."""
        return self._negative_pairs("test")
