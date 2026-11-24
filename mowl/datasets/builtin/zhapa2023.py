"""The benchmark data of CatE (Zhapa-Camacho and Hoehndorf, "Lattice-based ALC
ontology embeddings with saturation"), read from the authors' data release on Zenodo
(record 13766937, ``use_cases.tar.gz``).

* :class:`CatEORE1Dataset`: the ORE1 class-membership benchmark (Section 4.2, Table
  2). ORE1 comes from the SemREC challenge; the release splits its class assertions
  into 61,245 training, 7,578 validation and 15,157 test axioms (Table 1).
* :class:`CatESubsumptionDataset` (GO and FoodOn): the TBox completion benchmarks
  (Section 4.3, Table 3) on mOWL's go_subsumption / foodon_subsumption data (whose
  training ontologies are byte-identical to the release's) with the release's
  "cleaned" query lists: the published numbers rank the pairs of ``test_cleaned.csv``
  (11,893 of the 14,521 GO and 3,632 of the 5,957 FoodOn test pairs), selected by a
  script that is not part of the code release, so the lists are used as released.
"""

import hashlib
import os
import tarfile

from mowl.datasets.base import PathDataset, RemoteDataset, default_data_root
from mowl.datasets.builtin.subsumption import (FOODON_DATA_URL, GO_DATA_URL,
                                               SubsumptionDataset)

CATE_DATA_URL = "https://zenodo.org/api/records/13766937/files/use_cases.tar.gz/content"
#: MD5 published by Zenodo for ``use_cases.tar.gz``.
CATE_DATA_MD5 = "6b855827b7314f59a994d8488cc4d785"

_ORE1_MEMBERS = {
    "ontology.owl": "use_cases/ore1/data/_train_ORE1_wrapped.owl",
    "valid.owl": "use_cases/ore1/data/_valid_ORE1_wrapped.owl",
    "test.owl": "use_cases/ore1/data/_test_ORE1_wrapped.owl",
}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CatEORE1Dataset(PathDataset):
    """ORE1 membership prediction (axioms :math:`C(a)`) from the CatE data release.

    The archive (about 100 MB) is downloaded once, its MD5 is checked against the value
    published by Zenodo, and only the three ORE1 ontologies are extracted. The release's
    ontology headers carry the malformed IRI ``<owlapi:ontology:...">``; it is replaced
    by a well-formed one when extracting (no axiom is changed).

    :param data_root: Cache directory.
    :param tarball_path: Local copy of ``use_cases.tar.gz`` (checksum-verified).
    """

    def __init__(self, data_root=None, tarball_path=None):
        root = default_data_root() if data_root is None else data_root
        target = os.path.join(root, "cate_ore1")
        paths = {name: os.path.join(target, name) for name in _ORE1_MEMBERS}
        if not all(os.path.exists(p) for p in paths.values()):
            os.makedirs(target, exist_ok=True)
            tarball_path = tarball_path or self._download(root)
            digest = _md5(tarball_path)
            if digest != CATE_DATA_MD5:
                raise ValueError(f"{tarball_path} has MD5 {digest}, expected {CATE_DATA_MD5}.")
            with tarfile.open(tarball_path) as tf:
                for name, member in _ORE1_MEMBERS.items():
                    data = tf.extractfile(member).read().decode("utf-8")
                    split = name.split(".")[0]
                    data = data.replace(f'<owlapi:ontology:{split}">',
                                        f"<http://mowl.borg/benchmarks/cate/ore1/{split}>")
                    data = data.replace('<owlapi:ontology:train">',
                                        "<http://mowl.borg/benchmarks/cate/ore1/ontology>")
                    tmp = paths[name] + ".part"
                    with open(tmp, "w") as f:
                        f.write(data)
                    os.replace(tmp, paths[name])
        super().__init__(paths["ontology.owl"], paths["valid.owl"], paths["test.owl"])

    @staticmethod
    def _download(root):
        path = os.path.join(root, "cate_use_cases.tar.gz")
        if os.path.exists(path):
            return path
        partial = f"{path}.{os.getpid()}.part"
        try:
            with RemoteDataset._session() as session:
                with session.get(CATE_DATA_URL, stream=True, timeout=(10, 60)) as req:
                    req.raise_for_status()
                    with open(partial, "wb") as w:
                        for chunk in req.iter_content(chunk_size=1 << 16):
                            w.write(chunk)
        except BaseException:
            if os.path.exists(partial):
                os.remove(partial)
            raise
        os.replace(partial, path)
        return path

    @property
    def evaluation_classes(self):
        """All classes, as candidate concepts."""
        return self.classes, self.classes


def _tarball_member(tar, name):
    """Text content of ``name`` inside the verified ``use_cases.tar.gz``."""
    with tarfile.open(tar) as tf:
        return tf.extractfile(name).read().decode("utf-8")


def _cleaned_pairs_file(path):
    """Pairs of IRIs of a released ``*_cleaned.csv`` file (two columns, comma)."""
    out = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[0] and parts[1]:
                out.append((parts[0], parts[1]))
    return out


class CatESubsumptionDataset(SubsumptionDataset):
    """A GO or FoodOn TBox completion benchmark of CatE with the released query lists.

    The training, validation and test ontologies are those of the mOWL
    ``go_subsumption`` / ``foodon_subsumption`` datasets (the training ontologies are
    byte-identical to the CatE release's ``go.train.owl`` / ``foodon-merged.train.owl``,
    and the datasets' deductive-closure ontologies contain exactly the pairs of the
    release's ``inferred_ancestors.txt``, which the reference code uses for
    filtering). The queries that the paper ranked are the release's ``test_cleaned.csv``
    and ``valid_cleaned.csv``, exposed as :attr:`cate_test_query_pairs` and
    :attr:`cate_validation_query_pairs`.

    :param url: Download URL of the mOWL subsumption dataset.
    :param members: Mapping from attribute base name to tar member of the cleaned
        query lists (``test`` and ``validation``).
    """

    def __init__(self, url, members, data_root=None, tarball_path=None):
        super().__init__(url, data_root=data_root)
        root = default_data_root() if data_root is None else data_root
        cache = os.path.join(root, "cate_cleaned_pairs")
        # Prefix with the release directory (go, foodon): the cleaned files of the
        # two ontologies share their basenames.
        paths = {k: os.path.join(cache, "_".join(v.split("/")[-3:-1] + [v.split("/")[-1]]))
                 for k, v in members.items()}
        if not all(os.path.exists(v) for v in paths.values()):
            os.makedirs(cache, exist_ok=True)
            tarball_path = tarball_path or CatEORE1Dataset._download(root)
            digest = _md5(tarball_path)
            if digest != CATE_DATA_MD5:
                raise ValueError(f"{tarball_path} has MD5 {digest}, expected "
                                 f"{CATE_DATA_MD5}.")
            for k, v in paths.items():
                with open(v + ".part", "w") as f:
                    f.write(_tarball_member(tarball_path, members[k]))
                os.replace(v + ".part", v)
        self.cate_test_query_pairs = _cleaned_pairs_file(paths["test"])
        self.cate_validation_query_pairs = _cleaned_pairs_file(paths["validation"])


class CatEGOSubsumptionDataset(CatESubsumptionDataset):
    """The GO TBox completion benchmark of CatE (Table 3, left)."""

    _MEMBERS = {
        "test": "use_cases/go/data/test_cleaned.csv",
        "validation": "use_cases/go/data/valid_cleaned.csv",
    }

    def __init__(self, data_root=None, tarball_path=None):
        super().__init__(GO_DATA_URL, self._MEMBERS, data_root=data_root,
                         tarball_path=tarball_path)


class CatEFoodOnSubsumptionDataset(CatESubsumptionDataset):
    """The FoodOn TBox completion benchmark of CatE (Table 3, right)."""

    _MEMBERS = {
        "test": "use_cases/foodon/data/test_cleaned.csv",
        "validation": "use_cases/foodon/data/valid_cleaned.csv",
    }

    def __init__(self, data_root=None, tarball_path=None):
        super().__init__(FOODON_DATA_URL, self._MEMBERS, data_root=data_root,
                         tarball_path=tarball_path)
