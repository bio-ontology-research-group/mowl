"""The ORE1 class-membership benchmark used by CatE (Zhapa-Camacho and Hoehndorf,
"Lattice-based ALC ontology embeddings with saturation", Section 4.2, Table 2), read
from the authors' data release on Zenodo (record 13766937, ``use_cases.tar.gz``).

ORE1 comes from the SemREC challenge. The release splits its class assertions into
61,245 training, 7,578 validation and 15,157 test axioms (Table 1).
"""

import hashlib
import os
import tarfile

from mowl.datasets.base import PathDataset, RemoteDataset, default_data_root

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
