"""The Family subsets F_v1 - F_v4 of BoxLitE (Lourenço, Morgan, Ozaki, Pavlović, Sallinger,
KR 2026, Section 6 and Appendix G.2), read from the authors' repository
`AleksVap/BoxLitE <https://github.com/AleksVap/BoxLitE>`_ at commit ``7d53063``.

Each subset was sampled by the authors from the ABox of the Family dataset (Imenes,
Guimarães and Ozaki, RuleML+RR 2023; Wikidata kinship facts) with forest-fire sampling.
The release contains role assertions only, as tab-separated triples ``<subject>
<property> <object>`` with Wikidata IRIs, in ``Family_Dataset/{train,validation,test}``.
The validation and test sets are the role assertions entailed by the training ABox and
the TBox, minus the training ABox, split 20/80 (Appendix G.2).

The files are downloaded once from ``raw.githubusercontent.com`` at the pinned commit,
their SHA-256 checksums are verified, and they are converted to OWL functional syntax
(``ontology.owl`` = TBox + training assertions, ``valid.owl``, ``test.owl``).

The TBox is not part of the data files. Three choices are offered (``tbox=``):

``"reference"`` (default)
    The ten role inclusions hard-coded in the reference loader
    (``Family_Data_Loader.py``), which produced Table 2: spouse, hasSibling, hasChild,
    hasParent ⊑ relative; hasFather, hasMother ⊑ hasParent; relative ⊑ relative⁻;
    hasSibling ⊑ hasSibling⁻; spouse ⊑ spouse⁻; hasChild ⊑ hasParent⁻. No disjointness.
``"paper"``
    Figure 2 of the paper: the nine role inclusions above without hasChild ⊑ hasParent⁻,
    plus ∃hasFather⁻ ⊑ ¬∃hasMother⁻.
``"closure"``
    Figure 2 plus hasChild ⊑ hasParent⁻ and hasParent ⊑ hasChild⁻. Checked in mOWL: for
    each of F_v1 - F_v4, the closure of the training ABox under exactly these role
    inclusions, minus the training ABox, equals validation ∪ test (550, 1046, 2163 and
    6134 assertions); under the reference TBox 21, 44, 72 and 223 of them are not
    entailed, under Figure 2 alone 70, 112, 211 and 553.

Wikidata properties: P40 hasChild, P1038 relative, P26 spouse, P8810 hasParent,
P22 hasFather, P25 hasMother, P3373 hasSibling.
"""

import hashlib
import os

from mowl.datasets.base import PathDataset, RemoteDataset, default_data_root

#: Pinned commit of the authors' repository.
BOXLITE_COMMIT = "7d5306301118e87694de1fc080d873ea182a3094"
_BASE_URL = (f"https://raw.githubusercontent.com/AleksVap/BoxLitE/{BOXLITE_COMMIT}/"
             "Family_Dataset/{split}/F_v{version}.tsv")

#: SHA-256 of every file of the release.
SHA256 = {
    ("train", 1): "904fd9d4a72d8894a7de802c5b2ca9421500ab78bdcbd1622e113f8afdd05a14",
    ("train", 2): "e3ffd41203eabb2790ee4d9483943ef66f73bd05f0aa498944569feef3d205f9",
    ("train", 3): "12201637bcbb39ac026a8bf589e6f2a4dc961a2e4246f3304ca82d1be0af6ad4",
    ("train", 4): "8b5ccce7917f60497ddf83cfb5f8775aea13af4c1eddcb62c7836e2239ac1cf5",
    ("validation", 1): "ca8efdab9bcac2bdfe912d6a080a8a79f80781667d6f1bc718a4f1e8cbdf7ac5",
    ("validation", 2): "0bc34efee5585ed706fa9676a8639b9cfaf6caa8fe752b0e714853f70073a00d",
    ("validation", 3): "1d0e4e0abf9d9a794bc75a9c46705ffa2bc3a30546ca64d3bd543c54226a3d72",
    ("validation", 4): "96cda51a74ba4663d2fe267defbb65684968d1b5d13943911b8661cd6ec91982",
    ("test", 1): "c53c18288a01c7b9688b8e83326eeceb6ac320353f96250ebdf7f4930ef006f2",
    ("test", 2): "de5bc867a57357880e4c7e8d91769c2e48b2ac2712159ce714e3e686a17dc7a6",
    ("test", 3): "c01a37204f0313d9f8aec17cf10df53c425d4e791abea8626a3bcd034407d3fe",
    ("test", 4): "f6b6b162eb1570a2cde144ad310137f8e68ae6f701242adee75800c19153e8ce",
}

PROP = "https://www.wikidata.org/prop/"
#: Role names of the paper, by Wikidata property.
ROLE_NAMES = {"P40": "hasChild", "P1038": "relative", "P26": "spouse", "P8810": "hasParent",
              "P22": "hasFather", "P25": "hasMother", "P3373": "hasSibling"}
_P = {name: f"<{PROP}{pid}>" for pid, name in ROLE_NAMES.items()}

_FIGURE2_ROLES = [
    f"SubObjectPropertyOf({_P['spouse']} {_P['relative']})",
    f"SubObjectPropertyOf({_P['hasSibling']} {_P['relative']})",
    f"SubObjectPropertyOf({_P['hasChild']} {_P['relative']})",
    f"SubObjectPropertyOf({_P['hasParent']} {_P['relative']})",
    f"SubObjectPropertyOf({_P['hasFather']} {_P['hasParent']})",
    f"SubObjectPropertyOf({_P['hasMother']} {_P['hasParent']})",
    f"SubObjectPropertyOf({_P['relative']} ObjectInverseOf({_P['relative']}))",
    f"SubObjectPropertyOf({_P['hasSibling']} ObjectInverseOf({_P['hasSibling']}))",
    f"SubObjectPropertyOf({_P['spouse']} ObjectInverseOf({_P['spouse']}))",
]
_CHILD_PARENT = f"SubObjectPropertyOf({_P['hasChild']} ObjectInverseOf({_P['hasParent']}))"
_PARENT_CHILD = f"SubObjectPropertyOf({_P['hasParent']} ObjectInverseOf({_P['hasChild']}))"
_DISJOINT = (f"SubClassOf(ObjectSomeValuesFrom(ObjectInverseOf({_P['hasFather']}) owl:Thing) "
             f"ObjectComplementOf(ObjectSomeValuesFrom(ObjectInverseOf({_P['hasMother']}) "
             f"owl:Thing)))")

#: TBox axioms (OWL functional syntax) of each ``tbox`` choice.
TBOXES = {
    "reference": _FIGURE2_ROLES + [_CHILD_PARENT],
    "paper": _FIGURE2_ROLES + [_DISJOINT],
    "closure": _FIGURE2_ROLES + [_CHILD_PARENT, _PARENT_CHILD, _DISJOINT],
}

_HEADER = ("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n"
           "Ontology(<{iri}>\n")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_family_split(split, version, data_root=None, source_dir=None):
    """Path of ``<split>/F_v<version>.tsv`` of the pinned release, downloaded if needed.

    :param source_dir: Use the ``Family_Dataset`` directory of a local clone instead.
    :raises ValueError: If the checksum does not match the pinned release.
    """
    if (split, version) not in SHA256:
        raise ValueError(f"No file {split}/F_v{version}.")
    if source_dir is not None:
        path = os.path.join(source_dir, split, f"F_v{version}.tsv")
    else:
        root = os.path.join(default_data_root() if data_root is None else data_root,
                            "boxlite_family_release")
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, f"{split}_F_v{version}.tsv")
        if not os.path.exists(path):
            partial = f"{path}.{os.getpid()}.part"
            url = _BASE_URL.format(split=split, version=version)
            try:
                with RemoteDataset._session() as session:
                    with session.get(url, timeout=(10, 60)) as req:
                        req.raise_for_status()
                        with open(partial, "wb") as w:
                            w.write(req.content)
            except BaseException:
                if os.path.exists(partial):
                    os.remove(partial)
                raise
            os.replace(partial, path)
    digest = _sha256(path)
    if digest != SHA256[(split, version)]:
        raise ValueError(f"{path} has SHA-256 {digest}, expected {SHA256[(split, version)]}.")
    return path


def read_triples(path):
    """``[(subject IRI, property IRI, object IRI)]`` of a release file."""
    triples = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                triples.append(tuple(p.strip("<>") for p in parts))
    return triples


def _write(path, iri, lines):
    tmp = f"{path}.{os.getpid()}.part"
    with open(tmp, "w") as out:
        out.write(_HEADER.format(iri=iri))
        for line in lines:
            out.write(line + "\n")
        out.write(")\n")
    os.replace(tmp, path)


def _assertions(triples):
    return (f"ObjectPropertyAssertion(<{p}> <{s}> <{o}>)" for s, p, o in triples)


class BoxLitEFamilyDataset(PathDataset):
    """F_v1 - F_v4 of [lourenco2026]_ (Table 1: 300/110/440, 500/209/837, 1001/432/1731
    and 3014/1226/4908 train/validation/test role assertions).

    :param version: 1, 2, 3 or 4.
    :param tbox: ``"reference"``, ``"paper"`` or ``"closure"`` (module docstring).
    :param data_root: Cache directory (default
        :func:`~mowl.datasets.base.default_data_root`).
    :param source_dir: ``Family_Dataset`` directory of a local clone of the pinned
        repository (checksums are still verified).
    """

    def __init__(self, version=1, tbox="reference", data_root=None, source_dir=None):
        version = int(version)
        if version not in (1, 2, 3, 4):
            raise ValueError("version must be 1, 2, 3 or 4.")
        if tbox not in TBOXES:
            raise ValueError(f"tbox must be one of {sorted(TBOXES)}.")
        self.version, self.tbox = version, tbox
        root = default_data_root() if data_root is None else data_root
        self.dataset_dir = os.path.join(root, f"boxlite_family_v{version}_{tbox}")
        paths = [os.path.join(self.dataset_dir, f)
                 for f in ("ontology.owl", "valid.owl", "test.owl")]
        if not all(os.path.exists(p) for p in paths):
            os.makedirs(self.dataset_dir, exist_ok=True)
            files = {split: fetch_family_split(split, version, data_root, source_dir)
                     for split in ("train", "validation", "test")}
            iri = f"http://mowl.borg/benchmarks/boxlite/family/v{version}"
            _write(paths[0], f"{iri}/train/{tbox}",
                   list(TBOXES[tbox]) + list(_assertions(read_triples(files["train"]))))
            _write(paths[1], f"{iri}/valid", _assertions(read_triples(files["validation"])))
            _write(paths[2], f"{iri}/test", _assertions(read_triples(files["test"])))
        super().__init__(*paths)
