"""Subsumption benchmarks of EmEL++ (Mondal, Bhatia, Mutharaju, "EmEL++: Embeddings for
EL++ Description Logic", AAAI-MAKE 2021, Section 4.3), read from the data folder of the
reference implementation (`kracr/EmELpp <https://github.com/kracr/EmELpp>`_, commit
``267f2c7``, Apache-2.0).
"""

import os
import re

from mowl.datasets.base import PathDataset, RemoteDataset, default_data_root

EMELPP_COMMIT = "267f2c70865cc55b2000163b87a269780327068b"
EMELPP_BASE_URL = ("https://raw.githubusercontent.com/kracr/EmELpp/"
                   f"{EMELPP_COMMIT}/experiments/")

#: (normalised ontology, train, valid, test, inferences) per ontology.
EMELPP_FILES = {
    "GALEN": ("GALEN/data/gallen_norm_mod.owl", "GALEN/data/GALLEN_train.txt",
              "GALEN/data/GALLEN_valid.txt", "GALEN/data/GALLEN_test.txt",
              "GALEN/data/GALEN_inferences.txt"),
    "GO": ("GO/data/go_latest_norm_mod.owl", "GO/data/GO_train.txt", "GO/data/GO_valid.txt",
           "GO/data/GO_test.txt", "GO/data/GO_inferences.txt"),
    "ANATOMY": ("ANATOMY/data/anatomy_norm_mod.owl", "ANATOMY/data/ANATOMY_train.txt",
                "ANATOMY/data/ANATOMY_valid.txt", "ANATOMY/data/ANATOMY_test.txt",
                "ANATOMY/data/ANATOMY_inferences.txt"),
}

_ATOMIC_SUBCLASS = re.compile(r"^SubClassOf\((<[^<>\s]+>)\s+(<[^<>\s]+>)\)\s*$")


def _download(url, path):
    partial = f"{path}.{os.getpid()}.part"
    try:
        with RemoteDataset._session() as session:
            with session.get(url, stream=True, timeout=(10, 60)) as req:
                req.raise_for_status()
                with open(partial, "wb") as w:
                    for chunk in req.iter_content(chunk_size=1 << 16):
                        w.write(chunk)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    os.replace(partial, path)


def _pairs(path):
    out = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                out.append((parts[0], parts[1]))
    return out


def _write(path, iri, lines):
    # A per-process temporary name: jobs that build the same dataset at the same time
    # must not write into one file.
    tmp = f"{path}.{os.getpid()}.part"
    with open(tmp, "w") as f:
        f.write("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n")
        f.write(f"Ontology(<{iri}>\n")
        for line in lines:
            f.write(line.rstrip("\n") + "\n")
        f.write(")\n")
    os.replace(tmp, path)


#: The released ANATOMY ontology contains role chains with no roles
#: (``SubObjectPropertyOf(ObjectPropertyChain( ) R)``), which are not OWL and make the
#: file unparsable; such lines are dropped.
_EMPTY_CHAIN = re.compile(r"ObjectPropertyChain\(\s*\)")


class EmELppSubsumptionDataset(PathDataset):
    """The 70/20/10 subsumption split of EmEL++ for GALEN, GO or ANATOMY.

    The paper (Section 4.3) removes 30% of the atomic subclass pairs of the normalised
    ontology for validation (20%) and testing (10%) and trains on the rest. The release
    contains the normalised ontology (``*_norm_mod.owl``) and the three pair lists. The
    released ontology no longer contains the test pairs but still contains every
    validation pair, and the reference training script trains on it. mOWL follows the
    paper's text: the training ontology is the normalised ontology without the atomic
    subsumptions listed in the validation or test files. The training regime
    :class:`~mowl.training.ValidationAxiomsAsPositives` adds the validation pairs back.

    SNOMED CT is not provided: the release has no normalised SNOMED ontology, and
    SNOMED CT cannot be redistributed.

    :param ontology: ``"GALEN"``, ``"GO"`` or ``"ANATOMY"``.
    :param data_root: Cache directory.
    """

    def __init__(self, ontology="GALEN", data_root=None):
        ontology = ontology.upper()
        if ontology not in EMELPP_FILES:
            raise ValueError(f"ontology must be one of {sorted(EMELPP_FILES)}")
        self.ontology_name = ontology
        root = default_data_root() if data_root is None else data_root
        target = os.path.join(root, f"emelpp_{ontology.lower()}")
        os.makedirs(target, exist_ok=True)
        out = {n: os.path.join(target, n) for n in ("ontology.owl", "valid.owl", "test.owl")}
        if not all(os.path.exists(p) for p in out.values()):
            raw = {}
            for rel in EMELPP_FILES[ontology][:4]:
                path = os.path.join(target, "raw_" + os.path.basename(rel))
                if not os.path.exists(path):
                    _download(EMELPP_BASE_URL + rel, path)
                raw[rel] = path
            norm, _, valid, test = (raw[r] for r in EMELPP_FILES[ontology][:4])
            held_out = set(_pairs(valid)) | set(_pairs(test))
            iri = f"http://mowl.borg/benchmarks/emelpp/{ontology.lower()}"

            def train_lines():
                with open(norm) as f:
                    for line in f:
                        m = _ATOMIC_SUBCLASS.match(line.strip())
                        if m and (m.group(1), m.group(2)) in held_out:
                            continue
                        if line.strip() and not _EMPTY_CHAIN.search(line):
                            yield line
            _write(out["ontology.owl"], iri + "/train", train_lines())
            for name, src in (("valid.owl", valid), ("test.owl", test)):
                _write(out[name], f"{iri}/{name[:-4]}",
                       (f"SubClassOf({c} {d})" for c, d in _pairs(src)))
        super().__init__(out["ontology.owl"], out["valid.owl"], out["test.owl"])

    @property
    def evaluation_classes(self):
        return self.classes, self.classes
