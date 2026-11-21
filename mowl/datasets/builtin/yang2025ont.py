"""Benchmark data of OnT (Yang, Chen, He, Gao, Horrocks, "Language Models as Ontology
Encoders", ISWC 2025): GALEN, GO and ANATOMY (Uberon), normalised by OnT's own
name-preserving normaliser, for axiom prediction and inference (Section 5.1, Table 1).

The data are the public release of the authors on Zenodo (`record 17314434
<https://zenodo.org/records/17314434>`_, "OnT (Language Models as Ontology Encoder) -
Complete Datasets", DOI 10.5281/zenodo.17314434). Each archive is downloaded once into
the data root and its SHA-256 is checked. The release is textual: every concept and role
is given by its verbalisation (``concept_names.json``, ``role_names.json``), training
axioms are pairs of texts (``train.jsonl``, with the existential restrictions and
conjunctions they contain in ``train_exist.jsonl`` and ``train_conj.jsonl``), and
validation and test axioms are query texts with the identifier of the answer concept
(``val.json``, ``test.json``).

mOWL converts the release into OWL (functional syntax): concept ``i`` becomes the class
``<base>C<i>`` and role ``j`` the object property ``<base>R<j>``, each with its text as
``rdfs:label``; every concept is declared in the training ontology. Axioms are rebuilt
from the texts:

* a training pair (child, parent) is :math:`C \\sqcap D \\sqsubseteq E` if the child is a
  conjunction listed in ``train_conj.jsonl``, :math:`C \\sqsubseteq \\exists r.D` if the
  parent is a restriction listed in ``train_exist.jsonl``, :math:`\\exists r.C
  \\sqsubseteq D` if the child is, and :math:`C \\sqsubseteq D` otherwise. On the three
  prediction releases, where ``train.jsonl`` is the concatenation of
  ``train_nf1.jsonl`` ... ``train_nf4.jsonl``, this rule recovers the normal form of
  every line. Texts are mapped to concepts by name; a name shared by several concepts
  (292 in ANATOMY, 4,752 in GO, none in GALEN) is mapped to the smallest identifier.
* validation and test queries are mapped back to the normal forms of Section 5.1: NF1
  :math:`C \\sqsubseteq ?D`, NF2 :math:`C \\sqcap D \\sqsubseteq ?E`, NF3 :math:`?C
  \\sqsubseteq \\exists r.D`, NF4 :math:`\\exists r.C \\sqsubseteq ?D`, with the answer
  identifier of the release.

With ``verbalisation_style="release"`` (:class:`mowl.text.Verbaliser`), the
verbalisations of the rebuilt axioms are the texts of the release, except that the two
operands of a conjunction may appear in the other order (the OWL API sorts operands).

Differences between the release and the paper: the test sets and the inferred NF1
counts match Table 1 exactly, but the training and validation sets of the release are
smaller than Table 1 (e.g. GALEN prediction: 18,643 / 7,972 / 19,013 / 1 training
axioms of NF1-NF4 against 25,610 / 11,679 / 25,299 / 6,287; no NF4 validation axioms),
and the ``train_nf*.jsonl`` files of the inference releases do not match their
``train.jsonl``, which is the file the reference training script reads and the one mOWL
uses. The ``train_nf*.jsonl`` files are not used.

The candidate set of the OnT evaluation is every concept of ``concept_names.json``,
including the concepts introduced by normalisation; it is exposed as
:attr:`Yang2025OnTDataset.candidate_classes`.
"""

import hashlib
import json
import os
import zipfile

from mowl.datasets.base import PathDataset, RemoteDataset, default_data_root

#: Zenodo record of the OnT data release.
ZENODO_RECORD = "17314434"
#: URL template of the archives (``{name}`` is e.g. ``GALEN_predict``).
ONT_DATA_URL = "https://zenodo.org/api/records/" + ZENODO_RECORD + "/files/{name}.zip/content"
#: SHA-256 of the archives (the MD5 sums published by Zenodo were checked as well).
ONT_DATA_SHA256 = {
    "GALEN_predict": "abf2794a21395c94285cafcfb2e93a52848c5da09fa3fe0255fcd8c4a1692817",
    "GALEN_inference": "b0415f7790623c9661d4be9e108434f19838a153c217ffff1fc75b364959d390",
    "GO_predict": "8ee04eca0d86fae53d02676c5df0dcc816a77245067f3af2b532ce6c1a8a0e1c",
    "GO_inference": "b29e54fdb1c9fc295cda97d788bfd6f02140d913fa78fa9415f30024936643d5",
    "ANATOMY_predict": "013e86362cec381cac172d600de0e11f7da5dbbeb8fa4fac793dcb6f5359ba39",
    "ANATOMY_inference": "286179f1bb1ac44af60bfc5ab7a245f6269ed525c3a61981082e3103a34400b9",
}

ONTOLOGIES = ("GALEN", "GO", "ANATOMY")
TASKS = {"prediction": "predict", "inference": "inference"}

_HEADER = ("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n"
           "Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)\n"
           "Ontology(<{iri}>\n")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_ont_release(name, data_root=None, zip_path=None):
    """Path of one archive of the OnT release, downloaded if needed.

    :param name: ``"<ONTOLOGY>_predict"`` or ``"<ONTOLOGY>_inference"``.
    :param zip_path: Use this local copy instead of downloading.
    :raises ValueError: If the checksum does not match the pinned release.
    """
    if name not in ONT_DATA_SHA256:
        raise ValueError(f"Unknown archive {name!r}; known: {sorted(ONT_DATA_SHA256)}.")
    if zip_path is None:
        root = default_data_root() if data_root is None else data_root
        os.makedirs(root, exist_ok=True)
        zip_path = os.path.join(root, f"ont_{name}.zip")
        if not os.path.exists(zip_path):
            partial = f"{zip_path}.{os.getpid()}.part"
            try:
                with RemoteDataset._session() as session:
                    url = ONT_DATA_URL.format(name=name)
                    with session.get(url, stream=True, timeout=(10, 120)) as req:
                        req.raise_for_status()
                        with open(partial, "wb") as w:
                            for chunk in req.iter_content(chunk_size=1 << 16):
                                w.write(chunk)
            except BaseException:
                if os.path.exists(partial):
                    os.remove(partial)
                raise
            os.replace(partial, zip_path)
    digest = _sha256(zip_path)
    if digest != ONT_DATA_SHA256[name]:
        raise ValueError(f"{zip_path} has SHA-256 {digest}, expected the pinned release "
                         f"{ONT_DATA_SHA256[name]}.")
    return zip_path


def _literal(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


class _Release:
    """Reader of one directory of the release inside a zip archive."""

    def __init__(self, zf, folder):
        self.zf = zf
        self.folder = folder
        self.concepts = {int(k): v for k, v in self._json("concept_names.json").items()}
        self.roles = {int(k): v for k, v in self._json("role_names.json").items()}
        self.concept_id = {}
        for i in sorted(self.concepts):
            self.concept_id.setdefault(self.concepts[i], i)
        self.role_id = {}
        for i in sorted(self.roles):
            self.role_id.setdefault(self.roles[i], i)

    def _json(self, name):
        with self.zf.open(f"{self.folder}/{name}") as f:
            return json.load(f)

    def _jsonl(self, name):
        with self.zf.open(f"{self.folder}/{name}") as f:
            for raw in f:
                raw = raw.strip()
                if raw:
                    yield json.loads(raw)

    def training_axioms(self):
        """Rebuilt training axioms as tuples ``("nf1", C, D)``, ``("nf2", C, D, E)``,
        ``("nf3", C, r, D)``, ``("nf4", r, C, D)`` of identifiers."""
        conj = {x["Concept"]: (x["con1"], x["con2"]) for x in self._jsonl("train_conj.jsonl")}
        exist = {x["Concept"]: (x["role"], x["con"]) for x in self._jsonl("train_exist.jsonl")}
        cid, rid = self.concept_id, self.role_id
        out = []
        for x in self._jsonl("train.jsonl"):
            child, parent = x["child"], x["parent"]
            if child in conj and parent in cid:
                a, b = conj[child]
                out.append(("nf2", cid[a], cid[b], cid[parent]))
            elif parent in exist and child in cid:
                r, d = exist[parent]
                out.append(("nf3", cid[child], rid[r], cid[d]))
            elif child in exist and parent in cid:
                r, c = exist[child]
                out.append(("nf4", rid[r], cid[c], cid[parent]))
            else:
                out.append(("nf1", cid[child], cid[parent]))
        return out

    def evaluation_axioms(self, split):
        """Axioms of ``val.json`` or ``test.json`` (same tuples as
        :meth:`training_axioms`)."""
        data = self._json(f"{split}.json")
        cid, rid = self.concept_id, self.role_id
        out = []
        for nf, queries in data["query_sentences"].items():
            answers = data["answer_ids"][nf]
            for q, ans in zip(queries, answers):
                if nf == "nf1":
                    out.append(("nf1", cid[q["name"]], int(ans)))
                elif nf == "nf2":
                    out.append(("nf2", cid[q["con1"]], cid[q["con2"]], int(ans)))
                elif nf == "nf3":
                    out.append(("nf3", int(ans), rid[q["role"]], cid[q["con"]]))
                elif nf == "nf4":
                    out.append(("nf4", rid[q["role"]], cid[q["con"]], int(ans)))
        return out


class Yang2025OnTDataset(PathDataset):
    """GALEN, GO or ANATOMY as normalised and split by OnT [yang2025ont]_ (Section 5.1,
    Table 1), read from the authors' data release (see the module docstring).

    * ``task="prediction"``: random 80/10/10 split of the normalised axioms; training,
      validation and test axioms of all four normal forms.
    * ``task="inference"``: the whole normalised ontology for training, the inferred
      atomic subsumptions (NF1) for testing and 1,000 of them for validation.

    All classes have ``rdfs:label`` annotations (the release's concept texts), so
    text-based methods can be trained on it; it can equally be used by any EL model.

    :param ontology: ``"GALEN"``, ``"GO"`` or ``"ANATOMY"``.
    :param task: ``"prediction"`` or ``"inference"``.
    :param data_root: Cache directory (default :func:`~mowl.datasets.base.default_data_root`).
    :param zip_path: Local copy of the release archive (checksum-verified).
    """

    def __init__(self, ontology="GALEN", task="prediction", data_root=None, zip_path=None):
        ontology = ontology.upper()
        if ontology not in ONTOLOGIES:
            raise ValueError(f"ontology must be one of {ONTOLOGIES}")
        if task not in TASKS:
            raise ValueError(f"task must be one of {tuple(TASKS)}")
        self.ontology_name = ontology
        self.task = task
        self.base_iri = f"http://mowl.borg/datasets/yang2025ont/{ontology.lower()}/"
        root = default_data_root() if data_root is None else data_root
        self.dataset_dir = os.path.join(root, f"yang2025ont_{ontology.lower()}_{task}")
        paths = [os.path.join(self.dataset_dir, f)
                 for f in ("ontology.owl", "valid.owl", "test.owl")]
        names_path = os.path.join(self.dataset_dir, "concepts.json")
        if not all(os.path.exists(p) for p in paths + [names_path]):
            os.makedirs(self.dataset_dir, exist_ok=True)
            name = f"{ontology}_{TASKS[task]}"
            folder = f"{ontology}_OnT" + ("_inference" if task == "inference" else "")
            with zipfile.ZipFile(fetch_ont_release(name, root, zip_path)) as zf:
                self._convert(_Release(zf, folder), paths, names_path)
        with open(names_path) as f:
            self.num_concepts = json.load(f)["num_concepts"]
        super().__init__(*paths)

    def class_iri(self, i):
        """IRI of concept ``i`` of the release."""
        return f"{self.base_iri}C{i}"

    def role_iri(self, j):
        """IRI of role ``j`` of the release."""
        return f"{self.base_iri}R{j}"

    @property
    def candidate_classes(self):
        """IRIs of all concepts of the release, in identifier order: the candidate set
        of the OnT evaluation."""
        return [self.class_iri(i) for i in range(self.num_concepts)]

    def _expr(self, ax):
        c, r = self.class_iri, self.role_iri
        kind = ax[0]
        if kind == "nf1":
            return f"SubClassOf(<{c(ax[1])}> <{c(ax[2])}>)"
        if kind == "nf2":
            return (f"SubClassOf(ObjectIntersectionOf(<{c(ax[1])}> <{c(ax[2])}>) "
                    f"<{c(ax[3])}>)")
        if kind == "nf3":
            return f"SubClassOf(<{c(ax[1])}> ObjectSomeValuesFrom(<{r(ax[2])}> <{c(ax[3])}>))"
        return f"SubClassOf(ObjectSomeValuesFrom(<{r(ax[1])}> <{c(ax[2])}>) <{c(ax[3])}>)"

    def _write(self, path, suffix, lines):
        tmp = f"{path}.{os.getpid()}.part"
        with open(tmp, "w", encoding="utf-8") as out:
            out.write(_HEADER.format(iri=self.base_iri + suffix))
            for line in lines:
                out.write(line + "\n")
            out.write(")\n")
        os.replace(tmp, path)

    def _convert(self, rel, paths, names_path):
        ontology_path, valid_path, test_path = paths

        def training():
            for i in sorted(rel.concepts):
                yield f"Declaration(Class(<{self.class_iri(i)}>))"
                yield (f"AnnotationAssertion(rdfs:label <{self.class_iri(i)}> "
                       f"{_literal(rel.concepts[i])})")
            for j in sorted(rel.roles):
                yield f"Declaration(ObjectProperty(<{self.role_iri(j)}>))"
                yield (f"AnnotationAssertion(rdfs:label <{self.role_iri(j)}> "
                       f"{_literal(rel.roles[j])})")
            seen = set()
            for ax in rel.training_axioms():
                if ax not in seen:
                    seen.add(ax)
                    yield self._expr(ax)

        self._write(ontology_path, "train", training())
        for path, split in ((valid_path, "val"), (test_path, "test")):
            self._write(path, split, (self._expr(ax) for ax in rel.evaluation_axioms(split)))
        tmp = f"{names_path}.{os.getpid()}.part"
        with open(tmp, "w") as f:
            json.dump({"num_concepts": len(rel.concepts), "num_roles": len(rel.roles)}, f)
        os.replace(tmp, names_path)
