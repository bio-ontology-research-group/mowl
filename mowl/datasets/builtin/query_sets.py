"""The released logical query datasets of Query2Box and BetaE (FB15k, FB15k-237, NELL995).

The data are the archive ``KG_data.zip`` of the BetaE authors
(http://snap.stanford.edu/betae/KG_data.zip, about 1.4 GB, last modified 2020-10-23, the
SHA-256 is pinned below). For each knowledge graph it holds two folders: ``<KG>-betae``
(the queries of Ren and Leskovec 2020, used by BetaE, ConE and AConE, with the five
negation structures and a cap on the number of answers of validation and test queries) and
``<KG>-q2b`` (the queries of Ren et al. 2020, used by Query2Box: the nine positive
structures, validation and test queries without a cap). Both have the same training
graph and training queries.

Licence: the archive carries no licence file and the download page states none. The
knowledge graphs derive from Freebase and NELL; check their terms before redistributing.
mOWL does not redistribute the data: it downloads the archive when a dataset is first
used, verifies the checksum and converts it once into ``default_data_root()/query/``.

**Encoding.** An entity is a class ``<http://mowl.borg/query/<KG>/entity/<label>>`` and a
training triple ``(h, r, t)`` is the axiom ``h ⊑ ∃r.t`` (as in
:class:`~mowl.datasets.builtin.kg.KnowledgeGraphDataset`). The released files list every
edge together with its inverse (relation ``2k`` and ``2k+1``, the inverse named
``-<label>`` in FB15k-237 and ``<label>_reverse`` in NELL995); only the forward relation
``2k`` is a role of the ontology, the models add the inverse themselves. The training
ontology holds the training edges, the validation and testing ontologies the validation
and test edges (the new ones, not cumulative). The direction was checked on the data:
``(h, r, t)`` in ``train.txt`` means that ``t`` is an answer of the one-hop query
``(h, (r,))``.

**Queries.** The released pickles hold, for every query structure (a nested tuple such
as ``('e', ('r', 'r'))``), instances in which ``e`` is an anchor entity and ``r`` a relation
id, ``-2`` stands for negation and ``-1`` for union. Validation and test queries come with
*easy* answers (reachable in the graph known at that stage) and *hard* answers (those that
need an edge of the next split). :meth:`QueryAnsweringDataset.queries` returns them in the
format of :class:`mowl.evaluation.LogicalQueryAnswering` with entity and relation ids of
the model that asks. Queries with entities or relations the model does not know cannot
occur (every entity and every forward relation is declared in the training ontology).
The training queries and their answers on the training graph are returned by
:meth:`QueryAnsweringDataset.training_queries`.

**Format of other query sets.** Any directory in the same layout can be read with
``data_dir`` (e.g. WN18RR-QA of Huang et al. 2022, which AConE uses and which is published
on a Google Drive that mOWL does not fetch).

**Closed world.** The benchmarks are closed world with negation as absence: an entity is
not an answer of ``¬q`` if it is an answer of ``q`` in the stated graph, and every edge
that is not stated is false when the answers are computed.

The pickles are read with a restricted unpickler that only builds ``defaultdict``, ``set``
and ``frozenset``, so reading a tampered file cannot run code.

Conversion needs memory for one pickle at a time; the largest is the training answers of
FB15k (440 MB on disk, several GB in Python).
"""

import contextlib
import fcntl
import hashlib
import os
import pickle
import shutil
import tempfile
import zipfile
from urllib.parse import quote

import numpy as np

from mowl.datasets.base import (DOWNLOAD_CONNECT_TIMEOUT, DOWNLOAD_READ_TIMEOUT,
                                PathDataset, default_data_root)

IRI_PREFIX = "http://mowl.borg/query/"
URL = "http://snap.stanford.edu/betae/KG_data.zip"
SHA256 = "85959742a8d525c0026ad95ca154f327649564938142b43c83f1403b6d5d7caf"
#: The knowledge graphs of the archive, with their folder names.
NAMES = {"FB15k": "FB15k", "FB15k237": "FB15k-237", "NELL995": "NELL"}
#: ``"betae"``: queries of BetaE (positive, negation, union; capped answers); ``"q2b"``:
#: the queries of Query2Box.
VARIANTS = ("betae", "q2b")
SPLITS = ("train", "valid", "test")


class _SafeUnpickler(pickle.Unpickler):
    _ALLOWED = {("collections", "defaultdict"), ("builtins", "set"), ("builtins", "frozenset")}

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Refusing to unpickle {module}.{name} from a query "
                                     f"dataset.")


def _load_pickle(path):
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


@contextlib.contextmanager
def _locked(path):
    """An exclusive lock on ``path + '.lock'`` (processes on one machine)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _atomic(path, write):
    partial = f"{path}.{os.getpid()}.part"
    try:
        write(partial)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


# -- structures ----------------------------------------------------------------------------

def _kinds(structure):
    """Kind of every integer of a flattened instance: ``"e"`` anchor, ``"r"`` relation,
    ``"s"`` special (negation ``-2`` or union ``-1``)."""
    if structure == "e":
        return ["e"]
    if all(isinstance(x, str) for x in structure):
        return ["s" if x in ("n", "u") else "r" for x in structure]
    out = []
    for s in structure:
        out += _kinds(s)
    return out


def _flatten(instance):
    if isinstance(instance, tuple):
        out = []
        for x in instance:
            out += _flatten(x)
        return out
    return [int(instance)]


def _build(structure, values):
    if structure == "e":
        return next(values)
    if all(isinstance(x, str) for x in structure):
        return tuple(next(values) for _ in structure)
    return tuple(_build(s, values) for s in structure)


def _structure_names():
    from mowl.evaluation.query_answering import QUERY_STRUCTURES
    return {s: n for n, s in QUERY_STRUCTURES.items()}


# -- raw files -----------------------------------------------------------------------------

def _read_triples(path):
    with open(path, encoding="utf-8") as f:
        return np.array([[int(x) for x in line.split()] for line in f if line.strip()],
                        dtype=np.int64).reshape(-1, 3)


def _write_ontology(path, name, part, entities, relations, triples, declare):
    def write(partial):
        with open(partial, "w", encoding="utf-8") as out:
            out.write("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n")
            out.write(f"Ontology(<{IRI_PREFIX}{name}/{part}.ofn>\n")
            if declare:
                for iri in entities:
                    out.write(f"Declaration(Class(<{iri}>))\n")
                for iri in relations:
                    out.write(f"Declaration(ObjectProperty(<{iri}>))\n")
            for h, r, t in triples.tolist():
                out.write(f"SubClassOf(<{entities[h]}> ObjectSomeValuesFrom("
                          f"<{relations[r // 2]}> <{entities[t]}>))\n")
            out.write(")\n")
    _atomic(path, write)


def _convert_graph(src, kg_dir, name):
    """Entities, relations and the three ontologies of the knowledge graph."""
    id2ent = _load_pickle(os.path.join(src, "id2ent.pkl"))
    id2rel = _load_pickle(os.path.join(src, "id2rel.pkl"))
    if sorted(id2ent) != list(range(len(id2ent))) or sorted(id2rel) != list(range(len(id2rel))):
        raise ValueError("The entity and relation ids of the query dataset are not dense.")
    entities = [QueryAnsweringDataset.entity_iri(name, id2ent[i]) for i in range(len(id2ent))]
    relations = [QueryAnsweringDataset.relation_iri(name, id2rel[2 * k])
                 for k in range(len(id2rel) // 2)]
    os.makedirs(kg_dir, exist_ok=True)
    for part, fname, declare in (("ontology", "train.txt", True), ("valid", "valid.txt", False),
                                 ("test", "test.txt", False)):
        triples = _read_triples(os.path.join(src, fname))
        forward = triples[triples[:, 1] % 2 == 0]
        _write_ontology(os.path.join(kg_dir, f"{part}.ofn"), name, part, entities, relations,
                        forward, declare)

    def write_names(path, lines):
        def w(partial):
            with open(partial, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        _atomic(path, w)
    write_names(os.path.join(kg_dir, "entities.txt"), entities)
    write_names(os.path.join(kg_dir, "relations.txt"), relations)


def _answer_arrays(answer_sets):
    offsets = np.zeros(len(answer_sets) + 1, dtype=np.int64)
    values = []
    for i, s in enumerate(answer_sets):
        values.extend(sorted(s))
        offsets[i + 1] = len(values)
    return offsets, np.asarray(values, dtype=np.int32)


def _convert_queries(src, variant_dir):
    """One ``<split>.npz`` per split: for each structure name ``n``: ``n/q`` (flattened
    instances), ``n/easy_off``, ``n/easy_val``, ``n/hard_off``, ``n/hard_val``. For the
    training split the answers on the training graph are stored as ``easy``."""
    names = _structure_names()
    os.makedirs(variant_dir, exist_ok=True)
    for split in SPLITS:
        queries = _load_pickle(os.path.join(src, f"{split}-queries.pkl"))
        if split == "train":
            easy = _load_pickle(os.path.join(src, "train-answers.pkl"))
            hard = {}
        else:
            easy = _load_pickle(os.path.join(src, f"{split}-easy-answers.pkl"))
            hard = _load_pickle(os.path.join(src, f"{split}-hard-answers.pkl"))
        arrays = {}
        for structure, instances in queries.items():
            if structure not in names:
                raise ValueError(f"Unknown query structure {structure!r} in {split}-queries.pkl")
            n = names[structure]
            ordered = sorted(instances)
            arrays[f"{n}/q"] = np.asarray([_flatten(q) for q in ordered], dtype=np.int32)
            for tag, table in (("easy", easy), ("hard", hard)):
                off, val = _answer_arrays([table.get(q, ()) if table else () for q in ordered])
                arrays[f"{n}/{tag}_off"] = off
                arrays[f"{n}/{tag}_val"] = val
        del queries, easy, hard

        def write(partial, arrays=arrays):
            with open(partial, "wb") as f:
                np.savez(f, **arrays)
        _atomic(os.path.join(variant_dir, f"{split}.npz"), write)


# -- download ------------------------------------------------------------------------------

def _download_archive(root):
    """The archive in ``root/query/KG_data.zip`` (downloaded once, checked, atomic)."""
    path = os.path.join(root, "query", "KG_data.zip")
    with _locked(path):
        if os.path.exists(path):
            return path
        from mowl.datasets.base import RemoteDataset

        def write(partial):
            digest = hashlib.sha256()
            timeout = (DOWNLOAD_CONNECT_TIMEOUT, DOWNLOAD_READ_TIMEOUT)
            with RemoteDataset._session() as session:
                with session.get(URL, stream=True, timeout=timeout) as req:
                    req.raise_for_status()
                    with open(partial, "wb") as out:
                        for chunk in req.iter_content(chunk_size=1 << 20):
                            out.write(chunk)
                            digest.update(chunk)
            if digest.hexdigest() != SHA256:
                raise ValueError(f"The downloaded {URL} has SHA-256 {digest.hexdigest()}, "
                                 f"expected {SHA256}.")
        _atomic(path, write)
    return path


def _extract(archive, folder, target):
    with zipfile.ZipFile(archive) as z:
        prefix = folder + "/"
        members = [m for m in z.namelist() if m.startswith(prefix) and not m.endswith("/")]
        if not members:
            raise ValueError(f"{archive} has no folder {folder!r}.")
        for m in members:
            base = os.path.basename(m)
            if base != m[len(prefix):]:
                continue
            with z.open(m) as src, open(os.path.join(target, base), "wb") as dst:
                shutil.copyfileobj(src, dst)


class QueryAnsweringDataset(PathDataset):
    """A released logical query dataset of Query2Box / BetaE as an mOWL dataset.

    The ontology is the training graph (``h ⊑ ∃r.t`` between entity classes, see the module
    docstring); the released validation and test queries and the training queries are read
    with :meth:`queries` and :meth:`training_queries`.

    :param name: ``"FB15k"``, ``"FB15k237"`` or ``"NELL995"``; with ``data_dir`` the name of
        the data set.
    :param variant: ``"betae"`` (BetaE queries: positive and negation structures, unions;
        used by BetaE, ConE, AConE) or ``"q2b"`` (Query2Box queries: positive structures and
        unions).
    :param data_root: Where to cache the converted files.
    :param data_dir: A directory in the layout of one folder of ``KG_data.zip`` (files
        ``train.txt``, ``valid.txt``, ``test.txt``, ``id2ent.pkl``, ``id2rel.pkl``,
        ``<split>-queries.pkl``, ``train-answers.pkl``, ``<split>-easy-answers.pkl``,
        ``<split>-hard-answers.pkl``) to read instead of the archive.
    """

    def __init__(self, name="FB15k237", variant="betae", data_root=None, data_dir=None):
        if data_dir is None and name not in NAMES:
            raise ValueError(f"name must be one of {sorted(NAMES)}, or give data_dir (the "
                             f"data of {name!r} are not downloaded by mOWL).")
        if variant not in VARIANTS and data_dir is None:
            raise ValueError(f"variant must be one of {VARIANTS}.")
        self.name = name
        self.variant = "custom" if data_dir is not None else variant
        root = default_data_root() if data_root is None else data_root
        self.kg_directory = os.path.join(root, "query", name)
        self.directory = os.path.join(self.kg_directory, self.variant)
        paths = [os.path.join(self.kg_directory, f"{part}.ofn")
                 for part in ("ontology", "valid", "test")]
        ready = (all(os.path.exists(p) for p in paths)
                 and all(os.path.exists(os.path.join(self.directory, f"{s}.npz"))
                         for s in SPLITS))
        if not ready:
            with _locked(self.directory):
                self._build(root, data_dir, paths)
        with open(os.path.join(self.kg_directory, "entities.txt"), encoding="utf-8") as f:
            self.entity_iris = f.read().split("\n")[:-1]
        with open(os.path.join(self.kg_directory, "relations.txt"), encoding="utf-8") as f:
            self.relation_iris = f.read().split("\n")[:-1]
        self._arrays = {}
        super().__init__(*paths)

    @staticmethod
    def entity_iri(name, label):
        return f"{IRI_PREFIX}{name}/entity/{quote(str(label), safe='')}"

    @staticmethod
    def relation_iri(name, label):
        label = str(label)
        if label.startswith("+"):
            label = label[1:]
        return f"{IRI_PREFIX}{name}/relation/{quote(label, safe='')}"

    def _build(self, root, data_dir, paths):
        ready = (all(os.path.exists(p) for p in paths)
                 and all(os.path.exists(os.path.join(self.directory, f"{s}.npz"))
                         for s in SPLITS))
        if ready:
            return                       # another process finished while we waited
        if data_dir is not None:
            self._convert(os.path.abspath(data_dir))
            return
        archive = _download_archive(root)
        os.makedirs(os.path.join(root, "query"), exist_ok=True)
        scratch = tempfile.mkdtemp(prefix=f"{self.name}-{self.variant}-",
                                   dir=os.path.join(root, "query"))
        try:
            _extract(archive, f"{NAMES[self.name]}-{self.variant}", scratch)
            self._convert(scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _convert(self, src):
        _convert_graph(src, self.kg_directory, self.name)
        _convert_queries(src, self.directory)

    # -- queries -----------------------------------------------------------------------------
    def _split(self, split):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}.")
        if split not in self._arrays:
            with np.load(os.path.join(self.directory, f"{split}.npz")) as z:
                self._arrays[split] = {k: z[k] for k in z.files}
        return self._arrays[split]

    @property
    def structure_names(self):
        """Names of the query structures that the dataset has (in the test split)."""
        return sorted({k.split("/")[0] for k in self._split("test")})

    def _maps(self, entity_id, relation_id):
        ent = np.array([entity_id(iri) for iri in self.entity_iris], dtype=np.int64)
        rel = np.array([relation_id(self.relation_iris[j // 2], j % 2 == 1)
                        for j in range(2 * len(self.relation_iris))], dtype=np.int64)
        return ent, rel

    @staticmethod
    def _map_rows(structure, rows, ent, rel):
        rows = rows.astype(np.int64)
        for col, kind in enumerate(_kinds(structure)):
            if kind == "e":
                rows[:, col] = ent[rows[:, col]]
            elif kind == "r":
                rows[:, col] = rel[rows[:, col]]
        return rows

    def queries(self, split, entity_id, relation_id, names=None, limit=None, seed=0):
        """The released validation or test queries with their easy and hard answers.

        :param split: ``"valid"`` or ``"test"``.
        :param entity_id: ``f(iri) -> id`` of a class entity in the model that is evaluated
            (e.g. :meth:`mowl.models.composed.query.QueryEmbeddingModel.entity_id`).
        :param relation_id: ``f(role_iri, inverse) -> id`` of a relation of the model.
        :param names: Structure names to return (default: all the split has, except the
            De Morgan forms ``2u-DM`` and ``up-DM``).
        :param limit: At most this many queries per structure (a seeded random subset).
        :return: ``{name: [(query, easy_answers, hard_answers), ...]}``; queries in the
            nested-tuple format with the model's ids (``-2`` negation, ``-1`` union).
        """
        if split == "train":
            raise ValueError("Use training_queries() for the training split.")
        from mowl.evaluation.query_answering import QUERY_STRUCTURES
        data = self._split(split)
        ent, rel = self._maps(entity_id, relation_id)
        available = sorted({k.split("/")[0] for k in data})
        if names is None:
            names = [n for n in available if not n.endswith("-DM")]
        unknown = sorted(set(names) - set(available))
        if unknown:
            raise ValueError(f"The {split} split has no queries of {unknown}; it has "
                             f"{available}.")
        rng = np.random.default_rng(seed)
        out = {}
        for n in names:
            structure = QUERY_STRUCTURES[n]
            rows = data[f"{n}/q"]
            index = np.arange(len(rows))
            if limit is not None and len(rows) > limit:
                index = np.sort(rng.choice(len(rows), size=limit, replace=False))
            mapped = self._map_rows(structure, rows[index], ent, rel)
            sets = {}
            for tag in ("easy", "hard"):
                off, val = data[f"{n}/{tag}_off"], data[f"{n}/{tag}_val"]
                sets[tag] = [set(ent[val[off[i]:off[i + 1]]].tolist()) for i in index]
            items = []
            for row, easy, hard in zip(mapped.tolist(), sets["easy"], sets["hard"]):
                items.append((_build(structure, iter(row)), easy, hard))
            out[n] = items
        return out

    def training_queries(self, entity_id, relation_id, names=None):
        """The released training queries and their answers on the training graph.

        :return: ``{name: ReleasedTrainingQueries}``; ``names`` defaults to all the
            structures of the training split.
        """
        from mowl.evaluation.query_answering import QUERY_STRUCTURES
        data = self._split("train")
        ent, rel = self._maps(entity_id, relation_id)
        available = sorted({k.split("/")[0] for k in data})
        names = available if names is None else list(names)
        unknown = sorted(set(names) - set(available))
        if unknown:
            raise ValueError(f"The training split has no queries of {unknown}; it has "
                             f"{available}.")
        out = {}
        for n in names:
            structure = QUERY_STRUCTURES[n]
            out[n] = ReleasedTrainingQueries(
                structure, self._map_rows(structure, data[f"{n}/q"], ent, rel),
                data[f"{n}/easy_off"], ent[data[f"{n}/easy_val"]])
        return out


class ReleasedTrainingQueries:
    """Training queries of one structure with the ids of a model.

    :param structure: Nested-tuple structure.
    :param rows: ``(n, k)`` array, the flattened instances (anchors and relations mapped,
        ``-1`` / ``-2`` kept).
    :param offsets: ``(n + 1,)`` offsets into ``answers``.
    :param answers: Concatenated answer entities of the queries on the training graph.
    """

    def __init__(self, structure, rows, offsets, answers):
        self.structure = structure
        self.rows = np.asarray(rows, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.answers = np.asarray(answers, dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def answer_set(self, i):
        return self.answers[self.offsets[i]:self.offsets[i + 1]]

    def count(self, i):
        return int(self.offsets[i + 1] - self.offsets[i])
