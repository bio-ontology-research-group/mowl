"""Inductive knowledge graph completion benchmarks (GraIL splits of FB15k-237, WN18RR and
NELL-995, versions v1 to v4) as mOWL datasets."""

import os
import urllib.request
from dataclasses import dataclass
from urllib.parse import quote

import torch as th

from mowl.datasets.base import PathDataset, default_data_root

IRI_PREFIX = "http://mowl.borg/inductive-kg/"
#: The commit of https://github.com/kkteru/grail (the release of Teru et al., ICML 2020)
#: the files are read from. The repository states no licence; the data are derived from
#: FB15k-237, WN18RR and NELL-995, for which no licence is stated either (the paper of
#: RESHUFFLE, appendix, says the same).
COMMIT = "2a3dffa719518e7e6250e355a2fb37cd932de91e"
_BASE = f"https://raw.githubusercontent.com/kkteru/grail/{COMMIT}/data/"
#: Directory prefix of every dataset in the release.
_DIRECTORY = {"FB15k237": "fb237", "WN18RR": "WN18RR", "NELL995": "nell"}
NAMES = tuple(_DIRECTORY)
VERSIONS = ("v1", "v2", "v3", "v4")
#: Raw file of the dataset's cache -> (release directory suffix, release file).
_RAW = {"train.txt": ("", "train.txt"),            # training graph, training triples
        "valid.txt": ("", "valid.txt"),            # training graph, validation triples
        "inference.txt": ("_ind", "train.txt"),    # test graph, the facts a model sees
        "test.txt": ("_ind", "test.txt")}          # test graph, the triples to predict


def entity_iri(name, version, label):
    """The class IRI of an entity of a dataset."""
    return f"{IRI_PREFIX}{name}_{version}/entity/{quote(str(label), safe='')}"


def relation_iri(name, version, label):
    """The role IRI of a relation of a dataset (the same in the training and the test
    graph)."""
    return f"{IRI_PREFIX}{name}_{version}/relation/{quote(str(label), safe='')}"


@dataclass(frozen=True)
class InductiveGraph:
    """A knowledge graph given to a model at inference time: facts over entities the
    model has not seen in training.

    The entities and relations are IRIs. A model that can score triples over a new graph
    receives this object (see :mod:`mowl.evaluation.inductive_ranking`); relations are
    shared with the training graph by IRI, entities are not.

    :param entities: Entity IRIs; an entity's index is its position.
    :param relations: Relation (role) IRIs; a relation's index is its position.
    :param triples: ``(n, 3)`` long tensor of facts ``[head, relation, tail]`` as indices
        into ``entities`` and ``relations``. A fact is the axiom ``head ⊑ ∃relation.tail``.
    """

    entities: tuple
    relations: tuple
    triples: th.Tensor

    @property
    def entity_index(self):
        """Entity IRI -> index."""
        return {e: i for i, e in enumerate(self.entities)}

    @property
    def relation_index(self):
        """Relation IRI -> index."""
        return {r: i for i, r in enumerate(self.relations)}

    def index_triples(self, triples):
        """``(m, 3)`` long tensor of IRI triples ``(h, r, t)`` indexed in this graph.

        :raises KeyError: If an entity or relation is not in the graph."""
        e, r = self.entity_index, self.relation_index
        rows = [[e[h], r[x], e[t]] for h, x, t in triples]
        return th.tensor(rows, dtype=th.long).reshape(-1, 3)

    def iri_triples(self):
        """The facts as a list of IRI triples."""
        return [(self.entities[h], self.relations[r], self.entities[t])
                for h, r, t in self.triples.tolist()]


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]


class InductiveKnowledgeGraphDataset(PathDataset):
    """An inductive knowledge graph completion benchmark of Teru et al. (ICML 2020,
    "GraIL"), derived from FB15k-237, WN18RR and NELL-995: four versions (``v1`` to
    ``v4``) of each. A dataset has two graphs with disjoint entities and (mostly) the
    same relations: a training graph and a test graph.

    * ``ontology``: the facts of the training graph (``train.txt``). A triple ``(h, r, t)``
      is the axiom ``h ⊑ ∃r.t`` between two named classes, as in
      :class:`mowl.datasets.builtin.kg.KnowledgeGraphDataset`; the line ``h<TAB>r<TAB>t`` of
      the release is read head first (checked on the data).
    * ``validation``: the validation triples of the training graph (``valid.txt``). The
      RESHUFFLE paper selects hyper-parameters and stops training with them.
    * :attr:`inference_graph`: the facts of the test graph (``train.txt`` of the
      ``*_ind`` directory), an :class:`InductiveGraph`. **This is how a model receives
      the graph it is tested on**: the evaluation regime
      (:class:`mowl.evaluation.inductive_ranking.InductiveSampledRankingRegime`) passes
      it to ``model.inductive_costs(graph, rows)``. A model that scores triples over a
      graph it was not trained on implements that method; a transductive embedding has
      no vectors for the test entities and does not.
    * :attr:`test_triples`: the triples to predict, ``test.txt`` of the ``*_ind``
      directory, as IRI triples ``(h, r, t)``; their entities are entities of the
      inference graph.

    ``testing`` is ``None``: the test entities are not classes of the dataset, so a
    model built on the dataset has no (untrained) vector for them. Validation triples
    use training entities only.

    The files are downloaded from https://github.com/kkteru/grail at a fixed commit
    (:data:`COMMIT`; no licence is stated) and converted once into the cache directory
    ``<data_root>/inductive_kg/<name>_<version>``. The same data are in
    :mod:`pykeen.datasets.inductive.ilp_teru`, which the authors of RESHUFFLE use.

    :param name: ``"FB15k237"``, ``"WN18RR"`` or ``"NELL995"``.
    :param version: ``"v1"`` to ``"v4"``.
    :param data_root: Where to cache the files (default: :func:`default_data_root`).
    """

    def __init__(self, name="FB15k237", version="v1", data_root=None):
        if name not in NAMES:
            raise ValueError(f"name must be one of {NAMES}.")
        if version not in VERSIONS:
            raise ValueError(f"version must be one of {VERSIONS}.")
        self.name, self.version = name, version
        root = default_data_root() if data_root is None else data_root
        self.directory = os.path.join(root, "inductive_kg", f"{name}_{version}")
        paths = [os.path.join(self.directory, f"{part}.ofn") for part in ("ontology", "valid")]
        self._tsv = {part: os.path.join(self.directory, f"{part}.tsv")
                     for part in ("inference", "test")}
        if not all(os.path.exists(p) for p in (*paths, *self._tsv.values())):
            self._build(paths)
        super().__init__(*paths)
        self._inference_graph = None
        self._test_triples = None

    # -- construction -------------------------------------------------------------------
    def _raw(self, name):
        raw = os.path.join(self.directory, "raw")
        os.makedirs(raw, exist_ok=True)
        path = os.path.join(raw, name)
        if not os.path.exists(path):
            suffix, release_file = _RAW[name]
            url = f"{_BASE}{_DIRECTORY[self.name]}_{self.version}{suffix}/{release_file}"
            partial = f"{path}.{os.getpid()}.part"
            urllib.request.urlretrieve(url, partial)
            os.replace(partial, path)
        return path

    def _triples(self, name):
        return _read(self._raw(name))

    def _write_ofn(self, path, triples, declare_entities, declare_relations):
        partial = f"{path}.{os.getpid()}.part"
        with open(partial, "w", encoding="utf-8") as out:
            out.write("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n")
            out.write(f"Ontology(<{IRI_PREFIX}{self.name}_{self.version}/"
                      f"{os.path.basename(path)}>\n")
            for label in declare_entities:
                out.write(f"Declaration(Class(<{entity_iri(self.name, self.version, label)}>))\n")
            for label in declare_relations:
                out.write("Declaration(ObjectProperty("
                          f"<{relation_iri(self.name, self.version, label)}>))\n")
            for h, r, t in triples:
                out.write(f"SubClassOf(<{entity_iri(self.name, self.version, h)}> "
                          f"ObjectSomeValuesFrom(<{relation_iri(self.name, self.version, r)}> "
                          f"<{entity_iri(self.name, self.version, t)}>))\n")
            out.write(")\n")
        os.replace(partial, path)

    def _write_tsv(self, path, triples):
        partial = f"{path}.{os.getpid()}.part"
        with open(partial, "w", encoding="utf-8") as out:
            out.writelines(f"{h}\t{r}\t{t}\n" for h, r, t in triples)
        os.replace(partial, path)

    def _build(self, paths):
        os.makedirs(self.directory, exist_ok=True)
        train, valid = self._triples("train.txt"), self._triples("valid.txt")
        entities = sorted({x for h, _, t in train for x in (h, t)})
        relations = sorted({r for _, r, _ in train})
        self._write_ofn(paths[0], train, entities, relations)
        self._write_ofn(paths[1], valid, (), ())
        self._write_tsv(self._tsv["inference"], self._triples("inference.txt"))
        self._write_tsv(self._tsv["test"], self._triples("test.txt"))

    # -- the test graph -----------------------------------------------------------------
    def _iri_triples(self, path):
        return [(entity_iri(self.name, self.version, h),
                 relation_iri(self.name, self.version, r),
                 entity_iri(self.name, self.version, t)) for h, r, t in _read(path)]

    @property
    def inference_graph(self):
        """The facts of the test graph, an :class:`InductiveGraph`. Its entities are those
        of the test graph (the ``*_ind`` directory), in order of first occurrence."""
        if self._inference_graph is None:
            facts = self._iri_triples(self._tsv["inference"])
            entities = list(dict.fromkeys(x for h, _, t in facts for x in (h, t)))
            relations = list(dict.fromkeys(r for _, r, _ in facts))
            e = {x: i for i, x in enumerate(entities)}
            r = {x: i for i, x in enumerate(relations)}
            tensor = th.tensor([[e[h], r[x], e[t]] for h, x, t in facts],
                               dtype=th.long).reshape(-1, 3)
            self._inference_graph = InductiveGraph(tuple(entities), tuple(relations), tensor)
        return self._inference_graph

    @property
    def test_triples(self):
        """The triples to predict, as IRI triples ``(h, r, t)`` over the entities and
        relations of :attr:`inference_graph`."""
        if self._test_triples is None:
            self._test_triples = self._iri_triples(self._tsv["test"])
        return self._test_triples
