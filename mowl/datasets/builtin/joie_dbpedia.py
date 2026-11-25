"""The DBpedia two-view knowledge graph of JOIE (Hao et al., KDD 2019), as used by Concept2Box."""

import os
import random
import urllib.request
from urllib.parse import quote

from mowl.datasets.base import PathDataset, default_data_root

#: The commit of https://github.com/JunhengH/joie-kdd19 (MIT licence) the files are read from.
COMMIT = "6ce57ae30fae67bf6e2f48e7e6db35dbab3e0201"
_BASE = f"https://raw.githubusercontent.com/JunhengH/joie-kdd19/{COMMIT}/data/dbpedia/"
IRI_PREFIX = "http://mowl.borg/joie-dbpedia/"
TASKS = ("kg-completion", "concept-linking")


def _iri(kind, label):
    return f"{IRI_PREFIX}{kind}/{quote(label, safe='')}"


def _split(items, seed, salt):
    """80% / 10% / 10% of a list, by a seeded permutation."""
    order = list(range(len(items)))
    random.Random(f"{seed}-{salt}").shuffle(order)
    n_valid = n_test = round(0.1 * len(items))
    n_train = len(items) - n_valid - n_test
    pick = lambda idx: [items[i] for i in idx]  # noqa: E731
    return pick(order[:n_train]), pick(order[n_train:n_train + n_valid]), \
        pick(order[n_train + n_valid:])


class JOIEDBpediaDataset(PathDataset):
    """The DBpedia data of JOIE [hao2019joie]_ as a two-view knowledge graph in an ontology,
    the public dataset of Concept2Box [huang2023]_.

    * **Instance view**: relational triples between entities, ``ObjectPropertyAssertion(r h
      t)``; 658,504 distinct triples (658,505 rows), 298 relations, 99,744 entities.
    * **Ontology view**: triples between concepts, ``SubClassOf(h ObjectSomeValuesFrom(r t))``
      (every one of the 20 relations of the view, ``isa`` among them, is treated alike, as
      in the paper); 675 distinct triples (763 rows) over 174 concepts
      (:attr:`ontology_view_concepts`).
    * **Cross-view links**: an entity and the concept it is typed with,
      ``ClassAssertion(C a)``; 99,748 links over 242 concepts (every concept of the
      release is a class).

    The paper (Section 5.2) splits the triples of each view at random 80% / 10% / 10% for
    knowledge graph completion, with all cross-view links known (``task="kg-completion"``),
    and splits the cross-view links 80% / 10% / 10% for concept linking, with all triples
    known (``task="concept-linking"``). The split is not released; this class draws it
    with a seed. In the training ontology every entity, concept and relation is declared.

    Statistics differ from the paper's Table 2 (111,762 entities, 305 relations, 863,643
    instance triples; 99,748 links): the release has fewer entities, relations and
    triples (the paper's table repeats the full DB111K-174 statistics of the JOIE paper;
    this release is the subset known as DB99K-242 by Yu et al., 2023, used by EIKE). The
    paper's Recipe dataset is not public.

    The files are read from the authors' repository
    (https://github.com/JunhengH/joie-kdd19, MIT licence) at a fixed commit and converted
    once. The data are a sample of DBpedia (CC BY-SA) with concept relations from
    ConceptNet (CC BY-SA 4.0).

    :param task: ``"kg-completion"`` or ``"concept-linking"``.
    :param seed: Seed of the random split.
    :param data_root: Where to cache the files (default: :func:`default_data_root`).
    :param subsample: If given, keep only this many randomly chosen entities (with their
        triples and links) for smoke tests. Cached separately.
    """

    def __init__(self, task="kg-completion", seed=0, data_root=None, subsample=None):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}.")
        self.task, self.seed, self.subsample = task, seed, subsample
        root = default_data_root() if data_root is None else data_root
        self.root = os.path.join(root, "joie_dbpedia")
        tag = f"{task}-{int(seed)}" + ("" if subsample is None else f"-s{int(subsample)}")
        self.directory = os.path.join(self.root, tag)
        paths = [os.path.join(self.directory, f"{part}.ofn")
                 for part in ("ontology", "valid", "test")]
        meta = os.path.join(self.directory, "ontology_view_concepts.txt")
        if not all(os.path.exists(p) for p in paths + [meta]):
            self._build(paths, meta)
        with open(meta, encoding="utf-8") as f:
            #: IRIs of the concepts of the ontology view: the candidates of its completion.
            self.ontology_view_concepts = [line.strip() for line in f if line.strip()]
        super().__init__(*paths)

    def _raw(self, name):
        path = os.path.join(self.root, "raw", name + ".txt")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            partial = f"{path}.{os.getpid()}.part"
            urllib.request.urlretrieve(_BASE + name + ".txt", partial)
            os.replace(partial, path)
        return path

    def _triples(self, name):
        with open(self._raw(name), encoding="utf-8") as f:
            rows = [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]
        return list(dict.fromkeys(rows))  # the release repeats some rows

    @staticmethod
    def _atomic(path, text):
        partial = f"{path}.{os.getpid()}.part"
        with open(partial, "w", encoding="utf-8") as out:
            out.write(text)
        os.replace(partial, path)

    def _build(self, paths, meta):
        instance = self._triples("db_insnet")      # (head entity, relation, tail entity)
        links = self._triples("db_InsType_mini")   # (entity, "type", concept)
        onto = self._triples("db_onto_small_mini")  # (head concept, relation, tail concept)
        entities = {e for h, _, t in instance for e in (h, t)} | {e for e, _, _ in links}
        if self.subsample is not None:
            rng = random.Random(self.seed)
            kept = set(rng.sample(sorted(entities), min(int(self.subsample), len(entities))))
            instance = [x for x in instance if x[0] in kept and x[2] in kept]
            links = [x for x in links if x[0] in kept]
            entities = kept
        concepts = {c for _, _, c in links} | {c for h, _, t in onto for c in (h, t)}
        relations = sorted({r for _, r, _ in instance})
        onto_relations = sorted({r for _, r, _ in onto})

        def ent(x):
            return f"<{_iri('entity', x)}>"

        def con(x):
            return f"<{_iri('concept', x)}>"

        def instance_axiom(h, r, t):
            return f"ObjectPropertyAssertion(<{_iri('relation', r)}> {ent(h)} {ent(t)})"

        def onto_axiom(h, r, t):
            return (f"SubClassOf({con(h)} ObjectSomeValuesFrom(<{_iri('concept-relation', r)}> "
                    f"{con(t)}))")

        def link_axiom(e, _, c):
            return f"ClassAssertion({con(c)} {ent(e)})"

        views = (instance_axiom, instance), (onto_axiom, onto), (link_axiom, links)
        parts = [[], [], []]  # training, validation, testing axioms
        for number, ((axiom, rows)) in enumerate(views):
            # kg-completion splits the triples of both views; concept-linking the links
            splitting = (number < 2) == (self.task == "kg-completion")
            chunks = _split(rows, self.seed, number) if splitting else (rows, [], [])
            for part, chunk in zip(parts, chunks):
                part.extend(axiom(*x) for x in chunk)

        os.makedirs(self.directory, exist_ok=True)
        header = ["Prefix(owl:=<http://www.w3.org/2002/07/owl#>)"]
        declarations = [
            *(f"Declaration(Class({con(c)}))" for c in sorted(concepts)),
            *(f"Declaration(ObjectProperty(<{_iri('relation', r)}>))" for r in relations),
            *(f"Declaration(ObjectProperty(<{_iri('concept-relation', r)}>))"
              for r in onto_relations),
            *(f"Declaration(NamedIndividual({ent(e)}))" for e in sorted(entities))]
        for index, (path, axioms) in enumerate(zip(paths, parts)):
            name = f"{IRI_PREFIX}{self.task}/{os.path.basename(path)}"
            body = (declarations if index == 0 else []) + axioms
            self._atomic(path, "\n".join(header + [f"Ontology(<{name}>"] + body + [")"]) + "\n")
        in_view = sorted({_iri("concept", c) for h, _, t in onto for c in (h, t)})
        self._atomic(meta, "\n".join(in_view) + "\n")
