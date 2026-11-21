"""Knowledge graph completion benchmarks as mOWL datasets."""

import os
from urllib.parse import quote

from mowl.datasets.base import PathDataset, default_data_root

IRI_PREFIX = "http://mowl.borg/kg/"
#: Datasets of :mod:`pykeen.datasets` that are offered here.
NAMES = ("FB15k237", "WN18RR", "YAGO310", "Kinships", "UMLS", "Nations", "CoDExSmall")


class KnowledgeGraphDataset(PathDataset):
    """A knowledge graph completion benchmark of :mod:`pykeen.datasets` as an ontology.

    A triple ``(h, r, t)`` becomes the axiom ``h ⊑ ∃r.t`` between two named classes, the
    encoding mOWL uses for protein-protein interactions (an entity is a class, as a nominal
    would be). In the EL normal forms this is ``gci2``, so every method that scores
    ``C ⊑ ∃r.D`` can be trained and evaluated on it, and a graph projection of the ontology
    gives back the triples. Training, validation and test triples are the splits of PyKEEN
    (FB15k-237: 272,115 / 17,535 / 20,466; WN18RR: 86,835 / 3,034 / 3,034, of which the
    validation and test triples with entities unseen in training are dropped by PyKEEN).

    PyKEEN downloads the original files; mOWL converts them once.

    :param name: One of :data:`NAMES`.
    :param data_root: Where to cache the converted files.
    """

    def __init__(self, name="FB15k237", data_root=None):
        if name not in NAMES:
            raise ValueError(f"name must be one of {NAMES}.")
        self.name = name
        root = default_data_root() if data_root is None else data_root
        self.directory = os.path.join(root, "kg", name)
        paths = [os.path.join(self.directory, f"{part}.ofn")
                 for part in ("ontology", "valid", "test")]
        if not all(os.path.exists(p) for p in paths):
            self._build(paths)
        super().__init__(*paths)

    @staticmethod
    def entity_iri(name, label):
        return f"{IRI_PREFIX}{name}/entity/{quote(str(label), safe='')}"

    @staticmethod
    def relation_iri(name, label):
        return f"{IRI_PREFIX}{name}/relation/{quote(str(label), safe='')}"

    def _build(self, paths):
        from pykeen.datasets import get_dataset
        data = get_dataset(dataset=self.name)
        entities = {i: self.entity_iri(self.name, label)
                    for label, i in data.training.entity_to_id.items()}
        relations = {i: self.relation_iri(self.name, label)
                     for label, i in data.training.relation_to_id.items()}
        os.makedirs(self.directory, exist_ok=True)
        parts = (data.training, data.validation, data.testing)
        for path, factory, declare in zip(paths, parts, (True, False, False)):
            partial = f"{path}.{os.getpid()}.part"
            with open(partial, "w", encoding="utf-8") as out:
                out.write("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n")
                out.write(f"Ontology(<{IRI_PREFIX}{self.name}/{os.path.basename(path)}>\n")
                if declare:
                    for iri in entities.values():
                        out.write(f"Declaration(Class(<{iri}>))\n")
                    for iri in relations.values():
                        out.write(f"Declaration(ObjectProperty(<{iri}>))\n")
                for h, r, t in factory.mapped_triples.tolist():
                    out.write(f"SubClassOf(<{entities[h]}> ObjectSomeValuesFrom("
                              f"<{relations[r]}> <{entities[t]}>))\n")
                out.write(")\n")
            os.replace(partial, path)
