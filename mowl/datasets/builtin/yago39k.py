"""YAGO39K and M-YAGO39K (Lv et al., EMNLP 2018) as an ontology with individuals."""

import os
import random
import urllib.request
from urllib.parse import quote

from mowl.datasets.base import PathDataset, default_data_root

#: The commit of https://github.com/davidlvxin/TransC the files are read from.
COMMIT = "63b33769eedd25c396581ca260abd484e5d6e370"
_BASE = f"https://raw.githubusercontent.com/davidlvxin/TransC/{COMMIT}/data/YAGO39K/"
IRI_PREFIX = "http://mowl.borg/yago39k/"
VARIANTS = ("yago39k", "m-yago39k")
#: Negatives are stored per split in this order of forms.
FORMS = ("class_assertion", "gci0", "object_property_assertion")

_TRAIN = ("Train/concept2id", "Train/instance2id", "Train/relation2id", "Train/instanceOf2id",
          "Train/subClassOf2id", "Train/triple2id")
_ISA = ("instanceOf2id_positive", "instanceOf2id_negative", "subClassOf2id_positive",
        "subClassOf2id_negative")
_REL = ("triple2id_positive", "triple2id_negative")


def _iri(kind, label):
    return f"{IRI_PREFIX}{kind}/{quote(label.strip('<>'), safe='')}"


class YAGO39KDataset(PathDataset):
    """YAGO39K and M-YAGO39K of TransC [lv2018transc]_ as an ontology with an ABox, the data of
    EIKE [wang2024eike]_ (the paper's Table 1).

    The release has 39,374 instances, 46,109 concepts and 37 relations (the paper's Table 1
    counts the header lines). Training data (the same for both variants): 354,996 relational
    triples ``(h, t, r)``, 437,836 InstanceOf pairs and 29,181 SubClassOf pairs. In the
    ontology a relational triple is ``ObjectPropertyAssertion(r h t)``, an InstanceOf pair
    ``ClassAssertion(C a)`` and a SubClassOf pair ``SubClassOf(C D)``; instances are named
    individuals, concepts are classes, so every instance, concept and relation of the
    release is declared in the training ontology.

    Validation and test ontologies hold the positives of the release: the relational
    triples (9,341 and 9,364), and the InstanceOf and SubClassOf pairs (5,000 / 1,000 for
    ``"yago39k"``, 8,650 / 1,187 for ``"m-yago39k"``, whose pairs are the ones implied by
    transitivity of SubClassOf). The release also provides the negatives that the paper's
    triple classification uses; :meth:`negatives` returns them.

    The files are downloaded from the authors' repository
    (https://github.com/davidlvxin/TransC) at a fixed commit and converted once. The
    repository states no licence; the data are a sample of YAGO (a knowledge base from
    Wikipedia, WordNet and GeoNames, distributed under Creative Commons Attribution
    licences), so check the YAGO terms before redistributing the converted files.

    :param variant: ``"yago39k"`` or ``"m-yago39k"``.
    :param data_root: Where to cache the files (default: :func:`default_data_root`).
    :param subsample: If given, keep only this many randomly chosen instances (with their
        triples, and the validation and test triples among them) for smoke tests; all
        concepts and SubClassOf pairs are kept. Cached separately.
    :param seed: Seed of the subsample.
    """

    def __init__(self, variant="yago39k", data_root=None, subsample=None, seed=0):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}.")
        self.variant, self.subsample, self.seed = variant, subsample, seed
        root = default_data_root() if data_root is None else data_root
        self.root = os.path.join(root, "yago39k")
        tag = variant + ("" if subsample is None else f"-s{int(subsample)}-{int(seed)}")
        self.directory = os.path.join(self.root, tag)
        paths = [os.path.join(self.directory, f"{part}.ofn")
                 for part in ("ontology", "valid", "test")]
        self._negative_paths = {s: os.path.join(self.directory, f"negatives_{s}.tsv")
                                for s in ("valid", "test")}
        if not all(os.path.exists(p) for p in paths + list(self._negative_paths.values())):
            self._build(paths)
        super().__init__(*paths)

    # -- download ------------------------------------------------------------------------
    def _raw(self, name):
        path = os.path.join(self.root, "raw", name + ".txt")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            partial = f"{path}.{os.getpid()}.part"
            urllib.request.urlretrieve(_BASE + name + ".txt", partial)
            os.replace(partial, path)
        return path

    def _rows(self, name, skip=0):
        with open(self._raw(name), encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= skip and line.strip():
                    yield line.rstrip("\n").split("\t" if "\t" in line else " ")

    def _names(self, name, kind):
        return {int(i): _iri(kind, label) for label, i in self._rows(name, skip=1)}

    # -- conversion ----------------------------------------------------------------------
    def _build(self, paths):
        concepts = self._names("Train/concept2id", "concept")
        instances = self._names("Train/instance2id", "instance")
        relations = self._names("Train/relation2id", "relation")
        keep = None
        if self.subsample is not None:
            rng = random.Random(self.seed)
            ids = sorted(instances)
            keep = set(rng.sample(ids, min(int(self.subsample), len(ids))))
            instances = {i: v for i, v in instances.items() if i in keep}

        def ok(*ids):
            return keep is None or all(i in keep for i in ids)

        def pairs(name, skip):
            return [(int(a), int(b)) for a, b in self._rows(name, skip)]

        def triples(name):  # the release lists (head, tail, relation)
            return [(int(h), int(t), int(r)) for h, t, r in self._rows(name, 1)]

        def assertions(isa_inst, isa_sub, rel):
            lines = [f"ClassAssertion(<{concepts[c]}> <{instances[i]}>)"
                     for i, c in isa_inst if ok(i)]
            lines += [f"SubClassOf(<{concepts[a]}> <{concepts[b]}>)" for a, b in isa_sub]
            lines += [f"ObjectPropertyAssertion(<{relations[r]}> <{instances[h]}> "
                      f"<{instances[t]}>)" for h, t, r in rel if ok(h, t)]
            return lines

        train = assertions(pairs("Train/instanceOf2id", 0), pairs("Train/subClassOf2id", 0),
                           triples("Train/triple2id"))
        header = ["Prefix(owl:=<http://www.w3.org/2002/07/owl#>)"]
        os.makedirs(self.directory, exist_ok=True)
        self._write(paths[0], header, [
            *(f"Declaration(Class(<{v}>))" for v in concepts.values()),
            *(f"Declaration(ObjectProperty(<{v}>))" for v in relations.values()),
            *(f"Declaration(NamedIndividual(<{v}>))" for v in instances.values()),
            *train])
        for split, path, folder, mfolder in (("valid", paths[1], "Valid", "M-Valid"),
                                             ("test", paths[2], "Test", "M-Test")):
            isa = mfolder if self.variant == "m-yago39k" else folder
            pos, neg = {}, {}
            for kind, store in (("positive", pos), ("negative", neg)):
                store["isa"] = pairs(f"{isa}/instanceOf2id_{kind}", 1)
                store["sub"] = pairs(f"{isa}/subClassOf2id_{kind}", 1)
                store["rel"] = triples(f"{folder}/triple2id_{kind}")
            self._write(path, header, assertions(pos["isa"], pos["sub"], pos["rel"]))
            rows = [f"class_assertion\t{instances[i]}\t{concepts[c]}"
                    for i, c in neg["isa"] if ok(i)]
            rows += [f"gci0\t{concepts[a]}\t{concepts[b]}" for a, b in neg["sub"]]
            rows += [f"object_property_assertion\t{instances[h]}\t{relations[r]}\t"
                     f"{instances[t]}" for h, t, r in neg["rel"] if ok(h, t)]
            self._atomic(self._negative_paths[split], "\n".join(rows) + "\n")

    @staticmethod
    def _atomic(path, text):
        partial = f"{path}.{os.getpid()}.part"
        with open(partial, "w", encoding="utf-8") as out:
            out.write(text)
        os.replace(partial, path)

    def _write(self, path, header, axioms):
        name = f"{IRI_PREFIX}{self.variant}/{os.path.basename(path)}"
        self._atomic(path, "\n".join(header + [f"Ontology(<{name}>"] + axioms + [")"]) + "\n")

    # -- the released negatives ----------------------------------------------------------
    def negatives(self, split, form):
        """The released negatives of ``split`` (``"valid"`` or ``"test"``) for one form, as
        tuples of IRIs in the column layout of mOWL's normal forms: ``(individual, class)``
        for ``"class_assertion"``, ``(subclass, superclass)`` for ``"gci0"`` and
        ``(subject, role, object)`` for ``"object_property_assertion"``. Positives and
        negatives are in equal numbers."""
        if form not in FORMS:
            raise ValueError(f"form must be one of {FORMS}.")
        out = []
        with open(self._negative_paths[split], encoding="utf-8") as f:
            for line in f:
                kind, *rest = line.rstrip("\n").split("\t")
                if kind == form:
                    out.append(tuple(rest))
        return out
