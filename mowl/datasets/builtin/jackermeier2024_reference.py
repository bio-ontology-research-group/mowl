"""Benchmarks of Jackermeier, Chen and Horrocks (WWW 2024) that mOWL does not otherwise
ship: the "approximating deductive reasoning" data (Section 4.4) and the PPI data used in
Section 4.3, both read from the data release of the reference implementation
(``data.zip`` of `KRR-Oxford/BoxSquaredEL <https://github.com/KRR-Oxford/BoxSquaredEL>`_,
commit ``a715080``).

The archive is downloaded once into the data root, its SHA-256 is checked, and the
relevant files are converted into ``ontology.owl`` / ``valid.owl`` / ``test.owl`` in
OWL functional syntax. Nothing is generated: every axiom comes from the release.
"""

import hashlib
import json
import os
import re
import zipfile

from mowl.datasets.base import PathDataset, RemoteDataset, default_data_root, OWLClasses

#: Pinned URL of the reference data release.
BOX2EL_DATA_URL = ("https://raw.githubusercontent.com/KRR-Oxford/BoxSquaredEL/"
                   "a715080aa78e964b31d267da84906cc8bb1e9632/data.zip")
#: SHA-256 of that file (checked after download).
BOX2EL_DATA_SHA256 = "75530bf242ac915ac6dd487eb297cf2af99af427a720d29e9700702f988b09b2"

_HEADER = ("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n"
           "Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)\n"
           "Ontology(<{iri}>\n")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_box2el_release(data_root=None, zip_path=None):
    """Return the path of the reference ``data.zip``, downloading it if needed.

    :param zip_path: Use this local copy instead of downloading.
    :raises ValueError: If the checksum does not match the pinned release.
    """
    if zip_path is None:
        root = default_data_root() if data_root is None else data_root
        os.makedirs(root, exist_ok=True)
        zip_path = os.path.join(root, "box2el_reference_data.zip")
        if not os.path.exists(zip_path):
            partial = f"{zip_path}.{os.getpid()}.part"
            try:
                with RemoteDataset._session() as session:
                    with session.get(BOX2EL_DATA_URL, stream=True, timeout=(10, 60)) as req:
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
    if digest != BOX2EL_DATA_SHA256:
        raise ValueError(f"{zip_path} has SHA-256 {digest}, expected the pinned release "
                         f"{BOX2EL_DATA_SHA256}.")
    return zip_path


_OWL_BUILTINS = {"owl:Thing": "http://www.w3.org/2002/07/owl#Thing",
                 "owl:Nothing": "http://www.w3.org/2002/07/owl#Nothing"}


def reference_class_order(ontology, iri_of_index=None, data_root=None, zip_path=None):
    """Classes of the reference prediction data (``data/<ONTOLOGY>/prediction/
    classes.json``) in the order of their indices, as IRIs.

    The reference code ranks candidates by index; equal scores are therefore ordered by
    this index (see :class:`~mowl.evaluation.normal_form_ranking.NormalFormRankingRegime`,
    ``candidate_order="reference"``).

    :param ontology: ``"GALEN"``, ``"GO"`` or ``"ANATOMY"``.
    :param iri_of_index: Optional ``f(index) -> IRI`` for datasets whose IRIs are derived
        from the reference index (ANATOMY); ``owl:Thing``/``owl:Nothing`` are mapped to the
        OWL built-ins there. Otherwise the names of ``classes.json`` without angle
        brackets are used.
    """
    with zipfile.ZipFile(fetch_box2el_release(data_root, zip_path)) as zf:
        index = json.loads(zf.read(f"data/{ontology}/prediction/classes.json"))
    out = []
    for name in sorted(index, key=index.get):
        if iri_of_index is not None:
            out.append(_OWL_BUILTINS.get(name) or iri_of_index(index[name]))
        else:
            out.append(name[1:-1] if name.startswith("<") and name.endswith(">") else name)
    return out


def _write_ontology(path, iri, axiom_lines):
    tmp = f"{path}.{os.getpid()}.part"
    with open(tmp, "w") as out:
        out.write(_HEADER.format(iri=iri))
        for line in axiom_lines:
            line = line.strip()
            if line:
                out.write(line + "\n")
        out.write(")\n")
    os.replace(tmp, path)


_EMPTY_CHAIN = re.compile(r"ObjectPropertyChain\(\s*\)")


def _lines(zf, member):
    """Lines of an archive member. Role chains with an empty chain (``SubObjectPropertyOf(
    ObjectPropertyChain( ) r)``, four in ANATOMY) are not valid OWL; they are skipped, as
    the reference loader does."""
    with zf.open(member) as f:
        for raw in f:
            line = raw.decode("utf-8")
            if _EMPTY_CHAIN.search(line):
                continue
            yield line


#: Prefix of the IRIs given to normalisation helper classes by
#: :func:`separate_normalisation_helpers`.
HELPER_PREFIX = "http://mowl.borg/jackermeier2024/helper#"


def _named(expression):
    return not expression.isAnonymous() and not expression.isOWLThing() \
        and not expression.isOWLNothing()


def find_normalisation_helpers(ontologies):
    """The classes that the normalisation of the Box\\ :sup:`2`\\ EL release introduced as
    helpers, recognised by their pattern of use.

    A definition :math:`D \\equiv B \\sqcap \\exists r.C` is normalised to
    :math:`\\exists r.C \\sqsubseteq X` and :math:`B \\sqcap X \\sqsubseteq D` with a new
    class :math:`X`. In the release, :math:`X` carries the IRI of an existing, unrelated
    class (`issue 178 <https://github.com/bio-ontology-research-group/mowl/issues/178>`_).
    A class is taken to be such a helper if, over all the given ontologies, it is the
    superclass of exactly one axiom :math:`\\exists r.C \\sqsubseteq X` and a conjunct of
    exactly one axiom :math:`A \\sqcap B \\sqsubseteq D` (:math:`D \\neq \\bot`).

    :param ontologies: The training, validation and testing ontologies (``None`` entries
        are skipped).
    :return: ``(helpers, occurrences)``: the set of IRIs, and for every ontology the list
        of ``(axiom, kind)`` with ``kind`` in ``("head", "conjunct")`` in which a helper
        occurs in its helper role.
    """
    from org.semanticweb.owlapi.model import AxiomType, ClassExpressionType
    heads, conjuncts, found = {}, {}, []
    for ontology in ontologies:
        rows = []
        if ontology is not None:
            for axiom in ontology.getAxioms(AxiomType.SUBCLASS_OF):
                sub, sup = axiom.getSubClass(), axiom.getSuperClass()
                if not sub.isAnonymous() or sup.isAnonymous():
                    continue
                kind = sub.getClassExpressionType()
                if kind == ClassExpressionType.OBJECT_SOME_VALUES_FROM and _named(sup):
                    iri = str(sup.asOWLClass().toStringID())
                    heads[iri] = heads.get(iri, 0) + 1
                    rows.append((axiom, "head", (iri,)))
                elif kind == ClassExpressionType.OBJECT_INTERSECTION_OF \
                        and not sup.isOWLNothing():
                    iris = tuple(str(c.asOWLClass().toStringID())
                                 for c in sub.getOperandsAsList() if _named(c))
                    for iri in iris:
                        conjuncts[iri] = conjuncts.get(iri, 0) + 1
                    rows.append((axiom, "conjunct", iris))
        found.append(rows)
    helpers = {iri for iri, n in heads.items() if n == 1 and conjuncts.get(iri, 0) == 1
               and not iri.startswith(HELPER_PREFIX)}       # already separated
    occurrences = [[(axiom, kind) for axiom, kind, iris in rows
                    if any(i in helpers for i in iris)] for rows in found]
    return helpers, occurrences


def separate_normalisation_helpers(ontologies):
    """Give every normalisation helper of the Box\\ :sup:`2`\\ EL release an IRI of its
    own, in place.

    In the two axioms that use a class as a helper (see
    :func:`find_normalisation_helpers`) its IRI is replaced by
    :data:`HELPER_PREFIX` followed by the original IRI; every other axiom keeps the
    original class. The number of axioms and the split do not change.

    Checked with ELK on the training ontologies including the disjointness axioms: after
    the separation 19 of 53,730 classes of GO, 0 of 33,734 of GALEN and 22 of 102,574 of
    Anatomy are unsatisfiable, against 25,710 (GO) and 74,917 (Anatomy) before.

    :return: The set of helper IRIs (the original ones).
    """
    from java.util import HashSet
    from org.semanticweb.owlapi.model import IRI
    helpers, occurrences = find_normalisation_helpers(ontologies)
    for ontology, rows in zip(ontologies, occurrences):
        if ontology is None or not rows:
            continue
        manager = ontology.getOWLOntologyManager()
        factory = manager.getOWLDataFactory()

        def fresh(cls):
            if cls.isAnonymous():
                return cls
            iri = str(cls.asOWLClass().toStringID())
            return factory.getOWLClass(IRI.create(HELPER_PREFIX + iri)) \
                if iri in helpers else cls

        removed, added = HashSet(), HashSet()
        for axiom, kind in rows:
            sub, sup = axiom.getSubClass(), axiom.getSuperClass()
            if kind == "head":
                new = factory.getOWLSubClassOfAxiom(sub, fresh(sup))
            else:
                operands = HashSet()
                for operand in sub.getOperandsAsList():
                    operands.add(fresh(operand))
                if operands.size() < 2:
                    continue
                new = factory.getOWLSubClassOfAxiom(
                    factory.getOWLObjectIntersectionOf(operands), sup)
            removed.add(axiom)
            added.add(new)
        manager.removeAxioms(ontology, removed)
        manager.addAxioms(ontology, added)
    return helpers


class _Box2ELReferenceDataset(PathDataset):

    def __init__(self, name, data_root=None, zip_path=None):
        root = default_data_root() if data_root is None else data_root
        self.dataset_dir = os.path.join(root, name)
        paths = [os.path.join(self.dataset_dir, f)
                 for f in ("ontology.owl", "valid.owl", "test.owl")]
        if not all(os.path.exists(p) for p in paths):
            os.makedirs(self.dataset_dir, exist_ok=True)
            with zipfile.ZipFile(fetch_box2el_release(root, zip_path)) as zf:
                self._convert(zf, *paths)
        super().__init__(*paths)

    def _convert(self, zf, ontology_path, valid_path, test_path):
        raise NotImplementedError


class Jackermeier2024InferencesDataset(_Box2ELReferenceDataset):
    """"Approximating deductive reasoning" benchmark of [jackermeier2024]_ (Section 4.4,
    Table 6) for GALEN, GO or ANATOMY.

    * Training ontology: the full normalised ontology of the release
      (``<ONT>_norm_full.owl``, all asserted axioms).
    * Test / validation: the atomic subsumptions :math:`C \\sqsubseteq D` inferred by an
      :math:`\\mathcal{EL}^{++}` reasoner and not asserted (``inferences.json`` 90% and
      ``val.json`` 10%, split by the authors with seed 100).

    Differences from the reference loader: the reference parses the training file line
    by line and only understands ``SubClassOf`` and ``SubObjectPropertyOf``; the
    ANATOMY file also contains 21 ``EquivalentClasses(... ObjectOneOf(...))`` and 63
    ``ObjectPropertyRange`` axioms, which the reference misreads as atomic subsumptions
    between malformed names. mOWL loads the file with the OWL API, so these axioms keep
    their meaning (and the class count differs slightly from the reference). Four role
    chain axioms of ANATOMY with an empty chain are skipped, as by the reference loader.

    :param ontology: ``"GALEN"``, ``"GO"`` or ``"ANATOMY"``.
    :param data_root: Cache directory (default :func:`~mowl.datasets.base.default_data_root`).
    :param zip_path: Local copy of the reference ``data.zip`` (checksum-verified).
    """

    ONTOLOGIES = ("GALEN", "GO", "ANATOMY")

    def __init__(self, ontology="GALEN", data_root=None, zip_path=None):
        ontology = ontology.upper()
        if ontology not in self.ONTOLOGIES:
            raise ValueError(f"ontology must be one of {self.ONTOLOGIES}")
        self.ontology_name = ontology
        super().__init__(f"box2el_inferences_{ontology.lower()}", data_root, zip_path)

    def _convert(self, zf, ontology_path, valid_path, test_path):
        o = self.ontology_name
        base = f"data/{o}/inferences/"
        iri = f"http://mowl.borg/benchmarks/box2el/{o.lower()}/inferences"
        _write_ontology(ontology_path, iri + "/train", _lines(zf, base + f"{o}_norm_full.owl"))
        for member, path in (("val.json", valid_path), ("inferences.json", test_path)):
            pairs = json.loads(zf.read(base + member).decode("utf-8"))
            _write_ontology(path, f"{iri}/{member[:-5]}",
                            (f"SubClassOf({c} {d})" for c, d in pairs))

    @property
    def evaluation_classes(self):
        """All classes of the dataset, as heads and as tails."""
        return self.classes, self.classes


class Jackermeier2024PPIDataset(_Box2ELReferenceDataset):
    """PPI prediction data used by [jackermeier2024]_ (Section 4.3, Table 5): the yeast or
    human ontology of ELEmbeddings (GO plus STRING interactions and protein functions),
    as distributed in the reference release (``data/PPI/<organism>``).

    * Training ontology: ``train/<organism>.owl`` (proteins are classes; training
      interactions are axioms :math:`P_1 \\sqsubseteq \\exists \\mathit{interacts}.P_2`).
    * Validation / test: ``val`` and ``test`` ``protein_links.txt``, converted to
      :math:`P_1 \\sqsubseteq \\exists \\mathit{interacts}.P_2` with
      ``http://interacts`` and protein IRIs ``http://<taxon>.<id>``.
    * Evaluation classes: the proteins of ``proteins.json`` (5,502 for yeast).

    This is not the same data as :class:`~mowl.datasets.builtin.PPIYeastDataset` (which
    has 12,040 test interactions against 21,560 here).

    :param organism: ``"yeast"`` or ``"human"``.
    """

    def __init__(self, organism="yeast", data_root=None, zip_path=None):
        if organism not in ("yeast", "human"):
            raise ValueError("organism must be 'yeast' or 'human'.")
        self.organism = organism
        super().__init__(f"box2el_ppi_{organism}", data_root, zip_path)
        self._proteins_path = os.path.join(self.dataset_dir, "proteins.txt")

    def _convert(self, zf, ontology_path, valid_path, test_path):
        base = f"data/PPI/{self.organism}/"
        iri = f"http://mowl.borg/benchmarks/box2el/ppi/{self.organism}"
        _write_ontology(ontology_path, iri + "/train",
                        _lines(zf, base + f"train/{self.organism}.owl"))
        for split, path in (("val", valid_path), ("test", test_path)):
            axioms = []
            for line in _lines(zf, base + f"{split}/protein_links.txt"):
                parts = line.split()
                if len(parts) == 2:
                    axioms.append(f"SubClassOf(<http://{parts[0]}> ObjectSomeValuesFrom("
                                  f"<http://interacts> <http://{parts[1]}>))")
            _write_ontology(path, f"{iri}/{split}", axioms)
        proteins = json.loads(zf.read(base + "proteins.json").decode("utf-8"))
        with open(os.path.join(self.dataset_dir, "proteins.txt"), "w") as f:
            for p in proteins:
                f.write(p.strip("<>") + "\n")

    @property
    def evaluation_classes(self):
        """The proteins of the release, as heads and as tails."""
        if self._evaluation_classes is None:
            with open(self._proteins_path) as f:
                wanted = {line.strip() for line in f if line.strip()}
            prots = [c for name, c in self.classes.as_dict.items() if name in wanted]
            self._evaluation_classes = OWLClasses(prots), OWLClasses(prots)
        return self._evaluation_classes

    def get_evaluation_property(self):
        return "http://interacts"
