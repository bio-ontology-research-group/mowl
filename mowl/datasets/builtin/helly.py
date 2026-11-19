"""The Helly family :math:`\\mathcal{T}_k`: small propositional :math:`\\mathcal{EL}_\\bot`
ontologies that have no model by single axis-aligned boxes."""

import os
import tempfile

from mowl.datasets.base import PathDataset

HELLY_NS = "http://mowl.borg/geo#"


def build_tk_ontology(k, adapter=None):
    """Build :math:`\\mathcal{T}_k` with the OWL API adapter and return the ontology.

    For :math:`k \\ge 3`, :math:`\\mathcal{T}_k` has primitive classes
    :math:`X_1, \\dots, X_k`, defined classes

    .. math:: A_i \\equiv \\bigsqcap_{j \\ne i} X_j \\qquad (i = 1, \\dots, k)

    and the disjointness axioms :math:`A_i \\sqcap A_j \\sqsubseteq \\bot`
    (:math:`i < j`). Every :math:`k - 1` of the :math:`X_j` are jointly satisfiable but
    all :math:`k` are not, so (Helly number 2 of axis-aligned boxes) no single-box
    embedding is a non-degenerate model. For :math:`k = 3` this is the counterexample of
    Leemhuis and Kutz (NeSy 2025): three pairwise compatible but jointly incompatible
    concepts, identical to ``tests/geometric/fixtures/t3.owl`` up to that fixture's
    extra isolated class ``Z``.

    For :math:`k > 3` the :math:`(k-1)`-ary conjunctions are written as left-nested
    binary conjunctions with named prefix classes ``Aux_1_2``, ``Aux_1_2_3``, ... (as in
    ``examples/elmodels/plot_4_multibox_structural_tk.py``), so that normalisation adds
    no fresh names.
    """
    from mowl.owlapi import OWLAPIAdapter
    if k < 3:
        raise ValueError("T_k is defined for k >= 3.")
    adapter = adapter or OWLAPIAdapter()
    mgr = adapter.owl_manager
    ont = adapter.create_ontology(f"http://mowl.borg/geo/t{k}")
    xs = [adapter.create_class(f"{HELLY_NS}X{j + 1}") for j in range(k)]
    as_ = [adapter.create_class(f"{HELLY_NS}A{i + 1}") for i in range(k)]
    cache = {}

    def prefix(indices):
        if len(indices) == 1:
            return xs[indices[0]]
        if indices not in cache:
            aux = adapter.create_class(
                f"{HELLY_NS}Aux_" + "_".join(str(i + 1) for i in indices))
            conj = adapter.create_object_intersection_of(prefix(indices[:-1]),
                                                         xs[indices[-1]])
            mgr.addAxiom(ont, adapter.create_equivalent_classes(aux, conj))
            cache[indices] = aux
        return cache[indices]

    for i in range(k):
        others = tuple(j for j in range(k) if j != i)
        rhs = adapter.create_object_intersection_of(prefix(others[:-1]), xs[others[-1]])
        mgr.addAxiom(ont, adapter.create_equivalent_classes(as_[i], rhs))
    bot = adapter.data_factory.getOWLNothing()
    for i in range(k):
        for j in range(i + 1, k):
            mgr.addAxiom(ont, adapter.create_subclass_of(
                adapter.create_object_intersection_of(as_[i], as_[j]), bot))
    return ont


class HellyTkDataset(PathDataset):
    """:math:`\\mathcal{T}_k` (see :func:`build_tk_ontology`) as a dataset with only a
    training ontology. The ontology is written in OWL functional syntax to ``directory``
    (default: a new directory under ``$TMPDIR``, or ``~/tmp``).

    :param k: Number of primitive classes, at least 3.
    :param directory: Where to write ``t<k>.owl``.
    """

    def __init__(self, k=3, directory=None):
        import java.io
        from org.semanticweb.owlapi.formats import FunctionalSyntaxDocumentFormat
        from org.semanticweb.owlapi.model import IRI
        from mowl.owlapi import OWLAPIAdapter
        self.k = int(k)
        if directory is None:
            base = os.environ.get("TMPDIR") or os.path.expanduser("~/tmp")
            os.makedirs(base, exist_ok=True)
            directory = tempfile.mkdtemp(prefix="mowl_helly_", dir=base)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"t{self.k}.owl")
        adapter = OWLAPIAdapter()
        ont = build_tk_ontology(self.k, adapter)
        adapter.owl_manager.saveOntology(ont, FunctionalSyntaxDocumentFormat(),
                                         IRI.create(java.io.File(path).toURI()))
        super().__init__(path)

    @property
    def evaluation_classes(self):
        return self.classes, self.classes
