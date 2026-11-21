import os

from ..base import PathDataset, RemoteDataset, default_data_root
from .jackermeier2024_reference import (reference_class_order,
                                        separate_normalisation_helpers)


DATA_GALEN_URL = 'https://bio2vec.net/data/mowl/galen_jackermeier2024.tar.gz'
DATA_GO_URL = 'https://bio2vec.net/data/mowl/go_jackermeier2024.tar.gz'
DATA_ANATOMY_URL = 'https://bio2vec.net/data/mowl/anatomy_jackermeier2024.tar.gz'


_REPAIR_DOC = """

    .. warning::

       In the reference release the helper classes that the normalisation introduced
       carry the IRIs of existing, unrelated classes (`issue 178
       <https://github.com/bio-ontology-research-group/mowl/issues/178>`_), which makes
       tens of thousands of classes unsatisfiable. By default (``repaired=True``) mOWL
       gives every helper an IRI of its own
       (:func:`~mowl.datasets.builtin.jackermeier2024_reference.separate_normalisation_helpers`):
       the axioms, their number and the split stay the same, and the training ontology
       becomes a sound basis for reasoning. ``repaired=False`` serves the release as it
       is; the benchmarks that reproduce published numbers use it.

    :param repaired: Separate the normalisation helpers from the classes whose IRIs they
        share. Defaults to ``True``.
    :type repaired: bool, optional
    """


class _Jackermeier2024Dataset(RemoteDataset):
    """Base of the three datasets: the release, optionally with separated helpers."""

    URL = None

    def __init__(self, data_root=None, repaired=True):
        self.repaired = bool(repaired)
        if not self.repaired:
            super().__init__(url=self.URL, data_root=data_root)
            return
        root = default_data_root() if data_root is None else data_root
        name = self.URL.split("/")[-1].split(".")[0]
        directory = os.path.join(root, name, "repaired")
        paths = [os.path.join(directory, f) for f in ("ontology.owl", "valid.owl",
                                                      "test.owl")]
        if not all(os.path.exists(p) for p in paths):
            release = RemoteDataset(url=self.URL, data_root=data_root)
            ontologies = [release.ontology, release.validation, release.testing]
            separate_normalisation_helpers(ontologies)
            os.makedirs(directory, exist_ok=True)
            self._save(ontologies, paths)
        self.url, self.data_root = self.URL, root
        PathDataset.__init__(self, *paths)

    @staticmethod
    def _save(ontologies, paths):
        import java.io
        from org.semanticweb.owlapi.formats import FunctionalSyntaxDocumentFormat
        for ontology, path in zip(ontologies, paths):
            partial = f"{path}.{os.getpid()}.part"
            stream = java.io.FileOutputStream(partial)
            try:
                ontology.getOWLOntologyManager().saveOntology(
                    ontology, FunctionalSyntaxDocumentFormat(), stream)
            finally:
                stream.close()
            os.replace(partial, path)


class GALENJackermeier2024Dataset(_Jackermeier2024Dataset):
    """Dataset containing the \
    `GALEN ontology <http://www.co-ode.org/ontologies/galen>`_ \
    (Rector et al., 1996) as an :math:`\\mathcal{EL}^{++}` subsumption prediction benchmark.

    The axioms are provided already in the :doc:`normal forms </embedding_el/index>` of the \
    :math:`\\mathcal{EL}^{++}` language, split into training (80%), validation (10%) and \
    testing (10%) sets per normal form. The training ontology additionally contains the role \
    inclusion and role chain (transitive property) axioms of the ontology.

    The split is the one distributed with the reference implementation of \
    [jackermeier2024]_ (`KRR-Oxford/BoxSquaredEL \
    <https://github.com/KRR-Oxford/BoxSquaredEL>`_), which follows the subsumption \
    prediction protocol established by earlier :math:`\\mathcal{EL}^{++}` embedding work on \
    these three ontologies. It is also the benchmark used by Yang et al., "TransBox: \
    Geometric Interpretations and Logical Constraints of Concept Box Embeddings for the \
    Description Logic EL++" (WWW 2025).

    The axioms and the split are the frozen release of the reference benchmark, which \
    distributes them as integer-indexed splits plus the class and relation index files; \
    mOWL serves the same axioms serialized as OWL. The per-split normal form counts are \
    checked against the reference in ``tests/datasets/test_jackermeier2024_datasets.py``. \
    The terms of the underlying ontology and of the reference benchmark apply to this \
    redistribution.
    """

    URL = DATA_GALEN_URL

    def __init__(self, data_root=None, repaired=True):
        super().__init__(data_root=data_root, repaired=repaired)

    def reference_class_order(self):
        """Class IRIs in the index order of the reference release (``classes.json``),
        which decides ties in the reference evaluation. Reads the pinned ``data.zip`` of
        the Box2EL code (downloaded on first use)."""
        return reference_class_order("GALEN")


class GOJackermeier2024Dataset(_Jackermeier2024Dataset):
    """Dataset containing the \
    `Gene Ontology (GO) <http://geneontology.org/>`_ \
    (Ashburner et al., 2000) as an :math:`\\mathcal{EL}^{++}` subsumption prediction \
    benchmark.

    The axioms are provided already in the :doc:`normal forms </embedding_el/index>` of the \
    :math:`\\mathcal{EL}^{++}` language, split into training (80%), validation (10%) and \
    testing (10%) sets per normal form. The training ontology additionally contains the role \
    inclusion and role chain (transitive property) axioms of the ontology.

    The split is the one distributed with the reference implementation of \
    [jackermeier2024]_ (`KRR-Oxford/BoxSquaredEL \
    <https://github.com/KRR-Oxford/BoxSquaredEL>`_), which follows the subsumption \
    prediction protocol established by earlier :math:`\\mathcal{EL}^{++}` embedding work on \
    these three ontologies. It is also the benchmark used by Yang et al., "TransBox: \
    Geometric Interpretations and Logical Constraints of Concept Box Embeddings for the \
    Description Logic EL++" (WWW 2025).

    The axioms and the split are the frozen release of the reference benchmark, which \
    distributes them as integer-indexed splits plus the class and relation index files; \
    mOWL serves the same axioms serialized as OWL. The per-split normal form counts are \
    checked against the reference in ``tests/datasets/test_jackermeier2024_datasets.py``. \
    The terms of the underlying ontology and of the reference benchmark apply to this \
    redistribution.
    """

    URL = DATA_GO_URL

    def __init__(self, data_root=None, repaired=True):
        super().__init__(data_root=data_root, repaired=repaired)

    def reference_class_order(self):
        """Class IRIs in the index order of the reference release (``classes.json``),
        which decides ties in the reference evaluation. Reads the pinned ``data.zip`` of
        the Box2EL code (downloaded on first use)."""
        return reference_class_order("GO")


class AnatomyJackermeier2024Dataset(_Jackermeier2024Dataset):
    """Dataset containing the anatomy ontology of the :math:`\\mathcal{EL}^{++}` embedding \
    literature (an FMA-derived anatomy ontology, distributed as "ANATOMY" by the reference \
    benchmark) as a subsumption prediction benchmark.

    .. warning::
       **The class IRIs of this dataset are not anatomy identifiers.** The reference \
       benchmark does not ship class names for ANATOMY, so all of its 106,361 classes are \
       generated as opaque IRIs under \
       ``http://bio2vec.net/data/mowl/anatomy_jackermeier2024/aux/``. Ranking metrics over \
       the split are unaffected -- the axioms and the split are faithful -- but a prediction \
       cannot be mapped back to an anatomical term, and this dataset cannot be joined with \
       any other ontology. The object property IRIs *are* preserved. Use \
       :class:`GALENJackermeier2024Dataset` or :class:`GOJackermeier2024Dataset` when term \
       identity matters.

    The axioms are provided already in the :doc:`normal forms </embedding_el/index>` of the \
    :math:`\\mathcal{EL}^{++}` language, split into training (80%), validation (10%) and \
    testing (10%) sets per normal form. The training ontology additionally contains the role \
    inclusion and role chain (transitive property) axioms of the ontology.

    The split is the one distributed with the reference implementation of \
    [jackermeier2024]_ (`KRR-Oxford/BoxSquaredEL \
    <https://github.com/KRR-Oxford/BoxSquaredEL>`_), which follows the subsumption \
    prediction protocol established by earlier :math:`\\mathcal{EL}^{++}` embedding work on \
    these three ontologies. It is also the benchmark used by Yang et al., "TransBox: \
    Geometric Interpretations and Logical Constraints of Concept Box Embeddings for the \
    Description Logic EL++" (WWW 2025).

    The axioms and the split are the frozen release of the reference benchmark, which \
    distributes them as integer-indexed splits plus the class and relation index files; \
    mOWL serves the same axioms serialized as OWL. The per-split normal form counts are \
    checked against the reference in ``tests/datasets/test_jackermeier2024_datasets.py``. \
    The terms of the underlying ontology and of the reference benchmark apply to this \
    redistribution.
    """

    URL = DATA_ANATOMY_URL

    def __init__(self, data_root=None, repaired=True):
        super().__init__(data_root=data_root, repaired=repaired)

    def reference_class_order(self):
        """Class IRIs in the index order of the reference release (``classes.json``),
        which decides ties in the reference evaluation. Reads the pinned ``data.zip`` of
        the Box2EL code (downloaded on first use)."""
        base = "http://bio2vec.net/data/mowl/anatomy_jackermeier2024/aux/"
        return reference_class_order("ANATOMY", iri_of_index=lambda i: f"{base}{i}")


for _cls in (GALENJackermeier2024Dataset, GOJackermeier2024Dataset,
             AnatomyJackermeier2024Dataset):
    _cls.__doc__ = _cls.__doc__.rstrip() + _REPAIR_DOC
