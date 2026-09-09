from ..base import RemoteDataset


DATA_GALEN_URL = 'https://bio2vec.net/data/mowl/galen_jackermeier2024.tar.gz'
DATA_GO_URL = 'https://bio2vec.net/data/mowl/go_jackermeier2024.tar.gz'
DATA_ANATOMY_URL = 'https://bio2vec.net/data/mowl/anatomy_jackermeier2024.tar.gz'


class GALENJackermeier2024Dataset(RemoteDataset):
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

    def __init__(self, data_root=None):
        super().__init__(url=DATA_GALEN_URL, data_root=data_root)


class GOJackermeier2024Dataset(RemoteDataset):
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

    def __init__(self, data_root=None):
        super().__init__(url=DATA_GO_URL, data_root=data_root)


class AnatomyJackermeier2024Dataset(RemoteDataset):
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

    def __init__(self, data_root=None):
        super().__init__(url=DATA_ANATOMY_URL, data_root=data_root)
