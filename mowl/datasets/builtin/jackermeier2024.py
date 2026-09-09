from ..base import RemoteDataset


DATA_GALEN_URL = 'https://bio2vec.net/data/mowl/galen_jackermeier2024.tar.gz'
DATA_GO_URL = 'https://bio2vec.net/data/mowl/go_jackermeier2024.tar.gz'
DATA_ANATOMY_URL = 'https://bio2vec.net/data/mowl/anatomy_jackermeier2024.tar.gz'


class GALENJackermeier2024Dataset(RemoteDataset):
    """Dataset containing the \
    `GALEN ontology <http://www.co-ode.org/ontologies/galen>`_ \
    (Rector et al., 1996) as the :math:`\mathcal{EL}^{++}` subsumption prediction benchmark \
    introduced by Jackermeier, Chen and Horrocks in \
    "Dual Box Embeddings for the Description Logic EL++" (WWW 2024, \
    `BoxSquaredEL repository <https://github.com/KRR-Oxford/BoxSquaredEL>`_).

    The axioms are provided already in the \
    :doc:`normal forms </embedding_el/index>` of the \
    :math:`\mathcal{EL}^{++}` language, split into training (80%), validation (10%) and \
    testing (10%) sets per normal form. The training ontology additionally contains the \
    role inclusion and role chain (transitive property) axioms of the ontology.

    The benchmark was also used by Yang et al., "TransBox: Geometric Interpretations and \
    Logical Constraints of Concept Box Embeddings for the Description Logic EL++" (WWW 2025).
    """

    def __init__(self, data_root=None):
        super().__init__(url=DATA_GALEN_URL, data_root=data_root)


class GOJackermeier2024Dataset(RemoteDataset):
    """Dataset containing the \
    `Gene Ontology (GO) <http://geneontology.org/>`_ \
    (Ashburner et al., 2000) as the :math:`\mathcal{EL}^{++}` subsumption prediction \
    benchmark introduced by Jackermeier, Chen and Horrocks in \
    "Dual Box Embeddings for the Description Logic EL++" (WWW 2024, \
    `BoxSquaredEL repository <https://github.com/KRR-Oxford/BoxSquaredEL>`_).

    The axioms are provided already in the \
    :doc:`normal forms </embedding_el/index>` of the \
    :math:`\mathcal{EL}^{++}` language, split into training (80%), validation (10%) and \
    testing (10%) sets per normal form. The training ontology additionally contains the \
    role inclusion and role chain (transitive property) axioms of the ontology.

    The benchmark was also used by Yang et al., "TransBox: Geometric Interpretations and \
    Logical Constraints of Concept Box Embeddings for the Description Logic EL++" (WWW 2025).
    """

    def __init__(self, data_root=None):
        super().__init__(url=DATA_GO_URL, data_root=data_root)


class AnatomyJackermeier2024Dataset(RemoteDataset):
    """Dataset containing the \
    `Anatomy ontology (Uberon) <https://uberonontology.org/>`_ \
    (Mungall et al., 2012) as the :math:`\mathcal{EL}^{++}` subsumption prediction \
    benchmark introduced by Jackermeier, Chen and Horrocks in \
    "Dual Box Embeddings for the Description Logic EL++" (WWW 2024, \
    `BoxSquaredEL repository <https://github.com/KRR-Oxford/BoxSquaredEL>`_).

    The axioms are provided already in the \
    :doc:`normal forms </embedding_el/index>` of the \
    :math:`\mathcal{EL}^{++}` language, split into training (80%), validation (10%) and \
    testing (10%) sets per normal form. The training ontology additionally contains the \
    role inclusion and role chain (transitive property) axioms of the ontology. The \
    anonymous class expressions that the benchmark labels with Manchester-syntax strings \
    (normal forms of complex axioms) are materialized as generated atomic classes under \
    ``http://bio2vec.net/data/mowl/anatomy_jackermeier2024/aux/``.

    The benchmark was also used by Yang et al., "TransBox: Geometric Interpretations and \
    Logical Constraints of Concept Box Embeddings for the Description Logic EL++" (WWW 2025).
    """

    def __init__(self, data_root=None):
        super().__init__(url=DATA_ANATOMY_URL, data_root=data_root)
