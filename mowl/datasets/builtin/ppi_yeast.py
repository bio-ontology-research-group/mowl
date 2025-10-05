from ..base import RemoteDataset, OWLClasses
from deprecated.sphinx import versionchanged


YEAST_DATA_URL = 'https://bio2vec.net/data/mowl/ppi_yeast.tar.gz'
YEAST_SLIM_DATA_URL = 'https://bio2vec.net/data/mowl/ppi_yeast_slim.tar.gz'

HUMAN_DATA_URL = 'https://bio2vec.net/data/mowl/ppi_human.tar.gz'

class PPIYeastDataset(RemoteDataset):
    """
    Dataset containing protein-protein interactions in yeast.

    The data used for this dataset consists of the \
    `Gene Ontology <http://geneontology.org/docs/download-ontology/>`_ released on 20-10-2021 and \
    protein interaction data found in \
    `String Database <https://string-db.org/cgi/download?sessionId=bJLbZRTBIir4>`_ version 11.5. \
    Protein interaction data was randomly split 90:5:5 across training, validation and testing \
    ontologies and Gene Ontology functional annotations of proteins is part of the training \
    ontology only. Protein interactions are represented as an axiom of the form \
    :math:`protein_1 \sqsubseteq interacts\_with .protein_2.`

    This dataset was used as a benchmark in [chen2025]_.

    """

    def __init__(self, url=None):
        super().__init__(url=YEAST_DATA_URL if not url else url)

    @versionchanged(reason="Return pair of classes for evaluation", version="0.4.0")
    @property
    def evaluation_classes(self):
        """Classes that are used in evaluation
        """

        if self._evaluation_classes is None:
            proteins = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "http://4932" in owl_name:
                    proteins.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(proteins)

        return self._evaluation_classes

    def get_evaluation_property(self):
        return "http://interacts_with"


class PPIYeastSlimDataset(PPIYeastDataset):
    """
    Reduced version of :class:`PPIYeastDataset`. 

    Tranining ontology is built from the Slim Yeast subset of Gene Ontology.

    """

    def __init__(self):
        super().__init__(url=YEAST_SLIM_DATA_URL)



class PPIHumanDataset(RemoteDataset):
    """
    Dataset containing protein-protein interactions in human. This dataset was used as benchmark in [chen2025]_

    """

    def __init__(self, url=None):
        super().__init__(url=HUMAN_DATA_URL if not url else url)

    @versionchanged(reason="Return pair of classes for evaluation", version="0.4.0")
    @property
    def evaluation_classes(self):
        """Classes that are used in evaluation
        """

        if self._evaluation_classes is None:
            proteins = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "http://9606" in owl_name:
                    proteins.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(proteins)

        return self._evaluation_classes

    def get_evaluation_property(self):
        return "http://interacts_with"

