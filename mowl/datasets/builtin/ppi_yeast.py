import pathlib

from ..base import RemoteDataset, PathDataset
import math
import random
import numpy as np
import gzip
import os
from java.util import HashSet

DATA_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast.tar.gz'
SLIM_DATA_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast_slim.tar.gz'

class PPIYeastDataset(RemoteDataset):
    """
    This dataset represent protein--protein interactions on the yeast species.  The data used for this dataset consists of the `Gene Ontology <http://geneontology.org/docs/download-ontology/>`_ released on 20-10-2021 and protein interaction data found in `String Database <https://string-db.org/cgi/download?sessionId=bJLbZRTBIir4>`_ version 11.5. Protein interaction data was randomly split 90:5:5 across training, validation and testing ontologies and Gene Ontology functional annotations of proteins is part of the training ontology only. Protein interactions are represented as an axiom of the form :math:`protein_1 \sqsubseteq interacts\_with . protein_2.`
"""

    
    def __init__(self, url=None):
        super().__init__(url=DATA_URL if not url else url)
        self._evaluation_classes = None
        self._loaded_eval_data = False

    @property
    def evaluation_classes(self):
        """Classes that are used in evaluation
        """
        if self._loaded_eval_data:
            return self._evaluation_classes
        
        proteins = set()
        for owl_cls in self.classes:
            if "http://4932" in owl_cls:
                proteins.add(owl_cls)
        self._evaluation_classes = proteins
        self._loaded_eval_data = True
        proteins = list(proteins)
        proteins.sort()
        return proteins
    
    def get_evaluation_property(self):
        return "http://interacts_with"
    
class PPIYeastSlimDataset(PPIYeastDataset):
    """
    Reduced version of :class:`PPIYeastDataset`. Tranining ontology is built from the Slim Yeast subset of Gene Ontology.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(url=SLIM_DATA_URL)
