import pathlib

from ..base import RemoteDataset, PathDataset
import math
import random
import numpy as np
import gzip
import os
from java.util import HashSet

DATA_HUMAN_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/gda_human.tar.gz'
DATA_HUMAN_EL_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/gda_human_el.tar.gz'
DATA_MOUSE_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/gda_mouse.tar.gz'
DATA_MOUSE_EL_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/gda_mouse_el.tar.gz'

class GDADataset(RemoteDataset):
    """Abstract class for Gene--Disease association datasets. This dataset represent the gene-disease association in a particular species. This dataset is built using phenotypic annotations of genes and diseases. For genes annotations we used the `Mouse/Human Orthology with Phenotype Annotations <http://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt>`_ document. Disease annotations were obtained from the `HPO annotations for rare disease <http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa>`_ document. These annotations were added to the *Unified Phenotype Ontology*(uPheno) to build the training ontology. 
Futhermore, gene-disease associations were obtained from the `Associations of Mouse Genes with DO Diseases <http://www.informatics.jax.org/downloads/reports/MGI_DO.rpt>`_ file, from which associations for human and mouse were extracted (to build separate datasets) and each of them were randomly split 80:10:10, added to the training ontology and created the validation and testing ontologies, respectively.
    """
    
    
    def __init__(self, url=None):
        super().__init__(url=url)

    def _get_evaluation_classes(self):
        """Classes that are used in evaluation
        """
        genes = set()
        diseases = set()
        for owl_cls in self.classes:
            if owl_cls[7:].isnumeric():
                genes.add(owl_cls)
            if "OMIM_" in owl_cls:
                diseases.add(owl_cls)
                
        return genes, diseases

    def get_evaluation_property(self):
        return "http://is_associated_with"
    
class GDAHumanDataset(GDADataset):
    def __init__(self):
        super().__init__(url=DATA_HUMAN_URL)

class GDAHumanELDataset(GDADataset):
    """This dataset is a reduced version of :class:`GDAHumanDataset`. The training ontology contains axioms in the :math:`\mathcal{EL}` language.
    """
    def __init__(self):
        super().__init__(url=DATA_HUMAN_EL_URL)

class GDAMouseDataset(GDADataset):
    def __init__(self):
        super().__init__(url=DATA_MOUSE_URL)

class GDAMouseELDataset(GDADataset):
    """This dataset is a reduced version of :class:`GDAMouseDataset`. The training ontology contains axioms in the :math:`\mathcal{EL}` language.
    """
    def __init__(self):
        super().__init__(url=DATA_MOUSE_EL_URL)
