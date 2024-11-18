import pathlib

from ..base import RemoteDataset, PathDataset, OWLClasses
import math
import random
import numpy as np
import gzip
import os
from java.util import HashSet

DATA_HUMAN_URL = 'https://bio2vec.net/data/mowl/gda_human.tar.gz'
DATA_HUMAN_EL_URL = 'https://bio2vec.net/data/mowl/gda_human_el.tar.gz'
DATA_MOUSE_URL = 'https://bio2vec.net/data/mowl/gda_mouse.tar.gz'
DATA_MOUSE_EL_URL = 'https://bio2vec.net/data/mowl/gda_mouse_el.tar.gz'


class GDADataset(RemoteDataset):
    """Abstract class for gene-disease association datasets.

    This dataset represents the \
    gene-disease association in a particular species. This dataset is built using phenotypic \
    annotations of genes and diseases. For genes annotations we used the `Mouse/Human Orthology \
    with Phenotype Annotations \
    <http://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt>`_ document. Disease \
    annotations were obtained from the \
    `HPO annotations for rare disease <http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa>`_ \
    document. These annotations were added to the *Unified Phenotype Ontology* (uPheno) to build \
    the training ontology. Futhermore, gene-disease associations were obtained from the \
    `Associations of Mouse Genes with DO Diseases \
    <http://www.informatics.jax.org/downloads/reports/MGI_DO.rpt>`_ file, from which associations \
    for human and mouse were extracted (to build separate datasets) and each of them were \
    randomly split 80:10:10, added to the training ontology and created the validation and \
    testing ontologies, respectively.
    """

    def __init__(self, url=None):
        super().__init__(url=url)

    @property
    def evaluation_classes(self):
        """List of classes used for evaluation. Depending on the dataset, this method could \
        return a single :class:`OWLClasses` object \
        (as in :class:`PPIYeastDataset <mowl.datasets.builtin.PPIYeastDataset>`) \
        or a tuple of :class:`OWLClasses` objects \
        (as in :class:`GDAHumanDataset <mowl.datasets.builtin.GDAHumanDataset>`). If not \
        overriden, this method returns the classes in the testing ontology obtained from the \
        OWLAPI method ``getClassesInSignature()`` as a :class:`OWLClasses` object.
        """

        if self._evaluation_classes is None:
            genes = set()
            diseases = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if owl_name[7:].isnumeric():
                    genes.add(owl_cls)
                if "OMIM_" in owl_name:
                    diseases.add(owl_cls)

            genes = OWLClasses(genes)
            diseases = OWLClasses(diseases)
            self._evaluation_classes = (genes, diseases)

        return self._evaluation_classes

    def get_evaluation_property(self):
        return "http://is_associated_with"


class GDAHumanDataset(GDADataset):
    """
    Dataset containing gene-disease associations in human.
    """

    
    def __init__(self):
        super().__init__(url=DATA_HUMAN_URL)


class GDAHumanELDataset(GDADataset):
    """This dataset is a reduced version of :class:`GDAHumanDataset`. The training ontology \
    contains axioms in the :math:`\mathcal{EL}` language.
    """

    def __init__(self):
        super().__init__(url=DATA_HUMAN_EL_URL)


class GDAMouseDataset(GDADataset):
    """
    Dataset containing gene-disease associations in mouse.
    """
    
    def __init__(self):
        super().__init__(url=DATA_MOUSE_URL)


class GDAMouseELDataset(GDADataset):
    """This dataset is a reduced version of :class:`GDAMouseDataset`. The training ontology \
    contains axioms in the :math:`\mathcal{EL}` language.
    """

    def __init__(self):
        super().__init__(url=DATA_MOUSE_EL_URL)
