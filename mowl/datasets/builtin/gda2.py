from mowl.datasets import RemoteDataset, PathDataset, OWLClasses
import os

GDA2_URL = 'https://bio2vec.net/data/mowl/gda2.tar.gz'
GDA2_EL_URL = 'https://bio2vec.net/data/mowl/gda2_el.tar.gz'

class GDADatasetV2(RemoteDataset):
    def __init__(self, url=GDA2_URL):
        super().__init__(url=url)

    @property
    def evaluation_classes(self):
        
        if self._evaluation_classes is None:
            genes = set()
            diseases = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                
                if "mowl.borg" in owl_name and owl_name.split("/")[-1].isnumeric():
                    genes.add(owl_cls)
                if "OMIM_" in owl_name:
                    diseases.add(owl_cls)

            genes = OWLClasses(genes)
            diseases = OWLClasses(diseases)
            self._evaluation_classes = (genes, diseases)

        return self._evaluation_classes

    @property
    def evaluation_object_property(self):
        return "http://mowl.borg/associated_with"

class GDADatasetV2EL(GDADatasetV2):
    def __init__(self, url=GDA2_EL_URL):
        super().__init__(url=url)
                                            


