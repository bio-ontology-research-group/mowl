from mowl.datasets import RemoteDataset, PathDataset, OWLClasses

HPI_URL = 'https://bio2vec.net/data/mowl/hpi.tar.gz'

class HPIDataset(RemoteDataset):

    def __init__(self, url=HPI_URL):
        super().__init__(url=url)

    @property
    def evaluation_classes(self):
        
        if self._evaluation_classes is None:
            genes = set()
            viruses = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                
                if "mowl.borg" in owl_name:
                    genes.add(owl_cls)
                if "NCBITaxon_" in owl_name:
                    viruses.add(owl_cls)

            genes = OWLClasses(genes)
            viruses = OWLClasses(viruses)
            self._evaluation_classes = (genes, viruses)

        return self._evaluation_classes

    @property
    def evaluation_object_property(self):
        return "http://mowl.borg/associated_with"

