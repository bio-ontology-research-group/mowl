from ..base import RemoteDataset, PathDataset

DATA_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/family.tar.gz'


class FamilyDataset(RemoteDataset):
    """This dataset represents a family domain. It is a short ontology with 12 axioms."""

    def __init__(self, url=None):
        super().__init__(url=DATA_URL if not url else url)
        self._evaluation_classes = None
        self._loaded_eval_data = False

    @property
    def evaluation_classes(self):
        return self.classes
