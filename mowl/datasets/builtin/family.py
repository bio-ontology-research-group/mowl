from ..base import RemoteDataset, PathDataset

DATA_URL = 'https://bio2vec.net/data/mowl/family.tar.gz'


class FamilyDataset(RemoteDataset):
    r"""This dataset represents a family domain. 

    It is a short ontology with 12 axioms describing \
    family relationships. The axioms are:

    .. math::

        \begin{align}
            Male & \sqsubseteq Person \\
            Female & \sqsubseteq Person \\
            Father & \sqsubseteq Male \\
            Mother & \sqsubseteq Female \\
            Father & \sqsubseteq Parent \\
            Mother & \sqsubseteq Parent \\
            Female \sqcap Male & \sqsubseteq \perp \\
            Female \sqcap Parent & \sqsubseteq Mother \\
            Male \sqcap Parent & \sqsubseteq Father \\
            \exists hasChild.Person & \sqsubseteq Parent\\
            Parent & \sqsubseteq Person \\
            Parent & \sqsubseteq \exists hasChild.\top
        \end{align}

    """

    def __init__(self, url=None):
        super().__init__(url=DATA_URL if not url else url)
        self._evaluation_classes = None
        self._loaded_eval_data = False

    @property
    def evaluation_classes(self):
        return self.classes
