from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import ELEmModule


class ELEmbeddings(EmbeddingELModel):
    """
    Implementation based on [kulmanov2019]_.

    The idea of this paper is to embed EL by modeling ontology classes as :math:`n`-dimensional \
    balls (:math:`n`-balls) and ontology object properties as transformations of those \
    :math:`n`-balls. For each of the normal forms, there is a distance function defined that will \
    work as loss functions in the optimization framework.
    """

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0,
                 reg_norm=1,
                 learning_rate=0.001,
                 batch_size=4096 * 8,
                 model_filepath=None,
                 device='cpu'
                 ):
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate)

        self.margin = margin
        self.reg_norm = reg_norm
        self._loaded = False
        self.extended = False
        self.init_module()

    def init_module(self):
        self.module = ELEmModule(
            len(self.class_index_dict),  # number of ontology classes
            len(self.object_property_index_dict),  # number of ontology object properties
            len(self.individual_index_dict),  # number of individuals
            embed_dim=self.embed_dim,
            margin=self.margin
        ).to(self.device)

