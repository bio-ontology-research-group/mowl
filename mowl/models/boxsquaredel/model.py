from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import BoxSquaredELModule


class BoxSquaredEL(EmbeddingELModel):
    """
    Implementation based on [jackermeier2023]_.
    """

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0.02,
                 reg_norm=1,
                 learning_rate=0.001,
                 batch_size=4096 * 8,
                 delta=2.5,
                 reg_factor=0.2,
                 num_negs=4,
                 model_filepath=None,
                 device='cpu'
                 ):
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate)

        self.margin = margin
        self.reg_norm = reg_norm
        self.delta = delta
        self.reg_factor = reg_factor
        self.num_negs = num_negs
        self._loaded = False
        self.extended = False
        self.init_module()

    def init_module(self):
        self.module = BoxSquaredELModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            len(self.individual_index_dict),
            embed_dim=self.embed_dim,
            gamma=self.margin,
            delta=self.delta,
            reg_factor=self.reg_factor
        ).to(self.device)

