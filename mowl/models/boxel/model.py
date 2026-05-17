from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import BoxELModule


class BoxEL(EmbeddingELModel):
    """
    Implementation based on [xiong2022]_.

    This model uses box embeddings where each class is represented as an
    axis-aligned box defined by a minimum point and a delta (size) vector.
    """

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 min_bounds=(1e-4, 0.2),
                 delta_bounds=(-0.1, 0),
                 relation_bounds=(-0.1, 0.1),
                 scaling_bounds=(0.9, 1.1),
                 temperature=1.0,
                 learning_rate=0.001,
                 batch_size=4096 * 8,
                 model_filepath=None,
                 device='cpu',
                 neg_sampling_gcis=None
                 ):
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate,
                         neg_sampling_gcis=neg_sampling_gcis)

        self.min_bounds = list(min_bounds)
        self.delta_bounds = list(delta_bounds)
        self.relation_bounds = list(relation_bounds)
        self.scaling_bounds = list(scaling_bounds)
        self.temperature = temperature
        self.init_module()

    def init_module(self):
        self.module = BoxELModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            nb_inds=len(self.individual_index_dict) or None,
            embed_dim=self.embed_dim,
            min_bounds=self.min_bounds,
            delta_bounds=self.delta_bounds,
            relation_bounds=self.relation_bounds,
            scaling_bounds=self.scaling_bounds,
            temperature=self.temperature,
        ).to(self.device)
