
from mowl.nn import BoxSquaredELModule
from mowl.base_models.elmodel import EmbeddingELModel

import torch as th
from torch import nn


class BoxSquaredEL(EmbeddingELModel):
    """
    Implementation based on [peng2020]_.
    """

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0.02,
                 reg_norm=1,
                 learning_rate=0.001,
                 epochs=1000,
                 batch_size=4096 * 8,
                 delta=2.5,
                 reg_factor=0.2,
                 num_negs=4,
                 model_filepath=None,
                 device='cpu'
                 ):
        super().__init__(dataset, embed_dim, batch_size, extended=True, model_filepath=model_filepath)

        
        self.margin = margin
        self.reg_norm = reg_norm
        self.delta = delta
        self.reg_factor = reg_factor
        self.num_negs = num_negs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self._loaded = False
        self.extended = False
        self.init_module()

    def init_module(self):
        self.module = BoxSquaredELModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            embed_dim=self.embed_dim,
            gamma=self.margin,
            delta=self.delta,
            reg_factor=self.reg_factor

        ).to(self.device)

    def train(self):
        raise NotImplementedError

                                                                                                
    def eval_method(self, data):
        return self.module.gci2_score(data)

    def get_embeddings(self):
        self.init_module()

        print('Load the best model', self.model_filepath)
        self.load_best_model()
                
        ent_embeds = {k: v for k, v in zip(self.class_index_dict.keys(),
                                           self.module.class_embed.weight.cpu().detach().numpy())}
        rel_embeds = {k: v for k, v in zip(self.object_property_index_dict.keys(),
                                           self.module.rel_embed.weight.cpu().detach().numpy())}
        return ent_embeds, rel_embeds

    def load_best_model(self):
        self.init_module()
        self.module.load_state_dict(th.load(self.model_filepath))
        self.module.eval()

