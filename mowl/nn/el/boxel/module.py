import mowl.nn.el.boxel.losses as L
from mowl.nn import ELModule
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class BoxELModule(ELModule):
    """Implementation of BoxEL from [xiong2022]_.

    .. note::
        `Original implementation: <https://github.com/Box-EL/BoxEL>`_

    """
    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50,
                 min_bounds=[1e-4, 0.2], delta_bounds=[-0.1, 0],
                 relation_bounds=[-0.1, 0.1],
                 scaling_bounds=[0.9,1.1], temperature=1.0):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        self.min_embedding = self.init_entity_embedding(nb_ont_classes, embed_dim, min_bounds)
        self.delta_embedding = self.init_entity_embedding(nb_ont_classes, embed_dim, delta_bounds)
        self.relation_embedding = self.init_entity_embedding(nb_rels, embed_dim, relation_bounds)
        self.scaling_embedding = self.init_entity_embedding(nb_rels, embed_dim, scaling_bounds)

        if self.nb_inds is not None:
            self.ind_embedding = self.init_entity_embedding(nb_inds, embed_dim, min_bounds)
        else:
            self.ind_embedding = None
        

    def init_entity_embedding(self, num_entities, embed_dim, bounds):
        embed = nn.Embedding(num_entities, embed_dim)
        nn.init.uniform_(embed.weight, bounds[0], bounds[1])
        return embed


    
    def gci0_loss(self, data, neg=False):
        return L.gci0_loss(data, self.min_embedding, self.delta_embedding, self.temperature, neg=neg)

    def gci0_bot_loss(self, data, neg=False):
        return L.gci0_bot_loss(data, self.min_embedding, self.delta_embedding, self.temperature, neg=neg)

    def gci1_loss(self, data, neg=False):
        return L.gci1_loss(data, self.min_embedding, self.delta_embedding, self.temperature, neg=neg)

    def gci1_bot_loss(self, data, neg=False):
        return L.gci1_bot_loss(data, self.min_embedding, self.delta_embedding, self.temperature, neg=neg)

    def gci2_loss(self, data, neg=False):
        return L.gci2_loss(data, self.min_embedding, self.delta_embedding, self.relation_embedding, self.scaling_embedding, self.temperature, neg=neg)

    def gci3_loss(self, data, neg=False):
        return L.gci3_loss(data, self.min_embedding, self.delta_embedding, self.relation_embedding, self.scaling_embedding, self.temperature, neg=neg)

    def gci3_bot_loss(self, data, neg=False):
        return L.gci3_bot_loss(data, self.min_embedding, self.delta_embedding, self.relation_embedding, self.scaling_embedding, self.temperature, neg=neg)

    def regularization_loss(self):
        return L.regularization_loss(self.min_embedding, self.delta_embedding)
