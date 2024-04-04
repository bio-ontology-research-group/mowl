nimport mowl.nn.el.boxel.losses as L
from mowl.nn import ELModule
import torch as th
import torch.nn as nn
from torch.distributions import uniform
import torch.nn.functional as F

eps = 1e-8

class Box:
    def __init__(self, min_embed, delta_embed):
        self.min_embed = min_embed
        self.delta_embed = th.exp(delta_embed)
        self.max_embed = min_embed + self.delta_embed
        
    def l2_side_regularizer(self, log_scale: bool=True):
         min_x = self.min_embed
         delta_x = self.delta_embed
         if not log_scale:
             return th.mean(delta_x ** 2)
         else:
             return th.mean(F.relu(min_x + delta_x - 1 + eps )) +  th.mean(F.relu(-min_x - eps))


        
class BoxELModule(ELModule):
    """Implementation of BoxEL from [xiong2022]_.
    """
    def __init__(self, nb_ont_classes, nb_rels, embed_dim=50,
                 min_bounds=[1e-4, 0.2], delta_bounds=[-0.1, 0],
                 relation_bounds=[-0.1, 0.1],
                 scaling_bounds=[0.9,1.1], temperature=1.0):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        self.min_embedding = self.init_entity_embedding(nb_ont_classes, embed_dim, min_bounds)
        self.delta_embedding = self.init_entity_embedding(nb_ont_classes, embed_dim, delta_bounds)
        self.relation_embedding = self.init_entity_embedding(nb_rels, embed_dim, relation_bounds)
        self.scaling_embedding = self.init_entity_embedding(nb_rels, embed_dim, relation_bounds)
        

    def init_entity_embedding(self, num_entities, embed_dim, bounds):
        distribution = uniform.Uniform(bounds[0], bounds[1])
        box_embed = distribution.sample((num_entities, embed_dim))
        return box_embed


    
    def gci0_loss(self, data, neg=False):
        return L.gci0_loss(data, self.min_embedding, self.delta_embedding, self.temperature, neg=neg)
pp
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
