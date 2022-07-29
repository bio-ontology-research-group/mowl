import mowl.models.elembeddings.losses as L
from mowl.nn.elmodule import ELModule
from torch import nn
import torch as th

class ELEmModule(ELModule):

    def __init__(self, nb_ont_classes, nb_rels, embed_dim=50, margin=0.1, reg_norm = 1):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.reg_norm = reg_norm
        self.embed_dim = embed_dim

        self.class_embed = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_embed.weight, a=-1, b=1)
        self.class_embed.weight.data /= th.linalg.norm(self.class_embed.weight.data,axis=1).reshape(-1,1)

        self.class_rad = nn.Embedding(self.nb_ont_classes, 1)
        nn.init.uniform_(self.class_rad.weight, a=-1, b=1)
        self.class_rad.weight.data /= th.linalg.norm(self.class_rad.weight.data,axis=1).reshape(-1,1)
        
        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)
        self.rel_embed.weight.data /= th.linalg.norm(self.rel_embed.weight.data,axis=1).reshape(-1,1)

        self.margin = margin

    def class_reg(self, x):
        res = th.abs(th.linalg.norm(x, axis=1) - self.reg_norm) #force n-ball to be inside unit ball
        res = th.reshape(res, [-1, 1])
        return res
    
    def gci0_loss(self, data, neg = False):
        return L.gci0_loss(data, self.class_embed, self.class_rad, self.class_reg, self.margin, neg = neg)
    def gci1_loss(self, data, neg = False):
        return L.gci1_loss(data, self.class_embed, self.class_rad, self.class_reg, self.margin, neg = neg)
    def gci1_bot_loss(self, data, neg = False):
        return L.gci1_bot_loss(data, self.class_embed, self.class_rad, self.class_reg, self.margin, neg = neg)
    def gci2_loss(self, data, neg = False, idxs_for_negs = None):
        return L.gci2_loss(data, self.class_embed, self.class_rad, self.rel_embed, self.class_reg, self.margin, neg = neg)
    def gci3_loss(self, data, neg = False):
        return L.gci3_loss(data, self.class_embed, self.class_rad, self.rel_embed, self.class_reg, self.margin, neg = neg)









